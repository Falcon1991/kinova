#from home.discoverlab.kinova_drake.controllers.command_sequence import CommandSequence
from controllers.command_sequence_controller import *
from kinova_station.common import draw_open3d_point_cloud, draw_points

import open3d as o3d
from scipy.optimize import differential_evolution

class DepthControllerHw(CommandSequenceController):
    """
    A controller which uses point cloud data to plan
    and execute a grasp. 
    """
    def __init__(self, start_sequence=None, 
                       command_type=EndEffectorTarget.kTwist, 
                       Kp=10*np.eye(6), Kd=2*np.sqrt(10)*np.eye(6),
                       show_candidate_grasp=False,
                       hardware=False):
        """
        Parameters:

            start_sequence       : a CommandSequence object for moving around and building up
                                   a point cloud. 

            command_type         : the type of command that we'll send (kTwist or kWrench)

            Kp/Kd                : PD gains

            show_candidate_grasp : whether or not to display candidate grasps over meshcat each
                                   time the grasp cost function is evaluated. 

            hardware             : whether we're applying this on hardware (simulation default)
        """
        self.hardware = hardware

        if start_sequence is None:
            # Create a default starting command sequence for moving around and
            # building up the point cloud
            start_sequence = CommandSequence([])
            start_sequence.append(Command(
                name="front_view",
                target_pose=np.array([0.7*np.pi, 0.0, 0.5*np.pi, 0.5, 0.0, 0.15]),
                duration=3,
                gripper_closed=False))
            start_sequence.append(Command(
                name="left_view",
                target_pose=np.array([0.7*np.pi, 0.0, 0.3*np.pi, 0.6, 0.1, 0.15]),
                duration=3,
                gripper_closed=False))
            start_sequence.append(Command(
                name="front_view",
                target_pose=np.array([0.7*np.pi, 0.0, 0.5*np.pi, 0.5, 0.0, 0.15]),
                duration=3,
                gripper_closed=False))
            start_sequence.append(Command(
                name="right_view",
                target_pose=np.array([0.7*np.pi, 0.0, 0.8*np.pi, 0.6, -0.3, 0.15]),
                duration=3,
                gripper_closed=False))
            start_sequence.append(Command(
                name="home",
                target_pose=np.array([0.5*np.pi, 0.0, 0.5*np.pi, 0.5, 0.0, 0.2]),
                duration=3,
                gripper_closed=False))

        # Initialize the underlying command sequence controller
        CommandSequenceController.__init__(self, start_sequence, 
                                            command_type=command_type, Kp=Kp, Kd=Kd)

        # Create an additional input port for the point cloud
        self.point_cloud_input_port = self.DeclareAbstractInputPort(
                "point_cloud",
                AbstractValue.Make(PointCloud()))

        # Create an additional input port for the camera pose
        self.camera_transform_port = self.DeclareAbstractInputPort(
                "camera_transform",
                AbstractValue.Make(RigidTransform()))

        # Recorded point clouds from multiple different views
        self.stored_point_clouds = []
        self.merged_point_cloud = None

        # Drake model with just a floating gripper, used to evaluate grasp candidates
        builder = DiagramBuilder()
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)

        gripper_urdf = "./models/hande_gripper/urdf/robotiq_hande_static.urdf"
        self.gripper = Parser(plant=self.plant).AddModelFromFile(gripper_urdf, "gripper")

        self.plant.RegisterCollisionGeometry(  # add a flat ground that we can collide with
                self.plant.world_body(),
                RigidTransform(), HalfSpace(), 
                "ground_collision",
                CoulombFriction())

        # Connect to meshcat so we can show this floating gripper
        self.show_candidate_grasp = show_candidate_grasp
        if self.show_candidate_grasp:
            self.meshcat = ConnectMeshcatVisualizer(builder=builder, 
                                               zmq_url="tcp://127.0.0.1:6000",
                                               scene_graph=self.scene_graph,
                                               output_port=self.scene_graph.get_query_output_port(),
                                               prefix="candidate_grasp")
            self.meshcat.load()
        
        self.count = 4
        self.reverse = False
        self.plant.Finalize()
        self.diagram = builder.Build()
        self.diagram_context = self.diagram.CreateDefaultContext()
        self.plant_context = self.diagram.GetMutableSubsystemContext(self.plant, self.diagram_context)
        self.scene_graph_context = self.scene_graph.GetMyContextFromRoot(self.diagram_context)

    def StorePointCloud(self, point_cloud, camera_position):
        """
        Add the given Drake point cloud to our list of point clouds. 

        Converts to Open3D format, crops, and estimates normals before adding
        to self.stored_point_clouds.
        """
        # Convert to Open3D format
        indices = np.all(np.isfinite(point_cloud.xyzs()), axis=0)
        o3d_cloud = o3d.geometry.PointCloud()
        o3d_cloud.points = o3d.utility.Vector3dVector(point_cloud.xyzs()[:, indices].T)
        print(o3d_cloud)
        if point_cloud.has_rgbs():
            o3d_cloud.colors = o3d.utility.Vector3dVector(point_cloud.rgbs()[:, indices].T / 255.)

        # Crop to relevant area
        # x_min = 0.0; x_max = 1.0
        # y_min = -0.2; y_max = 0.2
        # z_min = 0.05; z_max = 0.3
        # o3d_cloud = o3d_cloud.crop(o3d.geometry.AxisAlignedBoundingBox(
        #                                         min_bound=[x_min, y_min, z_min],
        #                                         max_bound=[x_max, y_max, z_max]))

        try:
            # Estimate normals
            o3d_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
            o3d_cloud.orient_normals_towards_camera_location(camera_position)

            # Save
            self.stored_point_clouds.append(o3d_cloud)

        except RuntimeError:
            # We were unable to compute normals for this frame, so we'll just skip it. 
            # The most likely reason for this is simply that all the points were outside
            # the cropped region.
            pass

        print(self.stored_point_clouds[0])
 
    def AppendMovement(self, com):
        """
        Edit the command sequence to add an immediate movement
        """
        self.cs.clear()
        self.cs.append(com)

    def AppendPickupToStoredCommandSequence(self, grasp):
        """
        Given a viable grasp location, modify the stored command sequence to 
        include going to that grasp location and picking up the object. 
        """
        # we need to translate target grasps from the end_effector_link frame (G, at the wrist)
        # used to specify grasp poses and the end_effector frame (E, at fingertips)
        # associated with end-effector commands. This is slightly different on the hardware
        # and in simulation. 
        X_WG = RigidTransform(             
                RotationMatrix(RollPitchYaw(grasp[:3])),
                grasp[3:])
        if self.hardware:
            X_GE = RigidTransform(
                    RotationMatrix(np.eye(3)),
                    np.array([0,0,0.18]))
        else:
            X_GE = RigidTransform(
                    RotationMatrix(np.eye(3)),
                    np.array([0,0,0.13]))
        X_WE = X_WG.multiply(X_GE)
        grasp = np.hstack([RollPitchYaw(X_WE.rotation()).vector(), X_WE.translation()])

        # Compute a pregrasp location that is directly behind the grasp location
        X_WG = RigidTransform(             
                RotationMatrix(RollPitchYaw(grasp[:3])),
                grasp[3:])
        X_GP = RigidTransform(
                RotationMatrix(np.eye(3)),
                np.array([0,0,-0.1]))
        X_WP = X_WG.multiply(X_GP)
        pregrasp = np.hstack([RollPitchYaw(X_WP.rotation()).vector(), X_WP.translation()])

        self.cs.append(Command(
            name="pregrasp",
            target_pose=pregrasp,
            duration=4,
            gripper_closed=False))
        self.cs.append(Command(
            name="grasp",
            target_pose=grasp - np.array([0,0,0,.1,0,0]),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
            duration=3,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
            gripper_closed=False))
        self.cs.append(Command(
            name="grasp2",
            target_pose=grasp + np.array([0,0,0,.05,0,-.05]),
            duration=1,
            gripper_closed=False))
        self.cs.append(Command(
            name="close_gripper",
            target_pose=grasp,
            duration=0.5,
            gripper_closed=True))
        self.cs.append(Command(
            name="lift",
            target_pose = grasp + np.array([0,0,0,0,0,0.1]),
            duration=2,
            gripper_closed=True))

    def GenerateGraspCandidate(self, cloud=None):
        """
        Use some simple heuristics to generate a reasonable-ish candidate grasp
        """
        if cloud is None:
            cloud = self.merged_point_cloud

        # Pick a random point on the point cloud
        index = np.random.randint(0, len(cloud.points))

        p_WS = np.asarray(cloud.points[index])  # position of the [S]ample point in the [W]orld
        n_WS = np.asarray(cloud.normals[index])

        # Create a gripper pose consistent with this point
        y = np.array([0., 0., -1.])
        Gx = n_WS
        Gy = y - np.dot(y, Gx)*Gx
        Gz = np.cross(Gx, Gy)
        R_WG = RotationMatrix(np.vstack([Gx, Gy, Gz]).T)

        # Rotate the grasp angle 180 degrees. This seems to lead to upside-down grasps
        # less often. Note that this could be randomized as well.
        R_WG = R_WG.multiply(RotationMatrix(RollPitchYaw([-np.pi,0,0])))

        p_GS_G = np.array([0.02,0,0.13])   # position of the sample in the gripper frame
        p_SG_W = -R_WG.multiply(p_GS_G)
        p_WG = p_WS + p_SG_W

        ee_pose = np.hstack([RollPitchYaw(R_WG).vector(), p_WG])

        return ee_pose

    def ScoreGraspCandidate(self, ee_pose, cloud=None):
        """
        For the given point cloud (merged, downsampled, with normals) and
        end-effector pose corresponding to a candidate grasp, return the
        score associated with this grasp. 
        """
        cost = 0

        if cloud is None:
            cloud = self.merged_point_cloud

        # Set the pose of our internal gripper model
        gripper = self.plant.GetBodyByName("hande_base_link")
        R_WG = RotationMatrix(RollPitchYaw(ee_pose[:3]))
        X_WG = RigidTransform(
                R_WG, 
                ee_pose[3:])
        self.plant.SetFreeBodyPose(self.plant_context, gripper, X_WG)

        # Transform the point cloud to the gripper frame
        X_GW = X_WG.inverse()
        pts = np.asarray(cloud.points).T
        p_GC = X_GW.multiply(pts)

        # Select the points that are in between the fingers
        crop_min = [-0.025, -0.01, 0.12]
        crop_max = [0.025, 0.01, 0.14]
        indices = np.all((crop_min[0] <= p_GC[0,:], p_GC[0,:] <= crop_max[0],
                          crop_min[1] <= p_GC[1,:], p_GC[1,:] <= crop_max[1],
                          crop_min[2] <= p_GC[2,:], p_GC[2,:] <= crop_max[2]),
                         axis=0)
        p_GC_between = p_GC[:,indices]

        # Compute normals for those points between the fingers
        n_GC_between = X_GW.rotation().multiply(np.asarray(cloud.normals)[indices,:].T)

        # Reward normals that are alligned with the gripper
        cost -= np.sum(n_GC_between[0,:]**2)

        # Penalize collisions between the point cloud and the gripper
        self.diagram.Publish(self.diagram_context)   # updates scene_graph_context
        query_object = self.scene_graph.get_query_output_port().Eval(self.scene_graph_context)

        for pt in cloud.points:
            # Compute all distances from the gripper to the point cloud, ignoring any
            # that are over 0
            distances = query_object.ComputeSignedDistanceToPoint(pt, threshold=0)
            if distances:
                # Any (negative) distance found indicates that we're in collision, so
                # the resulting cost is infinite
                cost = np.inf

        # Penalize collisions between the gripper and the ground
        if query_object.HasCollisions():
            cost = np.inf

        ## Penalize deviations from a nominal orientation
        #rpy_nom = np.array([0.75, 0, 0.5])*np.pi
        #R_nom = RotationMatrix(RollPitchYaw(rpy_nom))
        #R_diff = R_WG.multiply(R_nom.transpose())
        #theta = np.arccos( (np.trace(R_diff.matrix()) - 1)/2 )  # angle between current and desired rotation

        #cost += 1*(theta**2)

        # Visualize the candidate grasp point with meshcat
        if self.show_candidate_grasp:

            # Draw the point cloud 
            v = self.meshcat.vis["merged_point_cloud"]
            draw_open3d_point_cloud(v, cloud, normals_scale=0.01)

            # Highlight the points on the point cloud that are between the
            # grippers
            v = self.meshcat.vis["grip_location"]
            p_WC_between = X_WG.multiply(p_GC_between)
            draw_points(v, p_WC_between, [1.,0.,0.], size=0.01)  # Red points
       
        return cost

    def FindGrasp(self, seed=None):
        """
        Use a genetic algorithm to find a suitable grasp.
        """
        print("===> Searching for a suitable grasp...")
        assert self.merged_point_cloud is not None, "Merged point cloud must be created before finding a grasp"

        # Generate several semi-random candidate grasps
        np.random.seed(seed)
        grasps = []
        for i in range(10):
            grasps.append(self.GenerateGraspCandidate())

        # Use a genetic algorithm to find a locally optimal grasp
        bounds = [(-2*np.pi,2*np.pi),
                  (-2*np.pi,2*np.pi),
                  (-2*np.pi,2*np.pi),
                  (-0.7, 0.7),
                  (-0.7, 0.7),
                  (0.0, 1.0)]
        init = np.array(grasps)
        res = differential_evolution(self.ScoreGraspCandidate, bounds, init=init)

        if res.success and res.fun < 0:
            print(res)
            print("===> Found locally optimal grasp with cost %s" % res.fun)
            return res.x
        else:
            print("===> Failed to converge to an optimal grasp: retrying.")
            return self.FindGrasp()

    def locateObject(self, cloud=None):
        """
        This method orients the gripper to align with the object and
        moves the gripper to line up with the object. This is done using an 
        algorithm to find the sides and the average values of the points and 
        normals of each side. Then it will travel .1 to the object and grab 
        it. Some trigonometry is used here to calculate the gripper poition 
        based on the point cloud.
        """

        if cloud is None:
            cloud = self.merged_point_cloud

        cloud = self.merged_point_cloud
        print(cloud)

        

        # Compare each point with all other points to group them together in similar groups of points and normals
        cluster = []
        cluster_p = [] 
        c = 0
        for i in range(len(cloud.points)):
            n1_WS = np.asarray(cloud.normals[i])
            p1_WS = np.asarray(cloud.points[i])
            c = 0
            for j in range(len(cloud.points)):
                n2_WS = np.asarray(cloud.normals[j])
                # The < .2 is a larger tolerance due to the error caused by the real world and the hardware
                close0 = (math.fabs(n1_WS[0]-n2_WS[0]) < .2)
                close1 = (math.fabs(n1_WS[1]-n2_WS[1]) < .2)
                close2 = (math.fabs(n1_WS[2]-n2_WS[2]) < .2)    
                if close0:
                    if close1:
                        if close2:
                            c += 1
            # A group is valid if it is a portion of the total points. A side should be large
            if c > (len(cloud.normals) / 8):
                cluster.append(n1_WS)
                cluster_p.append(p1_WS)

        # The groups are consolidated to represent the sides that are detected
        clusters = []
        clusters_p = []
        while len(cluster) > 0:
            temp = []
            temp_p = []
            ex = cluster[0]
            temp.append(ex)
            temp_p.append(cluster_p[0])
            del cluster[0]
            del cluster_p[0]
            toRemove = []
            for i in range(len(cluster)):
                act = cluster[i]
                close0 = (math.fabs(ex[0]-act[0]) < .1)
                close1 = (math.fabs(ex[1]-act[1]) < .1)
                close2 = (math.fabs(ex[2]-act[2]) < .1)    
                if close0:
                    if close1:
                        if close2:
                            temp.append(act)
                            temp_p.append(cluster_p[i])
                            toRemove.append(i)


            numDeleted = 0
            for i in range(len(toRemove)):
                del cluster[toRemove[i] - numDeleted]
                del cluster_p[toRemove[i] - numDeleted]
                numDeleted += 1
            clusters.append(temp)
            clusters_p.append(temp_p)

        # Average the normals and points of each side
        sides = []
        sides_p = []
        for i in range(len(clusters)):
            side = [0, 0, 0]
            side_p = [0, 0, 0]
            for j in range(len(clusters[i])):
                side[0] += clusters[i][j][0]
                side[1] += clusters[i][j][1]
                side[2] += clusters[i][j][2]

                side_p[0] += clusters_p[i][j][0]
                side_p[1] += clusters_p[i][j][1]
                side_p[2] += clusters_p[i][j][2]

            side[0] /= len(clusters[i])
            side[1] /= len(clusters[i])
            side[2] /= len(clusters[i])

            side_p[0] /= len(clusters_p[i])
            side_p[1] /= len(clusters_p[i])
            side_p[2] /= len(clusters_p[i])
            sides.append(side)
            sides_p.append(side_p)

        # Print the average points and normals of the detected sides
        print("Sides: ", sides)
        print("Point Sides: ", sides_p)
        found = False

        # Search for point to use to find grasp
        for i in range(len(sides)):
            n_WS = np.asarray(sides[i])
            p_WS = np.asarray(sides_p[i])
            if n_WS[1] > .01 and n_WS[1] < (np.pi - .01):
                break

        print("This is the point I am going for: ", p_WS)

        # Find the correct pitch/roll/yaw for the arm
        n_WS = (n_WS * [0, 0, 1]) + [.5*np.pi, 0, .8*np.pi]

        # Calculate the position for the arm to be .1 away from object and orient the arm using some trigonometry
        # X = p_WS[2] because the point cloud is translated for the hardware. So, X = Z
        Xo = p_WS[2]
        Yo = p_WS[1]
        Xg = .1 * math.sin(np.pi - n_WS[2])
        Yg = .1 * math.cos(np.pi - n_WS[2])
        X = Xo - Xg
        Y = Yo - Yg
        X1 = Xo + (Xg/3)
        Y1 = Yo + (Yg/3)


        # Orient gripper and move .1 straight away from the object
        # The Y+.2 is to correct the position on the hardware. Adjust as needed since the loop is open
        self.cs.append(Command(
            name="line_up",
            target_pose=np.hstack([n_WS, [X, Y+.2, .03]]),   # Change PitchRollYaw to vector of n_WS normal
            duration=2,
            gripper_closed=False))
        self.cs.append(Command(
            name="line_up2",
            target_pose=np.hstack([n_WS, [X, Y+.2, .03]]),   # The same command to kill time and make sure it is
            duration=2,                                      # in line with the object before moving further towards it
            gripper_closed=False))
        # Move the rest of the way towards the object and grab it
        self.cs.append(Command(
            name="move_towards",
            target_pose=np.hstack([n_WS, [X1, Y1+.2, .03]]),
            duration=2,
            gripper_closed=False))
        self.cs.append(Command(
            name="grip",
            target_pose=np.hstack([n_WS, [X1, Y1+.2, .03]]),
            duration=1,
            gripper_closed=True))
        # Move back to a position to confirm the object is picked up
        self.cs.append(Command(
            name="pick_up",
            target_pose=np.array([.5*np.pi, 0, .5*np.pi, .5, 0, .3]),
            duration=2,
            gripper_closed=True))

    def CalcEndEffectorCommand(self, context, output):
        """
        Compute and send an end-effector twist command.
        """

        t = context.get_time()
        if t < self.cs.total_duration() and len(self.stored_point_clouds) == 0:
            # Only fetch the point clouds infrequently, since this is slow
            point_cloud = self.point_cloud_input_port.Eval(context)

            # Convert to Open3D, crop, compute normals, and save
            X_camera = self.camera_transform_port.Eval(context)
            indices = np.all(np.isfinite(point_cloud.xyzs()), axis=0)
            
            # Only save the point cloud if there are points
            if " 0 " not in str(o3d.utility.Vector3dVector(point_cloud.xyzs()[:, indices].T)):
                self.StorePointCloud(point_cloud, X_camera.translation())
            
            # This is the current method to save space and add many different positions
            # Use this variable to determine number of positions of arm (ex: step = 1 means there are 10 steps in 180 degrees)
            step = 1

            # Count up or down depending on which position (left/right) the arm was in
            if t % .1 == 0 and not self.reverse:
                self.count += step
            elif t % .1 == 0:
                self.count -= step
            if self.count == 10:
                self.reverse == True
            elif self.count == 0:
                self.reverse == False

            # The command is adjusted every time step to conituously look left and right
            com = Command(
                    name="look_left/right",
                    target_pose=np.array([.55*np.pi, 0.0, (self.count/10)*np.pi, .4, 0, .1]),
                    duration=1,
                    gripper_closed=False)
            self.AppendMovement(com)

        elif self.merged_point_cloud is None:

            # Merge stored point clouds and downsample
            self.merged_point_cloud = self.stored_point_clouds[0]    # Just adding together may not
            for i in range(1, len(self.stored_point_clouds)):        # work very well on hardware...
                self.merged_point_cloud += self.stored_point_clouds[i]

            # This downsises the point cloud so that the algorithm to find the sides runs faster
            # Adjust the voxel_size as needed to increase potential accuracy (it was originall .005)
            self.merged_point_cloud = self.merged_point_cloud.voxel_down_sample(voxel_size=0.1)

            # Call a method to calculate the movements to pick up the object based on the point cloud
            self.locateObject()


        # Follow the command sequence stored in self.cs
        CommandSequenceController.CalcEndEffectorCommand(self, context, output)
