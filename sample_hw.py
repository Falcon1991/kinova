##
#
# Use the sample program on the Kinova arm. This allows us to use
# the camera_viewer class to crop the point cloud to only view
# the target object and determine the target.
#
##

from pydrake.all import *
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk

from kinova_station import KinovaStationHardwareInterface, EndEffectorTarget
from controllers import CommandSequenceController, CommandSequence, Command, DepthController
from observers.camera_viewer_hw import CameraViewer
from meshcat.servers.zmqserver import start_zmq_server_as_subprocess

########################### Parameters #################################

# Make a plot of the inner workings of the station
show_station_diagram = False

# Make a plot of the diagram for this example, where only the inputs
# and outputs of the station are shown
show_toplevel_diagram = False

########################################################################

with KinovaStationHardwareInterface() as station:

    if show_station_diagram:
        # Show the station's system diagram
        plt.figure()
        plot_system_graphviz(station,max_depth=1)
        plt.show()

    # Start assembling the overall system diagram
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    station = builder.AddSystem(station)

    # Create the controller and connect inputs and outputs appropriately
    #Kp = 1*np.diag([100, 100, 100, 200, 200, 200])  # high gains needed to overcome
    #Kd = 2*np.sqrt(0.5*Kp)                          # significant joint friction
    #.07
    Kp = .07*np.diag([100, 100, 100, 200, 200, 200])
    Kd = 3*np.sqrt(.5*Kp)

    # Custom start sequence
    start_sequence = CommandSequence([])
    start_sequence.append(Command(
        name="front_view",
        target_pose=np.array([0.7*np.pi, 0.0, 0.5*np.pi, 0.5, 0.0, 0.15]),
        duration=1,
        gripper_closed=False))
    start_sequence.append(Command(
        name="left_view",
        target_pose=np.array([0.7*np.pi, 0.0, 0.3*np.pi, 0.6, 0.1, 0.15]),
        duration=1,
        gripper_closed=False))
    start_sequence.append(Command(
        name="front_view",
        target_pose=np.array([0.7*np.pi, 0.0, 0.5*np.pi, 0.5, 0.0, 0.15]),
        duration=1,
        gripper_closed=False))
    start_sequence.append(Command(
        name="right_view",
        target_pose=np.array([0.7*np.pi, 0.0, 0.8*np.pi, 0.6, -0.3, 0.15]),
        duration=1,
        gripper_closed=False))
    start_sequence.append(Command(
        name="home",
        target_pose=np.array([0.5*np.pi, 0.0, 0.5*np.pi, 0.5, 0.0, 0.2]),
        duration=1,
        gripper_closed=False))
    
    cs = CommandSequence([])
    cs.append(Command(
        name="stay_still",
        target_pose=np.array([.55*np.pi, 0.0, .5*np.pi, .5, 0.0, .1]),
        duration=3,
        gripper_closed=False))

    controller = builder.AddSystem(DepthController(
        start_sequence=cs,
        command_type=EndEffectorTarget.kWrench,  # wrench commands work best on hardware
        show_candidate_grasp=True,
        hardware=True,
        Kp=Kp,
        Kd=Kd))

    controller.set_name("controller")
    controller.ConnectToStation(builder, station)
    
    # Setup camera_viewer 
    camera_viewer = builder.AddSystem(CameraViewer())
    camera_viewer.set_name("camera_viewer")

    builder.Connect(
        station.GetOutputPort("camera_rgb_image"),
        camera_viewer.GetInputPort("color_image"))
    builder.Connect(
        station.GetOutputPort("camera_depth_image"),
        camera_viewer.GetInputPort("depth_image"))
    

    # Add converter from depth images to point clouds
    point_cloud_generator = builder.AddSystem(DepthImageToPointCloud(
                                    CameraInfo(width=480, height=270, fov_y=np.radians(40)),
                                    pixel_type=PixelType.kDepth16U,
                                    scale=0.001,
                                    fields=BaseField.kXYZs))
    point_cloud_generator.set_name("point_cloud_generator")

    builder.Connect(
            camera_viewer.GetOutputPort("cropped_depth_image"),
            #station.GetOutputPort("camera_depth_image"),
            point_cloud_generator.depth_image_input_port())
    builder.Connect(
            station.GetOutputPort("camera_transform"),
            point_cloud_generator.GetInputPort("camera_pose"))

    # Send generated point cloud and camera transform to the controller
    builder.Connect(
            point_cloud_generator.point_cloud_output_port(),
            controller.GetInputPort("point_cloud"))
    builder.Connect(
            station.GetOutputPort("camera_transform"),
            controller.GetInputPort("camera_transform"))

    # Connect meshcat visualizer
    #proc, zmq_url, web_url = start_zmq_server_as_subprocess()  # start meshcat from here
    # Alternative: start meshcat (in drake dir) with bazel run @meshcat_python//:meshcat-server
    zmq_url = "tcp://127.0.0.1:6000"

    meshcat = ConnectMeshcatVisualizer(
            builder=builder, 
            scene_graph=scene_graph,
            zmq_url=zmq_url)

    meshcat_point_cloud = builder.AddSystem(MeshcatPointCloudVisualizer(
                                                meshcat,
                                                draw_period=0.2))
    meshcat_point_cloud.set_name("point_cloud_viz")
    builder.Connect(
            point_cloud_generator.point_cloud_output_port(),
            meshcat_point_cloud.get_input_port())

    # Build the system diagram
    diagram = builder.Build()
    diagram.set_name("system_diagram")
    plant.Finalize()
    diagram_context = diagram.CreateDefaultContext()

    if show_toplevel_diagram:
        # Show the overall system diagram
        plt.figure()
        plot_system_graphviz(diagram,max_depth=1)
        plt.show()

    # Set default arm positions
    #station.go_home(name="Home")
    station.send_pose_command((.55*np.pi, 0.0, .5*np.pi, .5, 0.0, .1))

    """
    # Set up simulation
    simulator = Simulator(diagram, diagram_context)
    simulator.set_target_realtime_rate(1.0)
    simulator.set_publish_every_time_step(False)

    integration_scheme = "explicit_euler"
    time_step = 0.10
    ResetIntegratorFromFlags(simulator, integration_scheme, time_step)

    # Run simulation
    simulator.Initialize()
    simulator.AdvanceTo(30.0)
    
    here, count = camera_viewer.isHere()
    if count == 0:
        # Look left. Is the object there?
        com = Command(
                name="look_left",
                target_pose=np.array([.55*np.pi, 0.0, .9*np.pi, .4, 0.1, .1]),
                duration=3,
                gripper_closed=False)
        controller.AppendMovement(com)
    
    # Check if the object is over here 
    if count != 0:
        # Look ahead. Is the object there?
        com = Command(
                name="look_ahead",
                target_pose=np.array([.7*np.pi, 0.0, .5*np.pi, .5, 0.0, .2]),
                duration=3,
                gripper_closed=False)
        controller.AppendMovement(com)
    
        # Check if the object is here
        here, count = camera_viewer.isHere()

        if not here:
            # Look right. Is the object there?
            com = Command(
                    name="look_right",
                    target_pose=np.array([.7*np.pi, 0.0, .3*np.pi, .4, -0.1, .1]),
                    duration=3,
                    gripper_closed=False)
            controller.AppendMovement(com)
    
    # If the object is found, find grasp
    if here:
        com = Command(
                name="look_right",
                target_pose=np.array([.7*np.pi, 0.0, .5*np.pi, .4, 0.0, .1]),
                duration=0,
                gripper_closed=False)
       controller.AppendMovement(com)
    """
    
    # Set up simulation
    simulator = Simulator(diagram, diagram_context)
    simulator.set_target_realtime_rate(1.0)
    simulator.set_publish_every_time_step(False)

    integration_scheme = "explicit_euler"
    time_step = 0.10
    ResetIntegratorFromFlags(simulator, integration_scheme, time_step)

    # Run simulation
    simulator.Initialize()
    simulator.AdvanceTo(30.0)
    

    # Print rate data
    print("")
    print("Target control frequency: %s Hz" % (1/time_step))
    print("Actual control frequency: %s Hz" % (1/time_step * simulator.get_actual_realtime_rate()))
