##
#
# First attempt at simulating the Kinova arm and getting
# the camera and depth outputs. The image is segmented to
# identify the target object.
#
#
#@misc{wu2019detectron2,
#  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
#                  Wan-Yen Lo and Ross Girshick},
#  title =        {Detectron2},
#  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
#  year =         {2019}
#}
#


##

# Some basic setup:

# import some common libraries
import os, json, cv2, random

from pydrake.all import *
import numpy as np
import matplotlib.pyplot as plt

from kinova_station import KinovaStation, EndEffectorTarget, GripperTarget
from controllers import CommandSequenceController, CommandSequence, Command, DepthController
from observers.camera_viewer import CameraViewer

# Parameters
show_station_diagram = False

show_toplevel_diagram = True

gripper_type = "hande"

######################################

n = np.zeros((1, 480, 270))
np.save('save.npy', n)

station = KinovaStation(time_step=.002)
station.SetupDualPegScenario(gripper_type=gripper_type, arm_damping=False, peg1_position=[.8, 0.1, .1], peg2_position=[.75,-0.1,0.1])
#station.SetupSinglePegScenario(gripper_type=gripper_type, arm_damping=False)
station.AddCamera()
station.ConnectToMeshcatVisualizer(start_server=False)

station.Finalize()



# Commands
cs = CommandSequence([])
cs.append(Command(
    name="FirstPos",
    target_pose=np.array([.5*np.pi, 0.0, .5*np.pi, .8, 0.0, .1]),
    duration=1,
    gripper_closed=True))
cs.append(Command(
    name="SecondPos",
    target_pose=np.array([.5*np.pi, 0.0, .5*np.pi, .8, 0.0, .1]),
    duration=1,
    gripper_closed=False))
cs.append(Command(
    name="ThirdPos",
    target_pose=np.array([.5*np.pi, 0.0, .5*np.pi, .8, 0.0, .1]),
    duration=1,
    gripper_closed=True))
cs.append(Command(
    name="Lift",
    target_pose=np.array([.5*np.pi, 0.0, .5*np.pi, .5, 0.0, .5]),
    duration=3,
    gripper_closed=True))
cs.append(Command(
    name="Setdown",
    target_pose=np.array([.5*np.pi, 1.0, .5*np.pi, 1.0, 0.0, .5]),
    duration=5,
    gripper_closed=True))
cs.append(Command(
    name="Setdown2",
    target_pose=np.array([.5*np.pi, 0.0, .5*np.pi, .7, 0.0, .1]),
    duration=5,
    gripper_closed=True))
cs.append(Command(
    name="LetGo",
    target_pose=np.array([.5*np.pi, 0.0, .5*np.pi, .7, 0.0, .1]),
    duration=1,
    gripper_closed=False))
cs.append(Command(
    name="MoveAway",
    target_pose=np.array([.5*np.pi, 0.0, .5*np.pi, .5, 0.0, .1]),
    duration=2,
    gripper_closed=False))


# Look around to find the 
c = CommandSequence([])
c.append(Command(
    name="SearchLeft",
    target_pose=np.array([.5*np.pi, 0.0, 1.0*np.pi, .5, 0.0, .1]),
    duration=2,
    gripper_closed=False))
c.append(Command(
    name="SearchRight",
    target_pose=np.array([.5*np.pi, 0.0, .1*np.pi, .5, 0.0, .1]),
    duration=4,
    gripper_closed=False))


builder = DiagramBuilder()
station = builder.AddSystem(station)

camera_viewer = builder.AddSystem(CameraViewer())
camera_viewer.set_name("camera_viewer")

builder.Connect(
        station.GetOutputPort("camera_rgb_image"),
        camera_viewer.GetInputPort("color_image"))
builder.Connect(
        station.GetOutputPort("camera_depth_image"),
        camera_viewer.GetInputPort("depth_image"))


# Add the controller
controller = builder.AddSystem(DepthController(start_sequence=None, show_candidate_grasp=False))

# Convert the depth image to a point cloud
# Note that this system block is slow
point_cloud_generator = builder.AddSystem(DepthImageToPointCloud(
                                    CameraInfo(width=480, height=270, fov_y=np.radians(40)),
                                    fields=BaseField.kXYZs | BaseField.kRGBs))
point_cloud_generator.set_name("point_cloud_generator")
builder.Connect(
#        station.GetOutputPort("camera_depth_image"),
        camera_viewer.GetOutputPort("cropped_depth_image"),
        point_cloud_generator.depth_image_input_port())
builder.Connect(
        station.GetOutputPort("camera_rgb_image"),
        point_cloud_generator.color_image_input_port())

# Connect camera pose to point cloud generator and controller
builder.Connect(
        station.GetOutputPort("camera_transform"),
        point_cloud_generator.GetInputPort("camera_pose"))
builder.Connect(
        station.GetOutputPort("camera_transform"),
        controller.GetInputPort("camera_transform"))

# Connect generated point cloud to the controller
builder.Connect(
        point_cloud_generator.point_cloud_output_port(),
        controller.GetInputPort("point_cloud"))

# Visualize the point cloud with meshcat
meshcat_point_cloud = builder.AddSystem(MeshcatPointCloudVisualizer(
                                            station.meshcat,
                                            draw_period=0.2))
meshcat_point_cloud.set_name("point_cloud_viz")
builder.Connect(
        point_cloud_generator.point_cloud_output_port(),
        meshcat_point_cloud.get_input_port())



Kp = 10*np.eye(6)
Kd = 2*np.sqrt(Kp)

#controller = builder.AddSystem(CommandSequenceController(
#    cs,
#    command_type=EndEffectorTarget.kTwist,
#    Kp=Kp,
#    Kd=Kd))
controller.set_name("controller")
controller.ConnectToStation(builder, station)


# Diagram
diagram = builder.Build()
diagram.set_name("system_diagram")
diagram_context = diagram.CreateDefaultContext()

if show_toplevel_diagram:
    plt.figure()
    plot_system_graphviz(diagram,max_depth=1)
    plt.show()

# Default position
station.go_home(diagram, diagram_context, name="Custom")
station.SetManipulandStartPositions(diagram, diagram_context)

simulator = Simulator(diagram, diagram_context)
simulator.set_target_realtime_rate(1.0)
simulator.set_publish_every_time_step(False)

# Run
simulator.Initialize()
simulator.AdvanceTo(30.0)
#type(var)
