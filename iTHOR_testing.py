from os import path
from ai2thor.controller import Controller
from PIL import Image

# RoboTHOR initialization
# controller = Controller(
#     agentMode="locobot",
#     visibilityDistance=1.5,
#     scene="FloorPlan_Train1_3",
#     gridSize=0.25,
#     movementGaussianSigma=0.005,
#     rotateStepDegrees=90,
#     rotateGaussianSigma=0.5,
#     renderDepthImage=False,
#     renderInstanceSegmentation=False,
#     width=300,
#     height=300,
#     fieldOfView=60
# )

# <- INITIALIZATION ------------------------------------------>
# iTHOR initialization
controller_test = Controller(
    agentMode="default",
    visibilityDistance=1.5,
    scene="FloorPlan212",

    # step sizes
    gridSize=0.25,
    snapToGrid=True,
    rotateStepDegrees=90,

    # image modalities
    renderDepthImage=False,
    renderInstanceSegmentation=False,

    # camera properties
    width=300,
    height=300,
    fieldOfView=90
)

# any Controller Initialization parameter can later be changed by calling the reset method
# controller.reset(scene="FloorPlan319", rotateStepDegrees=30, fieldOfView=100)

# <- OBJECT TYPES ------------------------------------------>
controller = Controller(scene='FloorPlan1')
for obj in controller.last_event.metadata["objects"]:
    # print(obj["objectTyope","pickupable"])
    print(obj["objectType"])

# <- SET OBJECT STATES ------------------------------------------>
# SetMassProperties to change the mass properties of any Pickupable or Moveable object
# controller.step(
#     action="SetMassProperties",
#     objectId="Apple|+1.25|+0.25|-0.75",
#     mass=22.5,
#     drag=15.9,
#     angularDrag=5.6
# )

# <- DOMAIN RANDOMIZATION ------------------------------------------>
# material randomization, randomizes all the materials in the scene
controller.step(
    action="RandomizeMaterials",
    useTrainMaterials=None,
    useValMaterials=None,
    useTestMaterials=None,
    inRoomTypes=None
)

# <- NAVIGATION ------------------------------------------>
# move the agent in a cardinal direction, relative to its forward facing direction
# controller.step(
#     action="MoveAhead",
#     moveMagnitude=None
# )
# controller.step("MoveBack")
# controller.step("MoveLeft")
# controller.step("MoveRight")

# rotation
controller.step(action="RotateLeft")

# camera rotation
controller.step("LookDown")

# teleportation
controller.step(
    action="Teleport",
    position=dict(x=1, y=0.9, z=-1.5),
    rotation=dict(x=0, y=270, z=0),
    horizon=30,
    standing=True
)

# returns a cleaned up event with respect to the metadata
controller.step(action="Done")

# <- OBJECT MOVEMENT ------------------------------------------>

controller.step(
    action="PickupObject",
    objectId="Apple|1|1|1",
    forceAction=True,
    manualInteract=False
)

# <- EXTRACTING IMAGES ------------------------------------------>

# Each scene is made in 3D space, which means there are an infinite number of possible images. Plus, there are several user settings that may alter the image dataset, such as the camera's field of view, the height/width/resolution, and the channels available (e.g., depth, instance segmentation, class segmentation, etc.).

# Thus, we do not have a dataset for every possible image. However, if we wanted to roam around the environments, and download the images, this can be done using Python's image library.

# with Controller() as c:
#     for _ in range(10):
#         event = c.step(action='MoveBack')

#         # save image
#         Image.fromarray(event.frame).save('image.png')
#         # '/Users/naomilee/Downloads'

# If we want a more curated dataset of all positions in each room, use GetReachablePositions and Teleport to each of the returned reachable positions

with Controller() as c:
    event = c.step(action='GetReachablePositions')
    # positions = event.metadata['reachablePositions']
    positions = event.metadata['actionReturn']
    for pos in positions:
        event = c.step(action='Teleport', **pos)
        Image.fromarray(event.frame).save('image.png')