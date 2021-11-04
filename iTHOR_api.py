from os import path
from typing import OrderedDict
from ai2thor.controller import Controller
from PIL import Image
import json

def collect_images():

    # all 120 scenes in iTHOR, evenly spread across kitchens, living rooms, bedrooms, and bathrooms
    kitchens = [f"FloorPlan{i}" for i in range(1, 31)]
    living_rooms = [f"FloorPlan{200 + i}" for i in range(1, 31)]
    bedrooms = [f"FloorPlan{300 + i}" for i in range(1, 31)]
    bathrooms = [f"FloorPlan{400 + i}" for i in range(1, 31)]
    scenes = kitchens + living_rooms + bedrooms + bathrooms

    # CODE FOR INITIALIZING AND GRABBING IMAGE/METADATA FROM ONE FLOORPLAN
    # # intialize controller
    # # controller = Controller('FloorPlan9')
    # # Quality: Very Low, Low, Medium, MediumCloseFitShadows, High, Very High, Ultra, High WebGL
    # controller = Controller(agentMode="default", visibilityDistance=0.0, scene="FloorPlan3", width=800,
    # height=800)
    # # controller.reset('FloorPlan3', visibilityDistance=1)
    # event = controller.step(dict(action='Initialize', gridSize=0.25))
    # print(event.metadata)
    # scene_objects = event.metadata["objects"]
    # for object in scene_objects:
    #     if object["visible"] == True:
    #         print("YAY!!!!!")
    #         object_strings.append(object["objectType"])
    # print(object_strings)
    # Image.fromarray(event.frame).save('/Users/naomilee/iTHOR/images/' + str(controller.scene) + 'high_res.png')


    # iterate through iTHOR scenes and save each image in folder '/images'
    for scene in scenes:
        object_strings = []
        controller = Controller(agentMode="default", visibilityDistance=0.0, scene=scene, width=800, height=800)
        event = controller.step(dict(action='Initialize', gridSize=0.25))
        # print(event.metadata["objects"])
        # uncomment lines below to gather metadata in a json for each scene
        with open('/Users/naomilee/iTHOR/jsons/'+ str(scene) + '.json', 'w') as f:
            # json.dump(event.metadata, f, indent=4, sort_keys=True)
            scene_objects = event.metadata["objects"]
            for object in scene_objects:
                if object["visible"] == True:
                    # print("YAY!!!!!")
                    object_strings.append(object["objectType"])
                    # json.dump(object, f, indent=4, sort_keys=True)
            # object strings
            object_strings = list(OrderedDict.fromkeys(object_strings))
            # print(object_strings)
            # print(scene_objects, "TYPE:", type(scene_objects))
            # return 0
            json.dump(object_strings, f, indent=4, sort_keys=True)
        Image.fromarray(event.frame).save('/Users/naomilee/iTHOR/images/' + str(scene) + '.png')

def main():
    collect_images()

if __name__ == "__main__":
    main()