from os import path
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

    # intialize controller
    controller = Controller()

    # iterate through iTHOR scenes and save each image in folder '/images'
    for scene in scenes:
        controller.reset(scene)
        event = controller.step(dict(action='Initialize', gridSize=0.25))
        # uncomment lines below to gather metadata in a json for each scene
        # with open('/Users/naomilee/iTHOR/images/'+ str(scene) + '.json', 'w') as f:
        #     json.dump(event.metadata, f, indent=4, sort_keys=True)
        Image.fromarray(event.frame).save('/Users/naomilee/iTHOR/images/' + str(scene) + '.png')

def main():
    collect_images()

if __name__ == "__main__":
    main()