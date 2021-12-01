import requests
import numpy as np
import pandas as pd

#TODO: refine these verbs here
#ai2thor_verbs = ["toggle","break","fill_with_liquid","dirty","use_up","cook","heat_up","make_cold","slice","open","pick_up","move"]
ai2thor_verbs = ["toggle","break","fill_with_liquid","dirty","use_up","cook","slice","open","pick_up","move"]

#THIS is not including ones that seem unrealistic to be spotted by the detectron
allNouns = ["alarm_clock", "aluminum_foil", "apple", "arm_chair", "baseball_bat", "basketball", "bathtub",
"bathtub_basin", "bed", "blinds", "book", "boots", "bottle", "bowl", "box", "bread", "butter_knife", "cabinet", 
"cabinet", "candle", "cd", "cell_phone", "chair", "cloth", "coffee_machine", "coffee_table", "counter_top",
"credit_card", "cup", "curtains", "desk", "desk_lamp", "desktop", "dining_table", "dish_sponge", "dog_bed",
"drawer", "dresser", "dumbbell", "egg", "faucet", "floor", "floor_lamp", "foot_stool", "fork", 
"fridge", "garbage_bag", "hand_towel", "hand_towel_holder", "house_plant", "kettle", "key_chain", "knife", 
"ladle", "laptop", "laundry_hamper", "lettuce", "light_switch", "microwave", "mirror", "mug", "newspaper", 
"ottoman", "painting", "pan", "paper_towel_roll", "pen", "pencil", "pepper_shaker", "pillow", "plate", "plunger",
"poster", "pot", "potato", "remote_control", "room_decor", "safe", "salt_shaker", "scrub_brush", "shelf", "shelving_unit", 
"shower_curtain", "shower_door", "shower_glass", "shower_head", "side_table", "sink", "sink_basin", "soap_bar", "soap_bottle", 
"soap_bottle", "sofa", "spatula", "spoon", "spray_bottle", "statue", "stool", "stove_burner", "stove_knob","table_top_decor",
"target_circle", "teddy_bear", "television", "tennis_racket", "tissue_box", "toaster", "toilet", "toilet_paper", 
"toilet_paper_hanger", "tomato","towel", "towel_holder", "tv_stand", "vacuum_cleaner", "vase","watch", "watering_can", "window", "wine_bottle",
"oven", "tooth_brush", "hair_drier", "keyboard","scissors","mouse"]

keyNouns = ["basket_ball", "baseball_bat", "tennis_racket", "bottle", "wine_bottle", 
"cup", "fork", "knife", "spoon", "bowl", "apple", "chair", "sofa", "house_plant", 
"dining_table", "toilet", "television", "laptop", "remote_control", "cell_phone", 
"microwave", "toaster", "sink", "fridge", "book", "alarm_clock", "vase", "teddy_bear"]

aithorNouns = ["alarm_clock", "aluminum_foil", "apple", "armchair", "baseball_bat", "basketball", "bathtub",
"bathtub_basin", "bed", "blinds", "book", "boots", "bottle", "bowl", "box", "bread", "butter_knife", 
"cabinet", "candle", "cd", "cell_phone", "chair", "cloth", "coffee_machine", "coffee_table", "counter_top",
"credit_card", "cup", "curtains", "desk", "desk_lamp", "desktop", "dining_table", "dish_sponge", "dog_bed",
"drawer", "dresser", "dumbbell", "egg", "faucet", "floor", "floor_lamp", "footstool", "fork", 
"fridge", "garbage_bag", "hand_towel", "hand_towel_holder", "house_plant", "kettle", "key_chain", "knife", 
"ladle", "laptop", "laundry_hamper", "lettuce", "light_switch", "microwave", "mirror", "mug", "newspaper", 
"ottoman", "painting", "pan", "paper_towel_roll", "pen", "pencil", "pepper_shaker", "pillow", "plate", "plunger",
"poster", "pot", "potato", "remote_control", "room_decor", "safe", "salt_shaker", "scrub_brush", "shelf", "shelving_unit", 
"shower_curtain", "shower_door", "shower_glass", "shower_head", "side_table", "sink", "sink_basin", "soap_bar", 
"soap_bottle", "sofa", "spatula", "spoon", "spray_bottle", "statue", "stool", "stove_burner", "stove_knob","table_top_decor",
"target_circle", "teddy_bear", "television", "tennis_racket", "tissue_box", "toaster", "toilet", "toilet_paper", 
"toilet_paper_hanger", "tomato","towel", "towel_holder", "tv_stand", "vacuum_cleaner", "vase","watch", "watering_can", "window", "wine_bottle"]
detectronNouns = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck","boat", "traffic_light",
"fire_hydrant", "stop_sign", "parking_meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow","elephant", "bear", 
"zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports_ball", 
"kite","baseball_glove", "skateboard", "surfboard", "wine_glass", "banana", "sandwich", "orange", "broccoli", "carrot", 
"hot_dog", "pizza", "donut", "cake", "couch","potted_plant", "mouse", "keyboard", "oven","clock","scissors", "hair_drier", "tooth_brush"]
#detectron nouns don't include duplicate in aithornouns already
testNouns = ["trafficlight","firehydrant","parkingmeter","back_pack","ski","snow_board","surf_board","wineglass",
"hotdog","pottedplant","blowdrier"]


basic_url = 'http://api.conceptnet.io'
relatedness1 = '/relatedness?node1=/c/en/'
relatedness2 = '&node2=/c/en/'
query_url1 = 'http://api.conceptnet.io/query?start=/c/en/'
query_url2 = '&rel=/r/UsedFor&limit=20'

def getRelatedness(word1, word2):
    relatedness = requests.get(basic_url + relatedness1 + word1 + relatedness2 + word2)
    if relatedness:
        return relatedness.json()['value']
    else:
        return -1

def getConceptnetQuery(word):
    relatedVerbs = requests.get(query_url1 + word + query_url2).json() 
    verbs = []
    for edge in relatedVerbs['edges']:
        verbs.append(edge['end']['label'])
    verbs = formatCorrectly(verbs)
    return verbs

def saveDictOfRelatedness():
    data = []
    for word in keyNouns[25:]:
        for word2 in keyNouns:
            print(word,word2)
            relatedness = getRelatedness(word,word2)
            data.append([word,word2,relatedness])
    print("Saving...")
    df = pd.DataFrame(data, columns = ['Object1', 'Object2','Relatedness']) 
    df.to_csv('relatedness3.csv')

def saveDictOfVerbs():
    data = []
    for word in testNouns:
        print(word)
        verbs = getConceptnetQuery(word)
        data.append([word,verbs])
    print("Saving...")
    df = pd.DataFrame(data, columns = ['Object', 'RelatedVerbs']) 
    df.to_csv('testnouns_usedfor.csv')

def formatCorrectly(verbs):
    return [verb.replace(" ","_") for verb in verbs]

saveDictOfVerbs()