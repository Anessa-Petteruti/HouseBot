import os
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, accuracy_score
from conceptnet import calculateProbBySimilarVerbs, calculateProbBySimilarity
#import ConceptNet

threshold = .001
#aithor_verbs = ["Toggleable","Breakable","Fillable","Dirtable","UsedUp","Cookable","Heatable","Coldable","Sliceable","Openable","Pickupable","Moveable"]
aithor_verbs = ["Toggleable","Breakable","Fillable","Dirtyable","UsedUp","Cookable","Sliceable","Openable","Pickupable","Moveable"]
properties = ["Toggleable","Breakable","Fillable","Dirtyable","UseUpable","Cookable","Sliceable","Openable","Pickupable","Moveable"]
#Dictionary mapping detectron verbs to ai2thor verbs (currently only ones with very similar aithor words included):
nounDict = {"sports ball": "BasketBall", "baseball bat": "BaseballBat", "tennis racket": "TennisRacket", "bottle": "Bottle",
"wine glass": "WineBottle", "cup": "Cup", "fork" : "Fork", "knife": "Knife", "spoon": "Spoon", "bowl": "Bowl",
"apple": "Apple", "chair": "Chair", "couch": "Sofa", "potted plant": "HousePlant", "bed": "Bed", "dining table" : "DiningTable",
"toilet": "Toilet", "tv": "Television", "laptop": "Laptop", "remote": "RemoteControl", "cell phone": "CellPhone",
"microwave": "Microwave", "toaster": "Toaster", "sink": "Sink", "refrigerator": "fridge", "book": "Book", "clock": "AlarmClock",
"vase": "Vase", "teddy bear": "Teddy Bear"}

aithorNouns = ["alarm_clock", "aluminum_foil", "apple", "armchair", "baseball_bat", "basketball", "bathtub",
"bathtub_basin", "bed", "blinds", "book", "boots", "bottle", "bowl", "box", "bread", "butter_knife", 
"cabinet", "candle", "cd", "cell_phone", "chair", "cloth", "coffee_machine", "coffee_table", "counter_top",
"credit_card", "cup", "curtains", "desk", "desk_lamp", "desktop", "dining_table", "dish_sponge", "dog_bed",
"drawer", "dresser", "dumbbell", "egg", "faucet", "floor", "floor_lamp", "footstool", "fork", 
"fridge", "garbage_bag", "garbage_can","hand_towel", "hand_towel_holder", "house_plant", "kettle", "key_chain", "knife", 
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

def process_object_labels():
    """
    Performs any necessary processing on object labels outputted by Detectron2.
    Detectron2 is run in a separate GoogleColab, dataset of images and labels are saved in repo.
    """
    pass

def find_actions():
    """
    Calls ConceptNet to output a set of actions for input object labels.
    """
    pass

def train():
    """
    """
    pass

def test():
    """
    Compare ConceptNet actions with AI2Thor verbs.
    """
    pass

def f1_score_us(true,score):
    """
    Evaluation metric that takes both precision and recall into account.
    """
    return f1_score(true,score)

def mean_avg_prec_us(true,score):
    """
    Calculates Mean Average Precision (MAP), the mean precision across each result.
    """
    return average_precision_score(true,score)

def getTrueLabels(object):
    df = pd.read_csv('ithor.csv')
    aiword = df.loc[df['Object Type'] == object, 'Actionable Properties']
    print(aiword)
    actions = df.loc[df['Object Type'] == object, 'Actionable Properties'].iloc[0]
    if pd.isna(actions):
        return -1
    actions = actions.replace(" (Some)","")
    actions = actions.split(", ")
    print(actions)
    return [verb in actions for verb in aithor_verbs]

def getAllTrueLabels():
    groundTruthLabels = {}
    df = pd.read_csv('groundtruthLabels.csv')
    for i,row in df.iterrows():
        object = row['Object Type']
        actions = row['Actionable Properties']
        if pd.isna(actions):
            label = [False for verb in aithor_verbs]
        else:
            actions = actions.split(", ")
            label = [verb in actions for verb in aithor_verbs]
        groundTruthLabels[object] = label
    return groundTruthLabels


def sample_test(object):
    trueLabels = getTrueLabels(object)
    if trueLabels == -1:
        print("This object, ", object, "has no ai2thor verbs. Skipping...")
        return
    probs = calculateProbBySimilarVerbs(object.lower())
    labels = [num > threshold for num in probs]
    print(object)
    print(aithor_verbs)
    print("Predicted Labels: ", labels)
    print("True Labels: ", trueLabels)
    print("F1 Score: ", f1_score_us(trueLabels,labels))
    print("MAP Score: ", mean_avg_prec_us(trueLabels,labels))

def getLabelsFromChart(threshold):
    dfCapable = pd.read_csv('overallAverageCapable.csv')
    dfUsed = pd.read_csv('overallAverageUsed.csv')
    concepnetLabels = {}
    for object in aithorNouns + detectronNouns:
        print(object)
        toUse = [dfUsed.loc[dfUsed['Object'] == object, verb].iloc[0] for verb in properties]
        capable = [dfCapable.loc[dfCapable['Object'] == object, verb].iloc[0] for verb in properties]
        if all(item == 0 for item in toUse):
            toUse = capable
        thresholded = [item > threshold for item in toUse]
        concepnetLabels[object] = thresholded
    return concepnetLabels

def get_acc(labels, pred):
    final_acc = 0
    labels = list(labels.values())
    pred = list(pred.values())
    for i in range(len(labels)):
        final_acc += accuracy_score(labels[i], pred[i])
    return final_acc / len(labels)

def getAccuracy(labels, pred):
    final_acc = 0
    nouns = list(labels.keys())
    for obj in nouns:
        final_acc += accuracy_score(labels[obj], pred[obj])
    return final_acc / len(labels)

def main():
    # Will need to pass things in to these functions here...
    #process_object_labels()
    conceptnetLabels = getLabelsFromChart(0.02)
    print(conceptnetLabels["sofa"])
    groundTruthLabels = getAllTrueLabels()
    print(groundTruthLabels["sofa"])
    acc1 = get_acc(conceptnetLabels, groundTruthLabels)
    acc2 = getAccuracy(conceptnetLabels, groundTruthLabels)
    print(acc1)
    print(acc2)

#RECEPTABCLE HEAT AND COLD NEED OT BE ADDED

if __name__ == "__main__":
    main()
