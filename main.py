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
#Dictionary mapping detectron verbs to ai2thor verbs (currently only ones with very similar aithor words included): NEEDED FOR INTEGRATION
nounDict = {"sports ball": "BasketBall", "baseball bat": "BaseballBat", "tennis racket": "TennisRacket", "bottle": "Bottle",
"wine glass": "WineBottle", "cup": "Cup", "fork" : "Fork", "knife": "Knife", "spoon": "Spoon", "bowl": "Bowl",
"apple": "Apple", "chair": "Chair", "couch": "Sofa", "potted plant": "HousePlant", "bed": "Bed", "dining table" : "DiningTable",
"toilet": "Toilet", "tv": "Television", "laptop": "Laptop", "remote": "RemoteControl", "cell phone": "CellPhone",
"microwave": "Microwave", "toaster": "Toaster", "sink": "Sink", "refrigerator": "fridge", "book": "Book", "clock": "AlarmClock",
"vase": "Vase", "teddy bear": "Teddy Bear"}

#Nouns present in aithor 
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

#Nouns that detectron can detect not including nouns already present in aithorNouns list
detectronNouns = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck","boat", "traffic_light",
"fire_hydrant", "stop_sign", "parking_meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow","elephant", "bear", 
"zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports_ball", 
"kite","baseball_glove", "skateboard", "surfboard", "wine_glass", "banana", "sandwich", "orange", "broccoli", "carrot", 
"hot_dog", "pizza", "donut", "cake", "couch","potted_plant", "mouse", "keyboard", "oven","clock","scissors", "hair_drier",
 "tooth_brush", "refrigerator", "remote", "tv"]

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

#Gets the true property lables for an object from the ithor chart - only works for aithor nouns currently
def getTrueLabels(object):
    df = pd.read_csv('ithor.csv')
    actions = df.loc[df['Object Type'] == object, 'Actionable Properties'].iloc[0]
    if pd.isna(actions):
        return -1
    actions = actions.replace(" (Some)","")
    actions = actions.split(", ")
    return [verb in actions for verb in aithor_verbs]

#get all true labels for all objects from the groundtruthLabels charts 
def getAllTrueLabels():
    groundTruthLabels = {}
    df = pd.read_csv('groundtruthLabelsReceptacle.csv')
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

#Gets all predicted labels from the average charts for all objects using one threshold across all properties
def getLabelsFromChart(threshold):
    dfCapable = pd.read_csv('overallAverageCapable.csv')
    dfUsed = pd.read_csv('overallAverageUsed.csv')
    concepnetLabels = {}
    for object in aithorNouns + detectronNouns:
        #print(object)
        toUse = [dfUsed.loc[dfUsed['Object'] == object, verb].iloc[0] for verb in properties]
        capable = [dfCapable.loc[dfCapable['Object'] == object, verb].iloc[0] for verb in properties]
        if all(item == 0 for item in toUse):
            toUse = capable
        thresholded = [item > threshold for item in toUse]
        concepnetLabels[object] = thresholded
    return concepnetLabels

#Gets all predicted labels from the average charts for all objects using one threshold across all properties
def getMaxLabelsFromChart():
    dfCapable = pd.read_csv('overallAverageCapable.csv')
    dfUsed = pd.read_csv('overallAverageUsed.csv')
    concepnetLabels = {}
    for object in aithorNouns + detectronNouns:
        toUse = [dfUsed.loc[dfUsed['Object'] == object, verb].iloc[0] for verb in properties]
        capable = [dfCapable.loc[dfCapable['Object'] == object, verb].iloc[0] for verb in properties]
        if all(item == 0 for item in capable):
            #if all(item == 0 for item in toUse):
            concepnetLabels[object] = 8
            continue
            #else:
                #capable = toUse
        maxLabel = np.argmax(capable)
        concepnetLabels[object] = maxLabel
    return concepnetLabels

#Gets all predicted labels from the average charts for all objects using a list of thresholds, one for each property
def getLabelsFromChartMultiThresh(threshold):
    dfCapable = pd.read_csv('overallAverageCapable.csv')
    dfUsed = pd.read_csv('overallAverageUsed.csv')
    concepnetLabels = {}
    for object in aithorNouns + detectronNouns:
        #print(object)
        toUse = [dfUsed.loc[dfUsed['Object'] == object, verb].iloc[0] for verb in properties]
        capable = [dfCapable.loc[dfCapable['Object'] == object, verb].iloc[0] for verb in properties]
        if all(item == 0 for item in toUse):
            toUse = capable
        thresholded = [item > threshold for item,threshold in zip(toUse,threshold)]
        concepnetLabels[object] = thresholded
    return concepnetLabels

#calculates accuracy, precision, and recall for 1 label and ground truth
def modified_accuracy_score(label,pred):
    matching = 0
    matchingFalse = 0
    falseNeg = 0
    falsePos = 0
    for i in range(len(label)):
        if label[i] and pred[i]:
            matching += 1
        elif label[i] and not pred[i]:
            falseNeg += 1
        elif not label[i] and pred[i]:
            falsePos += 1
        elif not label[i] and not pred[i]:
            matchingFalse += 1
    accuracy = (matching + matchingFalse) / len(label)
    if matching + falsePos == 0:
        precision = -1
    else:
        precision = matching / (matching + falsePos)
    if matching + falseNeg == 0:
        recall = -1
    else:
        recall = matching / (matching + falseNeg)
    #print(label,pred,accuracy,precision,recall, matching, falsePos, falseNeg, matchingFalse)
    return accuracy, precision, recall

#calculates accuracy, precision, and recall for multiple labels and corresponding ground truths
def get_acc_modified(labels,pred):
    final_acc = 0
    final_prec = 0
    final_recall = 0
    num_prec = 0
    num_recall = 0
    labels = list(labels.values())
    pred = list(pred.values())
    for i in range(len(labels)):
        acc, prec, rec = modified_accuracy_score(labels[i], pred[i])
        final_acc += acc
        if prec != -1:
            final_prec += prec
            num_prec += 1
        if rec != -1:
            final_recall += rec
            num_recall += 1
    return final_acc / len(labels), final_prec / num_prec, final_recall / num_recall

#calculates accuracy, precision, and recall for multiple labels and corresponding ground truths at one index j (property_j)
def get_single_acc_modified(labels,pred,j):
    final_pos = 0
    final_neg = 0
    final_prec = 0
    final_recall = 0
    labels = list(labels.values())
    pred = list(pred.values())
    for i in range(len(labels)):
        if labels[i][j] and pred[i][j]:
            final_pos += 1
            final_prec += 1
            final_recall += 1
        elif labels[i][j] and not pred[i][j]:
            final_recall += 1
        elif not labels[i][j] and pred[i][j]:
            final_neg += 1
        elif not labels[i][j] and not pred[i][j]:
            final_pos += 1
    if final_prec == 0:
        return (final_pos + final_neg) / len(labels), -1, final_pos / final_recall
    elif final_recall == 0:
        return (final_pos + final_neg) / len(labels), final_pos / final_prec, -1
    elif final_recall == 0 and final_prec == 0:
        return (final_pos + final_neg) / len(labels), -1, -1
    else:
        return (final_pos + final_neg) / len(labels), final_pos / final_prec, final_pos / final_recall

#calculates accuracy for multiple labels and ground truths
def get_acc(labels, pred):
    final_acc = 0
    labels = list(labels.values())
    pred = list(pred.values())
    for i in range(len(labels)):
        final_acc += accuracy_score(labels[i], pred[i])
    return final_acc / len(labels)

#calculates accuracy for multiple labels and ground truths
def getAccuracy(labels, pred):
    final_acc = 0
    nouns = list(labels.keys())
    for obj in nouns:
        final_acc += accuracy_score(labels[obj], pred[obj])
    return final_acc / len(labels)

#calculates accuracy for multiple labels and ground truths for one property j
def getSingleAccuracy(labels, pred,j):
    final_acc = 0
    labels = list(labels.values())
    pred = list(pred.values())
    for i in range(len(labels)):
        if labels[i][j] == pred[i][j]:
            final_acc += 1
    return final_acc / len(labels)

#calculates optimal accuracy using a single threshold
def findOptimalAccuracy():
    bestAcc = 0
    bestThresh = 0
    groundTruthLabels = getAllTrueLabels()
    for i in np.arange(0,.25,.01):
        conceptnetLabels = getLabelsFromChart(i)
        acc = get_acc(conceptnetLabels,groundTruthLabels)
        if acc > bestAcc:
            bestAcc = acc
            bestThresh = i
        print("Threshold: ", i, " Accuracy: ", acc)
    print(bestThresh, bestAcc)

#calculates optimal accuracy, precision, and recall using a single threshold
def findOptimalAccuracyModified():
    bestAcc = 0
    bestPrec = 0
    bestRec = 0
    bestThreshA = 0
    bestThreshP = 0
    bestThreshR = 0
    groundTruthLabels = getAllTrueLabels()
    for i in np.arange(-.05,.2,.01):
        conceptnetLabels = getLabelsFromChart(i)
        acc, prec, recall = get_acc_modified(groundTruthLabels, conceptnetLabels)
        if acc > bestAcc:
            bestAcc = acc
            bestThreshA = i
        if prec > bestPrec:
            bestPrec = prec
            bestThreshP = i
        if recall > bestRec:
            bestRec = recall
            bestThreshR = i
        print("Threshold: ", i, " Accuracy: ", acc, " Precision: ", prec, " Recall: ", recall)
    print(bestThreshA, bestAcc, bestThreshP, bestPrec, bestThreshR, bestRec)

#calculates optimal accuracy using list of thresholds, one for each property
def findOptimalMultiAccuracy():
    groundTruthLabels = getAllTrueLabels()
    bestThreshOverall = []
    bestAccOverall = []
    for j in range(10):
        thresh = [0,0,0,0,0,0,0,0,0,0]
        bestAcc = 0
        bestThresh = 0
        for i in np.arange(-.05,.25,.01):
            thresh[j] = i
            conceptnetLabels = getLabelsFromChartMultiThresh(thresh)
            acc = getSingleAccuracy(conceptnetLabels,groundTruthLabels,j)
            if acc > bestAcc:
                bestAcc = acc
                bestThresh = i
            print("j: ", j, " Threshold: ", i, " Accuracy: ", acc)
        print(bestThresh, bestAcc)
        bestThreshOverall.append(bestThresh)
        bestAccOverall.append(bestAcc)
    print(bestThreshOverall,bestAccOverall)

#calculates optimal accuracy, precision, and recall using list of thresholds, one for each property
def findOptimalMultiAccuracyModified():
    bestAccOverall = {}
    bestThreshOverall = {}
    groundTruthLabels = getAllTrueLabels()
    for j in range(10):
        bestAcc = 0
        bestPrec = 0
        bestRec = 0
        bestThreshA = 0
        bestThreshP = 0
        bestThreshR = 0
        thresh = [0,0,0,0,0,0,0,0,0,0]
        for i in np.arange(-.1,.25,.01):
            thresh[j] = i
            conceptnetLabels = getLabelsFromChartMultiThresh(thresh)
            acc, prec, recall = get_single_acc_modified(conceptnetLabels,groundTruthLabels,j)
            if acc > bestAcc:
                bestAcc = acc
                bestThreshA = i
            if prec > bestPrec:
                bestPrec = prec
                bestThreshP = i
            if recall > bestRec:
                bestRec = recall
                bestThreshR = i
            print("j: ", j ," Threshold: ", i, " Accuracy: ", acc, " Precision: ", prec, " Recall: ", recall)
        print(bestThreshA, bestAcc, bestThreshP, bestPrec, bestThreshR, bestRec)
        bestThreshOverall[j] = [bestThreshA,bestThreshP,bestThreshR]
        bestAccOverall[j] = [bestAcc,bestPrec,bestRec]
    print(bestThreshOverall,bestAccOverall)

def analyzeResults():
    groundTruthLabels = getAllTrueLabels()
    conceptnetLabels = getLabelsFromChart(.1) #random threshold - need to fill in with the best threshold
    objects = pd.read_csv('detectron_predictions_final.csv')
    overall_accuracy = 0
    overall_count = 0
    overall_synonym_accuracy = 0
    conceptnetLabels = getMaxLabelsFromChart()
    groundTruthLabels = getAllTrueLabels()
    overall_correct = 0
    for j,row in objects.iterrows():
        groundTruth = formatToList(row['Ground Truth Objects (actual objects in the image) JUST AI2THOR NOUNS'])
        predicted = formatToList(row['Detectron Predicted Objects'])
        if groundTruth == ['']:
            continue
        print("Ground Truth: ", groundTruth, " vs. ", predicted)
        assert(len(groundTruth) == len(predicted))
        extra = 0
        correct = 0
        incorrect = 0
        accuracy = 0
        syn_accuracy = 0
        numSyn = 0
        for i in range(len(groundTruth)):
            overall_count += 1
            gTObj = groundTruth[i]
            predObj = predicted[i]
            if gTObj == "extra":
                extra += 1
            elif gTObj == "incorrect":
                incorrect += 1
            elif gTObj != predObj:
                numSyn += 1
                if groundTruthLabels[gTObj][conceptnetLabels[predObj]]:
                    syn_accuracy += 1
                correct += 1
                if groundTruthLabels[gTObj][conceptnetLabels[gTObj]]:
                    accuracy += 1
            else:
                correct += 1
                if groundTruthLabels[gTObj][conceptnetLabels[gTObj]]:
                    accuracy += 1
                    syn_accuracy += 1
        print("For row ", j, " it found ", 100* correct / (correct + incorrect), "%, objects correctly, with ", extra, " extras.")
        if numSyn > 0:
            print("Per image accuracy (actual aithor word) is ", 100*accuracy/correct, "%, while using synonyms gives ", 100*syn_accuracy/correct, "%")
        elif correct > 0:
            print("Per image accuracy (actual aithor word) is ", 100*accuracy/correct, "%")
        else: 
            print("Nothing correct to run through conceptnet.")
        overall_accuracy += accuracy
        overall_synonym_accuracy += syn_accuracy
        overall_correct += correct
    print("Object detection accuracy: ", 100* overall_correct / overall_count, "%")
    print("overall direct nlp accuracy: ", 100*overall_accuracy/ overall_correct, "%")
    print("Overall accuracy using synonyms: ", 100*overall_synonym_accuracy/ overall_correct, "%")
    print("overall full accuracy direct: ", 100*overall_accuracy/overall_count, "%")
    print("overall full accuracy synonyms: ", 100*overall_synonym_accuracy / overall_count, "%")



def formatToList(text):
    text = text.replace("'","")
    text = text.replace("[","")
    text = text.replace("]","")
    text = text.replace(", ","/")
    text = text.replace(" ", "_")
    text = text.split("/")
    return text

def getMaxAccuracy():
    conceptnetIndices = getMaxLabelsFromChart()
    groundTruthLabels = getAllTrueLabels()
    fullAccuracy = 0
    possibleAccuracy = 0
    possibleAccuracyTotal = 0
    availableInfo = 0
    total = 0
    #could put a threshold on possible accuracy max values i guess
    for key in conceptnetIndices.keys():
        index = conceptnetIndices[key]
        total += 1
        if index != -1:
            availableInfo += 1
            result = groundTruthLabels[key][index]
            if result: 
                fullAccuracy += 1
            if not all(not item for item in groundTruthLabels[key]):
                possibleAccuracyTotal += 1
                if result:
                    possibleAccuracy += 1
    print("Total accuracy: ", fullAccuracy/total, " total for available info", fullAccuracy/availableInfo)
    print("possible on available accuracy: ", possibleAccuracy/availableInfo, " super possible accuracy: ", possibleAccuracy/ possibleAccuracyTotal)


def getMaxAccuracySimple():
    conceptnetIndices = getMaxLabelsFromChart()
    groundTruthLabels = getAllTrueLabels()
    fullAccuracy = 0
    possibleAccuracy = 0
    possibleAccuracyTotal = 0
    total = 0
    #could put a threshold on possible accuracy max values i guess
    for key in detectronNouns:
        index = conceptnetIndices[key]
        total += 1
        result = groundTruthLabels[key][index]
        if result: 
            fullAccuracy += 1
        if not all(not item for item in groundTruthLabels[key]):
            possibleAccuracyTotal += 1
            if result:
                possibleAccuracy += 1
    print("Total accuracy: ", fullAccuracy/total, " possible accuracy", possibleAccuracy/possibleAccuracyTotal)

#used for first: Total accuracy:  0.32571428571428573  possible accuracy:  0.3877551020408163
#capable of first: Total accuracy:  0.3485714285714286  possible accuracy:  0.41496598639455784
# not pickupable just dont count if no info used for: Total accuracy:  0.16  total for available info 0.23140495867768596
# possible on available accuracy:  0.23140495867768596  super possible accuracy:  0.27450980392156865
# not pickupable just dont count if no info capable of: Total accuracy:  0.18285714285714286  total for available info 0.2644628099173554
# possible on available accuracy:  0.2644628099173554  super possible accuracy:  0.3137254901960784 
#Simple no default capable: Total accuracy:  0.3942857142857143  possible accuracy 0.46938775510204084
#Simple no default to use: Total accuracy:  0.3314285714285714  possible accuracy 0.3945578231292517
#simple capable w/o receptacle: Total accuracy:  0.4114285714285714  possible accuracy 0.46153846153846156
#simple used for w/o receptacle: Total accuracy:  0.3314285714285714  possible accuracy 0.3717948717948718

##simple used for w/o receptacle aithornouns: Total accuracy:  0.3025210084033613  possible accuracy 0.3157894736842105
##simple used for aithornouns: Total accuracy:  0.3025210084033613  possible accuracy 0.34285714285714286
##simple capable of w/o receptacle aithornouns: Total accuracy:  0.44537815126050423  possible accuracy 0.4649122807017544
##simple capable of aithornouns: Total accuracy:  0.42016806722689076  possible accuracy 0.47619047619047616

##simple used for w/o receptacle detectronnouns: Total accuracy:  0.3392857142857143  possible accuracy 0.4523809523809524
##simple used for detectronnouns: Total accuracy:  0.39285714285714285  possible accuracy 0.5238095238095238
##simple capable of w/o receptacle detectronnouns: Total accuracy:  0.3392857142857143  possible accuracy 0.4523809523809524
##simple capable of detectronnouns: Total accuracy:  0.3392857142857143  possible accuracy 0.4523809523809524


# Otimal Accuracy threshold: [0.18000000000000005, 0.13000000000000006, 0.17000000000000004, 0.23000000000000004, 
# 0.15000000000000002, 0.24000000000000005, 0.19000000000000006, 0.12000000000000004, 0.13000000000000006, 
# 0.08000000000000003] Optimal Accuracies per thresh: [0.9142857142857143, 0.9085714285714286, 0.8857142857142857, 
# 0.9371428571428572, 0.9771428571428571, 0.9771428571428571, 0.9085714285714286, 0.8971428571428571, 
# 0.5428571428571428, 0.7771428571428571]

#Optimal Multi Thresholds: {0: [0.17999999999999985, -0.1, 0.17999999999999985], 1: [0.12999999999999987, -0.1, 0.12999999999999987], 
# 2: [0.16999999999999985, -0.1, -0.09000000000000001], 3: [0.23999999999999985, -0.1, 0.12999999999999987], 
# 4: [0.14999999999999988, -0.1, -0.01000000000000005], 5: [0.23999999999999985, -0.1, 0.049999999999999906], 
# 6: [0.18999999999999986, -0.1, 0.13999999999999987], 7: [0.11999999999999988, -0.1, 0.049999999999999906], 
# 8: [0.12999999999999987, -0.1, 0.14999999999999988], 9: [0.0799999999999999, -0.1, 0.0799999999999999]} 
# Optimal A/P/R per threshold: {0: [0.9142857142857143, 1.0, 1.0], 1: [0.9028571428571428, 1.0, 0.5], 
# 2: [0.8857142857142857, 1.0, 0.11494252873563218], 3: [0.9371428571428572, 1.0, 0.25], 
# 4: [0.9771428571428571, 1.0, 0.024242424242424242], 5: [0.9771428571428571, 1.0, 0.027777777777777776], 
# 6: [0.9085714285714286, 1.0, 0.25], 7: [0.8971428571428571, 1.0, 0.26666666666666666], 
# 8: [0.5428571428571428, 1.0, 0.7692307692307693], 9: [0.7771428571428571, 1.0, 0.5]}


def main():
    # Will need to pass things in to these functions here...
    #conceptnetLabels = getLabelsFromChart(0.02)
    #exampleThresh = [0,0,0,0,0,0,0,0,0,0]
    #conceptnetLabels = getLabelsFromChartMultiThresh(exampleThresh)
    #print(conceptnetLabels["sofa"])
    #groundTruthLabels = getAllTrueLabels()
    #print(groundTruthLabels["sofa"])
    #acc1 = get_acc(conceptnetLabels, groundTruthLabels)
    #acc2 = getAccuracy(conceptnetLabels, groundTruthLabels)
    #print(acc1)
    #print(acc2)
    analyzeResults()

#RECEPTABCLE HEAT AND COLD NEED OT BE ADDED

if __name__ == "__main__":
    main()


#Readout from multi thresholding and getting all of these values: 
#j:  0  Threshold:  -0.1  Accuracy:  0.11428571428571428  Precision:  1.0  Recall:  0.0935672514619883
# j:  0  Threshold:  -0.09000000000000001  Accuracy:  0.12  Precision:  1.0  Recall:  0.09411764705882353
# j:  0  Threshold:  -0.08000000000000002  Accuracy:  0.12571428571428572  Precision:  1.0  Recall:  0.09467455621301775
# j:  0  Threshold:  -0.07000000000000002  Accuracy:  0.13714285714285715  Precision:  1.0  Recall:  0.09580838323353294
# j:  0  Threshold:  -0.060000000000000026  Accuracy:  0.13714285714285715  Precision:  1.0  Recall:  0.09580838323353294
# j:  0  Threshold:  -0.05000000000000003  Accuracy:  0.15428571428571428  Precision:  1.0  Recall:  0.0975609756097561
# j:  0  Threshold:  -0.040000000000000036  Accuracy:  0.18857142857142858  Precision:  0.8125  Recall:  0.08552631578947369
# j:  0  Threshold:  -0.03000000000000004  Accuracy:  0.21714285714285714  Precision:  0.8125  Recall:  0.08843537414965986
# j:  0  Threshold:  -0.020000000000000046  Accuracy:  0.29714285714285715  Precision:  0.8125  Recall:  0.09774436090225563
# j:  0  Threshold:  -0.01000000000000005  Accuracy:  0.33714285714285713  Precision:  0.75  Recall:  0.0967741935483871
# j:  0  Threshold:  -5.551115123125783e-17  Accuracy:  0.42857142857142855  Precision:  0.75  Recall:  0.1111111111111111
# j:  0  Threshold:  0.00999999999999994  Accuracy:  0.7371428571428571  Precision:  0.4375  Recall:  0.1590909090909091
# j:  0  Threshold:  0.019999999999999934  Accuracy:  0.8171428571428572  Precision:  0.375  Recall:  0.21428571428571427
# j:  0  Threshold:  0.029999999999999943  Accuracy:  0.84  Precision:  0.3125  Recall:  0.22727272727272727
# j:  0  Threshold:  0.039999999999999925  Accuracy:  0.8742857142857143  Precision:  0.1875  Recall:  0.25
# j:  0  Threshold:  0.049999999999999906  Accuracy:  0.88  Precision:  0.125  Recall:  0.2222222222222222
# j:  0  Threshold:  0.059999999999999915  Accuracy:  0.9028571428571428  Precision:  0.125  Recall:  0.4
# j:  0  Threshold:  0.06999999999999992  Accuracy:  0.9028571428571428  Precision:  0.125  Recall:  0.4
# j:  0  Threshold:  0.0799999999999999  Accuracy:  0.8971428571428571  Precision:  0.0625  Recall:  0.25
# j:  0  Threshold:  0.08999999999999989  Accuracy:  0.8971428571428571  Precision:  0.0625  Recall:  0.25
# j:  0  Threshold:  0.0999999999999999  Accuracy:  0.8971428571428571  Precision:  0.0625  Recall:  0.25
# j:  0  Threshold:  0.1099999999999999  Accuracy:  0.9028571428571428  Precision:  0.0625  Recall:  0.3333333333333333
# j:  0  Threshold:  0.11999999999999988  Accuracy:  0.9028571428571428  Precision:  0.0625  Recall:  0.3333333333333333
# j:  0  Threshold:  0.12999999999999987  Accuracy:  0.9028571428571428  Precision:  0.0625  Recall:  0.3333333333333333
# j:  0  Threshold:  0.13999999999999987  Accuracy:  0.9085714285714286  Precision:  0.0625  Recall:  0.5
# j:  0  Threshold:  0.14999999999999988  Accuracy:  0.9085714285714286  Precision:  0.0625  Recall:  0.5
# j:  0  Threshold:  0.1599999999999999  Accuracy:  0.9085714285714286  Precision:  0.0625  Recall:  0.5
# j:  0  Threshold:  0.16999999999999985  Accuracy:  0.9085714285714286  Precision:  0.0625  Recall:  0.5
# j:  0  Threshold:  0.17999999999999985  Accuracy:  0.9142857142857143  Precision:  0.0625  Recall:  1.0
# j:  0  Threshold:  0.18999999999999986  Accuracy:  0.9142857142857143  Precision:  0.0625  Recall:  1.0
# j:  0  Threshold:  0.19999999999999982  Accuracy:  0.9142857142857143  Precision:  0.0625  Recall:  1.0
# j:  0  Threshold:  0.20999999999999983  Accuracy:  0.9142857142857143  Precision:  0.0625  Recall:  1.0
# j:  0  Threshold:  0.21999999999999983  Accuracy:  0.9085714285714286  Precision:  0.0  Recall:  -1
# j:  0  Threshold:  0.22999999999999984  Accuracy:  0.9085714285714286  Precision:  0.0  Recall:  -1
# j:  0  Threshold:  0.23999999999999985  Accuracy:  0.9085714285714286  Precision:  0.0  Recall:  -1
# 0.17999999999999985 0.9142857142857143 -0.1 1.0 0.17999999999999985 1.0
# j:  1  Threshold:  -0.1  Accuracy:  0.09714285714285714  Precision:  1.0  Recall:  0.09714285714285714
# j:  1  Threshold:  -0.09000000000000001  Accuracy:  0.09714285714285714  Precision:  1.0  Recall:  0.09714285714285714
# j:  1  Threshold:  -0.08000000000000002  Accuracy:  0.09714285714285714  Precision:  1.0  Recall:  0.09714285714285714
# j:  1  Threshold:  -0.07000000000000002  Accuracy:  0.09714285714285714  Precision:  1.0  Recall:  0.09714285714285714
# j:  1  Threshold:  -0.060000000000000026  Accuracy:  0.09714285714285714  Precision:  1.0  Recall:  0.09714285714285714
# j:  1  Threshold:  -0.05000000000000003  Accuracy:  0.10285714285714286  Precision:  1.0  Recall:  0.09770114942528736
# j:  1  Threshold:  -0.040000000000000036  Accuracy:  0.12  Precision:  1.0  Recall:  0.09941520467836257
# j:  1  Threshold:  -0.03000000000000004  Accuracy:  0.14285714285714285  Precision:  1.0  Recall:  0.10179640718562874
# j:  1  Threshold:  -0.020000000000000046  Accuracy:  0.17714285714285713  Precision:  1.0  Recall:  0.10559006211180125
# j:  1  Threshold:  -0.01000000000000005  Accuracy:  0.18857142857142858  Precision:  0.7058823529411765  Recall:  0.08053691275167785
# j:  1  Threshold:  -5.551115123125783e-17  Accuracy:  0.22285714285714286  Precision:  0.6470588235294118  Recall:  0.07801418439716312
# j:  1  Threshold:  0.00999999999999994  Accuracy:  0.5542857142857143  Precision:  0.47058823529411764  Recall:  0.1038961038961039
# j:  1  Threshold:  0.019999999999999934  Accuracy:  0.6057142857142858  Precision:  0.35294117647058826  Recall:  0.09375
# j:  1  Threshold:  0.029999999999999943  Accuracy:  0.6914285714285714  Precision:  0.17647058823529413  Recall:  0.06976744186046512
# j:  1  Threshold:  0.039999999999999925  Accuracy:  0.76  Precision:  0.058823529411764705  Recall:  0.037037037037037035
# j:  1  Threshold:  0.049999999999999906  Accuracy:  0.8171428571428572  Precision:  0.058823529411764705  Recall:  0.058823529411764705
# j:  1  Threshold:  0.059999999999999915  Accuracy:  0.8514285714285714  Precision:  0.058823529411764705  Recall:  0.09090909090909091
# j:  1  Threshold:  0.06999999999999992  Accuracy:  0.8628571428571429  Precision:  0.058823529411764705  Recall:  0.1111111111111111
# j:  1  Threshold:  0.0799999999999999  Accuracy:  0.88  Precision:  0.058823529411764705  Recall:  0.16666666666666666
# j:  1  Threshold:  0.08999999999999989  Accuracy:  0.8857142857142857  Precision:  0.058823529411764705  Recall:  0.2
# j:  1  Threshold:  0.0999999999999999  Accuracy:  0.8914285714285715  Precision:  0.058823529411764705  Recall:  0.25
# j:  1  Threshold:  0.1099999999999999  Accuracy:  0.8971428571428571  Precision:  0.058823529411764705  Recall:  0.3333333333333333
# j:  1  Threshold:  0.11999999999999988  Accuracy:  0.8971428571428571  Precision:  0.058823529411764705  Recall:  0.3333333333333333
# j:  1  Threshold:  0.12999999999999987  Accuracy:  0.9028571428571428  Precision:  0.058823529411764705  Recall:  0.5
# j:  1  Threshold:  0.13999999999999987  Accuracy:  0.9028571428571428  Precision:  0.0  Recall:  -1
# j:  1  Threshold:  0.14999999999999988  Accuracy:  0.9028571428571428  Precision:  0.0  Recall:  -1
# j:  1  Threshold:  0.1599999999999999  Accuracy:  0.9028571428571428  Precision:  0.0  Recall:  -1
# j:  1  Threshold:  0.16999999999999985  Accuracy:  0.9028571428571428  Precision:  0.0  Recall:  -1
# j:  1  Threshold:  0.17999999999999985  Accuracy:  0.9028571428571428  Precision:  0.0  Recall:  -1
# j:  1  Threshold:  0.18999999999999986  Accuracy:  0.9028571428571428  Precision:  0.0  Recall:  -1
# j:  1  Threshold:  0.19999999999999982  Accuracy:  0.9028571428571428  Precision:  0.0  Recall:  -1
# j:  1  Threshold:  0.20999999999999983  Accuracy:  0.9028571428571428  Precision:  0.0  Recall:  -1
# j:  1  Threshold:  0.21999999999999983  Accuracy:  0.9028571428571428  Precision:  0.0  Recall:  -1
# j:  1  Threshold:  0.22999999999999984  Accuracy:  0.9028571428571428  Precision:  0.0  Recall:  -1
# j:  1  Threshold:  0.23999999999999985  Accuracy:  0.9028571428571428  Precision:  0.0  Recall:  -1
# 0.12999999999999987 0.9028571428571428 -0.1 1.0 0.12999999999999987 0.5
# j:  2  Threshold:  -0.1  Accuracy:  0.11428571428571428  Precision:  1.0  Recall:  0.11428571428571428
# j:  2  Threshold:  -0.09000000000000001  Accuracy:  0.12  Precision:  1.0  Recall:  0.11494252873563218
# j:  2  Threshold:  -0.08000000000000002  Accuracy:  0.12  Precision:  0.95  Recall:  0.11046511627906977
# j:  2  Threshold:  -0.07000000000000002  Accuracy:  0.12  Precision:  0.95  Recall:  0.11046511627906977
# j:  2  Threshold:  -0.060000000000000026  Accuracy:  0.12  Precision:  0.8  Recall:  0.0963855421686747
# j:  2  Threshold:  -0.05000000000000003  Accuracy:  0.12571428571428572  Precision:  0.8  Recall:  0.09696969696969697
# j:  2  Threshold:  -0.040000000000000036  Accuracy:  0.14285714285714285  Precision:  0.75  Recall:  0.09375
# j:  2  Threshold:  -0.03000000000000004  Accuracy:  0.14857142857142858  Precision:  0.75  Recall:  0.09433962264150944
# j:  2  Threshold:  -0.020000000000000046  Accuracy:  0.15428571428571428  Precision:  0.7  Recall:  0.08974358974358974
# j:  2  Threshold:  -0.01000000000000005  Accuracy:  0.22285714285714286  Precision:  0.7  Recall:  0.09722222222222222
# j:  2  Threshold:  -5.551115123125783e-17  Accuracy:  0.2857142857142857  Precision:  0.6  Recall:  0.09302325581395349
# j:  2  Threshold:  0.00999999999999994  Accuracy:  0.6114285714285714  Precision:  0.25  Recall:  0.08620689655172414
# j:  2  Threshold:  0.019999999999999934  Accuracy:  0.6742857142857143  Precision:  0.15  Recall:  0.06976744186046512
# j:  2  Threshold:  0.029999999999999943  Accuracy:  0.76  Precision:  0.15  Recall:  0.10714285714285714
# j:  2  Threshold:  0.039999999999999925  Accuracy:  0.8057142857142857  Precision:  0.1  Recall:  0.1111111111111111
# j:  2  Threshold:  0.049999999999999906  Accuracy:  0.8228571428571428  Precision:  0.0  Recall:  0.0
# j:  2  Threshold:  0.059999999999999915  Accuracy:  0.8285714285714286  Precision:  0.0  Recall:  0.0
# j:  2  Threshold:  0.06999999999999992  Accuracy:  0.8571428571428571  Precision:  0.0  Recall:  0.0
# j:  2  Threshold:  0.0799999999999999  Accuracy:  0.8628571428571429  Precision:  0.0  Recall:  0.0
# j:  2  Threshold:  0.08999999999999989  Accuracy:  0.8628571428571429  Precision:  0.0  Recall:  0.0
# j:  2  Threshold:  0.0999999999999999  Accuracy:  0.8628571428571429  Precision:  0.0  Recall:  0.0
# j:  2  Threshold:  0.1099999999999999  Accuracy:  0.8742857142857143  Precision:  0.0  Recall:  0.0
# j:  2  Threshold:  0.11999999999999988  Accuracy:  0.8742857142857143  Precision:  0.0  Recall:  0.0
# j:  2  Threshold:  0.12999999999999987  Accuracy:  0.8742857142857143  Precision:  0.0  Recall:  0.0
# j:  2  Threshold:  0.13999999999999987  Accuracy:  0.88  Precision:  0.0  Recall:  0.0
# j:  2  Threshold:  0.14999999999999988  Accuracy:  0.88  Precision:  0.0  Recall:  0.0
# j:  2  Threshold:  0.1599999999999999  Accuracy:  0.88  Precision:  0.0  Recall:  0.0
# j:  2  Threshold:  0.16999999999999985  Accuracy:  0.8857142857142857  Precision:  0.0  Recall:  -1
# j:  2  Threshold:  0.17999999999999985  Accuracy:  0.8857142857142857  Precision:  0.0  Recall:  -1
# j:  2  Threshold:  0.18999999999999986  Accuracy:  0.8857142857142857  Precision:  0.0  Recall:  -1
# j:  2  Threshold:  0.19999999999999982  Accuracy:  0.8857142857142857  Precision:  0.0  Recall:  -1
# j:  2  Threshold:  0.20999999999999983  Accuracy:  0.8857142857142857  Precision:  0.0  Recall:  -1
# j:  2  Threshold:  0.21999999999999983  Accuracy:  0.8857142857142857  Precision:  0.0  Recall:  -1
# j:  2  Threshold:  0.22999999999999984  Accuracy:  0.8857142857142857  Precision:  0.0  Recall:  -1
# j:  2  Threshold:  0.23999999999999985  Accuracy:  0.8857142857142857  Precision:  0.0  Recall:  -1
# 0.16999999999999985 0.8857142857142857 -0.1 1.0 -0.09000000000000001 0.11494252873563218
# j:  3  Threshold:  -0.1  Accuracy:  0.05714285714285714  Precision:  1.0  Recall:  0.05714285714285714
# j:  3  Threshold:  -0.09000000000000001  Accuracy:  0.05142857142857143  Precision:  0.9  Recall:  0.05172413793103448
# j:  3  Threshold:  -0.08000000000000002  Accuracy:  0.05142857142857143  Precision:  0.9  Recall:  0.05172413793103448
# j:  3  Threshold:  -0.07000000000000002  Accuracy:  0.05714285714285714  Precision:  0.9  Recall:  0.05202312138728324
# j:  3  Threshold:  -0.060000000000000026  Accuracy:  0.05714285714285714  Precision:  0.9  Recall:  0.05202312138728324
# j:  3  Threshold:  -0.05000000000000003  Accuracy:  0.06857142857142857  Precision:  0.9  Recall:  0.05263157894736842
# j:  3  Threshold:  -0.040000000000000036  Accuracy:  0.08  Precision:  0.9  Recall:  0.05325443786982249
# j:  3  Threshold:  -0.03000000000000004  Accuracy:  0.09714285714285714  Precision:  0.9  Recall:  0.05421686746987952
# j:  3  Threshold:  -0.020000000000000046  Accuracy:  0.12571428571428572  Precision:  0.9  Recall:  0.055900621118012424
# j:  3  Threshold:  -0.01000000000000005  Accuracy:  0.2  Precision:  0.9  Recall:  0.060810810810810814
# j:  3  Threshold:  -5.551115123125783e-17  Accuracy:  0.2857142857142857  Precision:  0.8  Recall:  0.061068702290076333
# j:  3  Threshold:  0.00999999999999994  Accuracy:  0.6685714285714286  Precision:  0.8  Recall:  0.125
# j:  3  Threshold:  0.019999999999999934  Accuracy:  0.7485714285714286  Precision:  0.8  Recall:  0.16
# j:  3  Threshold:  0.029999999999999943  Accuracy:  0.7828571428571428  Precision:  0.5  Recall:  0.13157894736842105
# j:  3  Threshold:  0.039999999999999925  Accuracy:  0.8171428571428572  Precision:  0.3  Recall:  0.10714285714285714
# j:  3  Threshold:  0.049999999999999906  Accuracy:  0.8514285714285714  Precision:  0.3  Recall:  0.13636363636363635
# j:  3  Threshold:  0.059999999999999915  Accuracy:  0.8857142857142857  Precision:  0.2  Recall:  0.14285714285714285
# j:  3  Threshold:  0.06999999999999992  Accuracy:  0.8914285714285715  Precision:  0.2  Recall:  0.15384615384615385
# j:  3  Threshold:  0.0799999999999999  Accuracy:  0.9142857142857143  Precision:  0.2  Recall:  0.2222222222222222
# j:  3  Threshold:  0.08999999999999989  Accuracy:  0.9085714285714286  Precision:  0.1  Recall:  0.125
# j:  3  Threshold:  0.0999999999999999  Accuracy:  0.9142857142857143  Precision:  0.1  Recall:  0.14285714285714285
# j:  3  Threshold:  0.1099999999999999  Accuracy:  0.9142857142857143  Precision:  0.1  Recall:  0.14285714285714285
# j:  3  Threshold:  0.11999999999999988  Accuracy:  0.92  Precision:  0.1  Recall:  0.16666666666666666
# j:  3  Threshold:  0.12999999999999987  Accuracy:  0.9314285714285714  Precision:  0.1  Recall:  0.25
# j:  3  Threshold:  0.13999999999999987  Accuracy:  0.9314285714285714  Precision:  0.0  Recall:  0.0
# j:  3  Threshold:  0.14999999999999988  Accuracy:  0.9314285714285714  Precision:  0.0  Recall:  0.0
# j:  3  Threshold:  0.1599999999999999  Accuracy:  0.9314285714285714  Precision:  0.0  Recall:  0.0
# j:  3  Threshold:  0.16999999999999985  Accuracy:  0.9314285714285714  Precision:  0.0  Recall:  0.0
# j:  3  Threshold:  0.17999999999999985  Accuracy:  0.9314285714285714  Precision:  0.0  Recall:  0.0
# j:  3  Threshold:  0.18999999999999986  Accuracy:  0.9314285714285714  Precision:  0.0  Recall:  0.0
# j:  3  Threshold:  0.19999999999999982  Accuracy:  0.9314285714285714  Precision:  0.0  Recall:  0.0
# j:  3  Threshold:  0.20999999999999983  Accuracy:  0.9314285714285714  Precision:  0.0  Recall:  0.0
# j:  3  Threshold:  0.21999999999999983  Accuracy:  0.9314285714285714  Precision:  0.0  Recall:  0.0
# j:  3  Threshold:  0.22999999999999984  Accuracy:  0.9314285714285714  Precision:  0.0  Recall:  0.0
# j:  3  Threshold:  0.23999999999999985  Accuracy:  0.9371428571428572  Precision:  0.0  Recall:  0.0
# 0.23999999999999985 0.9371428571428572 -0.1 1.0 0.12999999999999987 0.25
# j:  4  Threshold:  -0.1  Accuracy:  0.022857142857142857  Precision:  1.0  Recall:  0.022857142857142857
# j:  4  Threshold:  -0.09000000000000001  Accuracy:  0.022857142857142857  Precision:  1.0  Recall:  0.022857142857142857
# j:  4  Threshold:  -0.08000000000000002  Accuracy:  0.022857142857142857  Precision:  1.0  Recall:  0.022857142857142857
# j:  4  Threshold:  -0.07000000000000002  Accuracy:  0.022857142857142857  Precision:  1.0  Recall:  0.022857142857142857
# j:  4  Threshold:  -0.060000000000000026  Accuracy:  0.022857142857142857  Precision:  1.0  Recall:  0.022857142857142857
# j:  4  Threshold:  -0.05000000000000003  Accuracy:  0.02857142857142857  Precision:  1.0  Recall:  0.022988505747126436
# j:  4  Threshold:  -0.040000000000000036  Accuracy:  0.02857142857142857  Precision:  1.0  Recall:  0.022988505747126436
# j:  4  Threshold:  -0.03000000000000004  Accuracy:  0.045714285714285714  Precision:  1.0  Recall:  0.023391812865497075
# j:  4  Threshold:  -0.020000000000000046  Accuracy:  0.06285714285714286  Precision:  1.0  Recall:  0.023809523809523808
# j:  4  Threshold:  -0.01000000000000005  Accuracy:  0.08  Precision:  1.0  Recall:  0.024242424242424242
# j:  4  Threshold:  -5.551115123125783e-17  Accuracy:  0.12  Precision:  0.75  Recall:  0.019230769230769232
# j:  4  Threshold:  0.00999999999999994  Accuracy:  0.45714285714285713  Precision:  0.0  Recall:  0.0
# j:  4  Threshold:  0.019999999999999934  Accuracy:  0.5371428571428571  Precision:  0.0  Recall:  0.0
# j:  4  Threshold:  0.029999999999999943  Accuracy:  0.6171428571428571  Precision:  0.0  Recall:  0.0
# j:  4  Threshold:  0.039999999999999925  Accuracy:  0.6914285714285714  Precision:  0.0  Recall:  0.0
# j:  4  Threshold:  0.049999999999999906  Accuracy:  0.7257142857142858  Precision:  0.0  Recall:  0.0
# j:  4  Threshold:  0.059999999999999915  Accuracy:  0.7771428571428571  Precision:  0.0  Recall:  0.0
# j:  4  Threshold:  0.06999999999999992  Accuracy:  0.8285714285714286  Precision:  0.0  Recall:  0.0
# j:  4  Threshold:  0.0799999999999999  Accuracy:  0.8742857142857143  Precision:  0.0  Recall:  0.0
# j:  4  Threshold:  0.08999999999999989  Accuracy:  0.9028571428571428  Precision:  0.0  Recall:  0.0
# j:  4  Threshold:  0.0999999999999999  Accuracy:  0.9314285714285714  Precision:  0.0  Recall:  0.0
# j:  4  Threshold:  0.1099999999999999  Accuracy:  0.9485714285714286  Precision:  0.0  Recall:  0.0
# j:  4  Threshold:  0.11999999999999988  Accuracy:  0.9657142857142857  Precision:  0.0  Recall:  0.0
# j:  4  Threshold:  0.12999999999999987  Accuracy:  0.9657142857142857  Precision:  0.0  Recall:  0.0
# j:  4  Threshold:  0.13999999999999987  Accuracy:  0.9714285714285714  Precision:  0.0  Recall:  0.0
# j:  4  Threshold:  0.14999999999999988  Accuracy:  0.9771428571428571  Precision:  0.0  Recall:  -1
# j:  4  Threshold:  0.1599999999999999  Accuracy:  0.9771428571428571  Precision:  0.0  Recall:  -1
# j:  4  Threshold:  0.16999999999999985  Accuracy:  0.9771428571428571  Precision:  0.0  Recall:  -1
# j:  4  Threshold:  0.17999999999999985  Accuracy:  0.9771428571428571  Precision:  0.0  Recall:  -1
# j:  4  Threshold:  0.18999999999999986  Accuracy:  0.9771428571428571  Precision:  0.0  Recall:  -1
# j:  4  Threshold:  0.19999999999999982  Accuracy:  0.9771428571428571  Precision:  0.0  Recall:  -1
# j:  4  Threshold:  0.20999999999999983  Accuracy:  0.9771428571428571  Precision:  0.0  Recall:  -1
# j:  4  Threshold:  0.21999999999999983  Accuracy:  0.9771428571428571  Precision:  0.0  Recall:  -1
# j:  4  Threshold:  0.22999999999999984  Accuracy:  0.9771428571428571  Precision:  0.0  Recall:  -1
# j:  4  Threshold:  0.23999999999999985  Accuracy:  0.9771428571428571  Precision:  0.0  Recall:  -1
# 0.14999999999999988 0.9771428571428571 -0.1 1.0 -0.01000000000000005 0.024242424242424242
# j:  5  Threshold:  -0.1  Accuracy:  0.005714285714285714  Precision:  1.0  Recall:  0.005714285714285714
# j:  5  Threshold:  -0.09000000000000001  Accuracy:  0.005714285714285714  Precision:  1.0  Recall:  0.005714285714285714
# j:  5  Threshold:  -0.08000000000000002  Accuracy:  0.011428571428571429  Precision:  1.0  Recall:  0.005747126436781609
# j:  5  Threshold:  -0.07000000000000002  Accuracy:  0.017142857142857144  Precision:  1.0  Recall:  0.005780346820809248
# j:  5  Threshold:  -0.060000000000000026  Accuracy:  0.02857142857142857  Precision:  1.0  Recall:  0.005847953216374269
# j:  5  Threshold:  -0.05000000000000003  Accuracy:  0.04  Precision:  1.0  Recall:  0.005917159763313609
# j:  5  Threshold:  -0.040000000000000036  Accuracy:  0.04  Precision:  1.0  Recall:  0.005917159763313609
# j:  5  Threshold:  -0.03000000000000004  Accuracy:  0.05142857142857143  Precision:  1.0  Recall:  0.005988023952095809
# j:  5  Threshold:  -0.020000000000000046  Accuracy:  0.08571428571428572  Precision:  1.0  Recall:  0.006211180124223602
# j:  5  Threshold:  -0.01000000000000005  Accuracy:  0.14285714285714285  Precision:  1.0  Recall:  0.006622516556291391
# j:  5  Threshold:  -5.551115123125783e-17  Accuracy:  0.21714285714285714  Precision:  1.0  Recall:  0.007246376811594203
# j:  5  Threshold:  0.00999999999999994  Accuracy:  0.5657142857142857  Precision:  1.0  Recall:  0.012987012987012988
# j:  5  Threshold:  0.019999999999999934  Accuracy:  0.64  Precision:  1.0  Recall:  0.015625
# j:  5  Threshold:  0.029999999999999943  Accuracy:  0.72  Precision:  1.0  Recall:  0.02
# j:  5  Threshold:  0.039999999999999925  Accuracy:  0.76  Precision:  1.0  Recall:  0.023255813953488372
# j:  5  Threshold:  0.049999999999999906  Accuracy:  0.8  Precision:  1.0  Recall:  0.027777777777777776
# j:  5  Threshold:  0.059999999999999915  Accuracy:  0.8514285714285714  Precision:  0.0  Recall:  0.0
# j:  5  Threshold:  0.06999999999999992  Accuracy:  0.88  Precision:  0.0  Recall:  0.0
# j:  5  Threshold:  0.0799999999999999  Accuracy:  0.9085714285714286  Precision:  0.0  Recall:  0.0
# j:  5  Threshold:  0.08999999999999989  Accuracy:  0.9085714285714286  Precision:  0.0  Recall:  0.0
# j:  5  Threshold:  0.0999999999999999  Accuracy:  0.9257142857142857  Precision:  0.0  Recall:  0.0
# j:  5  Threshold:  0.1099999999999999  Accuracy:  0.9371428571428572  Precision:  0.0  Recall:  0.0
# j:  5  Threshold:  0.11999999999999988  Accuracy:  0.9428571428571428  Precision:  0.0  Recall:  0.0
# j:  5  Threshold:  0.12999999999999987  Accuracy:  0.9428571428571428  Precision:  0.0  Recall:  0.0
# j:  5  Threshold:  0.13999999999999987  Accuracy:  0.9485714285714286  Precision:  0.0  Recall:  0.0
# j:  5  Threshold:  0.14999999999999988  Accuracy:  0.9542857142857143  Precision:  0.0  Recall:  0.0
# j:  5  Threshold:  0.1599999999999999  Accuracy:  0.9657142857142857  Precision:  0.0  Recall:  0.0
# j:  5  Threshold:  0.16999999999999985  Accuracy:  0.9714285714285714  Precision:  0.0  Recall:  0.0
# j:  5  Threshold:  0.17999999999999985  Accuracy:  0.9714285714285714  Precision:  0.0  Recall:  0.0
# j:  5  Threshold:  0.18999999999999986  Accuracy:  0.9714285714285714  Precision:  0.0  Recall:  0.0
# j:  5  Threshold:  0.19999999999999982  Accuracy:  0.9714285714285714  Precision:  0.0  Recall:  0.0
# j:  5  Threshold:  0.20999999999999983  Accuracy:  0.9714285714285714  Precision:  0.0  Recall:  0.0
# j:  5  Threshold:  0.21999999999999983  Accuracy:  0.9714285714285714  Precision:  0.0  Recall:  0.0
# j:  5  Threshold:  0.22999999999999984  Accuracy:  0.9714285714285714  Precision:  0.0  Recall:  0.0
# j:  5  Threshold:  0.23999999999999985  Accuracy:  0.9771428571428571  Precision:  0.0  Recall:  0.0
# 0.23999999999999985 0.9771428571428571 -0.1 1.0 0.049999999999999906 0.027777777777777776
# j:  6  Threshold:  -0.1  Accuracy:  0.08571428571428572  Precision:  1.0  Recall:  0.08571428571428572
# j:  6  Threshold:  -0.09000000000000001  Accuracy:  0.08571428571428572  Precision:  1.0  Recall:  0.08571428571428572
# j:  6  Threshold:  -0.08000000000000002  Accuracy:  0.09142857142857143  Precision:  1.0  Recall:  0.08620689655172414
# j:  6  Threshold:  -0.07000000000000002  Accuracy:  0.10857142857142857  Precision:  1.0  Recall:  0.08771929824561403
# j:  6  Threshold:  -0.060000000000000026  Accuracy:  0.12  Precision:  1.0  Recall:  0.08875739644970414
# j:  6  Threshold:  -0.05000000000000003  Accuracy:  0.14285714285714285  Precision:  0.9333333333333333  Recall:  0.08588957055214724
# j:  6  Threshold:  -0.040000000000000036  Accuracy:  0.1657142857142857  Precision:  0.9333333333333333  Recall:  0.0880503144654088
# j:  6  Threshold:  -0.03000000000000004  Accuracy:  0.18285714285714286  Precision:  0.9333333333333333  Recall:  0.08974358974358974
# j:  6  Threshold:  -0.020000000000000046  Accuracy:  0.21142857142857144  Precision:  0.9333333333333333  Recall:  0.09271523178807947
# j:  6  Threshold:  -0.01000000000000005  Accuracy:  0.2742857142857143  Precision:  0.8666666666666667  Recall:  0.09420289855072464
# j:  6  Threshold:  -5.551115123125783e-17  Accuracy:  0.4114285714285714  Precision:  0.8  Recall:  0.10714285714285714
# j:  6  Threshold:  0.00999999999999994  Accuracy:  0.7142857142857143  Precision:  0.26666666666666666  Recall:  0.09302325581395349
# j:  6  Threshold:  0.019999999999999934  Accuracy:  0.7828571428571428  Precision:  0.26666666666666666  Recall:  0.12903225806451613
# j:  6  Threshold:  0.029999999999999943  Accuracy:  0.8457142857142858  Precision:  0.26666666666666666  Recall:  0.2
# j:  6  Threshold:  0.039999999999999925  Accuracy:  0.8571428571428571  Precision:  0.2  Recall:  0.1875
# j:  6  Threshold:  0.049999999999999906  Accuracy:  0.8742857142857143  Precision:  0.2  Recall:  0.23076923076923078
# j:  6  Threshold:  0.059999999999999915  Accuracy:  0.8685714285714285  Precision:  0.06666666666666667  Recall:  0.1
# j:  6  Threshold:  0.06999999999999992  Accuracy:  0.8742857142857143  Precision:  0.06666666666666667  Recall:  0.1111111111111111
# j:  6  Threshold:  0.0799999999999999  Accuracy:  0.8742857142857143  Precision:  0.06666666666666667  Recall:  0.1111111111111111
# j:  6  Threshold:  0.08999999999999989  Accuracy:  0.88  Precision:  0.06666666666666667  Recall:  0.125
# j:  6  Threshold:  0.0999999999999999  Accuracy:  0.88  Precision:  0.06666666666666667  Recall:  0.125
# j:  6  Threshold:  0.1099999999999999  Accuracy:  0.8914285714285715  Precision:  0.06666666666666667  Recall:  0.16666666666666666
# j:  6  Threshold:  0.11999999999999988  Accuracy:  0.8971428571428571  Precision:  0.06666666666666667  Recall:  0.2
# j:  6  Threshold:  0.12999999999999987  Accuracy:  0.8971428571428571  Precision:  0.06666666666666667  Recall:  0.2
# j:  6  Threshold:  0.13999999999999987  Accuracy:  0.9028571428571428  Precision:  0.06666666666666667  Recall:  0.25
# j:  6  Threshold:  0.14999999999999988  Accuracy:  0.9028571428571428  Precision:  0.06666666666666667  Recall:  0.25
# j:  6  Threshold:  0.1599999999999999  Accuracy:  0.9028571428571428  Precision:  0.0  Recall:  0.0
# j:  6  Threshold:  0.16999999999999985  Accuracy:  0.9028571428571428  Precision:  0.0  Recall:  0.0
# j:  6  Threshold:  0.17999999999999985  Accuracy:  0.9028571428571428  Precision:  0.0  Recall:  0.0
# j:  6  Threshold:  0.18999999999999986  Accuracy:  0.9085714285714286  Precision:  0.0  Recall:  0.0
# j:  6  Threshold:  0.19999999999999982  Accuracy:  0.9085714285714286  Precision:  0.0  Recall:  0.0
# j:  6  Threshold:  0.20999999999999983  Accuracy:  0.9085714285714286  Precision:  0.0  Recall:  0.0
# j:  6  Threshold:  0.21999999999999983  Accuracy:  0.9085714285714286  Precision:  0.0  Recall:  0.0
# j:  6  Threshold:  0.22999999999999984  Accuracy:  0.9085714285714286  Precision:  0.0  Recall:  0.0
# j:  6  Threshold:  0.23999999999999985  Accuracy:  0.9085714285714286  Precision:  0.0  Recall:  0.0
# 0.18999999999999986 0.9085714285714286 -0.1 1.0 0.13999999999999987 0.25
# j:  7  Threshold:  -0.1  Accuracy:  0.10285714285714286  Precision:  1.0  Recall:  0.10285714285714286
# j:  7  Threshold:  -0.09000000000000001  Accuracy:  0.10857142857142857  Precision:  1.0  Recall:  0.10344827586206896
# j:  7  Threshold:  -0.08000000000000002  Accuracy:  0.10857142857142857  Precision:  1.0  Recall:  0.10344827586206896
# j:  7  Threshold:  -0.07000000000000002  Accuracy:  0.11428571428571428  Precision:  1.0  Recall:  0.10404624277456648
# j:  7  Threshold:  -0.060000000000000026  Accuracy:  0.12  Precision:  1.0  Recall:  0.10465116279069768
# j:  7  Threshold:  -0.05000000000000003  Accuracy:  0.11428571428571428  Precision:  0.9444444444444444  Recall:  0.09941520467836257
# j:  7  Threshold:  -0.040000000000000036  Accuracy:  0.13142857142857142  Precision:  0.8888888888888888  Recall:  0.0963855421686747
# j:  7  Threshold:  -0.03000000000000004  Accuracy:  0.16  Precision:  0.8888888888888888  Recall:  0.09937888198757763
# j:  7  Threshold:  -0.020000000000000046  Accuracy:  0.21714285714285714  Precision:  0.8888888888888888  Recall:  0.10596026490066225
# j:  7  Threshold:  -0.01000000000000005  Accuracy:  0.30857142857142855  Precision:  0.8888888888888888  Recall:  0.11851851851851852
# j:  7  Threshold:  -5.551115123125783e-17  Accuracy:  0.3657142857142857  Precision:  0.7777777777777778  Recall:  0.11570247933884298
# j:  7  Threshold:  0.00999999999999994  Accuracy:  0.7428571428571429  Precision:  0.6666666666666666  Recall:  0.23529411764705882
# j:  7  Threshold:  0.019999999999999934  Accuracy:  0.7771428571428571  Precision:  0.3333333333333333  Recall:  0.18181818181818182
# j:  7  Threshold:  0.029999999999999943  Accuracy:  0.8057142857142857  Precision:  0.2777777777777778  Recall:  0.19230769230769232
# j:  7  Threshold:  0.039999999999999925  Accuracy:  0.8228571428571428  Precision:  0.2222222222222222  Recall:  0.19047619047619047
# j:  7  Threshold:  0.049999999999999906  Accuracy:  0.8571428571428571  Precision:  0.2222222222222222  Recall:  0.26666666666666666
# j:  7  Threshold:  0.059999999999999915  Accuracy:  0.8685714285714285  Precision:  0.1111111111111111  Recall:  0.2222222222222222
# j:  7  Threshold:  0.06999999999999992  Accuracy:  0.8742857142857143  Precision:  0.05555555555555555  Recall:  0.16666666666666666
# j:  7  Threshold:  0.0799999999999999  Accuracy:  0.8857142857142857  Precision:  0.0  Recall:  0.0
# j:  7  Threshold:  0.08999999999999989  Accuracy:  0.8914285714285715  Precision:  0.0  Recall:  0.0
# j:  7  Threshold:  0.0999999999999999  Accuracy:  0.8914285714285715  Precision:  0.0  Recall:  0.0
# j:  7  Threshold:  0.1099999999999999  Accuracy:  0.8914285714285715  Precision:  0.0  Recall:  0.0
# j:  7  Threshold:  0.11999999999999988  Accuracy:  0.8971428571428571  Precision:  0.0  Recall:  -1
# j:  7  Threshold:  0.12999999999999987  Accuracy:  0.8971428571428571  Precision:  0.0  Recall:  -1
# j:  7  Threshold:  0.13999999999999987  Accuracy:  0.8971428571428571  Precision:  0.0  Recall:  -1
# j:  7  Threshold:  0.14999999999999988  Accuracy:  0.8971428571428571  Precision:  0.0  Recall:  -1
# j:  7  Threshold:  0.1599999999999999  Accuracy:  0.8971428571428571  Precision:  0.0  Recall:  -1
# j:  7  Threshold:  0.16999999999999985  Accuracy:  0.8971428571428571  Precision:  0.0  Recall:  -1
# j:  7  Threshold:  0.17999999999999985  Accuracy:  0.8971428571428571  Precision:  0.0  Recall:  -1
# j:  7  Threshold:  0.18999999999999986  Accuracy:  0.8971428571428571  Precision:  0.0  Recall:  -1
# j:  7  Threshold:  0.19999999999999982  Accuracy:  0.8971428571428571  Precision:  0.0  Recall:  -1
# j:  7  Threshold:  0.20999999999999983  Accuracy:  0.8971428571428571  Precision:  0.0  Recall:  -1
# j:  7  Threshold:  0.21999999999999983  Accuracy:  0.8971428571428571  Precision:  0.0  Recall:  -1
# j:  7  Threshold:  0.22999999999999984  Accuracy:  0.8971428571428571  Precision:  0.0  Recall:  -1
# j:  7  Threshold:  0.23999999999999985  Accuracy:  0.8971428571428571  Precision:  0.0  Recall:  -1
# 0.11999999999999988 0.8971428571428571 -0.1 1.0 0.049999999999999906 0.26666666666666666
# j:  8  Threshold:  -0.1  Accuracy:  0.5085714285714286  Precision:  1.0  Recall:  0.5085714285714286
# j:  8  Threshold:  -0.09000000000000001  Accuracy:  0.5085714285714286  Precision:  1.0  Recall:  0.5085714285714286
# j:  8  Threshold:  -0.08000000000000002  Accuracy:  0.5085714285714286  Precision:  1.0  Recall:  0.5085714285714286
# j:  8  Threshold:  -0.07000000000000002  Accuracy:  0.5085714285714286  Precision:  1.0  Recall:  0.5085714285714286
# j:  8  Threshold:  -0.060000000000000026  Accuracy:  0.5028571428571429  Precision:  0.9887640449438202  Recall:  0.5057471264367817
# j:  8  Threshold:  -0.05000000000000003  Accuracy:  0.5085714285714286  Precision:  0.9775280898876404  Recall:  0.5087719298245614
# j:  8  Threshold:  -0.040000000000000036  Accuracy:  0.5142857142857142  Precision:  0.9775280898876404  Recall:  0.5117647058823529
# j:  8  Threshold:  -0.03000000000000004  Accuracy:  0.5142857142857142  Precision:  0.9662921348314607  Recall:  0.5119047619047619
# j:  8  Threshold:  -0.020000000000000046  Accuracy:  0.49142857142857144  Precision:  0.9213483146067416  Recall:  0.5
# j:  8  Threshold:  -0.01000000000000005  Accuracy:  0.49714285714285716  Precision:  0.898876404494382  Recall:  0.5031446540880503
# j:  8  Threshold:  -5.551115123125783e-17  Accuracy:  0.5085714285714286  Precision:  0.8876404494382022  Recall:  0.5096774193548387
# j:  8  Threshold:  0.00999999999999994  Accuracy:  0.5028571428571429  Precision:  0.550561797752809  Recall:  0.5104166666666666
# j:  8  Threshold:  0.019999999999999934  Accuracy:  0.49142857142857144  Precision:  0.48314606741573035  Recall:  0.5
# j:  8  Threshold:  0.029999999999999943  Accuracy:  0.49714285714285716  Precision:  0.4606741573033708  Recall:  0.5061728395061729
# j:  8  Threshold:  0.039999999999999925  Accuracy:  0.49714285714285716  Precision:  0.42696629213483145  Recall:  0.5066666666666667
# j:  8  Threshold:  0.049999999999999906  Accuracy:  0.5085714285714286  Precision:  0.4157303370786517  Recall:  0.5211267605633803
# j:  8  Threshold:  0.059999999999999915  Accuracy:  0.4857142857142857  Precision:  0.33707865168539325  Recall:  0.4918032786885246
# j:  8  Threshold:  0.06999999999999992  Accuracy:  0.49142857142857144  Precision:  0.2696629213483146  Recall:  0.5
# j:  8  Threshold:  0.0799999999999999  Accuracy:  0.5314285714285715  Precision:  0.24719101123595505  Recall:  0.5945945945945946
# j:  8  Threshold:  0.08999999999999989  Accuracy:  0.5257142857142857  Precision:  0.2247191011235955  Recall:  0.5882352941176471
# j:  8  Threshold:  0.0999999999999999  Accuracy:  0.5257142857142857  Precision:  0.19101123595505617  Recall:  0.6071428571428571
# j:  8  Threshold:  0.1099999999999999  Accuracy:  0.52  Precision:  0.16853932584269662  Recall:  0.6
# j:  8  Threshold:  0.11999999999999988  Accuracy:  0.5257142857142857  Precision:  0.15730337078651685  Recall:  0.6363636363636364
# j:  8  Threshold:  0.12999999999999987  Accuracy:  0.5428571428571428  Precision:  0.15730337078651685  Recall:  0.7368421052631579
# j:  8  Threshold:  0.13999999999999987  Accuracy:  0.5371428571428571  Precision:  0.1348314606741573  Recall:  0.75
# j:  8  Threshold:  0.14999999999999988  Accuracy:  0.5314285714285715  Precision:  0.11235955056179775  Recall:  0.7692307692307693
# j:  8  Threshold:  0.1599999999999999  Accuracy:  0.5142857142857142  Precision:  0.07865168539325842  Recall:  0.7
# j:  8  Threshold:  0.16999999999999985  Accuracy:  0.5142857142857142  Precision:  0.07865168539325842  Recall:  0.7
# j:  8  Threshold:  0.17999999999999985  Accuracy:  0.5142857142857142  Precision:  0.07865168539325842  Recall:  0.7
# j:  8  Threshold:  0.18999999999999986  Accuracy:  0.5142857142857142  Precision:  0.07865168539325842  Recall:  0.7
# j:  8  Threshold:  0.19999999999999982  Accuracy:  0.49714285714285716  Precision:  0.033707865168539325  Recall:  0.6
# j:  8  Threshold:  0.20999999999999983  Accuracy:  0.49714285714285716  Precision:  0.033707865168539325  Recall:  0.6
# j:  8  Threshold:  0.21999999999999983  Accuracy:  0.4857142857142857  Precision:  0.011235955056179775  Recall:  0.3333333333333333
# j:  8  Threshold:  0.22999999999999984  Accuracy:  0.4857142857142857  Precision:  0.011235955056179775  Recall:  0.3333333333333333
# j:  8  Threshold:  0.23999999999999985  Accuracy:  0.4857142857142857  Precision:  0.011235955056179775  Recall:  0.3333333333333333
# 0.12999999999999987 0.5428571428571428 -0.1 1.0 0.14999999999999988 0.7692307692307693
# j:  9  Threshold:  -0.1  Accuracy:  0.22857142857142856  Precision:  1.0  Recall:  0.22413793103448276
# j:  9  Threshold:  -0.09000000000000001  Accuracy:  0.22857142857142856  Precision:  1.0  Recall:  0.22413793103448276
# j:  9  Threshold:  -0.08000000000000002  Accuracy:  0.2342857142857143  Precision:  1.0  Recall:  0.2254335260115607
# j:  9  Threshold:  -0.07000000000000002  Accuracy:  0.22285714285714286  Precision:  0.9487179487179487  Recall:  0.21637426900584794
# j:  9  Threshold:  -0.060000000000000026  Accuracy:  0.22285714285714286  Precision:  0.9487179487179487  Recall:  0.21637426900584794
# j:  9  Threshold:  -0.05000000000000003  Accuracy:  0.22285714285714286  Precision:  0.9230769230769231  Recall:  0.21301775147928995
# j:  9  Threshold:  -0.040000000000000036  Accuracy:  0.2342857142857143  Precision:  0.8974358974358975  Recall:  0.21212121212121213
# j:  9  Threshold:  -0.03000000000000004  Accuracy:  0.25142857142857145  Precision:  0.8974358974358975  Recall:  0.21604938271604937
# j:  9  Threshold:  -0.020000000000000046  Accuracy:  0.29714285714285715  Precision:  0.8974358974358975  Recall:  0.22727272727272727
# j:  9  Threshold:  -0.01000000000000005  Accuracy:  0.34285714285714286  Precision:  0.8974358974358975  Recall:  0.23972602739726026
# j:  9  Threshold:  -5.551115123125783e-17  Accuracy:  0.37142857142857144  Precision:  0.8461538461538461  Recall:  0.24087591240875914
# j:  9  Threshold:  0.00999999999999994  Accuracy:  0.5771428571428572  Precision:  0.48717948717948717  Recall:  0.2602739726027397
# j:  9  Threshold:  0.019999999999999934  Accuracy:  0.6571428571428571  Precision:  0.41025641025641024  Recall:  0.3018867924528302
# j:  9  Threshold:  0.029999999999999943  Accuracy:  0.7371428571428571  Precision:  0.41025641025641024  Recall:  0.41025641025641024
# j:  9  Threshold:  0.039999999999999925  Accuracy:  0.7142857142857143  Precision:  0.20512820512820512  Recall:  0.2962962962962963
# j:  9  Threshold:  0.049999999999999906  Accuracy:  0.7314285714285714  Precision:  0.15384615384615385  Recall:  0.3
# j:  9  Threshold:  0.059999999999999915  Accuracy:  0.7542857142857143  Precision:  0.1282051282051282  Recall:  0.35714285714285715
# j:  9  Threshold:  0.06999999999999992  Accuracy:  0.7657142857142857  Precision:  0.1282051282051282  Recall:  0.4166666666666667
# j:  9  Threshold:  0.0799999999999999  Accuracy:  0.7771428571428571  Precision:  0.07692307692307693  Recall:  0.5
# j:  9  Threshold:  0.08999999999999989  Accuracy:  0.7771428571428571  Precision:  0.05128205128205128  Recall:  0.5
# j:  9  Threshold:  0.0999999999999999  Accuracy:  0.7714285714285715  Precision:  0.02564102564102564  Recall:  0.3333333333333333
# j:  9  Threshold:  0.1099999999999999  Accuracy:  0.7771428571428571  Precision:  0.02564102564102564  Recall:  0.5
# j:  9  Threshold:  0.11999999999999988  Accuracy:  0.7771428571428571  Precision:  0.02564102564102564  Recall:  0.5
# j:  9  Threshold:  0.12999999999999987  Accuracy:  0.7771428571428571  Precision:  0.02564102564102564  Recall:  0.5
# j:  9  Threshold:  0.13999999999999987  Accuracy:  0.7714285714285715  Precision:  0.0  Recall:  0.0
# j:  9  Threshold:  0.14999999999999988  Accuracy:  0.7714285714285715  Precision:  0.0  Recall:  0.0
# j:  9  Threshold:  0.1599999999999999  Accuracy:  0.7714285714285715  Precision:  0.0  Recall:  0.0
# j:  9  Threshold:  0.16999999999999985  Accuracy:  0.7714285714285715  Precision:  0.0  Recall:  0.0
# j:  9  Threshold:  0.17999999999999985  Accuracy:  0.7714285714285715  Precision:  0.0  Recall:  0.0
# j:  9  Threshold:  0.18999999999999986  Accuracy:  0.7771428571428571  Precision:  0.0  Recall:  -1
# j:  9  Threshold:  0.19999999999999982  Accuracy:  0.7771428571428571  Precision:  0.0  Recall:  -1
# j:  9  Threshold:  0.20999999999999983  Accuracy:  0.7771428571428571  Precision:  0.0  Recall:  -1
# j:  9  Threshold:  0.21999999999999983  Accuracy:  0.7771428571428571  Precision:  0.0  Recall:  -1
# j:  9  Threshold:  0.22999999999999984  Accuracy:  0.7771428571428571  Precision:  0.0  Recall:  -1
# j:  9  Threshold:  0.23999999999999985  Accuracy:  0.7771428571428571  Precision:  0.0  Recall:  -1
# 0.0799999999999999 0.7771428571428571 -0.1 1.0 0.0799999999999999 0.5




# #Ground Truth:  ['toilet']  vs.  ['toilet']
# For row  0  it found  100.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  0.0 %
# Ground Truth:  ['bottle', 'bowl', 'oven', 'cup', 'fridge', 'bottle']  vs.  ['bottle', 'bowl', 'oven', 'cup', 'refrigerator', 'bottle']
# For row  1  it found  100.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  66.66666666666667 %, while using synonyms gives  66.66666666666667 %
# Ground Truth:  ['alarm_clock', 'chair']  vs.  ['clock', 'chair']
# For row  2  it found  100.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  50.0 %, while using synonyms gives  0.0 %
# Ground Truth:  ['chair']  vs.  ['chair']
# For row  3  it found  100.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  0.0 %
# Ground Truth:  ['extra', 'extra', 'incorrect', 'extra', 'dining_table', 'extra', 'incorrect', 'chair']  vs.  ['frisbee', 'tv', 'cup', 'tv', 'dining_table', 'kite', 'bed', 'chair']
# For row  4  it found  50.0 %, objects correctly, with  4  extras.
# Per image accuracy (actual aithor word) is  0.0 %
# Ground Truth:  ['house_plant', 'vase', 'sofa', 'sofa', 'dining_table', 'extra']  vs.  ['potted_plant', 'vase', 'couch', 'couch', 'dining_table', 'chair']
# For row  5  it found  100.0 %, objects correctly, with  1  extras.
# Per image accuracy (actual aithor word) is  20.0 %, while using synonyms gives  20.0 %
# Ground Truth:  ['house_plant', 'television', 'remote_control', 'vase']  vs.  ['potted_plant', 'tv', 'remote', 'vase']
# For row  7  it found  100.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  25.0 %, while using synonyms gives  50.0 %
# Ground Truth:  ['sink', 'incorrect']  vs.  ['sink', 'sink']
# For row  9  it found  50.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  0.0 %
# Ground Truth:  ['laptop', 'house_plant', 'chair', 'vase', 'dining_table', 'vase', 'chair']  vs.  ['laptop', 'potted_plant', 'chair', 'vase', 'dining_table', 'vase', 'chair']
# For row  10  it found  100.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  42.857142857142854 %, while using synonyms gives  42.857142857142854 %
# Ground Truth:  ['sink', 'cup', 'incorrect', 'bottle', 'incorrect', 'extra', 'extra']  vs.  ['sink', 'cup', 'sports_ball', 'bottle', 'sports_ball', 'sink', 'sink']
# For row  11  it found  60.0 %, objects correctly, with  2  extras.
# Per image accuracy (actual aithor word) is  66.66666666666667 %
# Ground Truth:  ['chair', 'dining_table', 'fridge']  vs.  ['chair', 'dining_table', 'refrigerator']
# For row  12  it found  100.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  0.0 %, while using synonyms gives  0.0 %
# Ground Truth:  ['laptop', 'chair', 'house_plant', 'book', 'vase', 'cup', 'extra', 'cell_phone', 'incorrect', 'extra', 'alarm_clock', 'extra']  vs.  ['laptop', 'chair', 'potted_plant', 'book', 'vase', 'cup', 'chair', 'cell_phone', 'tv', 'chair', 'clock', 'vase']
# For row  13  it found  88.88888888888889 %, objects correctly, with  3  extras.
# Per image accuracy (actual aithor word) is  62.5 %, while using synonyms gives  50.0 %
# Ground Truth:  ['extra', 'sofa', 'remote_control', 'dining_table']  vs.  ['bed', 'couch', 'remote', 'dining_table']
# For row  16  it found  100.0 %, objects correctly, with  1  extras.
# Per image accuracy (actual aithor word) is  0.0 %, while using synonyms gives  0.0 %
# Ground Truth:  ['incorrect']  vs.  ['traffic_light']
# For row  17  it found  0.0 %, objects correctly, with  0  extras.
# Nothing correct to run through conceptnet.
# Ground Truth:  ['fridge']  vs.  ['refrigerator']
# For row  18  it found  100.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  0.0 %, while using synonyms gives  0.0 %
# Ground Truth:  ['incorrect', 'incorrect']  vs.  ['airplane', 'airplane']
# For row  19  it found  0.0 %, objects correctly, with  0  extras.
# Nothing correct to run through conceptnet.
# Ground Truth:  ['chair']  vs.  ['chair']
# For row  20  it found  100.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  0.0 %
# Ground Truth:  ['television', 'incorrect', 'incorrect']  vs.  ['tv', 'person', 'refrigerator']
# For row  21  it found  33.333333333333336 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  0.0 %, while using synonyms gives  100.0 %
# Ground Truth:  ['incorrect']  vs.  ['mouse']
# For row  22  it found  0.0 %, objects correctly, with  0  extras.
# Nothing correct to run through conceptnet.
# Ground Truth:  ['incorrect', 'laptop', 'television', 'sofa', 'house_plant', 'vase', 'vase', 'incorrect', 'vase', 'laptop']  vs.  ['frisbee', 'laptop', 'tv', 'couch', 'potted_plant', 'vase', 'vase', 'toilet', 'vase', 'keyboard']
# For row  23  it found  80.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  62.5 %, while using synonyms gives  75.0 %
# Ground Truth:  ['house_plant', 'television', 'extra', 'vase', 'chair']  vs.  ['potted_plant', 'tv', 'couch', 'vase', 'chair']
# For row  24  it found  100.0 %, objects correctly, with  1  extras.
# Per image accuracy (actual aithor word) is  25.0 %, while using synonyms gives  50.0 %
# Ground Truth:  ['potted_plant', 'oven', 'bowl', 'vase', 'incorrect']  vs.  ['potted_plant', 'oven', 'bowl', 'vase', 'frisbee']
# For row  25  it found  80.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  50.0 %
# Ground Truth:  ['toilet', 'extra']  vs.  ['toilet', 'toilet']
# For row  26  it found  100.0 %, objects correctly, with  1  extras.
# Per image accuracy (actual aithor word) is  0.0 %
# Ground Truth:  ['laptop', 'bowl', 'cell_phone', 'book', 'dining_table']  vs.  ['laptop', 'bowl', 'cell_phone', 'book', 'dining_table']
# For row  27  it found  100.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  60.0 %
# Ground Truth:  ['chair', 'chair', 'laptop', 'chair']  vs.  ['chair', 'chair', 'laptop', 'chair']
# For row  28  it found  100.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  25.0 %
# Ground Truth:  ['vase', 'house_plant', 'chair', 'vase', 'sofa', 'remote_control', 'chair']  vs.  ['vase', 'potted_plant', 'chair', 'vase', 'couch', 'remote', 'chair']
# For row  31  it found  100.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  28.571428571428573 %, while using synonyms gives  28.571428571428573 %
# Ground Truth:  ['chair', 'incorrect', 'incorrect']  vs.  ['chair', 'sports_ball', 'toilet']
# For row  32  it found  33.333333333333336 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  0.0 %
# Ground Truth:  ['sink']  vs.  ['sink']
# For row  33  it found  100.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  0.0 %
# Ground Truth:  ['chair', 'incorrect', 'chair', 'chair', 'chair', 'chair', 'chair', 'extra']  vs.  ['chair', 'bed', 'chair', 'chair', 'chair', 'chair', 'chair', 'chair']
# For row  35  it found  85.71428571428571 %, objects correctly, with  1  extras.
# Per image accuracy (actual aithor word) is  0.0 %
# Ground Truth:  ['vase', 'house_plant', 'incorrect', 'chair']  vs.  ['vase', 'potted_plant', 'tv', 'chair']
# For row  36  it found  75.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  33.333333333333336 %, while using synonyms gives  33.333333333333336 %
# Ground Truth:  ['sink']  vs.  ['sink']
# For row  37  it found  100.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  0.0 %
# Ground Truth:  ['sink', 'bottle']  vs.  ['sink', 'bottle']
# For row  38  it found  100.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  50.0 %
# Ground Truth:  ['incorrect']  vs.  ['tv']
# For row  39  it found  0.0 %, objects correctly, with  0  extras.
# Nothing correct to run through conceptnet.
# Ground Truth:  ['sofa', 'sofa', 'television', 'incorrect', 'remote_control']  vs.  ['couch', 'couch', 'tv', 'tv', 'remote']
# For row  40  it found  80.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  0.0 %, while using synonyms gives  25.0 %
# Ground Truth:  ['oven', 'microwave']  vs.  ['oven', 'microwave']
# For row  41  it found  100.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  0.0 %
# Ground Truth:  ['chair', 'house_plant', 'sofa', 'television', 'vase', 'book']  vs.  ['chair', 'potted_plant', 'couch', 'tv', 'vase', 'book']
# For row  42  it found  100.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  16.666666666666668 %, while using synonyms gives  33.333333333333336 %
# Ground Truth:  ['cup', 'sink', 'microwave', 'bottle']  vs.  ['cup', 'sink', 'microwave', 'bottle']
# For row  43  it found  100.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  50.0 %
# Ground Truth:  ['vase', 'dining_table', 'incorrect', 'extra', 'chair', 'chair', 'incorrect']  vs.  ['vase', 'dining_table', 'frisbee', 'dining_table', 'chair', 'chair', 'frisbee']
# For row  44  it found  66.66666666666667 %, objects correctly, with  1  extras.
# Per image accuracy (actual aithor word) is  25.0 %
# Ground Truth:  ['vase', 'bed', 'alarm_clock']  vs.  ['vase', 'bed', 'clock']
# For row  45  it found  100.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  66.66666666666667 %, while using synonyms gives  33.333333333333336 %
# Ground Truth:  ['television', 'chair', 'chair', 'extra', 'sofa']  vs.  ['tv', 'chair', 'chair', 'chair', 'couch']
# For row  46  it found  100.0 %, objects correctly, with  1  extras.
# Per image accuracy (actual aithor word) is  0.0 %, while using synonyms gives  25.0 %
# Ground Truth:  ['incorrect', 'bottle', 'sink', 'house_plant', 'vase', 'fridge', 'dining_table', 'incorrect', 'cup']  vs.  ['umbrella', 'bottle', 'sink', 'potted_plant', 'vase', 'refrigerator', 'dining_table', 'umbrella', 'cup']
# For row  47  it found  77.77777777777777 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  42.857142857142854 %, while using synonyms gives  42.857142857142854 %
# Ground Truth:  ['incorrect']  vs.  ['stop_sign']
# For row  48  it found  0.0 %, objects correctly, with  0  extras.
# Nothing correct to run through conceptnet.
# Ground Truth:  ['sink']  vs.  ['sink']
# For row  49  it found  100.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  0.0 %
# Ground Truth:  ['fridge', 'bowl', 'sink', 'bowl']  vs.  ['refrigerator', 'bowl', 'sink', 'bowl']
# For row  50  it found  100.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  50.0 %, while using synonyms gives  50.0 %
# Ground Truth:  ['sink', 'bottle']  vs.  ['sink', 'bottle']
# For row  52  it found  100.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  50.0 %
# Ground Truth:  ['incorrect', 'bed', 'incorrect']  vs.  ['frisbee', 'bed', 'bench']
# For row  53  it found  33.333333333333336 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  0.0 %
# Ground Truth:  ['bottle', 'extra']  vs.  ['bottle', 'bottle']
# For row  56  it found  100.0 %, objects correctly, with  1  extras.
# Per image accuracy (actual aithor word) is  100.0 %
# Ground Truth:  ['laptop', 'chair', 'chair', 'extra']  vs.  ['laptop', 'chair', 'chair', 'couch']
# For row  57  it found  100.0 %, objects correctly, with  1  extras.
# Per image accuracy (actual aithor word) is  33.333333333333336 %
# Ground Truth:  ['chair', 'laptop', 'bowl', 'chair', 'dining_table']  vs.  ['chair', 'laptop', 'bowl', 'chair', 'dining_table']
# For row  58  it found  100.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  40.0 %
# Ground Truth:  ['house_plant', 'vase']  vs.  ['potted_plant', 'vase']
# For row  59  it found  100.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  50.0 %, while using synonyms gives  50.0 %
# Ground Truth:  ['bottle', 'chair', 'incorrect', 'microwave']  vs.  ['bottle', 'chair', 'mouse', 'microwave']
# For row  60  it found  75.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  33.333333333333336 %
# Ground Truth:  ['house_plant', 'television', 'incorrect', 'chair', 'extra']  vs.  ['potted_plant', 'tv', 'mouse', 'chair', 'couch']
# For row  62  it found  75.0 %, objects correctly, with  1  extras.
# Per image accuracy (actual aithor word) is  0.0 %, while using synonyms gives  33.333333333333336 %
# Ground Truth:  ['house_plant', 'incorrect', 'chair', 'vase', 'book']  vs.  ['potted_plant', 'person', 'chair', 'vase', 'book']
# For row  64  it found  80.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  25.0 %, while using synonyms gives  25.0 %
# Ground Truth:  ['house_plant', 'sink', 'vase']  vs.  ['potted_plant', 'sink', 'vase']
# For row  65  it found  100.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  33.333333333333336 %, while using synonyms gives  33.333333333333336 %
# Ground Truth:  ['house_plant', 'bottle', 'sink', 'bowl', 'cup', 'bottle', 'cup', 'incorrect', 'incorrect', 'bowl', 'bowl']  vs.  ['potted_plant', 'bottle', 'sink', 'bowl', 'wine_glass', 'bottle', 'wine_glass', 'frisbee', 'frisbee', 'bowl', 'bowl']
# For row  66  it found  81.81818181818181 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  77.77777777777777 %, while using synonyms gives  55.55555555555556 %
# Ground Truth:  ['toilet', 'incorrect']  vs.  ['toilet', 'toilet']
# For row  67  it found  50.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  0.0 %
# Ground Truth:  ['laptop']  vs.  ['laptop']
# For row  68  it found  100.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  100.0 %
# Ground Truth:  ['sink', 'vase', 'sink', 'cup']  vs.  ['sink', 'vase', 'sink', 'cup']
# For row  69  it found  100.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  50.0 %
# Ground Truth:  ['house_plant', 'sofa', 'incorrect', 'vase', 'incorrect', 'vase', 'laptop']  vs.  ['potted_plant', 'couch', 'mouse', 'vase', 'bird', 'vase', 'laptop']
# For row  70  it found  71.42857142857143 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  60.0 %, while using synonyms gives  60.0 %
# Ground Truth:  ['incorrect']  vs.  ['stop_sign']
# For row  71  it found  0.0 %, objects correctly, with  0  extras.
# Nothing correct to run through conceptnet.
# Ground Truth:  ['microwave', 'bottle', 'incorrect']  vs.  ['microwave', 'bottle', 'dining_table']
# For row  73  it found  66.66666666666667 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  50.0 %
# Ground Truth:  ['incorrect']  vs.  ['dining_table']
# For row  75  it found  0.0 %, objects correctly, with  0  extras.
# Nothing correct to run through conceptnet.
# Ground Truth:  ['incorrect']  vs.  ['refrigerator']
# For row  78  it found  0.0 %, objects correctly, with  0  extras.
# Nothing correct to run through conceptnet.
# Ground Truth:  ['incorrect']  vs.  ['chair']
# For row  79  it found  0.0 %, objects correctly, with  0  extras.
# Nothing correct to run through conceptnet.
# Ground Truth:  ['chair']  vs.  ['chair']
# For row  80  it found  100.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  0.0 %
# Ground Truth:  ['bottle', 'bottle', 'bottle', 'bottle', 'bottle', 'bottle', 'bed', 'remote_control']  vs.  ['bottle', 'bottle', 'bottle', 'bottle', 'bottle', 'bottle', 'bed', 'remote']
# For row  81  it found  100.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  75.0 %, while using synonyms gives  75.0 %
# Ground Truth:  ['incorrect']  vs.  ['chair']
# For row  82  it found  0.0 %, objects correctly, with  0  extras.
# Nothing correct to run through conceptnet.
# Ground Truth:  ['house_plant', 'book', 'alarm_clock', 'vase', 'book']  vs.  ['potted_plant', 'book', 'clock', 'vase', 'book']
# For row  84  it found  100.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  40.0 %, while using synonyms gives  20.0 %
# Ground Truth:  ['sofa', 'incorrect', 'chair', 'sofa', 'extra', 'extra']  vs.  ['couch', 'airplane', 'chair', 'couch', 'chair', 'couch']
# For row  85  it found  75.0 %, objects correctly, with  2  extras.
# Per image accuracy (actual aithor word) is  0.0 %, while using synonyms gives  0.0 %
# Ground Truth:  ['incorrect']  vs.  ['umbrella']
# For row  86  it found  0.0 %, objects correctly, with  0  extras.
# Nothing correct to run through conceptnet.
# Ground Truth:  ['incorrect', 'incorrect', 'laptop']  vs.  ['vase', 'dining_table', 'laptop']
# For row  89  it found  33.333333333333336 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  100.0 %
# Ground Truth:  ['fridge']  vs.  ['refrigerator']
# For row  91  it found  100.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  0.0 %, while using synonyms gives  0.0 %
# Ground Truth:  ['sink', 'bottle']  vs.  ['sink', 'bottle']
# For row  92  it found  100.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  50.0 %
# Ground Truth:  ['sink']  vs.  ['sink']
# For row  93  it found  100.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  0.0 %
# Ground Truth:  ['toilet', 'incorrect', 'sink']  vs.  ['toilet', 'chair', 'sink']
# For row  94  it found  66.66666666666667 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  0.0 %
# Ground Truth:  ['cup', 'cup']  vs.  ['cup', 'cup']
# For row  97  it found  100.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  100.0 %
# Ground Truth:  ['chair', 'incorrect']  vs.  ['chair', 'bird']
# For row  98  it found  50.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  0.0 %
# Ground Truth:  ['bed', 'alarm_clock']  vs.  ['bed', 'clock']
# For row  99  it found  100.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  50.0 %, while using synonyms gives  0.0 %
# Ground Truth:  ['chair', 'chair', 'chair', 'chair', 'sofa', 'chair', 'chair', 'extra', 'extra']  vs.  ['chair', 'chair', 'chair', 'chair', 'couch', 'chair', 'chair', 'chair', 'chair']
# For row  100  it found  100.0 %, objects correctly, with  2  extras.
# Per image accuracy (actual aithor word) is  0.0 %, while using synonyms gives  0.0 %
# Ground Truth:  ['television', 'chair', 'sofa', 'remote_control']  vs.  ['tv', 'chair', 'couch', 'remote']
# For row  102  it found  100.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  0.0 %, while using synonyms gives  25.0 %
# Ground Truth:  ['chair']  vs.  ['chair']
# For row  103  it found  100.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  0.0 %
# Ground Truth:  ['incorrect']  vs.  ['tv']
# For row  104  it found  0.0 %, objects correctly, with  0  extras.
# Nothing correct to run through conceptnet.
# Ground Truth:  ['sink']  vs.  ['sink']
# For row  105  it found  100.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  0.0 %
# Ground Truth:  ['fridge']  vs.  ['refrigerator']
# For row  106  it found  100.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  0.0 %, while using synonyms gives  0.0 %
# Ground Truth:  ['bottle', 'bottle', 'bottle', 'bottle', 'cup', 'sink', 'extra', 'dining_table', 'bottle', 'bottle', 'bottle', 'bottle', 'extra']  vs.  ['bottle', 'bottle', 'bottle', 'bottle', 'cup', 'sink', 'sink', 'dining_table', 'bottle', 'bottle', 'bottle', 'bottle', 'dining_table']
# For row  107  it found  100.0 %, objects correctly, with  2  extras.
# Per image accuracy (actual aithor word) is  81.81818181818181 %
# Ground Truth:  ['incorrect', 'teddy_bear']  vs.  ['dining_table', 'teddy_bear']
# For row  109  it found  50.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  0.0 %
# Ground Truth:  ['incorrect']  vs.  ['airplane']
# For row  110  it found  0.0 %, objects correctly, with  0  extras.
# Nothing correct to run through conceptnet.
# Ground Truth:  ['chair', 'chair', 'television', 'cup']  vs.  ['chair', 'chair', 'tv', 'cup']
# For row  111  it found  100.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  25.0 %, while using synonyms gives  50.0 %
# Ground Truth:  ['television', 'incorrect', 'house_plant']  vs.  ['tv', 'bed', 'potted_plant']
# For row  112  it found  66.66666666666667 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  0.0 %, while using synonyms gives  50.0 %
# Ground Truth:  ['toilet', 'extra', 'sink']  vs.  ['toilet', 'toilet', 'sink']
# For row  113  it found  100.0 %, objects correctly, with  1  extras.
# Per image accuracy (actual aithor word) is  0.0 %
# Ground Truth:  ['laptop', 'book', 'chair']  vs.  ['laptop', 'book', 'chair']
# For row  114  it found  100.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  33.333333333333336 %
# Ground Truth:  ['vase', 'vase', 'vase', 'sofa', 'extra']  vs.  ['vase', 'vase', 'vase', 'couch', 'chair']
# For row  115  it found  100.0 %, objects correctly, with  1  extras.
# Per image accuracy (actual aithor word) is  75.0 %, while using synonyms gives  75.0 %
# Ground Truth:  ['chair', 'chair', 'laptop', 'dining_table', 'chair', 'extra', 'extra']  vs.  ['chair', 'chair', 'laptop', 'dining_table', 'chair', 'chair', 'dining_table']
# For row  116  it found  100.0 %, objects correctly, with  2  extras.
# Per image accuracy (actual aithor word) is  20.0 %
# Ground Truth:  ['incorrect']  vs.  ['sink']
# For row  117  it found  0.0 %, objects correctly, with  0  extras.
# Nothing correct to run through conceptnet.
# Ground Truth:  ['bed', 'bed']  vs.  ['bed', 'bed']
# For row  118  it found  100.0 %, objects correctly, with  0  extras.
# Per image accuracy (actual aithor word) is  0.0 %
# Object detection accuracy:  76.0932944606414 %
# overall direct nlp accuracy:  36.39846743295019 %
# Overall accuracy using synonyms:  37.93103448275862 %
# overall full accuracy direct:  27.696793002915452 %
# overall full accuracy synonyms:  28.862973760932945 %