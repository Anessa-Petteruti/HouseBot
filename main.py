import os
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score
from conceptnet import calculateProbBySimilarVerbs, calculateProbBySimilarity
#import ConceptNet

threshold = .001
#aithor_verbs = ["Toggleable","Breakable","Fillable","Dirtable","UsedUp","Cookable","Heatable","Coldable","Sliceable","Openable","Pickupable","Moveable"]
aithor_verbs = ["Toggleable","Breakable","Fillable","Dirtable","UsedUp","Cookable","Sliceable","Openable","Pickupable","Moveable"]

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

def main():
    # Will need to pass things in to these functions here...
    #process_object_labels()

    sample_test('Mug')

#RECEPTABCLE HEAT AND COLD NEED OT BE ADDED

if __name__ == "__main__":
    main()
