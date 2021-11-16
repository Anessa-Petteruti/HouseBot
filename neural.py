import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from conceptnet import calculateProbBySimilarVerbs

aithor_verbs = ["Toggleable","Breakable","Fillable","Dirtable","UsedUp","Cookable","Sliceable","Openable","Pickupable","Moveable"]

def getData():
    df = pd.read_csv('ithor.csv')
    objects = []
    labels = []
    for index, row in df.iterrows():
        object = row['Object Type']
        verbLabels = row['Actionable Properties']
        if pd.isna(verbLabels):
            verbLabels = []
        else:
            verbLabels = verbLabels.replace(" (Some)","")
            verbLabels = verbLabels.split(", ")
        actions = [verb in verbLabels for verb in aithor_verbs]
        objects.append(object)
        labels.append(actions)
    return objects, labels

def splitData(data, labels):
    data,labels = shuffle(data, labels)
    #Using a 80/20 train, test split for now
    numTrain = round(len(data)*.8)
    return data[:numTrain], labels[:numTrain], data[numTrain:], labels[numTrain:]

def main():
    data, labels = getData()
    train_data, train_labels, test_data, test_labels = splitData(data, labels)
    #train on train data
    #test on test data

if __name__ == "__main__":
    main()