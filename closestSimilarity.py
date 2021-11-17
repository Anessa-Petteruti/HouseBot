import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
import requests
import re
basic_url = 'http://api.conceptnet.io'
relatedness1 = '/relatedness?node1=/c/en/'
relatedness2 = '&node2=/c/en/'
query_url1 = 'http://api.conceptnet.io/query?start=/c/en/'
query_url2 = '&rel=/r/CapableOf&limit=20'

aithor_verbs = ["Toggleable","Breakable","Fillable","Dirtable","UsedUp","Cookable","Sliceable","Openable","Pickupable","Moveable"]

def getData():
    df = pd.read_csv('ithor.csv')
    objects = []
    labels = []
    for index, row in df.iterrows():
        #object = row['Object Type']
        object = re.sub('(?<!^)(?=[A-Z])','_',row['Object Type']).lower()
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

def splitData(data, labels, percent):
    data,labels = shuffle(data, labels)
    #Using a 80/20 train, test split for now
    numTrain = round(len(data)*percent/100)
    return data[:numTrain], labels[:numTrain], data[numTrain:], labels[numTrain:]

def createDictionary(data,labels):
    dictionary = {}
    for i in range(len(data)):
        dictionary[data[i]] = labels[i]
    return dictionary

def train(data,labels,dictionary):
    accuracy = 0
    for i in range(len(data)):
        print(i,":", len(data), data[i])
        output = calculateLabels(data[i],dictionary)
        f1 = f1_score(output,labels[i])
        print(output,f1)
        accuracy += f1
    return accuracy / len(data)

def calculateLabels(data, dictionary):
    objects = []
    similarity = []
    for item, value in dictionary.items():
        objects.append(item)
        relatedness = requests.get(basic_url + relatedness1 + item + relatedness2 + data)
        if relatedness:
            similarity.append(relatedness.json()['value'])
            print(item, relatedness.json()['value'])
        else:
            similarity.append(0)
            print(item, -1)
    sorted = np.argsort(similarity)
    answer = [0,0,0,0,0,0,0,0,0,0]
    for i in range(5):
        dic = dictionary[objects[sorted[i]]]
        for n in range(len(answer)):
            if dic[n]:
                answer[n] += 1
    return [element > 2 for element in answer]

def main():
    data, labels = getData()
    train_data, train_labels, test_data, test_labels = splitData(data, labels, 20)
    #train on train data
    dictionary = createDictionary(train_data,train_labels)
    #test on test data
    acc = train(test_data,test_labels,dictionary)
    print("At ",percent, "% training data, accuracy is", acc)

if __name__ == "__main__":
    main()