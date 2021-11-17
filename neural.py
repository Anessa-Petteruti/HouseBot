import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle
from conceptnet import calculateProbBySimilarVerbs
from keras.models import Sequential
from keras.layers import Dense
import spacy

aithor_verbs = ["Toggleable","Breakable","Fillable","Dirtyable","UsedUp","Cookable","Sliceable","Openable","Pickupable","Moveable"]

train_test_split = 0.8

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
    numTrain = round(len(data)*train_test_split)
    return data[:numTrain], labels[:numTrain], data[numTrain:], labels[numTrain:]

def main():
    data, labels = getData()
    train_data, train_labels, test_data, test_labels = splitData(data, labels)
    print(train_data)
    print(test_data)

    # extract word embeddings for objects in train data
    # pip install spacy
    # python -m 
    nlp = spacy.load("en_core_web_md")
    for i in range(len(train_data)):
        train_data[i] = nlp(train_data[i])
        train_data[i] = train_data[i].vector
    
    for i in range(len(test_data)):
        test_data[i] = nlp(test_data[i])
        test_data[i] = test_data[i].vector

    train_data = np.stack(np.array(train_data), axis=0)
    test_data = np.stack(np.array(test_data), axis=0)
    train_labels = np.multiply(train_labels, 1)
    test_labels = np.multiply(test_labels, 1)

    model = Sequential()
    model.add(Dense(100, input_dim=300, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(train_data.shape)
    print(train_labels.shape)
    print(test_data.shape)
    print(test_labels.shape)
    model.fit(train_data, train_labels, epochs=30, batch_size=5)
    pred, accuracy = model.evaluate(test_data, test_labels)
    print(pred)
    print('Accuracy: %.2f' % (accuracy*100))

if __name__ == "__main__":
    main()