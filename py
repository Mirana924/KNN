import csv
import random
import math
import operator
import numpy as np
import pandas as pd

def normMin(dataset):
    classes = dataset['classes']
    dataset = dataset[['var1','var2','var3', 'var4']]
    dataNorm=((dataset-dataset.min())/(dataset.max()-dataset.min()))
    dataNorm['classes']=classes
    return dataNorm

def normAvg(dataset):
    classes = dataset['classes']
    dataset = dataset[['var1','var2','var3', 'var4']]
    dataNorm=((dataset-dataset.mean())/(dataset.max()-dataset.min()))
    dataNorm['classes']=classes
    return dataNorm

def normZscore(dataset):
    classes = dataset['classes']
    dataset = dataset[['var1','var2','var3', 'var4']]
    dataNorm=((dataset-dataset.mean())/dataset.std(ddof=0))
    dataNorm['classes']=classes
    return dataNorm


def splitDataset(df, split):
    trainingSet, testSet = [], []
    for x in range(len(df)-1):
        for y in range(4):
            df[x][y] = float(df[x][y])
        if random.random() < split:
            trainingSet.append(df[x])
        else:
            testSet.append(df[x])
    return trainingSet, testSet

 # use minkowski_distance instead below, just for reference
# def euclideanDistance(instance1, instance2, length):
#     distance = 0
#     for x in range(length):
#         distance += pow((instance1[x] - instance2[x]), 2)
#     return math.sqrt(distance)

# def manhattanDistance(instance1, instance2, length):
#     distance = 0
#     for i in range(length):
#         for j in range(i+1, length):
#             distance += (abs(instance1[i] - instance1[j]) + abs(instance2[i] - instance2[j]))
#     return distance


# minkowski_distance, p = 1 is Manhattan Distance, p = 2 is Euclidean Distance
def minkowskiDistance(p):
    def dist(ins1, ins2, length):
        distance = 0
        for x in range(length):
            distance +=pow(abs(ins1[x] - ins2[x]),p)
        return (pow(distance, 1/p))
    return dist


def getNeighbors(distfunc, trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1

    for x in range(len(trainingSet)):
        dist = distfunc(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))

    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def getPrecision(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(predictions))) * 100.0

if __name__ == "__main__":
    split = 0.80
    df = pd.read_csv("knn.csv", header = 0)

    # use closure
    # loop of three rescale approaches
    for normFunc in [normMin, normAvg, normZscore]:
        normdf = normFunc(df)
        normdf = normdf.values.tolist()
        trainingSet, testSet = splitDataset(normdf, split)
        # loop for 2 types of distances
        # minkowski_distance, p = 1 is Manhattan Distance, p = 2 is Euclidean Distance
        for distFunc in [minkowskiDistance(1), minkowskiDistance(2)]:
            predictions=[]
            k = 3
            for x in range(len(testSet)):
                neighbors = getNeighbors(distFunc, trainingSet, testSet[x], k)
                result = getResponse(neighbors)
                predictions.append(result)
            print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))

            precision = getPrecision(testSet, predictions)
            print('Precision: ' + repr(precision) + '%')
