import math
from utils import *

features = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
           'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']

featuresTraceDict = {}
for fea in features:
    featuresTraceDict[fea] = 0


class CriteriaCls():
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    @staticmethod
    def getProbOfList(list):
        staMap = {}
        for value in list:
            if value in staMap.keys():
                staMap[value] += 1
            else:
                staMap[value] = 1
        listLength = len(list)
        for key, value in staMap.items():
            staMap[key] = staMap[key] / listLength
        return staMap

    def entropy(self, data):
        probMap = self.getProbOfList(data)
        ent = 0
        for key, value in probMap.items():
            ent = ent + (-1) * math.log2(value) * value
        return ent

    def giniValue(self, data):
        firstValue = data[0]
        length = len(data)
        count = 0
        for val in data:
            if val == firstValue:
                count += 1
        return 2 * (firstValue / length) * (1 - (firstValue / length))

    def Gini(self, feature):
        featureIndex = features.index(feature)
        colData = getColValues(self.X_train, featureIndex)
        XProbMap = self.getProbOfList(colData)
        rowIdxMap = {}
        for idx, col in enumerate(colData):
            if col in rowIdxMap.keys():
                rowIdxMap[col].append(idx)
            else:
                rowIdxMap[col] = [idx]
        giniValueList = []
        for key in XProbMap.keys():
            rowIdxLIst = rowIdxMap[key]
            yValueList = []
            for row in rowIdxLIst:
                yValueList.append(self.y_train[row])
            giniValueList.append(self.giniValue(yValueList))

        pass

    def InfoGain(self, feature):
        originEnt = self.entropy(self.y_train)

        # get the distribution of one col in X
        featureIndex = features.index(feature)
        colData = getColValues(self.X_train, featureIndex)
        XProbMap = self.getProbOfList(colData)
        rowIdxMap = {}
        for idx, col in enumerate(colData):
            if col in rowIdxMap.keys():
                rowIdxMap[col].append(idx)
            else:
                rowIdxMap[col] = [idx]
        entOfFeatureList = []
        for key in XProbMap.keys():
            rowIdxLIst = rowIdxMap[key]
            yValueList = []
            for row in rowIdxLIst:
                yValueList.append(self.y_train[row])
            entOfFeatureList.append(self.entropy(yValueList))
        entOfFeature = getMulOfTwoList(entOfFeatureList, XProbMap.values())

        return originEnt - entOfFeature


class Node():

    def __init__(self, children, X, y, feature, label):
        self.children = children
        self.X = X
        self.y = y
        self.feature = feature
        self.label = label


class DecisionTreeCls():
    def __init__(self, X, y, criterion):
        self.X = X
        self.y = y
        self.criterion = criterion

    def bestSpliter(self, X, y, featuresTraceDict):
        if self.criterion == "entropy":
            criterioObj = CriteriaCls(X, y)
            criTraceList = []
            for feature in features:
                if featuresTraceDict[feature] == 0:
                    entroGain = criterioObj.InfoGain(feature)
                    criTraceList.append(entroGain)
        elif self.criterion == "gini":
            pass
        else:
            raise Exception(print("criterion does not exist!"))
        pass
        optiIndx = criTraceList.index(max(criTraceList))
        optiFea = features[optiIndx]
        # update the trace Dict of features
        featuresTraceDict[optiFea] = 1
        return optiFea

    def checkLeftFeatures(self, featuresTraceDict):
        count = 0
        for feature in features:
            if featuresTraceDict[feature] == 1:
                count += 1
            if count > 2:
                return count
        return count

    def checkLabels(self, labels):
        first = labels[0]
        for label in labels:
            if label != first:
                return False
        return True

    def depthFirstTree(self, X, y, usedFeaNum, featuresTraceDict):
        if len(features) - usedFeaNum == 1 or self.checkLabels(y):
            label = y[0]
            return Node(None, X, y, None, label)

        else:
            optiFea = self.bestSpliter(X, y, featuresTraceDict)
            optiFeaIndex = features.index(optiFea)
            usedFeaNum += 1
            colData = getColValues(X, optiFeaIndex)
            colUniData = set(colData)

            rowIdxMap = {}
            for idx, col in enumerate(colData):
                if col in rowIdxMap.keys():
                    rowIdxMap[col].append(idx)
                else:
                    rowIdxMap[col] = [idx]

            # generate children of one node in tree
            children = []
            for value in colUniData:
                rowList = rowIdxMap[value]
                XChildren = []
                yChildren = []
                for row in rowList:
                    XChildren.append(X[row])
                    yChildren.append(y[row])
                children.append(self.depthFirstTree(XChildren, yChildren, usedFeaNum, featuresTraceDict))
            return Node(children, None, None, optiFea, None)


def loadTrainData():
    data = []
    with open('originalData/adult.data') as file:
        for line in file:
            line = line.strip()
            if line == '':
                continue
            lineList = line.split(',')
            data.append(lineList)
    return data


if __name__ == '__main__':
    dataTrain = loadTrainData()
    X_train = dataTrain[:-1]
    y_train = dataTrain[-1]



