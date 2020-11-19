import math
from utils import *

features = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
           'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']


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
        rowIdxMap = getRowMap(colData)
        giniValueList = []
        for key in XProbMap.keys():
            rowIdxLIst = rowIdxMap[key]
            yValueList = []
            for row in rowIdxLIst:
                yValueList.append(self.y_train[row])
            giniValueList.append(self.giniValue(yValueList))
        avgGini = getMulOfTwoList(giniValueList, XProbMap.values())
        return avgGini

    def InfoGain(self, feature):
        originEnt = self.entropy(self.y_train)

        # get the distribution of one col in X
        featureIndex = features.index(feature)
        colData = getColValues(self.X_train, featureIndex)
        XProbMap = self.getProbOfList(colData)
        rowIdxMap = getRowMap(colData)
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

    def __init__(self, children, featureValue, X, y, feature, label):
        self.children = children
        self.featureValue = featureValue
        self.X = X
        self.y = y
        self.feature = feature
        self.label = label


class DecisionTreeCls():
    def __init__(self, X, y, criterion):
        self.X = X
        self.y = y
        self.criterion = criterion
        self.treeHead = None
        self.featuresTraceDict = {}
        for fea in features:
            self.featuresTraceDict[fea] = 0

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

    def depthFirstTree(self, X, y, usedFeaNum, featuresTraceDict, featureValue):
        if len(features) - usedFeaNum == 1 or self.checkLabels(y):
            label = y[0]
            return Node(None, featureValue, X, y, None, label)

        else:
            optiFea = self.bestSpliter(X, y, featuresTraceDict)
            optiFeaIndex = features.index(optiFea)
            usedFeaNum += 1
            colData = getColValues(X, optiFeaIndex)
            colUniData = set(colData)

            rowIdxMap = getRowMap(colData)

            # generate children of one node in tree
            children = []
            for value in colUniData:
                rowList = rowIdxMap[value]
                XChildren = []
                yChildren = []
                for row in rowList:
                    XChildren.append(X[row])
                    yChildren.append(y[row])
                children.append(self.depthFirstTree(XChildren, yChildren, usedFeaNum, featuresTraceDict, value))
            return Node(children, featureValue, None, None, optiFea, None)

    def fit(self):
        self.treeHead = self.depthFirstTree(self.X, self.y, usedFeaNum=0, featuresTraceDict=self.featuresTraceDict, featureValue=None)

    def predict(self, X_test):
        tempTreeHead = self.treeHead
        while tempTreeHead.children:
            for child in tempTreeHead.children:
                treeFeature = tempTreeHead.feature
                treeFeatureIndex = features.index(treeFeature)
                colValues = getColValues(X_test, treeFeatureIndex)
                # whether the feature value of X_test equals to the feature value of node
                if colValues[0] == child.featureValue:
                    tempTreeHead = child
                    break
        return tempTreeHead.label


def loadData(path):
    X = []
    y = []
    with open(path) as file:
        for line in file:
            line = line.strip()
            if line == '':
                continue
            lineList = line.split(',')
            X.append(lineList[:-1])
            y.append(lineList[-1])
    return X, y


if __name__ == '__main__':
    X_train, y_train = loadData('originalData/adult.data')
    dt = DecisionTreeCls(X_train, y_train, 'entropy')
    dt.fit()

    X_test, y_test = loadData('originalData/adult.test')
    y_pre = dt.predict(X_test)
    print(y_pre)
    print("test")



