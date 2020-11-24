import math
from utils import *

features = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
           'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week']

continuousFeatures = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']


class CriteriaCls():
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    @staticmethod
    def getProbOfList(dataList):
        staMap = {}
        for value in dataList:
            if value in staMap.keys():
                staMap[value] += 1
            else:
                staMap[value] = 1
        listLength = len(dataList)
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
        dataProb = self.getProbOfList(data)
        length = len(data)
        giniIndex = 1.0
        for key, value in dataProb.items():
            giniIndex -= pow(value / length, 2)
        return giniIndex

    def discreteGini(self, feature):
        featureIndex = features.index(feature)
        featureData = getColValues(self.X_train, featureIndex)
        XProbMap = self.getProbOfList(featureData)
        rowIdxMap = getRowMap(featureData)
        giniValueList = []
        for key in XProbMap.keys():
            rowIdxLIst = rowIdxMap[key]
            yValueList = []
            for row in rowIdxLIst:
                yValueList.append(self.y_train[row])
            giniValueList.append(self.giniValue(yValueList))
        avgGini = getMulOfTwoList(giniValueList, XProbMap.values())
        return avgGini

    def continuousGini(self, feature, continuousFeaList):
        featureIndex = features.index(feature)
        if feature == "fnlwgt":
            print("test")
        dataPairs, uniFeaValList = getValuesPairs(self.X_train, featureIndex, self.y_train)
        dataPairsSort = sorted(dataPairs, key=lambda x: x[0])

        # get the best split point among all points of feature
        # splitList has an increasing order which can help reduce time complexity
        splitList = getSplitList(feature, uniFeaValList, continuousFeaList)
        bestSplit, SplittedGini = self.getBestGiniPoint(dataPairsSort, splitList)
        return bestSplit, SplittedGini

    def getBestGiniPoint(self, dataPairsSort, splitList):
        length = len(dataPairsSort)
        dataIdx = 0
        minGini = 0
        bestSplit = 0
        for idx, split in enumerate(splitList):
            dataIdx = self.getSortedDataIdx(dataPairsSort, split, dataIdx)
            gini = dataIdx / length * self.giniValue(self.y_train[:dataIdx]) + (1 - dataIdx / length) * self.giniValue(self.y_train[dataIdx:])

            if idx == 0:
                minGini = gini
                bestSplit = split
            if gini < minGini:
                minGini = gini
                bestSplit = split
        return bestSplit, minGini

    def discreteInfoGain(self, feature):
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

    def continuousInfoGain(self, feature, continuousFeaList):
        originEnt = self.entropy(self.y_train)

        featureIndex = features.index(feature)
        dataPairs = getValuesPairs(self.X_train, featureIndex, self.y_train)
        dataPairsSort = sorted(dataPairs, key=lambda x: x[0])
        splitSet = set()

        # get all the split points
        dataLength = len(self.y_train)
        for i in range(dataLength - 1):
            point = (dataPairsSort[i][0] + dataPairsSort[i + 1][0]) / 2
            if (feature, point) in continuousFeaList:
                continue
            splitSet.add(point)
        # get the best split point among all points of feature
        # splitList has an increasing order which can help reduce time complexity
        splitList = sorted(list(splitSet))
        bestSplit, SplittedEnt = self.getBestEntPoint(dataPairsSort, splitList)
        return bestSplit, originEnt - SplittedEnt

    def getBestEntPoint(self, dataPairsSort, splitList):
        length = len(dataPairsSort)
        dataIdx = 0
        minEntro = 0
        bestSplit = 0
        for idx, split in enumerate(splitList):
            if idx == 0:
                dataIdx = self.getSortedDataIdx(dataPairsSort, split, dataIdx)
            else:
                dataIdx = self.getSortedDataIdx(dataPairsSort, split, dataIdx)
            entro = dataIdx / length * self.entropy(self.y_train[:dataIdx]) + (1 - dataIdx / length) * self.entropy(self.y_train[dataIdx:])

            if idx == 0:
                minEntro = entro
                bestSplit = split
            if entro < minEntro:
                minEntro = entro
                bestSplit = split
        return bestSplit, minEntro

    def getSortedDataIdx(self, dataPairsSort, splitPoint, startIdx):
        for idx, value in enumerate(dataPairsSort[startIdx:]):
            if splitPoint < value[0]:
                return idx + startIdx


class Node():

    def __init__(self, children, featureValue, relation, X, y, feature, label):
        self.children = children
        self.featureValue = featureValue
        self.relation = relation
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
        self.discreteFeaTraceDict = {}
        for fea in features:
            self.discreteFeaTraceDict[fea] = 0

    def bestSpliter(self, X, y, discreteFeaTraceDict, continuousFeaList):
        criTraceList = []
        feature2point = {}
        criterioObj = CriteriaCls(X, y)
        optiIdx = 0
        if self.criterion == "entropy":
            for feature in features:
                if feature not in continuousFeatures and discreteFeaTraceDict[feature] == 0:
                    entroGain = criterioObj.discreteInfoGain(feature)
                    criTraceList.append(entroGain)
                elif feature in continuousFeatures:
                    # for a continuous feature, a specified splitValue should be found out
                    bestSplitPoint, entroGain = criterioObj.continuousInfoGain(feature, continuousFeaList)
                    feature2point[feature] = bestSplitPoint
                    criTraceList.append(entroGain)
                else:
                    continue
            optiIdx = criTraceList.index(max(criTraceList))
        elif self.criterion == "gini":
            for feature in features:
                if feature not in continuousFeatures and discreteFeaTraceDict[feature] == 0:
                    gini = criterioObj.discreteGini(feature)
                    criTraceList.append(gini)
                elif feature in continuousFeatures:
                    bestSplitPoint, gini = criterioObj.continuousGini(feature, continuousFeaList)
                    feature2point[feature] = bestSplitPoint
                    criTraceList.append(gini)
                else:
                    continue
            optiIdx = criTraceList.index(min(criTraceList))

        optiFea = features[optiIdx]
        # update the trace Dict of features
        if optiFea in continuousFeatures:
            continuousFeaList.append((optiFea, feature2point[optiFea]))
            return optiFea, feature2point[optiFea], discreteFeaTraceDict, continuousFeaList
        else:
            discreteFeaTraceDict[optiFea] = 1
            return optiFea, None, discreteFeaTraceDict, continuousFeaList

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

    ## discreteFeaTraceDict: trace whether a discrete feature has been used, if the featureVallue belongs to a continuous feature, it should be None
    ## featureValue: the feature value of one feature that the node represents
    ## relation: if featureValue is contimuous, it represents the relation: <= (smaller), or > (larger) else None
    ## continuousFeaList: tuples mark continuous feature and feature has been used (feature, featureValue)
    def depthFirstTree(self, X, y, discreteFeaTraceDict, featureValue, relation, continuousFeaList):
        # if all labels in y are the same, then stop
        if self.checkLabels(y):
            label = y[0]
            # children, featureValue, relation, X, y, feature, label
            return Node(None, featureValue, relation, X, y, None, label)
        else:
            optiFea, splitPoint, discreteFeaTraceDict, continuousFeaList = self.bestSpliter(X, y, discreteFeaTraceDict, continuousFeaList)
            optiFeaIndex = features.index(optiFea)
            featureData = getColValues(X, optiFeaIndex)
            children = []

            if optiFea not in continuousFeatures:
                featureUniData = set(featureData)
                rowIdxMap = getRowMap(featureData)
                # generate children of one node in tree
                for value in featureUniData:
                    rowList = rowIdxMap[value]
                    XChildren = []
                    yChildren = []
                    for row in rowList:
                        XChildren.append(X[row])
                        yChildren.append(y[row])
                    children.append(self.depthFirstTree(XChildren, yChildren, discreteFeaTraceDict, value, None, continuousFeaList))
            else:
                XSmallerPart = []
                XLargerPart = []
                ySmallerPart = []
                yLargerPart = []
                for row, value in enumerate(featureData):
                    if value <= splitPoint:
                        XSmallerPart.append(X[row])
                        ySmallerPart.append(y[row])
                    else:
                        XLargerPart.append(X[row])
                        yLargerPart.append(y[row])
                children.append(self.depthFirstTree(XSmallerPart, ySmallerPart, discreteFeaTraceDict, splitPoint, "smaller", continuousFeaList))
                children.append(self.depthFirstTree(XLargerPart, yLargerPart, discreteFeaTraceDict, splitPoint, "larger", continuousFeaList))

            return Node(children, featureValue, relation, None, None, optiFea, None)

    def fit(self):
        self.treeHead = self.depthFirstTree(self.X, self.y, self.discreteFeaTraceDict, None, None, [])

    def predict(self, X_test):
        tempTreeHead = self.treeHead
        while tempTreeHead.children:
            for child in tempTreeHead.children:
                treeFeature = tempTreeHead.feature
                treeFeatureIndex = features.index(treeFeature)
                # whether the feature value of X_test equals to the feature value of node
                if treeFeature not in continuousFeatures:
                    if X_test[treeFeatureIndex] == child.featureValue:
                        tempTreeHead = child
                        break
                else:
                    if float(X_test[treeFeatureIndex]) <= child.featureValue and child.relation == "smaller":
                        tempTreeHead = child
                        break
                    if float(X_test[treeFeatureIndex]) > child.featureValue and child.relation == "larger":
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
            if '?' in lineList:
                continue
            X.append(lineList[:-2])
            y.append(lineList[-1])
    return X, y


if __name__ == '__main__':
    X_train, y_train = loadData('originalData/adult.data')
    dt = DecisionTreeCls(X_train, y_train, 'gini')
    dt.fit()

    X_test, y_test = loadData('originalData/adult.test')
    yPreList = []
    for test in X_test:
        yPreList.append(dt.predict(test))
    print(yPreList)



