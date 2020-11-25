import math
from utils import *

features = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
           'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week']

continuousFeatures = ['age', 'fnlwgt', 'capital-gain', 'education-num','capital-loss', 'hours-per-week']
CONTI_SPLIT_GROUP_NUMS = 10


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

    def continuousGini(self, feature, intervalNums):
        featureIndex = features.index(feature)
        dataPairs, uniFeaValList = getValuesPairs(self.X_train, featureIndex, self.y_train)
        dataPairsSort = sorted(dataPairs, key=lambda x: x[0])

        # get the best split point among all points of feature
        # splitList has an increasing order which can help reduce time complexity
        splitList = getSplitList(uniFeaValList, intervalNums)
        avgGini = self.countContiGini(dataPairsSort, splitList)
        return avgGini

    def countContiGini(self, dataPairsSort, splitList):
        length = len(dataPairsSort)
        startIdx = 0
        avgGini = 0.0
        for i in range(len(splitList) - 1):
            minInter = splitList[i]
            maxInter = splitList[i + 1]
            endIdx, valueCollect = self.getSortedDataIdx(dataPairsSort, maxInter, startIdx)
            tempGini = 1.0
            for key, value in valueCollect.items():
                tempGini -= pow(value / (endIdx - startIdx), 2)
            avgGini += (endIdx - startIdx) / length * tempGini
            startIdx = endIdx
        return avgGini

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

    def continuousInfoGain(self, feature):
        originEnt = self.entropy(self.y_train)

        featureIndex = features.index(feature)
        dataPairs = getValuesPairs(self.X_train, featureIndex, self.y_train)
        dataPairsSort = sorted(dataPairs, key=lambda x: x[0])
        splitSet = set()

        # get all the split points
        dataLength = len(self.y_train)
        for i in range(dataLength - 1):
            point = (dataPairsSort[i][0] + dataPairsSort[i + 1][0]) / 2
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
        valueCollect = {}
        for idx, value in enumerate(dataPairsSort[startIdx:]):
            if splitPoint < value[0]:
                return idx + startIdx, valueCollect
            else:
                if value[1] in valueCollect.keys():
                    valueCollect[value[1]] += 1
                else:
                    valueCollect[value[1]] = 1
        return len(dataPairsSort), valueCollect


class Node():

    def __init__(self, children, feature, featureValue, X, y, label, upper, lower):
        self.children = children
        self.feature = feature
        self.featureValue = featureValue
        self.X = X
        self.y = y
        self.label = label
        self.upper = upper
        self.lower = lower


class DecisionTreeCls():
    def __init__(self, X, y, criterion):
        self.X = X
        self.y = y
        self.criterion = criterion
        self.treeHead = None
        self.discreteFeaTraceDict = {}
        for fea in features:
            self.discreteFeaTraceDict[fea] = 0

    def bestSpliter(self, X, y, discreteFeaTraceDict):
        criTraceList = []
        # feature2point = {}
        criterioObj = CriteriaCls(X, y)
        optiIdx = 0
        if self.criterion == "entropy":
            for feature in features:
                if feature not in continuousFeatures and discreteFeaTraceDict[feature] == 0:
                    entroGain = criterioObj.discreteInfoGain(feature)
                    criTraceList.append(entroGain)
                elif feature in continuousFeatures:
                    # for a continuous feature, a specified splitValue should be found out
                    bestSplitPoint, entroGain = criterioObj.continuousInfoGain(feature)
                    # feature2point[feature] = bestSplitPoint
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
                    gini = criterioObj.continuousGini(feature, CONTI_SPLIT_GROUP_NUMS)
                    criTraceList.append(gini)
                else:
                    continue
            optiIdx = criTraceList.index(min(criTraceList))

        # update the trace Dict of features
        if features[optiIdx] not in continuousFeatures:
            discreteFeaTraceDict[features[optiIdx]] = 1
        return optiIdx, discreteFeaTraceDict

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

    def depthFirstTree(self, X, y, discreteFeaTraceDict, featureValue, upper, lower):
        # if all labels in y are the same, then stop
        if self.checkLabels(y) or len(y) == 1:
            label = y[0]
            # children, feature, featureValue, X, y, label, upper, lower
            return Node(children=None, feature=None, featureValue=featureValue, X=X, y=y, label=label, upper=upper, lower=lower)
        else:
            optiIdx, discreteFeaTraceDict = self.bestSpliter(X, y, discreteFeaTraceDict)
            optiFea = features[optiIdx]
            featureData = getColValues(X, optiIdx)
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
                    children.append(self.depthFirstTree(XChildren, yChildren, discreteFeaTraceDict, value, None, None))
            else:
                uniFeaData = list(set(featureData))
                minVal = min(uniFeaData)
                maxVal = max(uniFeaData) + 1
                interval = (maxVal - minVal) / CONTI_SPLIT_GROUP_NUMS

                group2row = getGroup2Row(featureData, minVal, interval)
                for key, value in group2row.items():
                    tempLower = minVal + interval * key
                    tempUpper = minVal + interval * (key + 1)
                    if len(value) > 0:
                        Xdata = getValueByRow(X, value)
                        ydata = getValueByRow(y, value)
                        children.append(self.depthFirstTree(Xdata, ydata, discreteFeaTraceDict, None, tempUpper, tempLower))
            # children, feature, featureValue, X, y, label, upper, lower
            return Node(children=children, feature=optiFea, featureValue=featureValue, X=None, y=None, label=None, upper=upper, lower=lower)

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
            for feature in continuousFeatures:
                idx = features.index(feature)
                lineList[idx] = float(lineList[idx])
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



