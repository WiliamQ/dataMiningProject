def getColValues(matrix, col):
    data = []
    for l in matrix:
        data.append(l[col])
    return data


def getValuesPairs(matrix, col, labels):
    dataPairs = []
    data = set()
    for idx, l in enumerate(matrix):
        dataPairs.append((l[col], labels[idx]))
        data.add(l[col])
    return dataPairs, sorted(list(data))


def getMulOfTwoList(list1, list2):
    func = lambda x, y: x * y
    result = map(func, list1, list2)
    return sum(result)


def getRowMap(colDataList):
    rowIdxMap = {}
    for idx, col in enumerate(colDataList):
        if col in rowIdxMap.keys():
            rowIdxMap[col].append(idx)
        else:
            rowIdxMap[col] = [idx]

    return rowIdxMap


def getGroup2Row(featureData, minVal, interval):
    group2row = {}
    for idx, value in enumerate(featureData):
        groupIdx = int((value - minVal) / interval)
        if groupIdx in group2row.keys():
            group2row[groupIdx].append(idx)
        else:
            group2row[groupIdx] = [idx]
    return group2row


def getSplitList(uniFeaValList, intervalNums):
    minVal = min(uniFeaValList)
    maxVal = max(uniFeaValList) + 1
    interval = (maxVal - minVal) / intervalNums
    splits = [minVal]
    start = minVal
    for i in range(intervalNums - 1):
        start += interval
        splits.append(start)
    splits.append(maxVal)
    return splits

def getValueByRow(dataLIst, rowList):
    data = []
    for row in rowList:
        data.append(dataLIst[row])
    return data