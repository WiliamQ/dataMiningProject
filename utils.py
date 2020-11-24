def getColValues(matrix, col):
    data = []
    for l in matrix:
        data.append(l[col])
    return data


def getValuesPairs(matrix, col, labels):
    dataPairs = []
    data = set()
    for idx, l in enumerate(matrix):
        dataPairs.append((float(l[col]), labels[idx]))
        data.add(float(l[col]))
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


def getSplitList(feature, uniFeaValList, continuousFeaList):
    splitSet = set()
    # get all the split points
    dataLength = len(uniFeaValList)
    for i in range(dataLength - 1):
        point = (uniFeaValList[i] + uniFeaValList[i + 1]) / 2
        if (feature, point) in continuousFeaList:
            continue
        splitSet.add(point)

    return sorted(list(splitSet))