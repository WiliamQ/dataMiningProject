def getColValues(matrix, col):
    data = []
    for l in matrix:
        data.append(l[col])
    return data


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