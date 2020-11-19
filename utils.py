def getColValues(matrix, col):
    data = []
    for l in matrix:
        data.append(l[col])
    return data


def getMulOfTwoList(list1, list2):
    func = lambda x, y: x * y
    result = map(func, list1, list2)
    return sum(result)