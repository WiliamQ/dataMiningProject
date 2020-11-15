import pandas as pd
import numpy as np

columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
           'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
           'Income']


# Data preprocess
# Remove all the records containing '?' (i.e., missing values). Also, remove the attribute "native-country".
# create the new binary columns to replace the label
# drop the Income columns
# TODO: encode dummies values to replace all the object attributes
def preprocess(data):
    data = data.drop('native-country', axis=1)
    data = data[(data.astype(str) != ' ?').all(axis=1)]
    data['Income_binary'] = data.apply(lambda row: 1 if '>50K' in row['Income'] else 0, axis=1)
    data = data.drop('Income', axis=1)
    return data


def loadData():
    data = pd.read_csv('originalData/adult.data', names=columns)
    print(data.head())
    return data


def loadTestData():
    test = pd.read_csv('originalData/adult.test', names=columns)
    test = test.drop([0], axis=0)


if __name__ == "__main__":
    m_data = loadData()
    preprocess(m_data)
