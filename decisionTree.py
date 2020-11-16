import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
           'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
           'Income']


# Data preprocess
# Remove all the records containing '?' (i.e., missing values). Also, remove the attribute "native-country".
# create the new binary columns to replace the label
# drop the Income columns
def preprocess(data):
    data = data.drop('native-country', axis=1)
    data = data[(data.astype(str) != ' ?').all(axis=1)]
    data['Income_binary'] = data.apply(lambda row: 1 if '>50K' in row['Income'] else 0, axis=1)
    data = data.drop('Income', axis=1)
    data = pd.get_dummies(data, columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship',
                                         'race', 'sex'])
    return data


def loadData():
    data = pd.read_csv('originalData/adult.data', names=columns)
    return data


def loadTestData():
    test = pd.read_csv('originalData/adult.test', names=columns)
    test = test.drop([0], axis=0)
    return test


def buildDecitionTree(data):
    X = data.drop('Income_binary', axis=1).values
    y = data['Income_binary'].values
    tree = DecisionTreeClassifier(max_depth=8)
    tree.fit(X, y)
    return tree


if __name__ == "__main__":
    m_data = loadData()
    m_data = preprocess(m_data)
    print(m_data.head())
    print(m_data.info())
    m_tree = buildDecitionTree(m_data)

    t_data = loadTestData()
    t_data = preprocess(t_data)

    test_label = t_data['Income_binary']
    test_data = t_data.drop('Income_binary', axis=1)
    print(test_label.shape)
    print(test_data.shape)

    # decision tree prediction
    predict_dt = m_tree.predict(test_data.values)

    # evaluate the accuracy of the model
    eval = pd.DataFrame(predict_dt, columns=['DecisionTree_label'])
    test_label = test_label.reset_index(drop=True)
    Eval = pd.concat([test_label, eval], axis=1)
    print(Eval.head())

    # visualization of the decision tree
    # command: dot -Tpng tree.dot -o filename.png
    with open("tree.dot", 'w') as f:
        f = tree.export_graphviz(m_tree, out_file=f)


