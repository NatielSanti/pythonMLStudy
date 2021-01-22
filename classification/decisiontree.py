import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':
    # read data
    my_data = pd.read_csv("../resources/drug200.csv")

    # Extract feature data
    X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
    y = my_data["Drug"]

    # transform labels to numeric data
    le_sex = preprocessing.LabelEncoder()
    le_sex.fit(['F', 'M'])  # F = 0; M = 1;
    X[:, 1] = le_sex.transform(X[:, 1])

    le_BP = preprocessing.LabelEncoder()
    le_BP.fit(['LOW', 'NORMAL', 'HIGH'])  # HIGH = 0; LOW = 1; NORMAL = 2;
    X[:, 2] = le_BP.transform(X[:, 2])

    le_Chol = preprocessing.LabelEncoder()
    le_Chol.fit(['NORMAL', 'HIGH'])  # HIGH = 0; NORMAL = 1;
    X[:, 3] = le_Chol.transform(X[:, 3])

    # Split data to train and test sets
    X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

    print('Shape of X training set {}'.format(X_trainset.shape), '&',
          ' Size of Y training set {}'.format(y_trainset.shape))
    # same thing
    # print('Train set:', X_trainset.shape, y_trainset.shape)
    print('Shape of X training set {}'.format(X_testset.shape), '&',
          ' Size of Y training set {}'.format(y_testset.shape))

    # Building a model
    drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
    drugTree.fit(X_trainset, y_trainset)

    # Trying to predict
    predTree = drugTree.predict(X_testset)

    # Metrics
    print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))
