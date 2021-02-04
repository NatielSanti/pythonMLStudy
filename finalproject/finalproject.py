import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# Final Course Project
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


def plot(data, name):
    bins = np.linspace(data[name].min(), data[name].max(), 10)
    g = sns.FacetGrid(data, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
    g.map(plt.hist, name, bins=bins, ec="k")
    g.axes[-1].legend()
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv("../resources/loan_train.csv")
    df['due_date'] = pd.to_datetime(df['due_date'])
    df['effective_date'] = pd.to_datetime(df['effective_date'])
    df['dayofweek'] = df['effective_date'].dt.dayofweek
    df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x > 3) else 0)

    # plot(df, 'Principal')
    # plot(df, 'age')
    # plot(df, 'dayofweek')
    print(df['loan_status'].value_counts())

    df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)
    df['Gender'].replace(to_replace=['male', 'female'], value=[0, 1], inplace=True)
    df.groupby(['education'])['loan_status'].value_counts(normalize=True)

    Feature = df[['Principal', 'terms', 'age', 'Gender', 'weekend']]
    Feature = pd.concat([Feature, pd.get_dummies(df['education'])], axis=1)
    Feature.drop(['Master or Above'], axis=1, inplace=True)

    X = preprocessing.StandardScaler().fit(Feature).transform(Feature)
    y = df['loan_status'].values
    print(X[5])

    # Split data to train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
    print('Train set:', X_train.shape, y_train.shape)
    print('Test set:', X_test.shape, y_test.shape)

    # K Nearest Neighbor(KNN)
    Ks = 12
    mean_acc = np.zeros((Ks - 1))
    jac_acc = np.zeros((Ks - 1))

    for n in range(1, Ks):
        neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
        yhat = neigh.predict(X_test)
        mean_acc[n - 1] = f1_score(y_test, yhat, average='weighted')
        jac_acc[n - 1] = jaccard_score(y_test, yhat, average='weighted')

    plt.plot(range(1, Ks), mean_acc, 'g')
    plt.plot(range(1, Ks), jac_acc, 'r')
    plt.ylabel('Accuracy ')
    plt.xlabel('Number of Neighbors (K)')
    plt.tight_layout()
    plt.show()
    print("Best K value 7")

    # Decision Tree
    max_depth = 10
    best_accuracy = 0
    best_metric_index = 0
    for n in range(1, max_depth):
        drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=n)
        drugTree.fit(X_train, y_train)
        predTree = drugTree.predict(X_test)
        metric = f1_score(y_test, predTree, average='weighted')
        if metric >= best_accuracy:
            best_accuracy = metric
            best_metric_index = n

    print("DecisionTrees's Best Accuracy: ", best_accuracy, " with depth:", best_metric_index)

    # Support Vector Machine
    clf = svm.SVC(kernel='rbf')
    clf.fit(X_train, y_train)
    yhat = clf.predict(X_test)
    print(f1_score(y_test, yhat, average='weighted'))

    # Logistic Regression
    LR = LogisticRegression(C=0.01, solver='lbfgs').fit(X_train, y_train)
    yhat = LR.predict(X_test)
    yhat_prob = LR.predict_proba(X_test)
    print(f1_score(y_test, yhat, average='weighted'))
