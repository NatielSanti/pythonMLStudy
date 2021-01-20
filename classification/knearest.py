import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

if __name__ == '__main__':
    df = pd.read_csv("../resources/teleCust1000t.csv")

    # show data
    print(df['custcat'].value_counts())
    df.hist(column='income', bins=50)
    plt.show()

    # features and labels
    X = df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed', 'employ', 'retire', 'gender',
            'reside']].values
    y = df['custcat'].values

    # normalize
    X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

    # Split data to train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
    print('Train set:', X_train.shape, y_train.shape)
    print('Test set:', X_test.shape, y_test.shape)

    # Train Model and Predict in different K (closest points)
    Ks = 10
    mean_acc = np.zeros((Ks - 1))
    std_acc = np.zeros((Ks - 1))

    for n in range(1, Ks):
        neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
        yhat = neigh.predict(X_test)
        mean_acc[n - 1] = metrics.accuracy_score(y_test, yhat)
        std_acc[n - 1] = np.std(yhat == y_test) / np.sqrt(yhat.shape[0])

    plt.plot(range(1, Ks), mean_acc, 'g')
    plt.fill_between(range(1, Ks), mean_acc - 1 * std_acc, mean_acc + 1 * std_acc, alpha=0.10)
    plt.fill_between(range(1, Ks), mean_acc - 3 * std_acc, mean_acc + 3 * std_acc, alpha=0.10, color="green")
    plt.legend(('Accuracy ', '+/- 1xstd', '+/- 3xstd'))
    plt.ylabel('Accuracy ')
    plt.xlabel('Number of Neighbors (K)')
    plt.tight_layout()
    plt.show()

    print("The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax() + 1)
