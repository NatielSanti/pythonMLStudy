import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import log_loss

if __name__ == '__main__':
    churn_df = pd.read_csv("../resources/ChurnData.csv")

    # Data pre-processing and selection
    churn_df = churn_df[
        ['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'callcard', 'wireless', 'churn']]
    churn_df['churn'] = churn_df['churn'].astype('int')
    X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
    y = np.asarray(churn_df['churn'])

    # Normalize data
    X = preprocessing.StandardScaler().fit(X).transform(X)

    # prepare test and train sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
    print('Train set:', X_train.shape, y_train.shape)
    print('Test set:', X_test.shape, y_test.shape)

    # Modeling and predicting
    LR = LogisticRegression(C=0.01, solver='lbfgs').fit(X_train, y_train)  # ‘newton-cg’, ‘lbfgs’, ‘liblinear’,
    # ‘sag’, ‘saga’
    yhat = LR.predict(X_test)
    yhat_prob = LR.predict_proba(X_test)

    # Evaluation #1
    # print(jaccard_similarity_score(y_test, yhat))

    # Evaluation #2
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, yhat, labels=[1, 0])
    print(cnf_matrix)

    # Plot non-normalized confusion matrix
    print(classification_report(y_test, yhat))

    # log loss
    print(log_loss(y_test, yhat_prob))
