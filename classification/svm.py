import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    cell_df = pd.read_csv("../resources/cell_samples.csv")

    # show data
    ax = cell_df[cell_df['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize',
                                                   color='DarkBlue', label='malignant')
    cell_df[cell_df['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize',
                                              color='Yellow', label='benign', ax=ax)
    plt.show()

    # Data preProcessing
    cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
    cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
    feature_df = cell_df[
        ['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
    X = np.asarray(feature_df)
    cell_df['Class'] = cell_df['Class'].astype('int')
    y = np.asarray(cell_df['Class'])

    # Train/Test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
    print('Train set:', X_train.shape, y_train.shape)
    print('Test set:', X_test.shape, y_test.shape)

    # Modeling
    clf = svm.SVC(kernel='rbf')  # Linear, Polynomial, Radial basis function (RBF), Sigmoid
    clf.fit(X_train, y_train)

    # Predict
    yhat = clf.predict(X_test)

    # Evaluation
    cnf_matrix = confusion_matrix(y_test, yhat, labels=[2, 4])
    print(cnf_matrix)
    print(classification_report(y_test, yhat))
    print(f1_score(y_test, yhat, average='weighted'))



