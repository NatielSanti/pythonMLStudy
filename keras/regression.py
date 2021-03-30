# imports
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing


# regression model
def regression_model():
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(1))

    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

if __name__ == '__main__':
    # load data
    csv_data = pd.read_csv('https://cocl.us/concrete_data')
    csv_data.head()

    all_data = csv_data.columns
    X = csv_data[all_data]
    X = X.drop(['Strength', 'Age'], axis=1)
    X = preprocessing.StandardScaler().fit(X).transform(X)
    y = csv_data['Strength']
    mean_arr = []
    model = regression_model()
    n_cols = X.shape[1]

    for i in range(50):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)
        model.fit(X_train, y_train, validation_split=0.3, epochs=50, verbose=0)
        scores = model.predict(X_test)
        mean_arr.append(mean_squared_error(scores, y_test))
        print('Iteration #', i, ' mean squared error:', mean_arr[i])

    mean = sum(mean_arr) / iter
    std_dev = np.std(mean_arr)
    print('Mean: ', mean, ' Standard deviation: ', std_dev)