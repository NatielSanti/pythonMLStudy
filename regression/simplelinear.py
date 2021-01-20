import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import r2_score

if __name__ == '__main__':
    df = pd.read_csv("../resources/FuelConsumptionCo2.csv")

    # Lets select some features to explore more.
    cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

    # We can plot each of these fearues
    viz = cdf[['CYLINDERS', 'ENGINESIZE', 'CO2EMISSIONS', 'FUELCONSUMPTION_COMB']]
    viz.hist()
    plt.show()
    plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
    plt.xlabel("Engine size")
    plt.ylabel("Emission")
    plt.show()

    # Creating train and test dataset
    msk = np.random.rand(len(df)) < 0.8
    train = cdf[msk]
    test = cdf[~msk]

    # Using sklearn package to model data
    regr = linear_model.LinearRegression()
    train_x = np.asanyarray(train[['ENGINESIZE']])
    train_y = np.asanyarray(train[['CO2EMISSIONS']])
    regr.fit(train_x, train_y)

    # The coefficients
    print('Coefficients: ', regr.coef_)
    print('Intercept: ', regr.intercept_)

    # Plot outputs
    plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
    plt.plot(train_x, regr.coef_[0][0] * train_x + regr.intercept_[0], '-r')
    plt.xlabel("Engine size")
    plt.ylabel("Emission")
    plt.show()

    test_x = np.asanyarray(test[['ENGINESIZE']])
    test_y = np.asanyarray(test[['CO2EMISSIONS']])
    test_y_ = regr.predict(test_x)

    print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
    print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
    print("R2-score: %.2f" % r2_score(test_y, test_y_))
