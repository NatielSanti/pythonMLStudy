import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

if __name__ == '__main__':
    df = pd.read_csv("../resources/FuelConsumptionCo2.csv")

    # Lets select some features to explore more.
    cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
    # cdf.head(9)

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
    train_x = np.asanyarray(train[['ENGINESIZE']])
    train_y = np.asanyarray(train[['CO2EMISSIONS']])

    test_x = np.asanyarray(test[['ENGINESIZE']])
    test_y = np.asanyarray(test[['CO2EMISSIONS']])

    poly = PolynomialFeatures(degree=2)
    train_x_poly = poly.fit_transform(train_x)

    # The coefficients
    clf = linear_model.LinearRegression()
    train_y_ = clf.fit(train_x_poly, train_y)
    # The coefficients
    print('Coefficients: ', clf.coef_)
    print('Intercept: ', clf.intercept_)

    # Plot outputs
    plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
    XX = np.arange(0.0, 10.0, 0.1)
    yy = clf.intercept_[0] + clf.coef_[0][1] * XX + clf.coef_[0][2] * np.power(XX, 2)
    plt.plot(XX, yy, '-r')
    plt.xlabel("Engine size")
    plt.ylabel("Emission")
    plt.show()

    # Metrics
    test_x_poly = poly.fit_transform(test_x)
    test_y_ = clf.predict(test_x_poly)

    print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
    print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
    print("R2-score: %.2f" % r2_score(test_y, test_y_))
