import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model

if __name__ == '__main__':
    df = pd.read_csv("../resources/FuelConsumptionCo2.csv")

    # Lets select some features to explore more.
    cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB',
              'CO2EMISSIONS']]
    cdf.head(9)

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
    train_x = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
    train_y = np.asanyarray(train[['CO2EMISSIONS']])
    regr.fit(train_x, train_y)

    # The coefficients
    print('Coefficients: ', regr.coef_)
    print('Intercept: ', regr.intercept_)

    y_hat = regr.predict(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
    x = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
    y = np.asanyarray(test[['CO2EMISSIONS']])
    print("Residual sum of squares: %.2f"
          % np.mean((y_hat - y) ** 2))

    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr.score(x, y))
