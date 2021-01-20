import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


def sigmoid(x, Beta_1, Beta_2):
    y = 1 / (1 + np.exp(-Beta_1 * (x - Beta_2)))
    return y


if __name__ == '__main__':
    # Non - Linear
    df = pd.read_csv("../resources/china_gdp.csv")

    # Plotting the Dataset
    plt.figure(figsize=(8, 5))
    x_data, y_data = (df["Year"].values, df["Value"].values)
    plt.plot(x_data, y_data, 'ro')
    plt.ylabel('GDP')
    plt.xlabel('Year')
    plt.show()

    # Choosing a model
    X = np.arange(-5.0, 5.0, 0.1)
    Y = 1.0 / (1.0 + np.exp(-X))
    plt.plot(X, Y)
    plt.ylabel('Dependent Variable')
    plt.xlabel('Independent Variable')
    plt.show()

    beta_1 = 0.10
    beta_2 = 1990.0

    # logistic function
    Y_pred = sigmoid(x_data, beta_1, beta_2)

    # plot initial prediction against datapoints
    plt.plot(x_data, Y_pred * 15000000000000.)
    plt.plot(x_data, y_data, 'ro')
    xdata = x_data / max(x_data)
    ydata = y_data / max(y_data)
    plt.show()

    popt, pcov = curve_fit(sigmoid, xdata, ydata)
    # print the final parameters
    print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))
    x = np.linspace(1960, 2015, 55)
    x = x / max(x)
    plt.figure(figsize=(8, 5))
    y = sigmoid(x, *popt)
    plt.plot(xdata, ydata, 'ro', label='data')
    plt.plot(x, y, linewidth=3.0, label='fit')
    plt.legend(loc='best')
    plt.ylabel('GDP')
    plt.xlabel('Year')
    plt.show()

    # TODO Split all data to test and train data before train and doing metrics
    print("Mean absolute error: %.2f" % np.mean(np.absolute(ydata - y)))
    print("Residual sum of squares (MSE): %.2f" % np.mean((ydata - y) ** 2))
    print("R2-score: %.2f" % r2_score(ydata, y))
