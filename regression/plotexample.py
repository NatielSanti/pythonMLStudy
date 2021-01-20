import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    x = np.arange(-5.0, 5.0, 0.1)

    y = 2 * (x) + 3
    y_noise = 2 * np.random.normal(size=x.size)
    ydata = y + y_noise
    plt.figure(figsize=(8, 6))
    plt.plot(x, ydata, 'bo')
    plt.plot(x, y, 'r')
    plt.ylabel('Dependent Variable')
    plt.xlabel('Independent Variable')
    plt.title('Linear')
    plt.show()

    y = 1 * (x ** 3) + 1 * (x ** 2) + 1 * x + 3
    y_noise = 20 * np.random.normal(size=x.size)
    ydata = y + y_noise
    plt.plot(x, ydata, 'bo')
    plt.plot(x, y, 'r')
    plt.ylabel('Dependent Variable')
    plt.xlabel('Independent Variable')
    plt.title('Polynomial')
    plt.show()

    # quadratic
    y = np.power(x, 2)
    y_noise = 2 * np.random.normal(size=x.size)
    ydata = y + y_noise
    plt.plot(x, ydata, 'bo')
    plt.plot(x, y, 'r')
    plt.ylabel('Dependent Variable')
    plt.xlabel('Independent Variable')
    plt.title('Quadratic')
    plt.show()

    # exponential
    Y = np.exp(x)
    plt.plot(x, Y)
    plt.ylabel('Dependent Variable')
    plt.xlabel('Independent Variable')
    plt.title('Exponential')
    plt.show()

    # logarithmic
    Y = np.log(x)
    plt.plot(x, Y)
    plt.ylabel('Dependent Variable')
    plt.xlabel('Independent Variable')
    plt.title('Logarithmic')
    plt.show()

    # Sigmoidal/Logistic
    Y = 1 - 4 / (1 + np.power(3, x - 2))
    plt.plot(x, Y)
    plt.ylabel('Dependent Variable')
    plt.xlabel('Independent Variable')
    plt.title('Sigmoidal/Logistic')
    plt.show()
