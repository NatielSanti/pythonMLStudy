from math import sqrt
from scipy.stats import kurtosis, skew


def average(lst):
    return sum(lst) / len(lst)


def stdDev(lst):
    sum1 = 0
    mean = average(lst)
    for a in lst:
        sum1 += (a - mean) * (a - mean)
    return sqrt(sum1 / len(lst))


def cov(lst1, lst2):
    sumCov = 0
    mean2 = average(lst1)
    mean3 = average(lst2)
    for i in range(0, len(lst1)):
        sumCov += (lst1[i] - mean2) * (lst2[i] - mean3)
    return sumCov / len(arr2)


def corr(lst1, lst2):
    return cov(lst1, lst2)/(stdDev(lst1)*stdDev(lst2))


if __name__ == '__main__':
    arr = [34, 1, 23, 4, 3, 3, 12, 4, 3, 1]
    arr.sort()
    print(len(arr), ' ', arr)

    print('Standart deviation ', stdDev(arr))
    print('Kurtosis ', kurtosis(arr))
    print('Skew ', skew(arr))

    arr2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    arr3 = [7, 6, 5, 4, 5, 6, 7, 8, 9, 10]
    print('Covariance ', cov(arr2, arr3))
    print('Corr ', corr(arr2, arr3))
    print(average([1,2,4,5,34,1,32,4,34,2,1,3]))