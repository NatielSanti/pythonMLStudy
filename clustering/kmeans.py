import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D


if __name__ == '__main__':
    # load data
    cust_df = pd.read_csv("../resources/Cust_Segmentation.csv")
    # drop Address column, bcz it not necessary in this case of ML
    df = cust_df.drop('Address', axis=1)
    # Normalizing data
    X = df.values[:, 1:]
    X = np.nan_to_num(X)
    Clus_dataSet = StandardScaler().fit_transform(X)
    # modeling
    clusterNum = 3
    k_means = KMeans(init="k-means++", n_clusters=clusterNum, n_init=12)
    k_means.fit(X)
    labels = k_means.labels_
    # Insights
    df["Clus_km"] = labels
    print(df)

    df.groupby('Clus_km').mean()
    area = np.pi * (X[:, 1]) ** 2
    plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(np.float), alpha=0.5)
    plt.xlabel('Age', fontsize=18)
    plt.ylabel('Income', fontsize=16)
    plt.show()

    fig = plt.figure(1, figsize=(8, 6))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    plt.cla()
    ax.set_xlabel('Education')
    ax.set_ylabel('Age')
    ax.set_zlabel('Income')
    ax.scatter(X[:, 1], X[:, 0], X[:, 3], c=labels.astype(np.float))
    plt.show()

