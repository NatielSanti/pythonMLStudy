import pandas as pd
import pylab
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # load data
    pdf = pd.read_csv("../resources/cars_clus.csv")
    print("Shape of dataset: ", pdf.shape)

    # clean data
    pdf[['sales', 'resale', 'type', 'price', 'engine_s',
         'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
         'mpg', 'lnsales']] = pdf[['sales', 'resale', 'type', 'price', 'engine_s',
                                   'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
                                   'mpg', 'lnsales']].apply(pd.to_numeric, errors='coerce')
    pdf = pdf.dropna()
    pdf = pdf.reset_index(drop=True)
    print("Shape of dataset after cleaning: ", pdf.shape)

    # feature selection and normalization
    x = pdf[['engine_s', 'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']].values
    feature_mtx = MinMaxScaler().fit_transform(x)

    dist_matrix = euclidean_distances(feature_mtx, feature_mtx)
    print(dist_matrix)

    Z_using_dist_matrix = hierarchy.linkage(dist_matrix, 'complete')
    fig = pylab.figure(figsize=(18, 50))


    def llf(id):
        return '[%s %s %s]' % (pdf['manufact'][id], pdf['model'][id], int(float(pdf['type'][id])))


    dendro = hierarchy.dendrogram(Z_using_dist_matrix, leaf_label_func=llf, leaf_rotation=0, leaf_font_size=12,
                                  orientation='right')

    agglom = AgglomerativeClustering(n_clusters=6, linkage='complete')
    agglom.fit(dist_matrix)
    pdf['cluster_'] = agglom.labels_
    n_clusters = max(agglom.labels_) + 1
    colors = cm.rainbow(np.linspace(0, 1, n_clusters))
    cluster_labels = list(range(0, n_clusters))

    # Create a figure of size 6 inches by 4 inches.
    plt.figure(figsize=(16, 14))

    for color, label in zip(colors, cluster_labels):
        subset = pdf[pdf.cluster_ == label]
        for i in subset.index:
            plt.text(subset.horsepow[i], subset.mpg[i], str(subset['model'][i]), rotation=25)
        plt.scatter(subset.horsepow, subset.mpg, s=subset.price * 10, c=color, label='cluster' + str(label), alpha=0.5)
    #    plt.scatter(subset.horsepow, subset.mpg)
    plt.legend()
    plt.title('Clusters')
    plt.xlabel('horsepow')
    plt.ylabel('mpg')
    pdf.groupby(['cluster_', 'type'])['cluster_'].count()
    agg_cars = pdf.groupby(['cluster_', 'type'])['horsepow', 'engine_s', 'mpg', 'price'].mean()
    plt.show()
