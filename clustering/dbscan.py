import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.utils
import sklearn.utils
from mpl_toolkits.basemap import Basemap
from pylab import rcParams
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Density-Based Spatial Clustering of Applications with Noise
if __name__ == '__main__':
    # load data
    pdf = pd.read_csv("../resources/weather-stations20140101-20141231.csv")
    # Cleaning
    pdf = pdf[pd.notnull(pdf["Tm"])]
    pdf = pdf.reset_index(drop=True)

    # Clustering of stations based on their location i.e. Lat & Lon
    sklearn.utils.check_random_state(1000)
    Clus_dataSet = pdf[['xm', 'ym']]
    Clus_dataSet = np.nan_to_num(Clus_dataSet)
    Clus_dataSet = StandardScaler().fit_transform(Clus_dataSet)

    # Compute DBSCAN
    db = DBSCAN(eps=0.15, min_samples=10).fit(Clus_dataSet)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    pdf["Clus_Db"] = labels

    realClusterNum = len(set(labels)) - (1 if -1 in labels else 0)
    clusterNum = len(set(labels))

    rcParams['figure.figsize'] = (14, 10)

    llon = -140
    ulon = -50
    llat = 40
    ulat = 65

    pdf = pdf[(pdf['Long'] > llon) & (pdf['Long'] < ulon) & (pdf['Lat'] > llat) & (pdf['Lat'] < ulat)]

    my_map = Basemap(projection='merc',
                     resolution='l', area_thresh=1000.0,
                     llcrnrlon=llon, llcrnrlat=llat,  # min longitude (llcrnrlon) and latitude (llcrnrlat)
                     urcrnrlon=ulon, urcrnrlat=ulat)  # max longitude (urcrnrlon) and latitude (urcrnrlat)

    my_map.drawcoastlines()
    my_map.drawcountries()
    # my_map.drawmapboundary()
    my_map.fillcontinents(color='white', alpha=0.3)
    my_map.shadedrelief()

    # To create a color map
    colors = plt.get_cmap('jet')(np.linspace(0.0, 1.0, clusterNum))

    # Visualization1
    for clust_number in set(labels):
        c = (([0.4, 0.4, 0.4]) if clust_number == -1 else colors[np.int(clust_number)])
        clust_set = pdf[pdf.Clus_Db == clust_number]
        my_map.scatter(clust_set.xm, clust_set.ym, color=c, marker='o', s=20, alpha=0.85)
        if clust_number != -1:
            cenx = np.mean(clust_set.xm)
            ceny = np.mean(clust_set.ym)
            plt.text(cenx, ceny, str(clust_number), fontsize=25, color='red', )
            print("Cluster " + str(clust_number) + ', Avg Temp: ' + str(np.mean(clust_set.Tm)))
