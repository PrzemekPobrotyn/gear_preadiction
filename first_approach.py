
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.ensemble import IsolationForest

# set the colour paletter for the graphs
colours = np.array(sns.color_palette('Set1',100))

# read in raw data
raw_data = pd.read_csv('rides_bootcamp.csv')

# number of unique vehicles
vehicle_no = len(raw_data['vehicle_id'].unique())

# iterate overal all vehicle ids
for n in range(1, vehicle_no + 1):

    raw_vehicle = (raw_data[(raw_data['vehicle_id'] == n)]).copy()

    trips = raw_vehicle['ride_id'].unique()

    #iterate over all trips of the vehicle
    for i in trips:

        #prepare the data for the model

        #separate speed and rpm's
        rpm = raw_vehicle[(raw_vehicle['type'] == 'rpm') & (raw_vehicle['ride_id'] == i)]
        speed = raw_vehicle[(raw_vehicle['type'] == 'speed') & (raw_vehicle['ride_id'] == i)]

        #clean up
        rpm = rpm.rename(index = str, columns = {'result':'rpm'})
        rpm.drop(['vehicle_id', 'ride_id', 'type'], axis = 1, inplace = True)

        speed = speed.rename(index = str, columns = {'result':'speed'})
        speed.drop(['vehicle_id', 'ride_id', 'type'], axis = 1, inplace = True)

        # merge
        vehicle = pd.merge(rpm, speed, how = 'outer', on = 'timestamp')

        # drop null values and zero speeds --> neutral gear
        # speed < 200 to remove outliers

        vh = vehicle.dropna(axis = 0)
        vh = vh[(vh['rpm'] > 0) & ((vh['speed'] > 0) & (vh['speed'] < 200))]

        # detect outliers using IsolationForest
        # assume contamination at 0.01 level

        distances = pairwise_distances(vh[['rpm','speed']],vh[['rpm','speed']], metric = 'cosine')
        clf = IsolationForest(max_samples = 100, contamination = 0.01, verbose = 1)
        clf.fit(distances)
        labels = clf.predict(distances)
        vh['outlier'] = labels

        # remove outliers found by IsolationForest
        vh = vh[['rpm','speed']][vh['outlier'] == 1]

        #recompute distances after outlier removal
        distances = pairwise_distances(vh[['rpm','speed']],vh[['rpm','speed']], metric = 'cosine')

        # initialize variable to keep best model, its silhouette score and predicted labels
        best_model = (None, -1, None)

        # iterate over possible number of gears
        # since we want to pick model with best silhouette score, can't start with single cluster (k=1)

        for k in range(2,7):

            model = KMeans(n_clusters = k, verbose = 1)
            model.fit(distances)
            labels = model.labels_
            model_score = silhouette_score(distances, labels, metric='precomputed')

            if model_score > best_model[1]:
                best_model = (model, model_score, labels)

        # plot and save the graph showing the clusterig found
        labels = best_model[2]
        plt.figure()
        plt.scatter(vh['speed'], vh['rpm'], c = colours[labels])
        plt.xlabel('Speed in km/h')
        plt.ylabel('RPM')
        plt.title('Vehicle {} Trip{} KMeans cosine metric'.format(n,i))
        plt.savefig('Vehicle_{}_Trip_{}.png'.format(n,i), bbox_inches='tight')
        plt.close()
