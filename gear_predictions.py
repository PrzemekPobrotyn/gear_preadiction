
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.ensemble import IsolationForest

# set the colour paletter for the graphs
colours = np.array(sns.color_palette('Set1',100))

#define necessary functions
def preprocess(raw_data):
    '''
    This function merges rpm and speed on a timestamp and discards mismatched rows.
    Zero speeds and rpms are discarded, as well as speed over 200.
    Ratio column is added and data frame is returend.
    '''

    rpm = raw_data[raw_data['type'] == 'rpm']
    speed = raw_data[raw_data['type'] == 'speed']

    rpm = rpm.rename(index = str, columns = {'result':'rpm'})
    rpm.drop(['type'], axis = 1, inplace = True)

    speed = speed.rename(index = str, columns = {'result':'speed'})
    speed.drop(['type'], axis = 1, inplace = True)

    data = pd.merge(rpm, speed, how = 'inner', on = ['timestamp','vehicle_id','ride_id'])
    data = data[(data['rpm'] > 0) & ((data['speed'] > 0) & (data['speed'] < 200))]
    data['ratio'] = data['rpm']/data['speed']

    return data


def get_ratio_converter(vehicle_id, ride_id, data, n_gears = 6, contamination = 0.01):
    '''
    Takes a model journey of vehicle_id, ride_id, preprocessed data, number of gears
    of the vehicle and contamination level and outputs a function which converts
    rpm/speed ratios to gears, as predicted by KMeans model with cosine metric.
    WARNING: May take some time to run.
    '''

    vh = data[(data['vehicle_id'] == vehicle_id) & (data['ride_id'] == ride_id)]

    distances = pairwise_distances(vh[['rpm','speed']],
                                    vh[['rpm','speed']],
                                    metric = 'cosine')

    # train a model to remove noisy data, so that clear theoretical lines are left

    clf = IsolationForest(max_samples=100, contamination = contamination)
    clf.fit(distances)
    labels = clf.predict(distances)
    vh['outlier'] = labels
    vh_no_outliers = vh[vh['outlier'] == 1]

    # recompute distances after outliers were removed
    distances_no_outlier = pairwise_distances(vh_no_outliers[['rpm','speed']],
                                              vh_no_outliers[['rpm','speed']],
                                              metric='cosine')

    # cluster data into gears
    model = KMeans(n_clusters=n_gears)
    model.fit(distances_no_outlier)
    vh_no_outliers['labels'] = model.labels_

    # find the mean rpm to speed raito for each label
    # and differences betweeen consecutive means
    gear_boundaries = []
    for i in vh_no_outliers['labels'].unique():
        temp = (vh_no_outliers[vh_no_outliers['labels'] == i]['ratio'].mean())
        gear_boundaries.append(temp)

    gear_boundaries = np.array(gear_boundaries)
    gear_boundaries = np.sort(gear_boundaries)[::-1]

    differences = abs(np.diff(gear_boundaries))
    differences = list(differences)
    temp = differences[-1]
    differences.append(temp)

    # create theoretical intervals for each gear

    gear_intervals = [(x-i/2,x+i/2) for (x,i) in zip(gear_boundaries,differences)]

    # define a function which converts a RPM to speed ratio into a gear
    # this function is specific to a vehicle of id = vehicle_id
    def ratio_converter(ratio):

        # some vehicles will have fewer than 6 gears, so avoid indexing errors by using try-except clause
        try:
            if ratio > gear_intervals[0][0] and ratio : return 1 #leaving out an upper bound cause usually there are a lot of high rpms near 0 speed
            if ratio > gear_intervals[1][0] and ratio < gear_intervals[1][1]: return 2
            if ratio > gear_intervals[2][0] and ratio < gear_intervals[2][1]: return 3
            if ratio > gear_intervals[3][0] and ratio < gear_intervals[3][1]: return 4
            if ratio > gear_intervals[4][0] and ratio < gear_intervals[4][1]: return 5
            if ratio > gear_intervals[5][0] and ratio < gear_intervals[5][1]: return 6
        except IndexError:
            return None # if not in any interval, it is an outlier
                        # will try to interpolate its value later

    return ratio_converter

def plot_graph(vehicle_id, ride_id, data):

    '''
    Takes vehicle_id, ride_id, and data with gears appened for the given
    vehicle. Saves to disc a graph of speed vs rpm with gears labelled by different colours.
    '''
    plt.figure()
    vh = data[(data['vehicle_id'] == vehicle_id) & (data['ride_id'] == ride_id)]
    vh.dropna(axis = 0, inplace=True)
    plt.scatter(vh['speed'],vh['rpm'], c = colours[vh['gear']])
    plt.xlabel('Speed in km/h')
    plt.ylabel('RPM')
    plt.title('Vehicle {} Trip {}'.format(vehicle_id,ride_id))
    plt.savefig('Vehicle_{}_Trip_{}.png'.format(vehicle_id,ride_id), bbox_inches='tight')
    plt.close()

#read in raw data
raw_data = pd.read_csv('rides_bootcamp.csv')

#merge speed with rpm on timestamp, vh id and ride id
data = preprocess(raw_data)

# (vehicle_id,ride_id, number of gears, contamination) tuples most representative of each vehicle
# if contamination not given, 0.01 is assumed
representatives = [
(1,180,6), (2,2,6), (3,194,5), (4,237,7), (5,201,6),(6,6,6),(7,17,6),(8,217,5),
(9,177,6), (10,21,3), (11,12,6), (12,15,6), (13,76,5), (14,132,6), (15,333,5),
(16,24,6), (17,155,6), (18,32,6), (19,371,6), (20,153,5, 0.05), (21,35,5), (22,38,6),
(23,40,6), (24, 43, 4), (25, 94, 5), (26,373,5), (27, 69, 5), (28,109,5, 0.05),
(29,84,6), (30,119,6),(31,160,6),(32,254,5),(33,148,5),(34,156,6),(35,310,6),
(36,186,6), (37,207, 5, 0.03), (38,261,5), (39,223,4), (40,227,5), (41,247,6),
(42,239,5), (43,301,6), (44,242,5, 0.05), (45,257,5), (46,260,5), (47,324,4), (48, 275,5),
(49, 276, 5, 0.05), (50, 453, 5), (51, 283, 6), (52, 329, 4), (53,296,5),
(54, 298,4), (55,304,6), (56, 313, 4), (57,381,5,0.05), (58,334,6), (59, 359, 5),
(60, 374, 5), (61, 379, 6,0.05), (62, 380, 5, 0.05), (63, 385, 5), (64, 426, 5),
(65, 397, 4), (66, 395,5), (67, 396, 5), (68,410,6), (69, 437,5), (70,403,6,0.1),
(71, 421, 5), (72, 425, 5), (73, 428, 4, 0.05), (74, 431, 5), (75, 460, 5), (76, 443, 5, 0.05),
(77, 445, 5), (78, 464, 6), (79, 462,5), (80, 477, 5), (81, 497, 5), (82, 500, 5)
]

# create and empty dictionary to hold {vehicle_id : ratio-to-gear converter function} pairs
converters = {}

# create a dictionary of (vehicle, ratio-to-gear converter) pairs
for quadruple in representatives:
    try:
        vehicle, ride, gears, contamination = quadruple
        print(quadruple)
        converters[vehicle] = get_ratio_converter(vehicle, ride, data, n_gears = gears, contamination = contamination)
    except ValueError: #fewer things to unpack
        vehicle, ride, gears = quadruple
        print(quadruple)
        converters[vehicle] = get_ratio_converter(vehicle, ride, data, n_gears = gears)

#apply ratio-to-gear converters
data['gear'] = data[['vehicle_id','ratio']].apply(
            lambda row: converters[row['vehicle_id']](row['ratio']), axis = 1)

# interpolate missing data based on surrounding gears
data['gear'] = data['gear'].fillna(method = 'pad', limit = 2)
data['gear'] = data['gear'].fillna(method = 'bfill', limit = 2)

#save findings to csv
data.to_csv("gears.csv", index = False)

#produce 500 graphs of all trips
vehicle_no = len(data['vehicle_id'].unique())
for n in range(1, vehicle_no + 1):
    vh = data[data['vehicle_id'] == n]
    trips = vh['ride_id'].unique()
    for j in trips:
        plot_graph(n, j, data)

# left-join data with raw_data
new_data = pd.merge(raw_data, data, how = 'left', on = ['timestamp', 'vehicle_id', 'ride_id'])

# sort the data by timestamp, vehicle_id and ride_id to allow interpolation
new_data.sort(columns=['timestamp', 'vehicle_id','ride_id'], inplace=True)

# interpolate some of the missing gears on entries where there was a timestamp mismatch
# these were discarded earlier, but we can infer their gear from surrounding gears now
new_data['gear'] = new_data['gear'].fillna(method = 'pad', limit = 5)
new_data['gear'] = new_data['gear'].fillna(method = 'bfill', limit = 5)

# save new dataset to a file
new_data.to_csv('rides_bootcamp_appended.csv')
