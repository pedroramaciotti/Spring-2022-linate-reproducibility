import sys

import pickle

if len(sys.argv) < 4:
    print('Usage: python choose_attitudinal_dimensions_with_significan_correlations.py <correlation_object_filename> <country_name> <correlation_threshold> [ideological_dimensions_under_consideration]. Aborting...')
    exit(-1)

correlation_object_filename = sys.argv[1]

dimension_correlations = pickle.load(open(correlation_object_filename, "rb"))

country = sys.argv[2]

if country not in dimension_correlations.keys():
    print('Non existent country. Aborting...')
    exit(-1)

corr = dimension_correlations[country]
corr = corr.reset_index()

if len(sys.argv) > 4:
    dims = sys.argv[4].split(':')
    corr = corr[corr['index'].isin(dims)]
corr.drop(columns = ['index'], inplace = True)

correlation_threshold = float(sys.argv[3])
attitudinal_dimensions = []
for c in corr.columns:
    cr = corr[c].copy()
    cr.sort_values(ascending = False, inplace = True)
    cr = cr.reset_index()
    if cr[c].values[0] >= correlation_threshold:
        print(c, cr[c].values[0])
