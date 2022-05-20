import sys

import pickle

if len(sys.argv) < 4:
    print('Usage: python choose_attitudinal_dimensions_with_significan_correlations.py <correlation_probability_object_filename> <country_name> <correlation_probability_threshold> [ideological_dimensions_under_consideration]. Aborting...')
    exit(-1)

correlation_object_filename = sys.argv[1]

dimension_correlations = pickle.load(open(correlation_object_filename, "rb"))

country = sys.argv[2]

if country not in dimension_correlations.keys():
    print('Non existent country. Aborting...')
    exit(-1)

corr_prob_df = dimension_correlations[country]
corr_prob_df = corr_prob_df.reset_index()

if len(sys.argv) > 4:
    dims = sys.argv[4].split(':')
    corr_prob_df = corr_prob_df[corr_prob_df['index'].isin(dims)]
corr_prob_df.drop(columns = ['index'], inplace = True)

correlation_probability_threshold = float(sys.argv[3])
attitudinal_dimensions = []
for c in corr_prob_df.columns:
    crp_df = corr_prob_df[c].copy()
    crp_df.sort_values(ascending = True, inplace = True)
    crp_df = crp_df.reset_index()
    if crp_df[c].values[0] < correlation_probability_threshold:
        print(c, crp_df[c].values[0])
