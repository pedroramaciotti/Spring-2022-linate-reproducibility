import sys

import pickle

import heapq

if len(sys.argv) < 3:
    print('Usage: python choose_attitudinal_dimensions_with_significan_correlations.py <correlation_object_filename> <country_name> [ideological_dimensions_under_consideration]. Aborting...')
    exit(-1)

correlation_object_filename = sys.argv[1]

dimension_correlations = pickle.load(open(correlation_object_filename, "rb"))

country = sys.argv[2]

if country not in dimension_correlations.keys():
    print('Non existent country. Aborting...')
    exit(-1)

corr_df = dimension_correlations[country]
corr_df = corr_df.reset_index()

if len(sys.argv) > 3:
    dims = sys.argv[3].split(':')
    corr_df = corr_df[corr_df['index'].isin(dims)]
corr_df.drop(columns = ['index'], inplace = True)
#print(corr_df)

heap = []
for _, row in corr_df.iterrows():
    for c in corr_df.columns:
        heapq.heappush(heap, (-float(row[c]), c))

i = 0
while heap:
    print(heapq.heappop(heap))
    i = i + 1
    if i >= 10:
        break
