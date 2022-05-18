from linate import AttitudinalEmbedding

import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from common_defs import *

data_dir = 'plos_asonam_exp_data'

countries = {'France': 'france_own', 'Italy': 'italy_own', 
        'Germany': 'germany_own', 'Spain': 'spain', 'UK': 'uk_own'}
#countries = {'Italy': 'italy_own'}

entity_to_group = {'France': 'FranceOwn_reference_group.csv',
        'Italy': 'ItalyOwn_reference_group.csv', 'Germany': 'GermanyOwn_reference_group.csv',
        'Spain': 'Spain_reference_group.csv', 'UK': 'UKOwn_reference_group.csv'} 

group_attitudes = {'France': 'FranceOwn_group_attitudes.csv',
        'Italy': 'ItalyOwn_group_attitudes.csv', 'Germany': 'GermanyOwn_group_attitudes.csv',
        'Spain': 'Spain_group_attitudes.csv', 'UK': 'UKOwn_group_attitudes.csv'} 

sweep_dimensions = np.arange(2, 11, 1)

error = {}

for country in countries.keys():
    print(country)

    ae_model = AttitudinalEmbedding(N = None)

    # load ideological dimensions at node level
    ideological_embedding_filename = os.path.join(data_dir, countries[country], 
            'exp_results/target_ideological_dimensions.csv')
    ideological_embedding_header_names = {'entity' : 'target_id'}
    X = ae_model.load_ideological_embedding_from_file(ideological_embedding_filename,
            ideological_embedding_header_names = ideological_embedding_header_names)
    X['entity'] = X['entity'].astype(float)
    X['entity'] = X['entity'].astype(int)
    X['entity'] = X['entity'].astype(str)

    # load node - party mapping
    entity_to_group_filename = os.path.join(data_dir, countries[country], entity_to_group[country])
    entity_to_group_mapping_header_names = {'group' : 'k', 'entity' : 'i'}
    XG = ae_model.load_entity_to_group_mapping_from_file(entity_to_group_filename,
            entity_to_group_mapping_header_names = entity_to_group_mapping_header_names)
    # compute group data in latent space
    entity_to_group_agg_fun = None
    X = ae_model.convert_to_group_ideological_embedding(X, XG, entity_to_group_agg_fun = entity_to_group_agg_fun)
    X_copy = X.copy()

    # selecting attitudinal columns and parties
    group_data_filename = os.path.join(data_dir, countries[country], group_attitudes[country])
    group_data = pd.read_csv(group_data_filename)
    group_data.rename({'k': 'entity'},  axis = 1, inplace = True)
    # choose a subset of the cols
    dataset_cols = [c for c in group_data.dropna().columns if c.startswith('ches')]
    nan_per_party = np.isnan(group_data[dataset_cols].values).sum(axis = 1) / len(dataset_cols)
    index_selector = nan_per_party < 0.5
    column_selector = group_data.loc[index_selector, dataset_cols].dropna(axis = 1).columns.tolist()
    Y = group_data.loc[index_selector, ['entity'] + column_selector]
    Y['entity'] = Y['entity'].astype(str)

    error[country] = []

    for Ndims in sweep_dimensions:

        # compute transformation
        ae_model = AttitudinalEmbedding(N = Ndims)
        ae_model.fit(X, Y)
        print('Number of considered ideological dimensions', ae_model.employed_N_)

        # apply transformation to create attitudinal embeddings
        group_ideological = X_copy.copy()
        group_ideological_id = group_ideological['entity']
        Y_id = Y['entity']
        shared_id = pd.merge(group_ideological_id, Y_id, on = 'entity', how = 'inner')
        group_ideological = pd.merge(group_ideological, shared_id, on = 'entity', how = 'inner')
        Y = pd.merge(Y, shared_id, on = 'entity', how = 'inner')
        latent_dims = group_ideological.columns[0:1 + ae_model.employed_N_]
        restricted_group_ideological = group_ideological[latent_dims].copy()

        group_attitudinal = ae_model.transform(restricted_group_ideological)

        group_attitudinal.sort_values(by = ['entity'], ascending = True, inplace = True)
        Y.sort_values(by = ['entity'], ascending = True, inplace = True)

        error_dims = []
        for d in group_attitudinal.columns:
            if d != 'entity':
                error_dims.append(d)
        errors = np.abs(Y[error_dims].values - group_attitudinal[error_dims].values)
        error[country].append(errors.mean())

        #break

# plotting global error and condition number
fig = plt.figure(figsize = bidimensional_figsize)
ax = {}
ax['error'] = fig.add_subplot(1,1,1)
ax['cond'] = fig.add_axes([0.51,0.5,0.4,0.3]) # left, bottom, width, height
for country in countries:
     ax['error'].plot(sweep_dimensions[:len(error[country])], error[country], color = country_colors[country])
     #ax['cond'].plot(sweep_dimensions[:len(cond[country])],cond[country],color=country_colors[country])
ax['error'].set_xlim((2,9))
ax['error'].set_ylim((-0.1,2))
ax['error'].legend(countries,loc='upper left')
ax['error'].set_title('Group position estimation error in '+r'$\mathcal{A}$')
ax['error'].axhline(0,linestyle=':',color='k')
ax['cond'].set_title('Condition number')
ax['cond'].set_yscale('log')
ax['cond'].set_xlim((2,9))
plt.tight_layout()
plt.savefig('lala.pdf')
plt.close()        
