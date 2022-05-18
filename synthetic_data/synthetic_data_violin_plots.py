# produce violin plots with of ideological errors

import synthetic_data_generation as gen

import numpy as np

import pandas as pd

import configparser

import sys

from tqdm import tqdm

from linate import IdeologicalEmbedding

from linate import AttitudinalEmbedding

from linate import compute_euclidean_distance

import matplotlib.pyplot as plt
import seaborn as sn

from common_defs import *

def main():

    params = configparser.ConfigParser()  # read parameters from file
    params.read(sys.argv[1])

    random_state = None # for random number generation

    N_referential = int(params['data_gen']['N_referential'])
    N_followers = int(params['data_gen']['N_followers'])

    standardize_mean = True
    in_degree_threshold = params['ideological_embedding']['in_degree_threshold']
    if in_degree_threshold == 'None':
        in_degree_threshold = None
    out_degree_threshold = params['ideological_embedding']['out_degree_threshold']
    if out_degree_threshold == 'None':
        out_degree_threshold = None
    force_bipartite = params['ideological_embedding']['force_bipartite']
    if force_bipartite == 'True':
        force_bipartite = True
    else:
        force_bipartite = False
    standardize_mean = params['ideological_embedding']['standardize_mean']
    if standardize_mean == 'True':
        standardize_mean = True
    else:
        standardize_mean = False
    standardize_std = params['ideological_embedding']['standardize_std']
    if standardize_std == 'True':
        standardize_std = True
    else:
        standardize_std = False

    # generate synthetic data in ideological space
    #
    # reference users in ideological space
    ref_mixmod = gen.load_gaussian_mixture_model_mu_and_cov_from_files(params['data_gen']['phi_mu'],
            params['data_gen']['phi_cov'])
    phi, phi_info, phi_group, phi_group_info = gen.generate_entities_in_idelogical_space(N_referential,
            ref_mixmod, entity_id_prefix = 'r_', random_state = random_state)

    phi_info_df = pd.DataFrame(phi_info, columns = ['index', 'group', 'id'])
    phi_group_df = pd.DataFrame(phi_group)
    phi_group_df.index.name = 'group_id'
    phi_group_df = phi_group_df.reset_index()

    # followers in ideological space
    fol_mixmod = gen.load_gaussian_mixture_model_mu_and_cov_from_files(params['data_gen']['theta_mu'],
            params['data_gen']['theta_cov'])
    theta, theta_info = gen.generate_entities_in_idelogical_space(N_followers, fol_mixmod,
            entity_id_prefix = 'f_', random_state = random_state, produce_group_dimensions = False)
    theta_info_df = pd.DataFrame(theta_info, columns = ['index', 'group', 'id'])

    # create unobservable attitudes from prescribed ideologies with error
    #
    # load the augmented transformation
    Tideo2att_tilde = gen.load_augmented_transformation_from_file(params['data_gen']['Tideo2att_tilde']) 

    etas = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    N_sims = 100 

    errors = {'References': np.zeros((len(etas), N_referential, N_sims)),
              'Followers': np.zeros((len(etas), N_followers, N_sims))}

    for i_eta, eta in tqdm(enumerate(etas)): 
        print(eta)
        #for n in tqdm(range(N_sims)): 
        for n in range(N_sims): 
            print(n)

            while True: 
                r, r_group = gen.transform_entity_dimensions_to_new_space(phi.T, Tideo2att_tilde,
                        entity_dimensions_info = phi_info.T, introduce_standard_error = True, error_std = eta)
                r_group_df = pd.DataFrame(r_group)
                r_group_df.rename({0: 'group_id'}, axis = 1, inplace = True)
                r_group_df['group_id'] = r_group_df['group_id'].astype(int) 

                f = gen.transform_entity_dimensions_to_new_space(theta.T, Tideo2att_tilde,
                        produce_group_dimensions = False, introduce_standard_error = True, error_std = eta) 

                # perform an additional transformation in the same space (can be I)
                A_P = gen.load_augmented_transformation_from_file(params['data_gen']['A_P']) 
                r_P, r_P_group = gen.transform_entity_dimensions_same_space(r.T, A_P, entity_dimensions_info = phi_info.T) 

                r_P_group_df = pd.DataFrame(r_P_group)
                r_P_group_df.rename({0: 'group_id'}, axis = 1, inplace = True)
                r_P_group_df['group_id'] = r_P_group_df['group_id'].astype(int) 
                r_P_info = phi_info.copy() 

                f_P_info = theta_info.copy()
                f_P = gen.transform_entity_dimensions_same_space(f.T, A_P, produce_group_dimensions = False) 

                # compute the social graph using distances within a given space 
                alpha = int(params['data_gen']['alpha'])
                beta = int(params['data_gen']['beta'])
                social_graph_df, all_edges = gen.compute_social_graph(f_P_info.T[0], f_P_info.T[2], f_P, r_P_info.T[0],
                        r_P_info.T[2], r_P, random_state, alpha, beta, output_all_distances = True) 

                # compute ideological embeddings
                ideological_embedding_model = IdeologicalEmbedding(n_latent_dimensions = int(params['ideological_embedding']
                    ['n_latent_dimensions']), engine = params['ideological_embedding']['engine'],
                    in_degree_threshold = in_degree_threshold, out_degree_threshold = out_degree_threshold,
                    force_bipartite = force_bipartite, standardize_mean = standardize_mean,
                    standardize_std = standardize_std, random_state = random_state)
                #
                X = social_graph_df.copy()
                ideological_embedding_model.fit(X) 

                # project to attitudinal space
                ae_model = AttitudinalEmbedding(N = None) 

                X = ideological_embedding_model.ideological_embedding_target_latent_dimensions_.copy()
                X = X.reset_index()
                X.rename(columns = {'target_id': 'entity'}, inplace = True)
                X['entity'] = X['entity'].astype(str)
                XG = phi_info_df.copy()
                XG = XG[['group', 'id']]
                XG.rename(columns = {'id': 'entity'}, inplace = True)
                X = ae_model.convert_to_group_ideological_embedding(X, XG) 

                Y = phi_group_df.copy()
                Y.rename(columns = {'group_id': 'entity'}, inplace = True)
                Y['entity'] = Y['entity'].astype(str)
                ae_model.fit(X, Y) 

                print('Computing Attitudinal Embedding...')
                phi_latent = ideological_embedding_model.ideological_embedding_target_latent_dimensions_.copy()
                phi_latent = phi_latent.reset_index()
                phi_latent.rename(columns = {'target_id': 'entity'}, inplace = True)
                phi_hat = ae_model.transform(phi_latent) 

                theta_latent = ideological_embedding_model.ideological_embedding_source_latent_dimensions_.copy()
                theta_latent = theta_latent.reset_index()
                theta_latent.rename(columns = {'source_id': 'entity'}, inplace = True)
                theta_hat = ae_model.transform(theta_latent) 

                # compute errors
                missing_nodes = False
                print('Computing Errors...')
                X_df = phi.copy()
                Y_df = phi_hat.copy()
                Y_df['sort'] = Y_df['entity'].apply(lambda d: int(d[d.index('_') + 1:]))
                Y_df = Y_df.sort_values(by = ['sort'], ascending = True)
                Y_df.drop(['sort', 'entity'], axis = 1, inplace = True)
                if X_df.shape[0] == Y_df.shape[0]:
                    error_np = compute_euclidean_distance(X_df, Y_df)
                    errors['References'][i_eta, :, n] = error_np
                else:
                    missing_nodes = True
                #
                X_df = theta.copy()
                Y_df = theta_hat.copy()
                Y_df['sort'] = Y_df['entity'].apply(lambda d: int(d[d.index('_') + 1:]))
                Y_df = Y_df.sort_values(by = ['sort'], ascending = True)
                Y_df.drop(['sort', 'entity'], axis = 1, inplace = True)
                if X_df.shape[0] == Y_df.shape[0]:
                    error_np = compute_euclidean_distance(X_df, Y_df)
                    errors['Followers'][i_eta, :, n] = error_np
                else:
                    missing_nodes = True

                if (not missing_nodes):
                    break

    error_df = pd.DataFrame(columns=['User type',r'$\eta$','Euclidian Error'])
    for user_type in ['References', 'Followers']:
        for i_eta, eta in enumerate(etas):
            for n in range(N_sims):
                df_aux = pd.DataFrame(columns = ['User type', r'$\eta$', 'Euclidian Error'])
                df_aux['Euclidian Error'] = errors[user_type][i_eta,:,n]
                df_aux[r'$\eta$'] = eta
                df_aux['User type'] = user_type
                error_df = pd.concat([error_df, df_aux], axis = 0)
    error_df.to_csv(params['error']['error_filename'], sep = ',', index = None)

    # Plotting the distribution of errors
    fig = plt.figure(figsize = bidimensional_figsize)
    ax = fig.add_subplot(1, 1, 1)
    # referential users
    sn.violinplot(x = r'$\eta$',y = 'Euclidian Error', hue = 'User type', 
            data = error_df, split = True, scale = 'area', palette = 'Paired')
    ax.set_title('Estimation error in ideological space (N=%d)'%N_sims,fontsize=bidimensional_title_fs)
    ax.xaxis.label.set_size(bidimensional_label_fs)
    ax.yaxis.label.set_size(bidimensional_label_fs)
    ax.tick_params(axis='both', which='major', labelsize=bidimensional_tick_fs)
    ax.legend(fontsize=bidimensional_legend_fs)
    plt.tight_layout()
    plt.savefig(params['error']['violin_graph_filename'])
    plt.close()

if __name__ == "__main__":
    main()
