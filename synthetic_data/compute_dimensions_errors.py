from linate import compute_euclidean_distance

import pandas as pd

import configparser

import numpy as np

import sys

def main():

    params = configparser.ConfigParser()  # read parameters from file
    params.read(sys.argv[1])

    # phi errors
    # read first set of dimensions
    X_df = pd.read_csv(params['errors']['phi_file'], header = None)
    #print(X_df.shape)
    # read second set of dimensions
    Y_df = pd.read_csv(params['errors']['phi_hat_file'])
    Y_df['sort'] = Y_df['entity'].apply(lambda d: int(d[d.index('_') + 1:]))
    Y_df = Y_df.sort_values(by = ['sort'], ascending = True)
    Y_df.drop(['sort', 'entity'], axis = 1, inplace = True)
    #print(Y_df.shape)
    error_np = compute_euclidean_distance(X_df, Y_df)
    np.savetxt(params['errors']['phi_hat_error_file'], error_np, delimiter = ",", fmt = "%f")

    # theta errors
    X_df = pd.read_csv(params['errors']['theta_file'], header = None)
    #print(X_df.shape)
    Y_df = pd.read_csv(params['errors']['theta_hat_file'])
    Y_df['sort'] = Y_df['entity'].apply(lambda d: int(d[d.index('_') + 1:]))
    Y_df = Y_df.sort_values(by = ['sort'], ascending = True)
    Y_df.drop(['sort', 'entity'], axis = 1, inplace = True)
    #print(Y_df.shape)
    error_np = compute_euclidean_distance(X_df, Y_df)
    np.savetxt(params['errors']['theta_hat_error_file'], error_np, delimiter = ",", fmt = "%f")

if __name__ == "__main__":
    main()
