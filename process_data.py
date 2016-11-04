'''
process_data.py
'''
import pandautils as pup
import numpy as np
import cPickle as pickle
import pandas as pd
import os
import deepdish.io as io
import utils
import logging

def main(inputfiles, treename, ftrain, max_n_pairs):
    '''
    Args:
    -----
        inputfiles: list of strings with the paths to root files
        treename: string, name of the TTree that contains the branches
        ftrain: float in range [0, 1], training fraction
        max_n_pairs: int, maximum number of jet pairs to consider per event
    Returns:
    --------
    '''
    # -- configure logging
    utils.configure_logging()
    logger = logging.getLogger('main')

    # -- concatenate all files into a pandas df
    short_filenames = [f.split('/')[-1] for f in inputfiles]
    logger.info('Creating pandas dataframes from: {}'.format(
        short_filenames)
    )
    #df = pd.concat([pup.root2panda(f, treename) for f in inputfiles], ignore_index=True)
    df_list = []
    for f in inputfiles:
        df_temp = pup.root2panda(f, treename)
        df_temp['sample'] = f.split('/')[-1].split('.')[0]
        df_list.append(df_temp)
    df = pd.concat(df_list, ignore_index=True)

    # -- remove events with more than one correct jet pair
    # -- because that shouldn't happen and complicates the task
    # -- of finding the correct jet pair
    logger.info('Removing events with more than one correct jet pair')
    keep = np.array([sum(yen) for yen in df['isCorrect'].values]) <= 1
    df = df[keep].reset_index(drop=True)

    # -- target
    logger.info('Building one-hot target')
    y = df['isCorrect'].values

    # -- extract array of names of sample of origin
    sample = df['sample'].values

    # -- prepend 1 to all entries in y where there is no correct jet pair,
    # -- 0 if there exists a correct jet pair already
    # -- each entry in y will now have length (n_jet_pairs + 1)
    y_long = np.array([np.insert(yev, 0, 1) if sum(yev) == 0 
        else np.insert(yev, 0, 0) 
        for yev in y]
    )

    # -- weights
    logger.info('Extracting weights from event_weight')
    w = df['event_weight'].values
    del df['event_weight'], df['isCorrect'], df['sample']

    # -- matrix of predictors
    X = df.values
    varlist = df.columns.values.tolist()

    # -- maximum number of jet pairs to consider in each event
    # -- can be set to whatever number makes sense
    #max_length = max([len(b) for b in df['Delta_eta_jb']]) + 1
    max_length = max_n_pairs + 1
    logger.info('The max number of jet pairs per event will be {}'.format(
        max_n_pairs)
    )

    X_train, X_test, y_train, y_test, w_train, w_test,\
    sample_train, sample_test, scaler_list = shuffle_split_scale_pad(
        X, y_long, w, sample, ftrain, max_length
    )
    
    logger.info('Saving processed data as hdf5 in data/')
    io.save(
        os.path.join('data', 'train_dict.hdf5'),
        {
            'X' : X_train,
            'y' : y_train,
            'w' : w_train,
            'vars' : varlist,
            'sample' : sample_train.tolist(),
            'scalers' : scaler_list
        }
    )

    io.save(
        os.path.join('data', 'test_dict.hdf5'),
        {
            'X' : X_test,
            'y' : y_test,
            'w' : w_test,
            'vars' : varlist,
            'sample' : sample_test.tolist(),
            'scalers' : scaler_list
        }
    )


# ----------------

def _paddingX(X, max_length, value=-999):
    '''
    Transforms X to a 3D array where the dimensions correspond
    to [n_ev, n_jet_pairs, n_features].
    n_jet_pairs is now fixed and equal to max_length.
    If the number of jet pairs in an event was < max_length, 
    the missing jet pairs will be filled with default values.
    If the number of jet pairs in an event was > max_length,
    the excess jet pairs will be removed.
    Args:
    -----
        X: ndarray [n_ev, n_features] with an arbitrary # of jet pairs per event
        max_length: int, the number of jet pairs to keep per event 
        value (optional): the value to input in case there are not enough 
                          jet pairs in the event, default=-999
    Returns:
    --------
        X_pad: ndarray [n_ev, n_jet_pairs, n_features],
               padded version of X with fixed number of jet pairs
    Note:
    -----
        Use Masking to avoid the jet pairs with artificial entries = -999
    '''
    X_pad = value*np.ones((X.shape[0], max_length, X.shape[1]), dtype='float32')
    for i, row in enumerate(X):
        X_pad[i, :min(len(row[0]), max_length), :] = np.array(row.tolist()).T[:min(len(row[0]), max_length), :]

    return X_pad


def _paddingy(y, max_length, value=0):
    '''
    Pads y with zeros.
    If the number of jet pairs in an event was < max_length,
    the missing jet pairs will be filled with default values.
    If the number of jet pairs in an event was > max_length,
    the excess jet pairs will be removed.
    Args:
    -----
        y: ndarray [n_ev, ...] with an arbitrary number of jet pairs per event
        max_length: int, the number of jet pairs to keep per event 
        value (optional): the value to input in case there are not enough jet 
                          pairs in the event, default=0
    Returns:
    --------
        y_pad: ndarray [n_ev, max_length], padded version of y
               with fixed number of jet pairs
    '''
    y_pad = np.zeros((y.shape[0], max_length), dtype='float32')
    for i, row in enumerate(y):
        y_pad[i, :min(len(row), max_length)] = np.array(row[:min(len(row), max_length)])

    return y_pad


def _scale(matrix_train, matrix_test):
    '''
    Use scikit learn to scale features to 0 mean, 1 std. 
    Because of event-level structure, we need to flatten X, scale,
    and then reshape back into event format.
    Args:
    -----
        matrix_train: X_train [n_ev_train, n_jetpair_features], numpy ndarray of
                      unscaled features of events allocated for training
        matrix_test: X_test [n_ev_test, n_jetpair_features], numpy ndarray of 
                      unscaled features of events allocated for testing
    Returns:
    --------
        the same matrices after scaling
        and the scaler_list
    '''
    from sklearn.preprocessing import StandardScaler
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        ref_test = matrix_test[:, 0]
        ref_train = matrix_train[:, 0]
        scaler_list = []
        for col in xrange(matrix_train.shape[1]):
            scaler = StandardScaler()
            matrix_train[:, col] = pup.match_shape(
                scaler.fit_transform(
                    pup.flatten(
                        matrix_train[:, col]
                    ).reshape(-1, 1)
                ).ravel(),
                ref_train
            )
            matrix_test[:, col] = pup.match_shape(
                scaler.transform(
                    pup.flatten(
                        matrix_test[:, col]
                    ).reshape(-1, 1)
                ).ravel(),
                ref_test
            )
            scaler_list.append(scaler)

    return matrix_train, matrix_test, scaler_list


def shuffle_split_scale_pad(X, y, w, sample, ftrain, max_length):
    '''
    Shuffle data, split it into train and test sets, scale X, 
    pads X it with -999, pads y with 0
    Args:
    -----
        X: ndarray [n_ev_train, n_jetpair_features], containing unscaled predictors
        y: ndarray [n_ev, 1] containing the truth labels
        w: ndarray [n_ev, 1] containing the event weights
        sample:
        ftrain:
        max_length:
    Returns:
    --------
        all X, y, w, sample ndarrays for both train and test:
        X_train, X_test, y_train, y_test, w_train, w_test,
        sample_train, sample_test
    '''
    # -- configure logger
    logger = logging.getLogger('shuffle_split_scale_pad')
    from collections import OrderedDict
    from sklearn.model_selection import train_test_split
    
    logger.info('Splitting data randomly into train ({}%) and test ({}%)'.format(
        ftrain * 100, (1 - ftrain) * 100
        )
    )
    X_train, X_test, y_train, y_test, w_train, w_test,\
    sample_train, sample_test = train_test_split(
        X, y, w, sample, train_size=ftrain
    )

    logger.info('Scaling X')
    X_train, X_test, scaler_list = _scale(X_train, X_test)

    logger.info('Padding X')
    X_train = _paddingX(X_train, max_length=max_length)
    X_test = _paddingX(X_test, max_length=max_length)
    
    logger.info('Padding y')
    y_train = _paddingy(y_train, max_length=max_length)
    y_test = _paddingy(y_test, max_length=max_length)

    return X_train, X_test, y_train, y_test, w_train, w_test,\
    sample_train, sample_test, scaler_list

# ----------------

if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Process data for 1-Tag RNN")
    parser.add_argument("--input",
        required=True, type=str, nargs="+", 
        help="List of input root file paths")
    parser.add_argument("--tree",
        type=str, default="events_1tag",
        help="Name of the tree in the ntuples. Default: events_1tag")
    parser.add_argument("--ftrain", type=float, default=0.7,
        help="Fraction of events to allocate for training. Default: 0.7.")
    parser.add_argument("--max_n_pairs", type=int, default=8,
        help="Maximum number of jet pairs to consider per event. Default: 8")
    args = parser.parse_args()

    sys.exit(main(args.input, args.tree, args.ftrain, args.max_n_pairs))
