'''
model.py
'''
import deepdish.io as io
import os
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from keras.layers import Input, LSTM, Dense, Masking, Activation, Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import utils

def main(train_data_path, test_data_path, weights_path):
    '''
    Args:
    -----
        train_data_path:
        test_data_path:
        weights_path:
    '''
    # -- configure logging
    utils.configure_logging()
    logger = logging.getLogger('main')

    # -- import data
    logger.info('Importing data from {} and {}'.format(
        train_data_path, test_data_path)
    )
    train_d = io.load(train_data_path)
    X_train, y_train, w_train = train_d['X'], train_d['y'], abs(train_d['w'])
    # ^^ convert neg weights to positive
    test_d = io.load(test_data_path)
    X_test, y_test, w_test = test_d['X'], test_d['y'], test_d['w']

    # -- balancing classes for training
    for k in range(y_train.shape[1]):
        class_weights = sum(w_train) / (float(y_train.shape[1]) * sum(w_train[np.argmax(y_train, axis=-1) == k]))
        w_train[np.argmax(y_train, axis=-1) == k] *= class_weights
        logger.info('Class {} is scaled by a factor of {}'.format(
            k, class_weights)
        )

    # -- design model
    input_layer = Input(shape=X_train.shape[1:], dtype='float32', name='input')
    output_layer = Activation('softmax')(
        Dense(X_train.shape[1], name='dense')(
            Dropout(0.3)(
                LSTM(output_dim=25, name='LSTM')(
                    Masking(mask_value=-999, input_shape=X_train.shape[1:], name='masking')(
                        input_layer
                    )
                )
            )
        )
    )

    model = Model(input=[input_layer], output=[output_layer])
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    logger.info('Model structure:')
    model.summary()
    from keras.utils.visualize_util import plot
    diagram_path = os.path.join('plots', 'model.png')
    logger.info('Saving diagram of model to ' + diagram_path)
    plot(model, to_file=diagram_path, show_shapes=True)

    # -- train
    try:
        model.load_weights(weights_path)
        logger.info('Pre-trained weights found and loaded from {}'.format(weights_path))
    except IOError:
        logger.info('Pre-trained weights not found in {}'.format(weights_path))

    print 'Training:'
    try:
        model.fit(X_train, 
            y_train,
            batch_size=16, 
            sample_weight=w_train,
            callbacks=[
                EarlyStopping(verbose=True, patience=200, monitor='val_loss'),
                ModelCheckpoint(weights_path,
                monitor='val_loss', verbose=True, save_best_only=True)
            ],
            nb_epoch=1000,
            validation_split = 0.2
        ) 

    except KeyboardInterrupt:
        logger.info('Training ended early')

    logger.info('Reloading best weights')
    model.load_weights(weights_path)

    logger.info('Evaluating performance')
    evaluate_perfromance(model, X_test, y_test, w_test)

# ------------------

def evaluate_perfromance(model, X_test, y_test, w_test):
    '''
    Args:
    -----
        model:
        X_test:
        y_test:
        w_test:
    '''
    logger = logging.getLogger('evaluate_perfromance')
    yhat = model.predict(X_test, verbose=True, batch_size=1024) 
    yhat_top1 = np.argmax(yhat, axis=1) # index of jet pair with highest prob.
    y_true = np.argmax(y_test, axis=1) # index of correct jet pair

    logger.info('Accuracy: {} %'.format(
        100 * sum(y_true == yhat_top1) / float(len(y_true)))
    )

    matplotlib.rcParams.update({'font.size': 14})
    fig = plt.figure(figsize=(11.69, 8.27), dpi=100)
    bins = bins=np.linspace(0, 11, 12)
    _ = plt.hist(y_true[y_true == yhat_top1],
        weights=abs(w_test[y_true == yhat_top1]), # abs value of weights
        bins=bins
    )
    plt.ylabel('Weighted # of events with correctly identified jet pair')
    plt.xlabel('True jet pair index')
    plt.savefig(os.path.join('plots', 'correct.pdf'))

    matplotlib.rcParams.update({'font.size': 14})
    fig = plt.figure(figsize=(11.69, 8.27), dpi=100)
    _ = plt.hist(y_true[y_true != yhat_top1],
        weights=abs(w_test[y_true != yhat_top1]), # abs value of weights
        bins=bins
    )
    plt.ylabel('Weighted # of events with incorrectly identified jet pair')
    plt.xlabel('True jet pair index')
    plt.savefig(os.path.join('plots', 'incorrect.pdf'))

    logger.info('Saved plots to {} and {}'.format(
        os.path.join('plots', 'correct.pdf'),
        os.path.join('plots', 'incorrect.pdf')
        )
    )
    np.save('yhat.npy', yhat)

# ------------------

if __name__ == '__main__':
    import argparse
    import sys
    parser = argparse.ArgumentParser(
        description='Build RNN for jet selection in the 1 tag category'
    )

    parser.add_argument("--train_data",
        required=True, type=str, 
        help="Path to hdf5 data for training"
    )
    parser.add_argument("--test_data",
        required=True, type=str, 
        help="Path to hdf5 data for testing"
    )
    parser.add_argument("--weights",
        required=True, type=str, 
        help="Path to hdf5 weights"
    )
    args = parser.parse_args()
    sys.exit(main(args.train_data, args.test_data, args.weights))
