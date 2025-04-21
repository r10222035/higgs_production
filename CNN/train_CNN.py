#!/usr/bin/env python
# coding: utf-8
# python train_CNN.py <config_path.json>
# python train_CNN.py config_files/config_01.json

import os
import sys
import json
import shutil
import datetime

import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def create_mix_sample_from(npy_dirs: list, nevents: tuple, ratios=(0.8, 0.2), seed=0):
    # npy_dirs: list of npy directories
    # nevents: tuple of (n_VBF_SR, n_VBF_BR, n_GGF_SR, n_GGF_BR)
    # ratios: tuple of (r_train, r_val)
    data_tr, data_vl, data_te = None, None, None
    label_tr, label_vl, label_te = None, None, None

    npy_dir0 = Path(npy_dirs[0])

    data_VBF_SR = np.load(npy_dir0 / 'VBF_in_SR-data.npy')
    data_VBF_BR = np.load(npy_dir0 / 'VBF_in_BR-data.npy')
    data_GGF_SR = np.load(npy_dir0 / 'GGF_in_SR-data.npy')
    data_GGF_BR = np.load(npy_dir0 / 'GGF_in_BR-data.npy')

    n_VBF_SR, n_GGF_SR, n_VBF_BR, n_GGF_BR = nevents
    n_test = 10000
    n_VBF_SR_test = int(data_VBF_SR.shape[0] / (data_VBF_SR.shape[0] + data_VBF_BR.shape[0]) * n_test)
    n_VBF_BR_test = n_test - n_VBF_SR_test
    n_GGF_SR_test = int(data_GGF_SR.shape[0] / (data_GGF_SR.shape[0] + data_GGF_BR.shape[0]) * n_test)
    n_GGF_BR_test = n_test - n_GGF_SR_test

    r_tr, r_vl = ratios

    np.random.seed(seed)
    idx_VBF_SR = np.random.choice(data_VBF_SR.shape[0], n_VBF_SR + n_VBF_SR_test, replace=False)
    idx_VBF_BR = np.random.choice(data_VBF_BR.shape[0], n_VBF_BR + n_VBF_BR_test, replace=False)
    idx_GGF_SR = np.random.choice(data_GGF_SR.shape[0], n_GGF_SR + n_GGF_SR_test, replace=False)
    idx_GGF_BR = np.random.choice(data_GGF_BR.shape[0], n_GGF_BR + n_GGF_BR_test, replace=False)

    idx_VBF_SR_tr = idx_VBF_SR[:int(n_VBF_SR*r_tr)]
    idx_VBF_BR_tr = idx_VBF_BR[:int(n_VBF_BR*r_tr)]
    idx_GGF_SR_tr = idx_GGF_SR[:int(n_GGF_SR*r_tr)]
    idx_GGF_BR_tr = idx_GGF_BR[:int(n_GGF_BR*r_tr)]
    idx_VBF_SR_vl = idx_VBF_SR[int(n_VBF_SR*r_tr):n_VBF_SR]
    idx_VBF_BR_vl = idx_VBF_BR[int(n_VBF_BR*r_tr):n_VBF_BR]
    idx_GGF_SR_vl = idx_GGF_SR[int(n_GGF_SR*r_tr):n_GGF_SR]
    idx_GGF_BR_vl = idx_GGF_BR[int(n_GGF_BR*r_tr):n_GGF_BR]
    idx_VBF_SR_te = idx_VBF_SR[n_VBF_SR:]
    idx_VBF_BR_te = idx_VBF_BR[n_VBF_BR:]
    idx_GGF_SR_te = idx_GGF_SR[n_GGF_SR:]
    idx_GGF_BR_te = idx_GGF_BR[n_GGF_BR:]

    print(f'Preparing dataset from {npy_dirs}')
    for npy_dir in npy_dirs:

        npy_dir = Path(npy_dir)
        data_VBF_SR = np.load(npy_dir / 'VBF_in_SR-data.npy')
        data_VBF_BR = np.load(npy_dir / 'VBF_in_BR-data.npy')
        data_GGF_SR = np.load(npy_dir / 'GGF_in_SR-data.npy')
        data_GGF_BR = np.load(npy_dir / 'GGF_in_BR-data.npy')

        new_data_tr = np.concatenate([
            data_VBF_SR[idx_VBF_SR_tr],
            data_GGF_SR[idx_GGF_SR_tr],
            data_VBF_BR[idx_VBF_BR_tr],
            data_GGF_BR[idx_GGF_BR_tr]
        ], axis=0)
        new_data_vl = np.concatenate([
            data_VBF_SR[idx_VBF_SR_vl],
            data_GGF_SR[idx_GGF_SR_vl],
            data_VBF_BR[idx_VBF_BR_vl],
            data_GGF_BR[idx_GGF_BR_vl]
        ], axis=0)
        # new_data_te = np.concatenate([
        #     data_VBF_SR[idx_VBF_SR_te],
        #     data_VBF_BR[idx_VBF_BR_te],
        #     data_GGF_SR[idx_GGF_SR_te],
        #     data_GGF_BR[idx_GGF_BR_te],
        # ], axis=0)

        if data_tr is None:
            data_tr = new_data_tr
            data_vl = new_data_vl
            # data_te = new_data_te
        else:
            data_tr = np.concatenate([data_tr, new_data_tr], axis=0)
            data_vl = np.concatenate([data_vl, new_data_vl], axis=0)
            # data_te = np.concatenate([data_te, new_data_te], axis=0)

        new_label_tr = np.zeros(new_data_tr.shape[0])
        new_label_tr[:idx_VBF_SR_tr.shape[0] + idx_GGF_SR_tr.shape[0]] = 1
        new_label_vl = np.zeros(new_data_vl.shape[0])
        new_label_vl[:idx_VBF_SR_vl.shape[0] + idx_GGF_SR_vl.shape[0]] = 1
        # new_label_te = np.zeros(new_data_te.shape[0])
        # new_label_te[:n_test] = 1

        if label_tr is None:
            label_tr = new_label_tr
            label_vl = new_label_vl
            # label_te = new_label_te
        else:
            label_tr = np.concatenate([label_tr, new_label_tr])
            label_vl = np.concatenate([label_vl, new_label_vl])
            # label_te = np.concatenate([label_te, new_label_te])

    new_data_te = np.concatenate([
        data_VBF_SR[idx_VBF_SR_te],
        data_VBF_BR[idx_VBF_BR_te],
        data_GGF_SR[idx_GGF_SR_te],
        data_GGF_BR[idx_GGF_BR_te],
    ], axis=0)
    data_te = new_data_te

    new_label_te = np.zeros(new_data_te.shape[0])
    new_label_te[:n_test] = 1
    label_te = new_label_te

    return data_tr, data_vl, data_te, label_tr, label_vl, label_te


def compute_nevent_in_SR_BR(GGF_cutflow_file='../Sample/selection_results_GGF_300_3.1.npy', VBF_cutflow_file='../Sample/selection_results_VBF_300_3.1.npy', L=300, cut_type='mjj'):
    # https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageAt14TeV
    cross_section_GGF = 54.67 * 1000
    cross_section_VBF = 4.278 * 1000
    # https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageBR
    BR_Haa = 0.00227

    GGF_selection = np.load(GGF_cutflow_file, allow_pickle=True).item()
    VBF_selection = np.load(VBF_cutflow_file, allow_pickle=True).item()

    if cut_type == 'mjj':
        n_GGF_SR = cross_section_GGF * GGF_selection['cutflow_number']['mjj: sig region'] / GGF_selection['cutflow_number']['Total'] * BR_Haa * L
        n_GGF_BR = cross_section_GGF * GGF_selection['cutflow_number']['mjj: bkg region'] / GGF_selection['cutflow_number']['Total'] * BR_Haa * L
        n_VBF_SR = cross_section_VBF * VBF_selection['cutflow_number']['mjj: sig region'] / VBF_selection['cutflow_number']['Total'] * BR_Haa * L
        n_VBF_BR = cross_section_VBF * VBF_selection['cutflow_number']['mjj: bkg region'] / VBF_selection['cutflow_number']['Total'] * BR_Haa * L
    elif cut_type == 'deta':
        n_GGF_SR = cross_section_GGF * GGF_selection['cutflow_number']['deta: sig region'] / GGF_selection['cutflow_number']['Total'] * BR_Haa * L
        n_GGF_BR = cross_section_GGF * GGF_selection['cutflow_number']['deta: bkg region'] / GGF_selection['cutflow_number']['Total'] * BR_Haa * L
        n_VBF_SR = cross_section_VBF * VBF_selection['cutflow_number']['deta: sig region'] / VBF_selection['cutflow_number']['Total'] * BR_Haa * L
        n_VBF_BR = cross_section_VBF * VBF_selection['cutflow_number']['deta: bkg region'] / VBF_selection['cutflow_number']['Total'] * BR_Haa * L
    elif cut_type == 'mjj, deta':
        n_GGF_SR = cross_section_GGF * GGF_selection['cutflow_number']['mjj, deta: sig region'] / GGF_selection['cutflow_number']['Total'] * BR_Haa * L
        n_GGF_BR = cross_section_GGF * GGF_selection['cutflow_number']['mjj, deta: bkg region'] / GGF_selection['cutflow_number']['Total'] * BR_Haa * L
        n_VBF_SR = cross_section_VBF * VBF_selection['cutflow_number']['mjj, deta: sig region'] / VBF_selection['cutflow_number']['Total'] * BR_Haa * L
        n_VBF_BR = cross_section_VBF * VBF_selection['cutflow_number']['mjj, deta: bkg region'] / VBF_selection['cutflow_number']['Total'] * BR_Haa * L
    elif cut_type == 'gluon_jet_2':
        n_GGF_SR = cross_section_GGF * GGF_selection['cutflow_number']['two gluon jet: sig region'] / GGF_selection['cutflow_number']['Total'] * BR_Haa * L
        n_GGF_BR = cross_section_GGF * GGF_selection['cutflow_number']['two gluon jet: bkg region'] / GGF_selection['cutflow_number']['Total'] * BR_Haa * L
        n_VBF_SR = cross_section_VBF * VBF_selection['cutflow_number']['two gluon jet: sig region'] / VBF_selection['cutflow_number']['Total'] * BR_Haa * L
        n_VBF_BR = cross_section_VBF * VBF_selection['cutflow_number']['two gluon jet: bkg region'] / VBF_selection['cutflow_number']['Total'] * BR_Haa * L
    elif cut_type == 'gluon_jet_1':
        n_GGF_SR = cross_section_GGF * GGF_selection['cutflow_number']['one gluon jet: sig region'] / GGF_selection['cutflow_number']['Total'] * BR_Haa * L
        n_GGF_BR = cross_section_GGF * GGF_selection['cutflow_number']['one gluon jet: bkg region'] / GGF_selection['cutflow_number']['Total'] * BR_Haa * L
        n_VBF_SR = cross_section_VBF * VBF_selection['cutflow_number']['one gluon jet: sig region'] / VBF_selection['cutflow_number']['Total'] * BR_Haa * L
        n_VBF_BR = cross_section_VBF * VBF_selection['cutflow_number']['one gluon jet: bkg region'] / VBF_selection['cutflow_number']['Total'] * BR_Haa * L
    elif cut_type == 'quark_jet_2':
        n_GGF_SR = cross_section_GGF * GGF_selection['cutflow_number']['two quark jet: sig region'] / GGF_selection['cutflow_number']['Total'] * BR_Haa * L
        n_GGF_BR = cross_section_GGF * GGF_selection['cutflow_number']['two quark jet: bkg region'] / GGF_selection['cutflow_number']['Total'] * BR_Haa * L
        n_VBF_SR = cross_section_VBF * VBF_selection['cutflow_number']['two quark jet: sig region'] / VBF_selection['cutflow_number']['Total'] * BR_Haa * L
        n_VBF_BR = cross_section_VBF * VBF_selection['cutflow_number']['two quark jet: bkg region'] / VBF_selection['cutflow_number']['Total'] * BR_Haa * L
    elif cut_type == 'quark_jet_1':
        n_GGF_SR = cross_section_GGF * GGF_selection['cutflow_number']['one quark jet: sig region'] / GGF_selection['cutflow_number']['Total'] * BR_Haa * L
        n_GGF_BR = cross_section_GGF * GGF_selection['cutflow_number']['one quark jet: bkg region'] / GGF_selection['cutflow_number']['Total'] * BR_Haa * L
        n_VBF_SR = cross_section_VBF * VBF_selection['cutflow_number']['one quark jet: sig region'] / VBF_selection['cutflow_number']['Total'] * BR_Haa * L
        n_VBF_BR = cross_section_VBF * VBF_selection['cutflow_number']['one quark jet: bkg region'] / VBF_selection['cutflow_number']['Total'] * BR_Haa * L
    elif cut_type == 'quark_gluon_jet_2':
        n_GGF_SR = cross_section_GGF * GGF_selection['cutflow_number']['two quark jet: sig region'] / GGF_selection['cutflow_number']['Total'] * BR_Haa * L
        n_GGF_BR = cross_section_GGF * GGF_selection['cutflow_number']['two gluon jet: bkg region'] / GGF_selection['cutflow_number']['Total'] * BR_Haa * L
        n_VBF_SR = cross_section_VBF * VBF_selection['cutflow_number']['two quark jet: sig region'] / VBF_selection['cutflow_number']['Total'] * BR_Haa * L
        n_VBF_BR = cross_section_VBF * VBF_selection['cutflow_number']['two gluon jet: bkg region'] / VBF_selection['cutflow_number']['Total'] * BR_Haa * L
    else:
        raise ValueError('cut_type must be mjj, deta, or mjj, or deta, or gluon_jet')
    return n_VBF_SR, n_GGF_SR, n_VBF_BR, n_GGF_BR


def get_sample_size(y):
    if len(y.shape) == 1:
        ns = (y == 1).sum()
        nb = (y == 0).sum()
    else:
        ns = (y.argmax(axis=1) == 1).sum()
        nb = (y.argmax(axis=1) == 0).sum()
    print(ns, nb)
    return ns, nb


class CNN(tf.keras.Model):
    def __init__(self, name='CNN'):
        super(CNN, self).__init__(name=name)

        self.bn = tf.keras.layers.BatchNormalization()

        self.sub_network = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (5, 5), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Conv2D(64, (5, 5), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ])

    @tf.function
    def call(self, inputs, training=False):

        output = self.bn(inputs)
        output = self.sub_network(output)

        return output


def get_highest_accuracy(y_true, y_pred):
    _, _, thresholds = roc_curve(y_true, y_pred)
    # compute highest accuracy
    thresholds = np.array(thresholds)
    if len(thresholds) > 1000:
        thresholds = np.percentile(thresholds, np.linspace(0, 100, 1001))
    accuracy_scores = []
    for threshold in thresholds:
        accuracy_scores.append(accuracy_score(y_true, y_pred > threshold))

    accuracies = np.array(accuracy_scores)
    return accuracies.max()


def get_tpr_from_fpr(passing_rate, fpr, tpr):
    n_th = (fpr < passing_rate).sum()
    return tpr[n_th]


def pt_normalization(X):
    # input shape: (n, res, res, 3)
    mean = np.mean(X, axis=(1, 2), keepdims=True)
    std = np.std(X, axis=(1, 2), keepdims=True)
    epsilon = 1e-8
    std = np.where(std < epsilon, epsilon, std)
    return (X - mean) / std


def pt_scaling(X):
    # input shape: (n, res, res, 3)
    # the total scaling of the input
    mean = np.mean(X, axis=(0, 1, 2), keepdims=True)
    std = np.std(X, axis=(0, 1, 2), keepdims=True)
    epsilon = 1e-8
    std = np.where(std < epsilon, epsilon, std)
    return (X - mean) / std

def main():
    config_path = sys.argv[1]

    # Read config file
    with open(config_path, 'r') as f:
        config = json.load(f)

    npy_paths = config['npy_paths']
    seed = config['seed']
    luminosity = config['luminosity']
    cut_type = config['cut_type']
    model_name = config['model_name']
    sample_type = config['sample_type']

    GGF_cutflow_file = config['GGF_cutflow_file']
    VBF_cutflow_file = config['VBF_cutflow_file']

    # Training parameters
    with open('params.json', 'r') as f:
        params = json.load(f)

    BATCH_SIZE = params['BATCH_SIZE']
    EPOCHS = params['EPOCHS']
    patience = params['patience']
    min_delta = params['min_delta']
    learning_rate = params['learning_rate']

    save_model_name = f'./CNN_models/last_model_GGF_VBF_CWoLa_{model_name}/'

    # Sampling dataset
    r_train, r_val = 0.8, 0.2
    n_SR_VBF, n_SR_GGF, n_BR_VBF, n_BR_GGF = compute_nevent_in_SR_BR(GGF_cutflow_file, VBF_cutflow_file, luminosity, cut_type)
    n_events = (int(n_SR_VBF), int(n_SR_GGF), int(n_BR_VBF), int(n_BR_GGF))

    X_train, X_val, X_test, y_train, y_val, y_test = create_mix_sample_from(npy_paths, n_events, (r_train, r_val), seed=seed)

    # normalize the datasets
    X_train = pt_normalization(X_train)
    X_val = pt_normalization(X_val)
    X_test = pt_normalization(X_test)

    train_size = get_sample_size(y_train)
    val_size = get_sample_size(y_val)
    test_size = get_sample_size(y_test)

    class_weight = {0: 1.0, 1: train_size[1] / train_size[0]}

    with tf.device('CPU'):
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=len(y_train)).batch(BATCH_SIZE)
        # del X_train, y_train

        valid_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        valid_dataset = valid_dataset.batch(BATCH_SIZE)

    model = CNN()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=min_delta, verbose=1, patience=patience)
    check_point = tf.keras.callbacks.ModelCheckpoint(save_model_name, monitor='val_loss', verbose=1, save_best_only=True)

    history = model.fit(train_dataset, validation_data=valid_dataset, epochs=EPOCHS, class_weight=class_weight, callbacks=[early_stopping, check_point])

    # Training results
    best_model_name = f'./CNN_models/best_model_GGF_VBF_CWoLa_{model_name}/'
    if not os.path.isdir(best_model_name):
        shutil.copytree(save_model_name, best_model_name, dirs_exist_ok=True)
        print('Save to best model')
    best_model = tf.keras.models.load_model(best_model_name)
    best_results = best_model.evaluate(valid_dataset, batch_size=BATCH_SIZE)
    print(f'Testing Loss = {best_results[0]:.3}, Testing Accuracy = {best_results[1]:.3}')

    loaded_model = tf.keras.models.load_model(save_model_name)
    results = loaded_model.evaluate(valid_dataset, batch_size=BATCH_SIZE)
    print(f'Testing Loss = {results[0]:.3}, Testing Accuracy = {results[1]:.3}')

    if results[0] < best_results[0]:
        shutil.copytree(save_model_name, best_model_name, dirs_exist_ok=True)
        print('Save to best model')

    # Compute ACC & AUC
    y_pred = loaded_model.predict(X_val, batch_size=BATCH_SIZE)
    ACC = get_highest_accuracy(y_val, y_pred)
    AUC = roc_auc_score(y_val, y_pred)

    # Testing results on true label sample
    true_label_results = loaded_model.evaluate(x=X_test, y=y_test, batch_size=BATCH_SIZE)
    print(f'True label: Testing Loss = {true_label_results[0]:.3}, Testing Accuracy = {true_label_results[1]:.3}')

    # Compute ACC & AUC
    y_pred = loaded_model.predict(X_test, batch_size=BATCH_SIZE)
    true_label_ACC = get_highest_accuracy(y_test, y_pred)
    true_label_AUC = roc_auc_score(y_test, y_pred)

    # Write results
    now = datetime.datetime.now()
    file_name = 'GGF_VBF_CWoLa_training_results.csv'
    data_dict = {
                'Train signal size': [train_size[0]],
                'Train background size': [train_size[1]],
                'Validation signal size': [val_size[0]],
                'Validation background size': [val_size[1]],
                'Test signal size': [test_size[0]],
                'Test background size': [test_size[1]],
                'Loss': [results[0]],
                'ACC': [ACC],
                'AUC': [AUC],
                'Loss-true': [true_label_results[0]],
                'ACC-true': [true_label_ACC],
                'AUC-true': [true_label_AUC],
                'Sample Type': [sample_type],
                'Model Name': [model_name],
                'Training epochs': [len(history.history['loss']) + 1],
                'time': [now],
                }

    df = pd.DataFrame(data_dict)
    if os.path.isfile(file_name):
        training_results_df = pd.read_csv(file_name)
        pd.concat([training_results_df, df], ignore_index=True).to_csv(file_name, index=False)
    else:
        df.to_csv(file_name, index=False)


if __name__ == '__main__':
    main()
