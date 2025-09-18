#!/usr/bin/env python
# coding: utf-8
# python train_ParT.py <config_path.json>

import os
import sys
import json
import h5py
import shutil
import datetime

import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score

from model_tf import ParT_Light

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


MAX_CONSTI = {
    'Jet': 50,
    'Tower': 250,
    'Track': 150,
    'Photon': 2,
    'Lepton': 4,
}


def prepare_feature_from_h5(h5_file, remove_decay_products=False):
    with h5py.File(h5_file, 'r') as f:
        event_pt = np.concatenate([f['TOWER/pt'][:], f['TRACK/pt'][:], f['PHOTON/pt'][:]], axis=1)
        event_eta = np.concatenate([f['TOWER/eta'][:], f['TRACK/eta'][:], f['PHOTON/eta'][:]], axis=1)
        event_phi = np.concatenate([f['TOWER/phi'][:], f['TRACK/phi'][:], f['PHOTON/phi'][:]], axis=1)
        event_mask = np.concatenate([f['TOWER/mask'][:], f['TRACK/mask'][:], np.tile([True, True], (event_pt.shape[0], 1))], axis=1)

        if remove_decay_products:
            photon_eta = f['PHOTON/eta'][:]
            photon_phi = f['PHOTON/phi'][:]
            indices = np.where((event_eta[:, :, None] == photon_eta[:, None, :]) & (event_phi[:, :, None] == photon_phi[:, None, :]))
            event_mask[indices[0], indices[1]] = False

        event_pt[event_mask == False] = float('nan')
        event_eta[event_mask == False] = float('nan')
        event_phi[event_mask == False] = float('nan')

        event_particle_type_0 = np.array([1] * MAX_CONSTI['Tower'] + [0] * MAX_CONSTI['Track'] + [0] * MAX_CONSTI['Photon'])
        event_particle_type_0 = np.tile(event_particle_type_0, (event_pt.shape[0], 1))
        event_particle_type_1 = np.array([0] * MAX_CONSTI['Tower'] + [1] * MAX_CONSTI['Track'] + [0] * MAX_CONSTI['Photon'])
        event_particle_type_1 = np.tile(event_particle_type_1, (event_pt.shape[0], 1))
        event_particle_type_2 = np.array([0] * MAX_CONSTI['Tower'] + [0] * MAX_CONSTI['Track'] + [1] * MAX_CONSTI['Photon'])
        event_particle_type_2 = np.tile(event_particle_type_2, (event_pt.shape[0], 1))

        features = np.stack([event_pt, event_eta, event_phi, event_particle_type_0, event_particle_type_1, event_particle_type_2], axis=-1)

    return features


def create_mix_sample_from(h5_dirs: list, nevents: tuple, ratios=(0.8, 0.2), seed=0, remove_decay_products=False):
    # h5_dirs: list of npy directories
    # nevents: tuple of (n_VBF_SR, n_VBF_BR, n_GGF_SR, n_GGF_BR)
    # ratios: tuple of (r_train, r_val)
    data_tr, data_vl, data_te = None, None, None
    label_tr, label_vl, label_te = None, None, None

    h5_dir0 = Path(h5_dirs[0])
    data_VBF_SR = prepare_feature_from_h5(h5_dir0 / 'VBF_in_SR.h5', remove_decay_products)
    data_VBF_BR = prepare_feature_from_h5(h5_dir0 / 'VBF_in_BR.h5', remove_decay_products)
    data_GGF_SR = prepare_feature_from_h5(h5_dir0 / 'GGF_in_SR.h5', remove_decay_products)
    data_GGF_BR = prepare_feature_from_h5(h5_dir0 / 'GGF_in_BR.h5', remove_decay_products)
    n_data_VBF_SR = data_VBF_SR.shape[0]
    n_data_VBF_BR = data_VBF_BR.shape[0]
    n_data_GGF_SR = data_GGF_SR.shape[0]
    n_data_GGF_BR = data_GGF_BR.shape[0]
    print(f'Number of events in VBF_SR: {n_data_VBF_SR}, VBF_BR: {n_data_VBF_BR}, GGF_SR: {n_data_GGF_SR}, GGF_BR: {n_data_GGF_BR}')

    n_VBF_SR, n_GGF_SR, n_VBF_BR, n_GGF_BR = nevents
    n_test = 10000
    n_VBF_SR_test = int(n_data_VBF_SR / (n_data_VBF_SR + n_data_VBF_BR) * n_test)
    n_VBF_BR_test = n_test - n_VBF_SR_test
    n_GGF_SR_test = int(n_data_GGF_SR / (n_data_GGF_SR + n_data_GGF_BR) * n_test)
    n_GGF_BR_test = n_test - n_GGF_SR_test

    r_tr, r_vl = ratios

    np.random.seed(seed)
    idx_VBF_SR = np.random.choice(n_data_VBF_SR, n_VBF_SR + n_VBF_SR_test, replace=False)
    idx_VBF_BR = np.random.choice(n_data_VBF_BR, n_VBF_BR + n_VBF_BR_test, replace=False)
    idx_GGF_SR = np.random.choice(n_data_GGF_SR, n_GGF_SR + n_GGF_SR_test, replace=False)
    idx_GGF_BR = np.random.choice(n_data_GGF_BR, n_GGF_BR + n_GGF_BR_test, replace=False)

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

    print(f'Preparing dataset from {h5_dirs}')
    for h5_dir in h5_dirs:

        h5_dir = Path(h5_dir)
        data_VBF_SR = prepare_feature_from_h5(h5_dir / 'VBF_in_SR.h5', remove_decay_products)
        data_VBF_BR = prepare_feature_from_h5(h5_dir / 'VBF_in_BR.h5', remove_decay_products)
        data_GGF_SR = prepare_feature_from_h5(h5_dir / 'GGF_in_SR.h5', remove_decay_products)
        data_GGF_BR = prepare_feature_from_h5(h5_dir / 'GGF_in_BR.h5', remove_decay_products)

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

        if data_tr is None:
            data_tr = new_data_tr
            data_vl = new_data_vl
        else:
            data_tr = np.concatenate([data_tr, new_data_tr], axis=0)
            data_vl = np.concatenate([data_vl, new_data_vl], axis=0)

        new_label_tr = np.zeros(new_data_tr.shape[0])
        new_label_tr[:idx_VBF_SR_tr.shape[0] + idx_GGF_SR_tr.shape[0]] = 1
        new_label_vl = np.zeros(new_data_vl.shape[0])
        new_label_vl[:idx_VBF_SR_vl.shape[0] + idx_GGF_SR_vl.shape[0]] = 1

        if label_tr is None:
            label_tr = new_label_tr
            label_vl = new_label_vl
        else:
            label_tr = np.concatenate([label_tr, new_label_tr])
            label_vl = np.concatenate([label_vl, new_label_vl])

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


def create_pure_sample_from(h5_dir, n_events, remove_decay_products=False):

    h5_dir = Path(h5_dir)
    features_GGF = prepare_feature_from_h5(h5_dir / 'GGF.h5', remove_decay_products)
    features_VBF = prepare_feature_from_h5(h5_dir / 'VBF.h5', remove_decay_products)

    n_train, n_val, n_test = n_events

    # Split the dataset into training and validation sets
    X_train = np.concatenate([features_GGF[:n_train], features_VBF[:n_train]], axis=0)
    y_train = np.array([0] * n_train + [1] * n_train)
    X_val = np.concatenate([features_GGF[n_train:n_train+n_val], features_VBF[n_train:n_train+n_val]], axis=0)
    y_val = np.array([0] * n_val + [1] * n_val)
    X_test = np.concatenate([features_GGF[n_train+n_val:n_train+n_val+n_test], features_VBF[n_train+n_val:n_train+n_val+n_test]], axis=0)
    y_test = np.array([0] * n_test + [1] * n_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


def compute_nevent_in_SR_BR(GGF_cutflow_file, VBF_cutflow_file, L, cut_type, BR=0.00227):
    # https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageAt14TeV
    cross_section_GGF = 54.67 * 1000
    cross_section_VBF = 4.278 * 1000

    GGF_selection = np.load(GGF_cutflow_file, allow_pickle=True).item()
    VBF_selection = np.load(VBF_cutflow_file, allow_pickle=True).item()

    if cut_type == 'quark_jet_2':
        n_GGF_SR = cross_section_GGF * GGF_selection['cutflow_number']['two quark jet: sig region'] / GGF_selection['cutflow_number']['Total'] * BR * L
        n_GGF_BR = cross_section_GGF * GGF_selection['cutflow_number']['two quark jet: bkg region'] / GGF_selection['cutflow_number']['Total'] * BR * L
        n_VBF_SR = cross_section_VBF * VBF_selection['cutflow_number']['two quark jet: sig region'] / VBF_selection['cutflow_number']['Total'] * BR * L
        n_VBF_BR = cross_section_VBF * VBF_selection['cutflow_number']['two quark jet: bkg region'] / VBF_selection['cutflow_number']['Total'] * BR * L
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
    print(f'Sample size: ns = {ns}, nb = {nb}')
    return ns, nb


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


def pt_normalization(X):
    slices = [slice(0, 250), slice(250, 400), slice(400, 402)]
    for i, s in enumerate(slices):
        mean = np.nanmean(X[:, s, 0], axis=(1), keepdims=True)
        std = np.nanstd(X[:, s, 0], axis=(1), keepdims=True)
        mean[np.isnan(mean)] = 0
        std[np.isnan(std)] = 1
        epsilon = 1e-8
        std = np.where(std < epsilon, epsilon, std)

        X[:, s, 0] = (X[:, s, 0] - mean) / std


class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_schedule_fn, warmup_steps):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_schedule_fn = decay_schedule_fn
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        # Linear warmup
        warmup_lr = self.initial_learning_rate * (tf.cast(step, tf.float32) / tf.cast(self.warmup_steps, tf.float32))
        # After warmup → decay schedule
        return tf.cond(
            step < self.warmup_steps,
            lambda: warmup_lr,
            lambda: self.decay_schedule_fn(step - self.warmup_steps)
        )

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_schedule_fn": self.decay_schedule_fn,
            "warmup_steps": self.warmup_steps,
        }


def main():
    config_path = sys.argv[1]

    # Read config file
    with open(config_path, 'r') as f:
        config = json.load(f)

    npy_paths = config['npy_paths']
    seed = config['seed']
    model_name = config['model_name']
    model_structure = config['model_structure']
    training_method = config['training_method']
    sample_type = config['sample_type']
    remove_decay_products = config['remove_decay_products']

    # Training parameters
    with open('params.json', 'r') as f:
        params = json.load(f)

    BATCH_SIZE = params['BATCH_SIZE']
    EPOCHS = params['EPOCHS']
    patience = params['patience']
    min_delta = params['min_delta']
    learning_rate = params['learning_rate']

    save_model_name = f'./ParT_models/last_model_{training_method}_{model_name}/'

    # Sampling dataset
    if training_method == 'CWoLa':
        luminosity = config['luminosity']
        cut_type = config['cut_type']

        GGF_cutflow_file = config['GGF_cutflow_file']
        VBF_cutflow_file = config['VBF_cutflow_file']
        decay_channel = config['decay_channel']

        if decay_channel == 'ZZ4l':
            BR = 0.0001240
        elif decay_channel == 'aa':
            BR = 0.00227
        else:
            raise ValueError(f'Unknown decay channel: {decay_channel}')

        r_train, r_val = 0.8, 0.2
        n_SR_VBF, n_SR_GGF, n_BR_VBF, n_BR_GGF = compute_nevent_in_SR_BR(GGF_cutflow_file, VBF_cutflow_file, luminosity, cut_type, BR)
        n_events = (int(n_SR_VBF), int(n_SR_GGF), int(n_BR_VBF), int(n_BR_GGF))
        X_train, X_val, X_test, y_train, y_val, y_test = create_mix_sample_from(npy_paths, n_events, (r_train, r_val), seed=seed, remove_decay_products=remove_decay_products)
    elif training_method == 'supervised':
        n_events = config['n_train'], config['n_val'], config['n_test']
        X_train, X_val, X_test, y_train, y_val, y_test = create_pure_sample_from(npy_paths[0], n_events, remove_decay_products=remove_decay_products)
    else:
        raise ValueError(f'Unknown training method: {training_method}')

    pt_normalization(X_train)
    pt_normalization(X_val)
    pt_normalization(X_test)

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

    if model_structure == 'ParT_Light':
        model = ParT_Light(num_channels=3)
    else:
        raise ValueError(f'Unknown model name: {model_name}')

    # # Learning rate schedule
    # steps_per_epoch = len(y_train) // BATCH_SIZE
    # warmup_epochs, decay_epochs = 5, 10
    # warmup_steps = warmup_epochs * steps_per_epoch
    # decay_steps = decay_epochs * steps_per_epoch
    # decay_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=learning_rate, decay_steps=decay_steps, decay_rate=0.9, staircase=False
    # )
    # lr_schedule = WarmUp(learning_rate, decay_schedule, warmup_steps=warmup_steps)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=min_delta, verbose=1, patience=patience)
    check_point = tf.keras.callbacks.ModelCheckpoint(save_model_name, monitor='val_loss', verbose=1, save_best_only=True)

    history = model.fit(train_dataset, validation_data=valid_dataset, epochs=EPOCHS, class_weight=class_weight, callbacks=[early_stopping, check_point])

    # Training results
    best_model_name = f'./ParT_models/best_model_{training_method}_{model_name}/'
    if not os.path.isdir(best_model_name):
        shutil.copytree(save_model_name, best_model_name, dirs_exist_ok=True)
        print('Save to best model')
    with tf.keras.utils.custom_object_scope({'WarmUp': WarmUp}):
        best_model = tf.keras.models.load_model(best_model_name)
    best_results = best_model.evaluate(valid_dataset, batch_size=BATCH_SIZE)
    print(f'Testing Loss = {best_results[0]:.3}, Testing Accuracy = {best_results[1]:.3}')

    with tf.keras.utils.custom_object_scope({'WarmUp': WarmUp}):
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
    file_name = 'GGF_VBF_training_results.csv'
    data_dict = {
                'Train signal size': [train_size[0]],
                'Train background size': [train_size[1]],
                'Validation signal size': [val_size[0]],
                'Validation background size': [val_size[1]],
                'Test signal size': [test_size[0]],
                'Test background size': [test_size[1]],
                'Training method': [training_method],
                'Model structure': [model_structure],
                'Model Name': [model_name],
                'Loss': [results[0]],
                'ACC': [ACC],
                'AUC': [AUC],
                'Loss-true': [true_label_results[0]],
                'ACC-true': [true_label_ACC],
                'AUC-true': [true_label_AUC],
                'Sample Type': [sample_type],
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
