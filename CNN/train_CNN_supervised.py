#!/usr/bin/env python
# coding: utf-8
# python train_CNN_supervised.py <config_path.json>
# python train_CNN_supervised.py config_files/config_01.json

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
    # input shape: (n, res, res, 2)
    mean = np.mean(X, axis=(1, 2), keepdims=True)
    std = np.std(X, axis=(1, 2), keepdims=True)
    epsilon = 1e-8
    std = np.where(std < epsilon, epsilon, std)
    return (X - mean) / std


def main():
    config_path = sys.argv[1]

    # Read config file
    with open(config_path, 'r') as f:
        config = json.load(f)

    npy_path = Path(config['npy_path'])
    seed = config['seed']
    n_train = config['n_train']
    n_val = config['n_val']
    n_test = config['n_test']

    model_name = config['model_name']
    sample_type = config['sample_type']

    with open('params.json', 'r') as f:
        params = json.load(f)

    # 從參數設定中獲取變數
    BATCH_SIZE = params['BATCH_SIZE']
    EPOCHS = params['EPOCHS']
    patience = params['patience']
    min_delta = params['min_delta']
    learning_rate = params['learning_rate']

    save_model_name = f'./CNN_models/last_model_GGF_VBF_supervised_{model_name}/'

    # Load data
    data_b = np.load(npy_path / 'GGF-data.npy', allow_pickle=True)
    data_s = np.load(npy_path / 'VBF-data.npy', allow_pickle=True)

    # Split the dataset
    X_train = np.concatenate((data_b[:n_train], data_s[:n_train]))
    X_val = np.concatenate((data_b[n_train:n_train+n_val], data_s[n_train:n_train+n_val]))
    X_test = np.concatenate((data_b[-n_test:], data_s[-n_test:]))
    y_train = np.concatenate((np.zeros(n_train), np.ones(n_train)))
    y_val = np.concatenate((np.zeros(n_val), np.ones(n_val)))
    y_test = np.concatenate((np.zeros(n_test), np.ones(n_test)))

    # normalize the datasets
    X_train = pt_normalization(X_train)
    X_val = pt_normalization(X_val)
    X_test = pt_normalization(X_test)

    train_size = get_sample_size(y_train)
    val_size = get_sample_size(y_val)
    test_size = get_sample_size(y_test)

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

    history = model.fit(train_dataset, validation_data=valid_dataset, epochs=EPOCHS, callbacks=[early_stopping, check_point])

    # Training results
    best_model_name = f'./CNN_models/best_model_GGF_VBF_supervised_{model_name}/'
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
