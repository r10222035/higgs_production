#!/usr/bin/env python3
import os
import json
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score

import gc
import time
import random
import argparse

# 優先使用外部設定的 GPU，否則預設使用 GPU 0
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/r10222035/.conda/envs/tf2'
import tensorflow as tf

# 啟用 memory growth 避免佔滿整張 GPU 顯存
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

MAX_CONSTI = {
    'Jet': 50,
    'Tower': 250,
    'Track': 150,
    'Photon': 2,
    'PHOTON': 2,
    'Lepton': 4,
    'LEPTON': 4,
    'Photon_Za': 1,
    'Lepton_Za': 2,
}

BATCH_SIZE = 128

# 1. 物理事件數計算 (Za 頻道使用專屬 BR)
def compute_nevent_in_SR_BR_Za(GGF_cutflow_file, VBF_cutflow_file, L=300, cut_type='quark_jet_2'):
    cross_section_GGF = 54.67 * 1000
    cross_section_VBF = 4.278 * 1000
    # Za 分支比: BR(H->Za) * (BR(Z->ee) + BR(Z->mumu)) = 1.533e-3 * (3.36e-2 + 3.36e-2) = 1.030176e-4
    BR_Za = 1.533e-3 * (3.36e-2 + 3.36e-2)

    GGF_selection = np.load(GGF_cutflow_file, allow_pickle=True).item()
    VBF_selection = np.load(VBF_cutflow_file, allow_pickle=True).item()

    if cut_type == 'quark_jet_2':
        n_GGF_SR = cross_section_GGF * GGF_selection['cutflow_number']['two quark jet: sig region'] / GGF_selection['cutflow_number']['Total'] * BR_Za * L
        n_GGF_BR = cross_section_GGF * GGF_selection['cutflow_number']['two quark jet: bkg region'] / GGF_selection['cutflow_number']['Total'] * BR_Za * L
        n_VBF_SR = cross_section_VBF * VBF_selection['cutflow_number']['two quark jet: sig region'] / VBF_selection['cutflow_number']['Total'] * BR_Za * L
        n_VBF_BR = cross_section_VBF * VBF_selection['cutflow_number']['two quark jet: bkg region'] / VBF_selection['cutflow_number']['Total'] * BR_Za * L
    else:
        raise ValueError('cut_type must be quark_jet_2')
    return n_VBF_SR, n_GGF_SR, n_VBF_BR, n_GGF_BR

# 2. 輔助函數
def get_highest_accuracy(y_true, y_pred):
    _, _, thresholds = roc_curve(y_true, y_pred)
    thresholds = np.array(thresholds)
    if len(thresholds) > 1000:
        thresholds = np.percentile(thresholds, np.linspace(0, 100, 1001))
    accuracy_scores = []
    for threshold in thresholds:
        accuracy_scores.append(accuracy_score(y_true, y_pred > threshold))
    return np.array(accuracy_scores).max()

# 2a. CNN Pt Normalization
def pt_normalization_CNN(X):
    mean = np.mean(X, axis=(1, 2), keepdims=True)
    std = np.std(X, axis=(1, 2), keepdims=True)
    epsilon = 1e-8
    std = np.where(std < epsilon, epsilon, std)
    return (X - mean) / std

_cnn_data_cache = {}

# 2b. CNN Data Preparation
def create_cnn_test_sample(npy_dirs, nevents, seed=0):
    npy_dir0 = Path(npy_dirs[0])
    cache_key = str(npy_dir0)
    
    if cache_key not in _cnn_data_cache:
        data_VBF_SR = np.load(npy_dir0 / 'VBF_in_SR-data.npy')
        data_VBF_BR = np.load(npy_dir0 / 'VBF_in_BR-data.npy')
        data_GGF_SR = np.load(npy_dir0 / 'GGF_in_SR-data.npy')
        data_GGF_BR = np.load(npy_dir0 / 'GGF_in_BR-data.npy')
        _cnn_data_cache[cache_key] = (data_VBF_SR, data_VBF_BR, data_GGF_SR, data_GGF_BR)
    else:
        data_VBF_SR, data_VBF_BR, data_GGF_SR, data_GGF_BR = _cnn_data_cache[cache_key]

    n_VBF_SR, n_GGF_SR, n_VBF_BR, n_GGF_BR = nevents
    n_test = 10000
    n_VBF_SR_test = int(data_VBF_SR.shape[0] / (data_VBF_SR.shape[0] + data_VBF_BR.shape[0]) * n_test)
    n_VBF_BR_test = n_test - n_VBF_SR_test
    n_GGF_SR_test = int(data_GGF_SR.shape[0] / (data_GGF_SR.shape[0] + data_GGF_BR.shape[0]) * n_test)
    n_GGF_BR_test = n_test - n_GGF_SR_test

    np.random.seed(seed)
    idx_VBF_SR = np.random.choice(data_VBF_SR.shape[0], n_VBF_SR + n_VBF_SR_test, replace=False)
    idx_VBF_BR = np.random.choice(data_VBF_BR.shape[0], n_VBF_BR + n_VBF_BR_test, replace=False)
    idx_GGF_SR = np.random.choice(data_GGF_SR.shape[0], n_GGF_SR + n_GGF_SR_test, replace=False)
    idx_GGF_BR = np.random.choice(data_GGF_BR.shape[0], n_GGF_BR + n_GGF_BR_test, replace=False)

    new_data_te = np.concatenate([
        data_VBF_SR[idx_VBF_SR[n_VBF_SR:]],
        data_VBF_BR[idx_VBF_BR[n_VBF_BR:]],
        data_GGF_SR[idx_GGF_SR[n_GGF_SR:]],
        data_GGF_BR[idx_GGF_BR[n_GGF_BR:]],
    ], axis=0)

    new_label_te = np.zeros(new_data_te.shape[0])
    new_label_te[:n_test] = 1
    return new_data_te, new_label_te

# 2c. ParT Pt Normalization
def pt_normalization_ParT(X):
    slices = [slice(0, 250), slice(250, 400), slice(400, 402)]
    for s in slices:
        mean = np.nanmean(X[:, s, 0], axis=1, keepdims=True)
        std = np.nanstd(X[:, s, 0], axis=1, keepdims=True)
        mean[np.isnan(mean)] = 0
        std[np.isnan(std)] = 1
        epsilon = 1e-8
        std = np.where(std < epsilon, epsilon, std)
        X[:, s, 0] = (X[:, s, 0] - mean) / std

_part_data_cache = {}

# 2d. ParT Data Preparation
def prepare_feature_from_h5(h5_file, remove_decay_products=True):
    cache_key = (str(h5_file), remove_decay_products)
    if cache_key in _part_data_cache:
        return _part_data_cache[cache_key]
        
    with h5py.File(h5_file, 'r') as f:
        # Za 包含 PHOTON 與 LEPTON
        event_pt = np.concatenate([f['TOWER/pt'][:], f['TRACK/pt'][:], f['PHOTON/pt'][:], f['LEPTON/pt'][:]], axis=1)
        event_eta = np.concatenate([f['TOWER/eta'][:], f['TRACK/eta'][:], f['PHOTON/eta'][:], f['LEPTON/eta'][:]], axis=1)
        event_phi = np.concatenate([f['TOWER/phi'][:], f['TRACK/phi'][:], f['PHOTON/phi'][:], f['LEPTON/phi'][:]], axis=1)
        
        # 由於是 Za 頻道，光子為 1，Lepton 為 2
        total_decay_len = MAX_CONSTI['Photon_Za'] + MAX_CONSTI['Lepton_Za']
        event_mask = np.concatenate([
            f['TOWER/mask'][:],
            f['TRACK/mask'][:],
            np.tile([True] * total_decay_len, (event_pt.shape[0], 1))
        ], axis=1)

        if remove_decay_products:
            # 移除所有衰變產物以進行公平對比
            decay_product_eta = np.concatenate([f['PHOTON/eta'][:], f['LEPTON/eta'][:]], axis=1)
            decay_product_phi = np.concatenate([f['PHOTON/phi'][:], f['LEPTON/phi'][:]], axis=1)
            indices = np.where((event_eta[:, :, None] == decay_product_eta[:, None, :]) & (event_phi[:, :, None] == decay_product_phi[:, None, :]))
            event_mask[indices[0], indices[1]] = False

        event_pt[event_mask == False] = float('nan')
        event_eta[event_mask == False] = float('nan')
        event_phi[event_mask == False] = float('nan')

        # 這裡的 Particle Type 分別是 Tower, Track, Decay Products
        event_particle_type_0 = np.array([1] * MAX_CONSTI['Tower'] + [0] * MAX_CONSTI['Track'] + [0] * total_decay_len)
        event_particle_type_0 = np.tile(event_particle_type_0, (event_pt.shape[0], 1))
        event_particle_type_1 = np.array([0] * MAX_CONSTI['Tower'] + [1] * MAX_CONSTI['Track'] + [0] * total_decay_len)
        event_particle_type_1 = np.tile(event_particle_type_1, (event_pt.shape[0], 1))
        event_particle_type_2 = np.array([0] * MAX_CONSTI['Tower'] + [0] * MAX_CONSTI['Track'] + [1] * total_decay_len)
        event_particle_type_2 = np.tile(event_particle_type_2, (event_pt.shape[0], 1))

        features = np.stack([event_pt, event_eta, event_phi, event_particle_type_0, event_particle_type_1, event_particle_type_2], axis=-1)
    
    _part_data_cache[cache_key] = features
    return features

def create_part_test_sample(h5_dirs, nevents, seed=0, remove_decay_products=True):
    h5_dir0 = Path(h5_dirs[0])
    data_VBF_SR = prepare_feature_from_h5(h5_dir0 / 'VBF_in_SR.h5', remove_decay_products)
    data_VBF_BR = prepare_feature_from_h5(h5_dir0 / 'VBF_in_BR.h5', remove_decay_products)
    data_GGF_SR = prepare_feature_from_h5(h5_dir0 / 'GGF_in_SR.h5', remove_decay_products)
    data_GGF_BR = prepare_feature_from_h5(h5_dir0 / 'GGF_in_BR.h5', remove_decay_products)

    n_data_VBF_SR = data_VBF_SR.shape[0]
    n_data_VBF_BR = data_VBF_BR.shape[0]
    n_data_GGF_SR = data_GGF_SR.shape[0]
    n_data_GGF_BR = data_GGF_BR.shape[0]

    n_VBF_SR, n_GGF_SR, n_VBF_BR, n_GGF_BR = nevents
    n_test = 10000
    n_VBF_SR_test = int(n_data_VBF_SR / (n_data_VBF_SR + n_data_VBF_BR) * n_test)
    n_VBF_BR_test = n_test - n_VBF_SR_test
    n_GGF_SR_test = int(n_data_GGF_SR / (n_data_GGF_SR + n_data_GGF_BR) * n_test)
    n_GGF_BR_test = n_test - n_GGF_SR_test

    np.random.seed(seed)
    idx_VBF_SR = np.random.choice(n_data_VBF_SR, n_VBF_SR + n_VBF_SR_test, replace=False)
    idx_VBF_BR = np.random.choice(n_data_VBF_BR, n_VBF_BR + n_VBF_BR_test, replace=False)
    idx_GGF_SR = np.random.choice(n_data_GGF_SR, n_GGF_SR + n_GGF_SR_test, replace=False)
    idx_GGF_BR = np.random.choice(n_data_GGF_BR, n_GGF_BR + n_GGF_BR_test, replace=False)

    new_data_te = np.concatenate([
        data_VBF_SR[idx_VBF_SR[n_VBF_SR:]],
        data_VBF_BR[idx_VBF_BR[n_VBF_BR:]],
        data_GGF_SR[idx_GGF_SR[n_GGF_SR:]],
        data_GGF_BR[idx_GGF_BR[n_GGF_BR:]],
    ], axis=0)

    # 裁剪特徵維度至 402，以適應 H -> aa 模型輸入形狀
    if new_data_te.shape[1] > 402:
        new_data_te = new_data_te[:, :402, :]

    new_label_te = np.zeros(new_data_te.shape[0])
    new_label_te[:n_test] = 1
    return new_data_te, new_label_te

# 3. CSV 合併更新函數 (並行安全)
def update_summary_csv(model_name, model_type, luminosity, acc_za, auc_za, file_name='GGF_VBF_CWoLa_summary.csv'):
    max_retries = 10
    for attempt in range(max_retries):
        try:
            if os.path.isfile(file_name):
                df = pd.read_csv(file_name)
            else:
                df = pd.DataFrame(columns=[
                    'Di-photon ACC', 'Di-photon AUC', 'ZZ->4l ACC', 'ZZ->4l AUC',
                    'Za ACC', 'Za AUC', 'Luminosity (fb^-1)', 'Model type', 'Model Name'
                ])
            
            # 檢查是否已經有這一行 (Model Name 相同且 Model type 相同)
            mask = (df['Model Name'] == model_name) & (df['Model type'] == model_type)
            if mask.any():
                idx = df[mask].index[0]
                df.at[idx, 'Za ACC'] = acc_za
                df.at[idx, 'Za AUC'] = auc_za
            else:
                # 如果沒有，則新增一行
                new_row = {
                    'Di-photon ACC': np.nan,
                    'Di-photon AUC': np.nan,
                    'ZZ->4l ACC': np.nan,
                    'ZZ->4l AUC': np.nan,
                    'Za ACC': acc_za,
                    'Za AUC': auc_za,
                    'Luminosity (fb^-1)': luminosity,
                    'Model type': model_type,
                    'Model Name': model_name
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                
            df.to_csv(file_name, index=False)
            print(f"  Saved Za results for {model_name} (ACC={acc_za:.4f}, AUC={auc_za:.4f})")
            return
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  Failed to save Za results for {model_name} after {max_retries} attempts: {e}")
                raise e
            # 遭遇 file write lock 衝突時，隨機 sleep 0.1 ~ 0.5 秒後重試
            time.sleep(random.uniform(0.1, 0.5))

# 4. CNN 評估核心
def evaluate_cnn_transfer(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    seed = config['seed']
    luminosity = config['luminosity']
    cut_type = config['cut_type']
    model_name = config['model_name']
    
    # 進行名稱轉換以匹配硬碟上的真實資料夾名稱
    # 例如：quark_jet_2_cut_eventCNN_L_100_wo_a_case1_01 -> quark_jet_2_cut_eventCNN_L_100_aa_wo_product_case_5_01
    # 例如：quark_jet_2_cut_eventCNN_L_100_w_a_01 -> quark_jet_2_cut_eventCNN_L_100_aa_w_product_01
    real_model_name = model_name
    if 'wo_a_case1' in model_name:
        real_model_name = model_name.replace('wo_a_case1', 'aa_wo_product_case_5')
    elif 'w_a' in model_name:
        real_model_name = model_name.replace('w_a', 'aa_w_product')

    save_model_name = f'./CNN/CNN_models/best_model_GGF_VBF_CWoLa_{real_model_name}/'
    if not os.path.exists(save_model_name):
        save_model_name = f'./CNN/CNN_models/last_model_GGF_VBF_CWoLa_{real_model_name}/'
        
    if not os.path.exists(save_model_name):
        print(f"Model {save_model_name} (neither best nor last) not found. Skipping.")
        return

    # 檢查是否已在結果 summary 中完成評估
    summary_csv = 'GGF_VBF_CWoLa_summary.csv'
    if os.path.exists(summary_csv):
        try:
            df = pd.read_csv(summary_csv)
            mask = (df['Model Name'] == model_name) & (df['Model type'] == 'Event-CNN')
            if mask.any() and not pd.isna(df.loc[mask, 'Za AUC'].values[0]):
                print(f"  Model {model_name} already evaluated in summary. Skipping.")
                return
        except Exception as e:
            pass
        
    GGF_cutflow = './Sample/selection_cut_results/selection_results_GGF_Za_2l_quark_jet.npy'
    VBF_cutflow = './Sample/selection_cut_results/selection_results_VBF_Za_2l_quark_jet.npy'
    
    # 計算 Za 頻道在該亮度下的預期事件數
    n_VBF_SR, n_GGF_SR, n_VBF_BR, n_GGF_BR = compute_nevent_in_SR_BR_Za(GGF_cutflow, VBF_cutflow, luminosity, cut_type)
    n_events = (int(n_VBF_SR), int(n_GGF_SR), int(n_VBF_BR), int(n_GGF_BR))
    
    # 載入 Za CWoLa 數據集的原始預處理資料 (排除 decay products)
    npy_paths = ['./Sample/data/Za_2l/quark_jet_2_cut/pre-processing/remove_product_case_5/40x40/']
    
    X_test, y_test = create_cnn_test_sample(npy_paths, n_events, seed=seed)
    X_test = pt_normalization_CNN(X_test)
    
    # 預測並計算
    loaded_model = tf.keras.models.load_model(save_model_name)
    y_pred = loaded_model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    acc = get_highest_accuracy(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    
    # 釋放記憶體避免記憶體洩漏
    tf.keras.backend.clear_session()
    del loaded_model
    gc.collect()
    
    # 更新 CSV
    update_summary_csv(model_name, 'Event-CNN', luminosity, acc, auc)

# 5. ParT 評估核心
def evaluate_part_transfer(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    seed = config['seed']
    luminosity = config['luminosity']
    cut_type = config['cut_type']
    model_name = config['model_name']
    training_method = config['training_method']
    remove_decay_products = config['remove_decay_products']
    
    save_model_name = f'./Particle_transformer/ParT_models/best_model_{training_method}_{model_name}/'
    if not os.path.exists(save_model_name):
        save_model_name = f'./Particle_transformer/ParT_models/last_model_{training_method}_{model_name}/'
        
    if not os.path.exists(save_model_name):
        print(f"Model {save_model_name} (neither best nor last) not found. Skipping.")
        return

    # 檢查是否已在結果 summary 中完成評估
    summary_csv = 'GGF_VBF_CWoLa_summary.csv'
    if os.path.exists(summary_csv):
        try:
            df = pd.read_csv(summary_csv)
            mask = (df['Model Name'] == model_name) & (df['Model type'] == 'ParT')
            if mask.any() and not pd.isna(df.loc[mask, 'Za AUC'].values[0]):
                print(f"  Model {model_name} already evaluated in summary. Skipping.")
                return
        except Exception as e:
            pass
        
    GGF_cutflow = './Sample/selection_cut_results/selection_results_GGF_Za_2l_quark_jet.npy'
    VBF_cutflow = './Sample/selection_cut_results/selection_results_VBF_Za_2l_quark_jet.npy'
    
    n_VBF_SR, n_GGF_SR, n_VBF_BR, n_GGF_BR = compute_nevent_in_SR_BR_Za(GGF_cutflow, VBF_cutflow, luminosity, cut_type)
    n_events = (int(n_VBF_SR), int(n_GGF_SR), int(n_VBF_BR), int(n_GGF_BR))
    
    # 載入 Za H5 資料集原始預處理資料
    h5_dirs = ['./Sample/data/Za_2l/quark_jet_2_cut/pre-processing/']
    
    X_test, y_test = create_part_test_sample(h5_dirs, n_events, seed=seed, remove_decay_products=remove_decay_products)
    pt_normalization_ParT(X_test)
    
    # 預測並計算
    loaded_model = tf.keras.models.load_model(save_model_name)
    y_pred = loaded_model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    acc = get_highest_accuracy(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    
    # 釋放記憶體避免記憶體洩漏
    tf.keras.backend.clear_session()
    del loaded_model
    gc.collect()
    
    # 更新 CSV
    update_summary_csv(model_name, 'ParT', luminosity, acc, auc)

# 6. 主迴圈遍歷
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard-id', type=int, default=0, help='Shard ID for data parallelism')
    parser.add_argument('--num-shards', type=int, default=1, help='Total number of shards')
    args = parser.parse_args()

    print(f"=================== STARTING TRANSFER EVALUATION TO Za (Shard {args.shard_id}/{args.num_shards}) ===================")
    
    # 6a. 遍歷 CNN 的 H -> aa configs
    cnn_config_dir = './CNN/config_files'
    if os.path.exists(cnn_config_dir):
        # 篩選出原先 H -> aa 頻道的 CWoLa 訓練 config 檔 (檔名不含 Za，且屬於 CWoLa)
        all_cnn_configs = sorted([
            f for f in os.listdir(cnn_config_dir) 
            if f.endswith('.json') and 'remove_photon_case1' in f and 'Za' not in f
        ])
        
        # 進行分片
        cnn_configs = [c for i, c in enumerate(all_cnn_configs) if i % args.num_shards == args.shard_id]
        
        print(f"Found {len(cnn_configs)} (out of {len(all_cnn_configs)}) CNN configs for evaluation.")
        for idx, cfg in enumerate(cnn_configs, 1):
            cfg_path = os.path.join(cnn_config_dir, cfg)
            print(f"[{idx}/{len(cnn_configs)}] Evaluating CNN: {cfg}...")
            try:
                evaluate_cnn_transfer(cfg_path)
            except Exception as e:
                print(f"  Error evaluating {cfg}: {e}")
                
    # 6b. 遍歷 ParT 的 H -> aa configs
    part_config_dir = './Particle_transformer/config_files'
    if os.path.exists(part_config_dir):
        # 篩選出原先 H -> aa 頻道的 CWoLa 訓練 config 檔 (檔名不含 Za，且屬於 CWoLa)
        all_part_configs = sorted([
            f for f in os.listdir(part_config_dir)
            if f.endswith('.json') and 'remove_decay_products' in f and 'Za' not in f and 'CWoLa_ParT_L' in f
        ])
        
        # 進行分片
        part_configs = [c for i, c in enumerate(all_part_configs) if i % args.num_shards == args.shard_id]
        
        print(f"\nFound {len(part_configs)} (out of {len(all_part_configs)}) ParT configs for evaluation.")
        for idx, cfg in enumerate(part_configs, 1):
            cfg_path = os.path.join(part_config_dir, cfg)
            print(f"[{idx}/{len(part_configs)}] Evaluating ParT: {cfg}...")
            try:
                evaluate_part_transfer(cfg_path)
            except Exception as e:
                print(f"  Error evaluating {cfg}: {e}")
                
    print("\n=================== TRANSFER EVALUATION COMPLETED ===================")

if __name__ == '__main__':
    main()
