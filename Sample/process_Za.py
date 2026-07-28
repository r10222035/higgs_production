#!/usr/bin/env python
# coding: utf-8

import os
import sys
import h5py
import shutil
import numpy as np
from pathlib import Path

# 導入同目錄下的工具函數
from utils import phi_shift_and_flipping, std_phi, flipping_only
from pixelation import from_h5_to_npy

MAX_CONSTI = {
    'Jet': 50,
    'Tower': 250,
    'Track': 150,
    'Photon': 2,
    'Lepton': 4,
    'Photon_Za': 1,
    'Lepton_Za': 2,
}

def create_dataset_Za(f, nevent, MAX_CONSTI):
    f.create_dataset('J1/mask', (nevent, MAX_CONSTI['Jet']), maxshape=(None, MAX_CONSTI['Jet']), dtype='|b1')
    f.create_dataset('J1/pt', (nevent, MAX_CONSTI['Jet']), maxshape=(None, MAX_CONSTI['Jet']), dtype='<f4')
    f.create_dataset('J1/eta', (nevent, MAX_CONSTI['Jet']), maxshape=(None, MAX_CONSTI['Jet']), dtype='<f4')
    f.create_dataset('J1/phi', (nevent, MAX_CONSTI['Jet']), maxshape=(None, MAX_CONSTI['Jet']), dtype='<f4')
    f.create_dataset('J1/flavor', (nevent,), maxshape=(None,), dtype='<i8')

    f.create_dataset('J2/mask', (nevent, MAX_CONSTI['Jet']), maxshape=(None, MAX_CONSTI['Jet']), dtype='|b1')
    f.create_dataset('J2/pt', (nevent, MAX_CONSTI['Jet']), maxshape=(None, MAX_CONSTI['Jet']), dtype='<f4')
    f.create_dataset('J2/eta', (nevent, MAX_CONSTI['Jet']), maxshape=(None, MAX_CONSTI['Jet']), dtype='<f4')
    f.create_dataset('J2/phi', (nevent, MAX_CONSTI['Jet']), maxshape=(None, MAX_CONSTI['Jet']), dtype='<f4')
    f.create_dataset('J2/flavor', (nevent,), maxshape=(None,), dtype='<i8')

    f.create_dataset('TOWER/mask', (nevent, MAX_CONSTI['Tower']), maxshape=(None, MAX_CONSTI['Tower']), dtype='|b1')
    f.create_dataset('TOWER/pt', (nevent, MAX_CONSTI['Tower']), maxshape=(None, MAX_CONSTI['Tower']), dtype='<f4')
    f.create_dataset('TOWER/eta', (nevent, MAX_CONSTI['Tower']), maxshape=(None, MAX_CONSTI['Tower']), dtype='<f4')
    f.create_dataset('TOWER/phi', (nevent, MAX_CONSTI['Tower']), maxshape=(None, MAX_CONSTI['Tower']), dtype='<f4')

    f.create_dataset('TRACK/mask', (nevent, MAX_CONSTI['Track']), maxshape=(None, MAX_CONSTI['Track']), dtype='|b1')
    f.create_dataset('TRACK/pt', (nevent, MAX_CONSTI['Track']), maxshape=(None, MAX_CONSTI['Track']), dtype='<f4')
    f.create_dataset('TRACK/eta', (nevent, MAX_CONSTI['Track']), maxshape=(None, MAX_CONSTI['Track']), dtype='<f4')
    f.create_dataset('TRACK/phi', (nevent, MAX_CONSTI['Track']), maxshape=(None, MAX_CONSTI['Track']), dtype='<f4')

    f.create_dataset('PHOTON/pt', (nevent, MAX_CONSTI['Photon_Za']), maxshape=(None, MAX_CONSTI['Photon_Za']), dtype='<f4')
    f.create_dataset('PHOTON/eta', (nevent, MAX_CONSTI['Photon_Za']), maxshape=(None, MAX_CONSTI['Photon_Za']), dtype='<f4')
    f.create_dataset('PHOTON/phi', (nevent, MAX_CONSTI['Photon_Za']), maxshape=(None, MAX_CONSTI['Photon_Za']), dtype='<f4')

    f.create_dataset('LEPTON/pt', (nevent, MAX_CONSTI['Lepton_Za']), maxshape=(None, MAX_CONSTI['Lepton_Za']), dtype='<f4')
    f.create_dataset('LEPTON/eta', (nevent, MAX_CONSTI['Lepton_Za']), maxshape=(None, MAX_CONSTI['Lepton_Za']), dtype='<f4')
    f.create_dataset('LEPTON/phi', (nevent, MAX_CONSTI['Lepton_Za']), maxshape=(None, MAX_CONSTI['Lepton_Za']), dtype='<f4')
    f.create_dataset('LEPTON/flavor', (nevent, MAX_CONSTI['Lepton_Za']), maxshape=(None, MAX_CONSTI['Lepton_Za']), dtype='<i8')

    f.create_dataset('EVENT/mjj', (nevent,), maxshape=(None,), dtype='<f4')
    f.create_dataset('EVENT/deta', (nevent,), maxshape=(None,), dtype='<f4')
    f.create_dataset('EVENT/type', (nevent,), maxshape=(None,), dtype='<i8')

def get_dataset_keys(f):
    keys = []
    f.visit(lambda key : keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
    return keys

def split_SR_BR(h5_path, output_path, cut_type='quark_jet', cut_value=2, mode='Za_2l'):
    print(f"Splitting SR and BR for {h5_path} (cut_type={cut_type}, cut_value={cut_value})")
    with h5py.File(h5_path, 'r') as f:
        j1_flavor = f['J1/flavor'][:]
        j2_flavor = f['J2/flavor'][:]
        mjj = f['EVENT/mjj'][:]
        deta = f['EVENT/deta'][:]
        
        if cut_type == 'quark_jet':
            quark_jet = (j1_flavor < 6).astype(int) + (j2_flavor < 6).astype(int)
            SR_range = quark_jet >= cut_value
            BR_range = quark_jet < cut_value
        else:
            raise ValueError(f'cut_type {cut_type} not supported')

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        root = output_path.with_suffix('')
        SR_path = f'{root}_in_SR.h5'
        BR_path = f'{root}_in_BR.h5'

        with h5py.File(SR_path, 'w') as f_SR, h5py.File(BR_path, 'w') as f_SB:
            if mode == 'Za_2l':
                create_dataset_Za(f_SR, SR_range.sum(), MAX_CONSTI)
                create_dataset_Za(f_SB, BR_range.sum(), MAX_CONSTI)
            else:
                raise ValueError(f'mode {mode} not supported')

            keys = get_dataset_keys(f_SR)
            for key in keys:
                f_SR[key][:] = f[key][:][SR_range]
                f_SB[key][:] = f[key][:][BR_range]
                
        print(f"  SR size: {SR_range.sum()}, BR size: {BR_range.sum()}")

def to_event_image_h5_Za(h5_path, out_h5):
    out_h5 = Path(out_h5)
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(h5_path, out_h5)

    with h5py.File(out_h5, 'a') as f_out:
        event_pt = np.concatenate([f_out['TOWER/pt'][:], f_out['TRACK/pt'][:], f_out['PHOTON/pt'][:], f_out['LEPTON/pt'][:]], axis=1)
        event_eta = np.concatenate([f_out['TOWER/eta'][:], f_out['TRACK/eta'][:], f_out['PHOTON/eta'][:], f_out['LEPTON/eta'][:]], axis=1)
        event_phi = np.concatenate([f_out['TOWER/phi'][:], f_out['TRACK/phi'][:], f_out['PHOTON/phi'][:], f_out['LEPTON/phi'][:]], axis=1)

        _, new_eta, new_phi = phi_shift_and_flipping(event_pt, event_eta, event_phi)

        f_out['TOWER/eta'][:] = new_eta[:, :MAX_CONSTI['Tower']]
        f_out['TRACK/eta'][:] = new_eta[:, MAX_CONSTI['Tower']:MAX_CONSTI['Tower'] + MAX_CONSTI['Track']]
        f_out['PHOTON/eta'][:] = new_eta[:, MAX_CONSTI['Tower'] + MAX_CONSTI['Track']:MAX_CONSTI['Tower'] + MAX_CONSTI['Track'] + MAX_CONSTI['Photon_Za']]
        f_out['LEPTON/eta'][:] = new_eta[:, MAX_CONSTI['Tower'] + MAX_CONSTI['Track'] + MAX_CONSTI['Photon_Za']:]

        f_out['TOWER/phi'][:] = new_phi[:, :MAX_CONSTI['Tower']]
        f_out['TRACK/phi'][:] = new_phi[:, MAX_CONSTI['Tower']:MAX_CONSTI['Tower'] + MAX_CONSTI['Track']]
        f_out['PHOTON/phi'][:] = new_phi[:, MAX_CONSTI['Tower'] + MAX_CONSTI['Track']:MAX_CONSTI['Tower'] + MAX_CONSTI['Track'] + MAX_CONSTI['Photon_Za']]
        f_out['LEPTON/phi'][:] = new_phi[:, MAX_CONSTI['Tower'] + MAX_CONSTI['Track'] + MAX_CONSTI['Photon_Za']:]

def phi_shifting_Za(h5_path, out_h5, shift_range=np.pi):
    out_h5 = Path(out_h5)
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(h5_path, out_h5)

    with h5py.File(out_h5, 'a') as f_out:
        event_pt = np.concatenate([f_out['TOWER/pt'][:], f_out['TRACK/pt'][:], f_out['PHOTON/pt'][:], f_out['LEPTON/pt'][:]], axis=1)
        event_eta = np.concatenate([f_out['TOWER/eta'][:], f_out['TRACK/eta'][:], f_out['PHOTON/eta'][:], f_out['LEPTON/eta'][:]], axis=1)
        event_phi = np.concatenate([f_out['TOWER/phi'][:], f_out['TRACK/phi'][:], f_out['PHOTON/phi'][:], f_out['LEPTON/phi'][:]], axis=1)

        _, new_eta, new_phi = phi_shift_and_flipping(event_pt, event_eta, event_phi)

        f_out['TOWER/eta'][:] = new_eta[:, :MAX_CONSTI['Tower']]
        f_out['TRACK/eta'][:] = new_eta[:, MAX_CONSTI['Tower']:MAX_CONSTI['Tower'] + MAX_CONSTI['Track']]
        f_out['PHOTON/eta'][:] = new_eta[:, MAX_CONSTI['Tower'] + MAX_CONSTI['Track']:MAX_CONSTI['Tower'] + MAX_CONSTI['Track'] + MAX_CONSTI['Photon_Za']]
        f_out['LEPTON/eta'][:] = new_eta[:, MAX_CONSTI['Tower'] + MAX_CONSTI['Track'] + MAX_CONSTI['Photon_Za']:]

        f_out['TOWER/phi'][:] = new_phi[:, :MAX_CONSTI['Tower']]
        f_out['TRACK/phi'][:] = new_phi[:, MAX_CONSTI['Tower']:MAX_CONSTI['Tower'] + MAX_CONSTI['Track']]
        f_out['PHOTON/phi'][:] = new_phi[:, MAX_CONSTI['Tower'] + MAX_CONSTI['Track']:MAX_CONSTI['Tower'] + MAX_CONSTI['Track'] + MAX_CONSTI['Photon_Za']]
        f_out['LEPTON/phi'][:] = new_phi[:, MAX_CONSTI['Tower'] + MAX_CONSTI['Track'] + MAX_CONSTI['Photon_Za']:]

        # 全範圍隨機角度旋轉
        nevent = f_out['EVENT/type'].shape[0]
        phi_shift = np.random.uniform(-shift_range, shift_range, size=nevent)[:, None]

        f_out['TOWER/phi'][:] += phi_shift
        f_out['TRACK/phi'][:] += phi_shift
        f_out['PHOTON/phi'][:] += phi_shift
        f_out['LEPTON/phi'][:] += phi_shift

        f_out['TOWER/phi'][:] = std_phi(f_out['TOWER/phi'][:])
        f_out['TRACK/phi'][:] = std_phi(f_out['TRACK/phi'][:])
        f_out['PHOTON/phi'][:] = std_phi(f_out['PHOTON/phi'][:])
        f_out['LEPTON/phi'][:] = std_phi(f_out['LEPTON/phi'][:])

def main():
    cut_type = 'quark_jet'
    cut_value = 2
    
    base_dir = './data/Za_2l'
    
    # 1. 劃分 SR / BR
    print("\n--- Step 1: Splitting SR and BR samples ---")
    h5_path_ggf = os.path.join(base_dir, 'GGF.h5')
    output_path_ggf = os.path.join(base_dir, f'{cut_type}_{cut_value}_cut', 'GGF.h5')
    split_SR_BR(h5_path_ggf, output_path_ggf, cut_type, cut_value, mode='Za_2l')
    
    h5_path_vbf = os.path.join(base_dir, 'VBF.h5')
    output_path_vbf = os.path.join(base_dir, f'{cut_type}_{cut_value}_cut', 'VBF.h5')
    split_SR_BR(h5_path_vbf, output_path_vbf, cut_type, cut_value, mode='Za_2l')
    
    # 2. Event image 預處理 (to_event_image_h5_Za)
    print("\n--- Step 2: Creating pre-processing (original) event-image samples ---")
    cut_dir = os.path.join(base_dir, f'{cut_type}_{cut_value}_cut')
    for name in ['VBF_in_SR', 'VBF_in_BR', 'GGF_in_SR', 'GGF_in_BR']:
        h5_path = os.path.join(cut_dir, f'{name}.h5')
        out_h5 = os.path.join(cut_dir, 'pre-processing', f'{name}.h5')
        print(f"Processing: {h5_path} -> {out_h5}")
        to_event_image_h5_Za(h5_path, out_h5)
        
    # 3. Phi shifting 數據擴增 (10份隨機旋轉)
    print("\n--- Step 3: Creating phi_shifting (random rotation) augmented samples ---")
    for n in range(1, 11):
        out_dir = os.path.join(cut_dir, 'phi_shifting', f'{n:02}')
        print(f"Generating phi shifting augmentation index {n:02}...")
        for name in ['VBF_in_SR', 'VBF_in_BR', 'GGF_in_SR', 'GGF_in_BR']:
            h5_path = os.path.join(cut_dir, f'{name}.h5')
            out_h5 = os.path.join(out_dir, f'{name}.h5')
            phi_shifting_Za(h5_path, out_h5)
            
    # 4. Pixelation (轉為 40x40 .npy 檔案)
    print("\n--- Step 4: Generating npy files via Pixelation (case=5) ---")
    res = 40
    case = 5
    
    # 4a. 為 pre-processing 進行 pixelation
    h5_pre_dir = os.path.join(cut_dir, 'pre-processing')
    npy_pre_dir = os.path.join(cut_dir, 'pre-processing', 'remove_product_case_5', f'{res}x{res}')
    os.makedirs(npy_pre_dir, exist_ok=True)
    
    for name in ['VBF_in_SR', 'VBF_in_BR', 'GGF_in_SR', 'GGF_in_BR']:
        h5_path = os.path.join(h5_pre_dir, f'{name}.h5')
        npy_path = os.path.join(npy_pre_dir, f'{name}.npy')
        print(f"Pixelating pre-processing: {h5_path} -> {npy_path}")
        from_h5_to_npy(h5_path, npy_path, res, case)
        
    # 4b. 為 phi_shifting 進行 pixelation
    h5_phi_dir = os.path.join(cut_dir, 'phi_shifting')
    npy_phi_dir = os.path.join(cut_dir, 'phi_shifting', 'remove_product_case_5', f'{res}x{res}')
    
    for n in range(1, 11):
        h5_n_dir = os.path.join(h5_phi_dir, f'{n:02}')
        npy_n_dir = os.path.join(npy_phi_dir, f'{n:02}')
        os.makedirs(npy_n_dir, exist_ok=True)
        for name in ['VBF_in_SR', 'VBF_in_BR', 'GGF_in_SR', 'GGF_in_BR']:
            h5_path = os.path.join(h5_n_dir, f'{name}.h5')
            npy_path = os.path.join(npy_n_dir, f'{name}.npy')
            print(f"Pixelating phi_shifting {n:02}: {h5_path} -> {npy_path}")
            from_h5_to_npy(h5_path, npy_path, res, case)

    print("\n--- Data Preprocessing & Split completed successfully! ---")

if __name__ == '__main__':
    main()
