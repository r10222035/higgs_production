import os
import h5py
import shutil
import random

import numpy as np

def get_dataset_keys(f):
    # 取得所有 Dataset 的名稱
    keys = []
    f.visit(lambda key: keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
    return keys


def print_h5_info(file, event=0):
    # 印出所有 Dataset
    print(file)
    with h5py.File(file, 'r') as f:
        dataset_keys = get_dataset_keys(f)
        print('Dataset size:', f[dataset_keys[0]].shape[0])
        for key in dataset_keys:
            print(key, end=' ')
            print(f[key][event])


def split_h5_size(main_file, size=1000):
    # 將輸入的 HDF5 檔案以 size 分成兩個
    root, ext = os.path.splitext(main_file)
    split_file1 = root + '_split1' + ext
    split_file2 = root + '_split2' + ext

    with h5py.File(main_file, 'r') as f_main:
        dataset_keys = get_dataset_keys(f_main)
        key0 = dataset_keys[0]
        total_size = f_main[key0].shape[0]
        print(f'Size of {main_file}: {total_size}')
        
        if size > total_size:
            print(f'Split size {size} is greater than the input file size {total_size}.')
            
        with h5py.File(split_file1, 'w') as f_sp1, h5py.File(split_file2, 'w') as f_sp2:    
            sp_size = size
            for key in dataset_keys:
                maxShape = list(f_main[key].maxshape)
                maxShape[0] = None
                f_sp1.create_dataset(key, maxshape=maxShape, data=f_main[key][:sp_size])
                f_sp2.create_dataset(key, maxshape=maxShape, data=f_main[key][sp_size:])
            
            print(f'Size of {split_file1}: {f_sp1[key0].shape[0]}')
            print(f'Size of {split_file2}: {f_sp2[key0].shape[0]}')

            
def split_h5_file(main_file, r=0.9):
    # 將輸入的 HDF5 檔案以 r 的比例分成兩個
    root, ext = os.path.splitext(main_file)
    split_file1 = root + '_split1' + ext
    split_file2 = root + '_split2' + ext

    with h5py.File(main_file, 'r') as f_main:
        dataset_keys = get_dataset_keys(f_main)
        key0 = dataset_keys[0]
        total_size = f_main[key0].shape[0]
        print(f'Size of {main_file}: {total_size}')
        
        size = int(total_size * r)
        if size > total_size:
            print(f'Split size {size} is greater than the input file size {total_size}.')

        with h5py.File(split_file1, 'w') as f_sp1, h5py.File(split_file2, 'w') as f_sp2:    
            sp_size = size
            for key in dataset_keys:
                maxShape = list(f_main[key].maxshape)
                maxShape[0] = None
                f_sp1.create_dataset(key, maxshape=maxShape, data=f_main[key][:sp_size])
                f_sp2.create_dataset(key, maxshape=maxShape, data=f_main[key][sp_size:])
                    
            print(f'Size of {split_file1}: {f_sp1[key0].shape[0]}')
            print(f'Size of {split_file2}: {f_sp2[key0].shape[0]}')
    

def merge_h5_file(main_file, *arg):
    # 合併傳入的 HDF5 檔案    
    
    # 檢查傳入檔案結構是否都相同
    same_structure = True
    with h5py.File(main_file, 'r') as f_main:
        main_dataset_keys = get_dataset_keys(f_main)
        for append_file in arg:
            with h5py.File(append_file, 'r') as f_append:
                append_dataset_keys = get_dataset_keys(f_append)    
                if set(main_dataset_keys) != set(append_dataset_keys):
                    same_structure = False
                    print(f"'{main_file}' and '{append_file}' are not same structure, can not be merged.")
                    break

    # 檢查檔案結構是否都相同
    if not same_structure:
        return
    print(f"'{main_file}' and {arg} are same structure, can be merged.")

    root, ext = os.path.splitext(main_file)
    new_file = root + '_merged' + ext

    # 檢查合併檔案是否存在
    if os.path.isfile(new_file):
        print(f'{new_file} exist. Can not copy {main_file} to {new_file}')
        return

    print(f'{new_file} not exist. Copy {main_file} to {new_file}')
    shutil.copyfile(main_file, new_file)

    with h5py.File(new_file, 'a') as f_main:
        key0 = main_dataset_keys[0]
        total_size = f_main[key0].shape[0]
        print(f'Size of {main_file}: {total_size}')

        for append_file in arg:
            with h5py.File(append_file, 'r') as f_append:
                size_of_append = f_append[key0].shape[0]            
                print(f'Size of {append_file}: {size_of_append}')

                total_size += size_of_append   
                for key in main_dataset_keys:
                    f_main[key].resize(total_size, axis=0)
                    f_main[key][-size_of_append:] = f_append[key]
                
                print(f'Size of {new_file}: {f_main[key0].shape[0]}')
    return new_file


def shuffle_h5(file_path):
    with h5py.File(file_path, 'a') as f:
        dataset_keys = get_dataset_keys(f)
        nevent = f[dataset_keys[0]].shape[0]
        print(f'Dataset size: {nevent}')
        
        ind_list = list(range(nevent))
        random.shuffle(ind_list)
        for key in dataset_keys:      
            f[key][...] = np.array(f[key])[ind_list]


def get_particle_mask(quark_jet, particle_quark_idx):
    # 粒子夸克：由特定粒子篩變而成的夸克
    # quark_jet: 每個夸克對應的 jet 編號，shape 為 (n_events, n_quarks)
    # particle_quark_idx: 粒子夸克的位置編號，shape 為 (n_particle_quarks,)

    # particle_jets: 粒子夸克對應的 jet 編號，shape 為 (n_events, n_particle_quarks)
    particle_jets = quark_jet[:, particle_quark_idx]

    # 檢查是否每個粒子夸克都有對應的 jet
    mask1 = np.all(particle_jets != -1, axis=1)

    # 對每一個事件，檢查每個粒子夸克對應的 jet 都不重複
    count = np.sum(quark_jet[:, np.newaxis, :] == particle_jets[:, :, np.newaxis], axis=2)
    mask2 = np.all(count == 1, axis=1)

    return mask1 & mask2