#!/usr/bin/env python
# coding: utf-8
# python pixelation.py <h5_path> <output_path> <resolution>
# python pixelation.py ./data/mjj_cut/pre-processing/GGF_in_BR.h5 ./data/mjj_cut/pre-processing/40x40/GGF_in_BR.npy 40

import os
import sys
import h5py

import numpy as np


def pixelization(pts, etas, phis, res=75):
    # pixelate jet constituents
    # res: resolution of the image

    nevent = pts.shape[0]

    # 計算 bin 的邊界
    bins_eta = np.linspace(-5.0, 5.0, res + 1)
    bins_phi = np.linspace(-np.pi, np.pi, res + 1)

    # 計算每個數據點在直方圖中的位置
    # shape: (nevent, MAX_JETS)
    bin_idx_eta = np.digitize(etas, bins_eta) - 1
    bin_idx_phi = np.digitize(phis, bins_phi) - 1

    # 計算每個 bin 的權重總和
    hpT = np.zeros((nevent, res + 1, res + 1))
    np.add.at(hpT, (np.arange(nevent)[:, None], bin_idx_eta, bin_idx_phi), pts)

    hpT = hpT[:, :res, :res]

    return hpT


def from_h5_to_npy(h5_path, output_path, res=75):
    # Generate the event image from h5 file and save it to npy file
    # res: resolution of the jet image
    with h5py.File(h5_path, 'r') as f:

        print('Computing the histogram')
        hpT0 = pixelization(f['TOWER/pt'], f['TOWER/eta'], f['TOWER/phi'], res)
        hpT1 = pixelization(f['TRACK/pt'], f['TRACK/eta'], f['TRACK/phi'], res)
        hpT2 = pixelization(f['PHOTON/pt'], f['PHOTON/eta'], f['PHOTON/phi'], res)

        # 將結果堆疊起來
        # data shpae: (nevent, res, res, 3)
        # label shape: (nevent,)
        data = np.stack([hpT0, hpT1, hpT2], axis=-1)
        label = f['EVENT/type'][:]

    root, _ = os.path.splitext(output_path)

    print(f'Saving data to {root}-data.npy')
    np.save(f'{root}-data.npy', data)
    np.save(f'{root}-label.npy', label)


def main():

    h5_path = sys.argv[1]
    output_path = sys.argv[2]
    res = int(sys.argv[3])

    from_h5_to_npy(h5_path, output_path, res)


if __name__ == '__main__':
    main()