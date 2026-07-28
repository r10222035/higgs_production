#!/usr/bin/env python
# coding: utf-8

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

ROOT = Path("/home/r10222035/NTUHEPML-CWoLa")
INFERENCE_DIR = ROOT / "output" / "inference"
FIGURES_DIR = Path("./figures/paper")
os.makedirs(FIGURES_DIR, exist_ok=True)

def load_inference_results(sub_dir_name):
    dir_path = INFERENCE_DIR / sub_dir_name
    if not dir_path.exists():
        return pd.DataFrame()
    dfs = []
    for csv_file in dir_path.glob("*.csv"):
        df = pd.read_csv(csv_file)
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    res = pd.concat(dfs, ignore_index=True)
    return res

df_trans_zz = load_inference_results("transfer_from_diphoton_to_zz4l")
df_trans_za = load_inference_results("transfer_from_diphoton_to_za2l")
df_direct_zz = load_inference_results("direct_zz4l")
df_direct_za = load_inference_results("direct_za2l")

try:
    df_cnn_direct = pd.read_csv('./CNN/GGF_VBF_training_results.csv')
    df_part_direct = pd.read_csv('./Particle_transformer/GGF_VBF_training_results.csv')
    df_summary = pd.read_csv('./GGF_VBF_CWoLa_summary.csv')
except Exception:
    df_cnn_direct = pd.DataFrame()
    df_part_direct = pd.DataFrame()
    df_summary = pd.DataFrame()

def get_auc_summary(df, model_name, num_rot_aug, channel_title=None, is_direct=False):
    lumi_list = [100, 300, 900, 1800, 3000]
    means, ci95s = [], []
    target_aug = str(num_rot_aug).replace("+", "").strip()
    
    if df is not None and not df.empty and "model" in df.columns:
        sub_df = df[(df["model"] == model_name) & (df["num_rot_aug"].astype(str).str.replace("+", "", regex=False).str.strip() == target_aug)]
        for l in lumi_list:
            vals = sub_df[sub_df["luminosity"] == l]["test_auc"].dropna()
            if len(vals) == 0:
                means.append(None)
                ci95s.append(None)
            else:
                n = len(vals)
                std = vals.std()
                sem = std / np.sqrt(n) if n > 1 else 0
                means.append(vals.mean())
                ci95s.append(1.96 * sem)
        if any(x is not None for x in means):
            return means, ci95s

    if is_direct and "CNN" in model_name and channel_title in ["Za", "Za2l"]:
        df_csv = df_cnn_direct
        aug_suffix = "" if target_aug in ["0", ""] else f", phi shifting: +{target_aug}"
        for l in lumi_list:
            st = f"quark jet: = 2, L: {l} fb^-1, Za2l, w/o decay product{aug_suffix}"
            vals = df_csv[df_csv['Sample Type'] == st]['AUC-true'].dropna() if not df_csv.empty else []
            if len(vals) == 0:
                means.append(None)
                ci95s.append(None)
            else:
                n = len(vals)
                std = vals.std()
                sem = std / np.sqrt(n) if n > 1 else 0
                means.append(vals.mean())
                ci95s.append(1.96 * sem)
        return means, ci95s

    return [None]*5, [None]*5

def plot_comparison_chart(
    model_name,
    channel_title,
    df_transfer,
    df_direct,
    save_filename,
    ylim=(0.46, 0.77),
    title_suffix=""
):
    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(5, 4))
    
    x_vals = np.array([100, 300, 900, 1800, 3000])
    x_labels = ["100", "300", "900", "1800", "3000"]
    
    colors = {"+0": "#4C72B0", "+5": "#DD8452", "+10": "#55A868"}
    markers = {"+0": "o", "+5": "X", "+10": "s"}
    
    def draw_curve(mean, std, label, color, linestyle, marker):
        valid_indices = [i for i, m in enumerate(mean) if m is not None]
        if not valid_indices:
            return
        valid_x = x_vals[valid_indices]
        valid_mean = np.array([mean[i] for i in valid_indices])
        valid_std = np.array([std[i] for i in valid_indices])
        
        ax.plot(valid_x, valid_mean, linestyle=linestyle, marker=marker, markersize=6, color=color,
                label=label, markeredgecolor="w", markeredgewidth=1 if marker in ["o", "X", "s"] else 0)
        if len(valid_x) > 1:
            ax.fill_between(valid_x, valid_mean - valid_std, valid_mean + valid_std, color=color, alpha=0.15)
            
    has_transfer_data = False
    for aug in ["+0", "+5", "+10"]:
        m, s = get_auc_summary(df_transfer, model_name, aug, channel_title=channel_title, is_direct=False)
        if any(x is not None for x in m):
            has_transfer_data = True
        draw_curve(m, s, f"Transfer {aug}", colors[aug], "-", markers[aug])
        
    has_direct_data = False
    for aug in ["+0", "+5", "+10"]:
        m, s = get_auc_summary(df_direct, model_name, aug, channel_title=channel_title, is_direct=True)
        if any(x is not None for x in m):
            has_direct_data = True
        draw_curve(m, s, f"Direct {aug}", colors[aug], "--", markers[aug])
            
    ax.set_xscale("log")
    ax.set_xlim(80, 3800)
    ax.set_ylim(ylim)
    ax.set_xticks(x_vals)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Luminosity [fb$^{-1}$]", fontsize=12)
    ax.set_ylabel("AUC", fontsize=12)
    
    h_orig = mlines.Line2D([], [], color=colors["+0"], marker="o", linestyle="-", label="Original", markersize=6, markeredgecolor="w", markeredgewidth=1)
    h_aug5 = mlines.Line2D([], [], color=colors["+5"], marker="X", linestyle="-", label="Augment +5", markersize=6, markeredgecolor="w", markeredgewidth=1)
    h_aug10 = mlines.Line2D([], [], color=colors["+10"], marker="s", linestyle="-", label="Augment +10", markersize=6, markeredgecolor="w", markeredgewidth=1)
    h_transfer = mlines.Line2D([], [], color="gray", linestyle="-", label="Transfer")
    h_direct = mlines.Line2D([], [], color="gray", linestyle="--", label="Direct")
    
    handles = [h_orig, h_transfer, h_aug5, h_direct, h_aug10] if has_direct_data else [h_orig, h_aug5, h_aug10, h_transfer]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8.5, ncol=2 if not has_direct_data else 3, columnspacing=0.8, handletextpad=0.3)
    
    plt.tight_layout()
    pdf_path = FIGURES_DIR / save_filename
    png_path = FIGURES_DIR / save_filename.replace(".pdf", ".png")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.savefig(png_path, format="png", bbox_inches="tight")
    print(f"Saved figures: {pdf_path.name}, {png_path.name}")
    plt.close(fig)

if __name__ == '__main__':
    plot_comparison_chart("CNN_EventCNN", "ZZ", df_trans_zz, df_direct_zz, "AUC_comparison_CNN_ZZ.pdf")
    plot_comparison_chart("CNN_EventCNN", "Za", df_trans_za, df_direct_za, "AUC_comparison_CNN_Za.pdf")
    plot_comparison_chart("ParT_Light", "ZZ", df_trans_zz, df_direct_zz, "AUC_comparison_ParT_ZZ.pdf")
    plot_comparison_chart("ParT_Light", "Za", df_trans_za, df_direct_za, "AUC_comparison_ParT_Za.pdf")
