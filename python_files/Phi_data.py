import glob
import numpy as np
import pathlib
import sys
import os
import pandas as pd
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
#import generate_data as gd

# 現在のパスの取得
cwd = os.getcwd()
p_file = pathlib.Path(cwd)
sys.path.append(str(p_file) + "/python_files")
import generate_data as gd

warnings.filterwarnings('ignore')

sns.set()
sns.set_style("white")
# fileに入っているデータを読み込む

scale = 3
tpast_list = [200, 400, 600]

def dataset(file):
    new = []
    for f in file:
        try:
            X = np.loadtxt(f, delimiter=",")
        except UserWarning as warning:
            pass
        if len(X) != 0:
            new.append(X)
        else:
            new.append(0)
    return new

def data_unite(kind="PHI", s2="dshita"):
    # 同ディレクトリのファイルからファイル名を全て収得
    cwd = os.getcwd()
    #p_file = pathlib.Path(cwd)
    title = cwd + '/T_max_' + str(400) + '/'

    # 　ファイル名の読み込み
    filelist = glob.glob(title + '*')
    if s2 == "dshita":
        if kind == "PHI":
            name = title + 'MC_PHI_dtheta_'
        elif kind == "Complex":
            name = title + 'MC_dtheta_'
    elif s2 == "speed":
        if kind == "PHI":
            name = title + 'MC_PHI_dspeed_'
        elif kind == "Complex":
            name = title + 'MC_dspeed_'


    new_list = [s for s in filelist if s.startswith(name)]
    new_list.sort()

    return dataset(new_list)

def complex_binary(x):
    # 10進数xを10桁の2進数を返すプログラム
    return np.array([int(i) for i in format(int(x), 'b').zfill(10)])[::-1]

def transform(x, index, time):
    N = 10
    main_comp = np.zeros((time, N), dtype=int)
    for i in range(time):
        main_comp[i, :] = complex_binary(x[index][i])
    return main_comp


def data_for_IIT(index):
    X_dir_phi = data_unite(kind="PHI", s2="dshita")
    data = data_unite(kind="Complex", s2="dshita")
    time = data[index].shape[0]
    main_comp_dir = transform(data, index, time)

    X_dir_sp = data_unite(kind="PHI", s2="speed")
    data = data_unite(kind="Complex", s2="speed")
    time = data[index].shape[0]
    main_comp_sp = transform(data, index, time)

    return X_dir_phi[index], main_comp_dir, X_dir_sp[index], main_comp_sp

def data_pair_generate(index):
    school = gd.Schools(10, index, 1)
    [PHI_dir, Main_complex_dir, PHI_sp, Main_complex_sp] = data_for_IIT(index)
    return [PHI_dir, Main_complex_dir, PHI_sp, Main_complex_sp], school



if __name__ == '__main__':
    vari = ["dshita", "speed"]
    output = ["PHI", "Complex"]
    #X = data_unite(kind="Complex", s2="dshita")
    X = data_pair_generate(0)
