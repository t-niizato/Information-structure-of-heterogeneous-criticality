import pandas as pd
import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib
import os
sns.set()

# fileに入っているデータを読み込む
def fish_dataset(file):
    new = []
    for f in file:
        new.append(pd.read_table(f, header=None))
    return new

# ファイルの種類を決定して上の関数を実行
def fish_data(number):
    # 上の階層のディレクトリからデータ読み出し
    cwd = os.getcwd()
    p_file = pathlib.Path(cwd)
    title = str(p_file) + '/fish/*'

    #　ファイル名の読み込み
    filelist = glob.glob(title)
    name = str(p_file) + '/fish/' + str(number)
    new_list = [s for s in filelist if s.startswith(name)]
    new_list.sort()

    return fish_dataset(new_list)

# 得られたリストを入れて軌跡データのみを取り出す
def trajectory(data, number):
    return np.delete(np.array(data[number]), [0, 1], axis=1)

# 軌跡データのグラフ化：確認用
def plot_trajectory(data, number):
    plt.plot(data[:, 0:number-1], data[:, number:2*number-1])
    plt.show()
