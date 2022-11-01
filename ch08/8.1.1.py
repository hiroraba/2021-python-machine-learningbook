""" 8.1.1 - 映画レビューデータセットを取得する """

import tarfile
import pandas as pd

# tarファイルを展開
with tarfile.open('aclImdb_v1.tar.gz', 'r:gz') as tar:
    def is_within_directory(directory, target):
        
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
    
        prefix = os.path.commonprefix([abs_directory, abs_target])
        
        return prefix == abs_directory
    
    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
    
        for member in tar.getmembers():
            member_path = os.path.join(path, member.name)
            if not is_within_directory(path, member_path):
                raise Exception("Attempted Path Traversal in Tar File")
    
        tar.extractall(path, members, numeric_owner=numeric_owner) 
        
    
    safe_extract(tar)

# 映画レビューデータセットを便利なフォーマットに変換する
import pyprind
import os

basepath = 'aclImdb'
labels = {'pos':1, 'neg':0}
pbar = pyprind.ProgBar(50000)
df = pd.DataFrame()

# 訓練データとテストデータ
for s in ('test', 'train'):
    # ポジネガ
    for l in ('pos', 'neg'):
        path = os.path.join(basepath, s, l)
        # フォルダ内のファイルを1つずつ読み込む
        for file in sorted(os.listdir(path)):
            with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
                txt = infile.read()
                # テキストと、ラベル（ポジ or ネガ）をdataframeに追加する
                df = df.append([[txt, labels[l]]], ignore_index=True)
                pbar.update()

df.columns = ['review', 'sentiment']

# dataframeはクラスラベルでソートされているため、順番をシャッフルしてcsvに出力する
import numpy as np
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('movie_data.csv', index=False, encoding='utf-8')