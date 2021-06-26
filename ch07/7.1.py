""" 7.1 - アンサンブルによる学習 """

# 確率質量関数を実装する
# - 離散的な確率変数が特定の値を取る確率を表す関数

from os import error
from scipy.special import comb
import math

def emsemble_error(n_classifier, error):
    k_start = int(math.ceil(n_classifier/2.))
    probs = [comb(n_classifier, k) * error**k * (1 - error)**(n_classifier - k) 
             for k in range(k_start, n_classifier + 1)]
    return sum(probs)

import numpy as np
import matplotlib.pyplot as plt

error_range = np.arange(0.0, 1.01, 0.01)
ens_errors = [emsemble_error(n_classifier=11, error=error) for error in error_range]

plt.plot(error_range, ens_errors, label='Emsemble error', linewidth=2)
plt.plot(error_range, error_range, label='Base error', linewidth=2, linestyle='--')

# ベース分類器の誤分類率が0.5より低い場合は常にアンサンブル法の誤分類率のほうがひくい
plt.xlabel('Base Error')
plt.ylabel('Base/Emsemble Error')
plt.legend(loc='upper left')
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()