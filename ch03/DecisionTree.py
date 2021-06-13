""" 3.6 - 決定木学習 """

import matplotlib.pyplot as plt
import numpy as np

# ジニ不純度
def gini(p):
    return (p)*(1-(p)) + (1 - p)*(1 - (1 - p))

# エントロピー
def entropy(p):
    return - p*np.log2(p) - (1 - p)*np.log2((1 - p))

# 分類誤差
def error(p):
    return 1 - np.max([p, 1 - p])

# 確率を表す配列
x = np.arange(0, 1, 0.01)

# エントロピー
ent = [entropy(p) if p != 0 else None for p in x]
# エントロピーを2で割ったもの
sc_ent = [e*0.5 if e else None for e in ent]
# 分類誤差
err = [error(i) for i in x]

fig = plt.figure()
ax = plt.subplot(111)

for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err], ['Entropy', 'Entorpy(1/2)', 'gini', 'miss classifier'],['-','-','--','-.'],['black','lightGray','red','green','cyan']):
    line = ax.plot(x,i,label=lab,linestyle=ls,lw=2,color=c)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol = 5)
ax.axhline(y=0.5, linewidth = 1, color='k', linestyle='--')
ax.axhline(y=1, linewidth = 1, color='k', linestyle='--')
plt.show()