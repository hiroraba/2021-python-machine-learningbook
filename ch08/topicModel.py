""" 8.4 - 潜在ディリクレ配分によるトピックモデルの構築 """

# トピックモデル
#  - 一般的な用途: 大規模なテキストコーパスから文書を分類する
#  - 教師なし学習のクラスタリングタスクと捉えることができる

""" 8.4.1 - 潜在ディリクレ配分を使ってテキスト文書を分解する """

# 潜在ディリクレ分布(LDA)
#  - 確率分布モデルであり、様々な文書に同時に出現する単語をトピックとする 
#  - 単語の出現頻度を表す特徴ベクトル(BoW)を入力として、2つの新しい行列に分解する
#    - 文書からトピックへの行列
#    - 単語からトピックへの行列
#  - 分解された行列をかけ合わせた場合に元のBoWが再現できるように分解する

# データの読み込み
import pandas as pd
df = pd.read_csv('movie_data.csv', encoding='utf-8')

# LDAへの入力を作成するためにCountVectonizerを使う
from sklearn.feature_extraction.text import CountVectorizer

# stop_words: 学習に有益でない単語を除外
# max_df: 単語の最大文書頻度を10%にし多くの文書に出現しすぎている単語を除外する
# max_features: 特徴量の最大数。データセットの最大次元を制限するために、5000に設定
count = CountVectorizer(stop_words='english', max_df=.1, max_features=5000)
X = count.fit_transform(df['review'].values)

from sklearn.decomposition import LatentDirichletAllocation
# LatentDirichletAllocation
# n_components: トピック数
# learning_method: 'batch'を指定すると、一回のイテレーションですべての訓練データに基づいて推定する
lda = LatentDirichletAllocation(n_components=10, random_state=123, learning_method='batch')

X_topics = lda.fit_transform(X)

# 10種類のトピックごとに単語の重要度(5000個)を含んだ行列
print(lda.components_.shape) # (10, 5000)

# 10個のトピックの重要度上位5個の単語を表示

n_top_words = 5
feature_names = count.get_feature_names

for topic_idx, topic in enumerate(lda.components_):
    print("Topic %d:" % (topic_idx + 1))
    print(" ".join([feature_names[i] 
                    for i in topic.argsort() \
                        [:-n_top_words - 1:-1]]))