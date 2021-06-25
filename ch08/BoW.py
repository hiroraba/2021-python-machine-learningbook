""" 8.2.1 - 単語を特徴量ベクトルに変換する """

import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

count = CountVectorizer()
docs = np.array(['The sun is shining',
                       'The weather is sweet',
                       'The sun is shining, The weather is sweet, and one and one is two'])
bag = count.fit_transform(docs)
print(count.vocabulary_)

""" 8.2.2 TF-IDFを使って単語の関連性を評価する """
# 特徴量ベクトルに頻繁に出てくる単語の重みを減らす
from sklearn.feature_extraction.text import TfidfTransformer

# TfidfTransformer:
# smooth_idf: すべての文書に出現する単語の重みを0にする
# norm: 正則化の方法（今回はl2）
tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
np.set_printoptions(precision=2)
print(tfidf.fit_transform(bag).toarray())

""" 8.2.3 - テキストデータのクレンジング """
# 正規表現で不要な文字列を取り除く
import re
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text

import pandas as pd

# レビューに適用させる
df = pd.read_csv('movie_data.csv', encoding='utf-8')
df.columns = ['review', 'sentiment']
df['review'] = df['review'].apply(preprocessor)


""" 8.2.4 文書をトークン化する """

def tokenizer(text):
    return text.split()

# スペースで区切って配列化する
print(tokenizer('runners like running and thus they run'))

# ワードステミング: 関連する単語を同じ語幹にマッピングする
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]
print(tokenizer_porter('runners like running and thus they run'))

# ストップワード（学習に有益ではない単語）を除去する
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
stop = stopwords.words('english')
[w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:] if w not in stop]

""" 8.2.5 - 文書を分類するロジスティック回帰モデルの訓練 """

# 訓練用データ（前半25000個）
X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values

# テスト用データ（後半25000個）
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

# GridSearchCVを使ってロジスティック回帰モデルの最適なパラメータ集合を求める
# 5分割交差検証する
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# CountVectoizerとTfidftransformerを組み合わせたTfidfVectorizerを利用
tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
param_grid = [{'vect__ngram_range': [(1,1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer_porter, tokenizer],
               'clf__penalty': ['l2', 'l1'],
               'clf__C':[1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1,1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer_porter, tokenizer],
               'vect__use_idf': [None],
               'vect__norm': [None],
               'clf__penalty': ['l2', 'l1'],
               'clf__C':[1.0, 10.0, 100.0]}]

lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(random_state=0, solver='liblinear'))])
gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid=param_grid, scoring='accuracy', cv=5, verbose=2, n_jobs=1)
gs_lr_tfidf.fit(X_train, y_train)

print('Best Parameter set: %s' % gs_lr_tfidf.best_params_)
print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)

clf = gs_lr_tfidf.best_estimator_

# ベストスコアを出した分類器でテストデータを予測
print('Test Accuracy: %.3f' % clf.score(X_test, y_test))