""" 8.3 - 大規模なデータの処理: オンラインアルゴリズムとアウトオブコア学習 """
import numpy as np
import re
from nltk.corpus import stopwords
from sklearn import preprocessing
stop = stopwords.words('english')

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv) # ヘッダーを読み飛ばす
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label

def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y

# 逐次的な学習にCountVectorやTFIDFは使えないので、HashingVectorを使う
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

vect = HashingVectorizer(decode_error='ignore', 
                         n_features=2**21,
                         preprocessor=None,
                         tokenizer=tokenizer)
# 損失引数をlogに設定することで、ロジスティック回帰分類器を初期化
clf = SGDClassifier(loss='log', random_state=1)

# csvのストリームを生成
doc_stream = stream_docs(path='movie_data.csv')

# 一行ずつ読み込みながら逐次的にモデルを学習する
import pyprind
pbar = pyprind.ProgBar(45)

classes = np.array([0, 1])

for _ in range(45):
    # 1000件ずつ読み込む
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes)
    pbar.update()

# 残りの5000個を使って、モデルの性能を評価する
X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)
print('Accuracy: %.3f' % clf.score(X_test, y_test))