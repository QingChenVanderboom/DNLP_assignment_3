# -*- coding: utf-8 -*-
import os
import warnings
import random
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

warnings.filterwarnings("ignore")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负数显示问题

# 参数设置
data_dir = "data"
stopwords_file = "cn_stopwords.txt"
book_titles_list = [
    "倚天屠龙记", "神雕侠侣", "射雕英雄传", "天龙八部", "笑傲江湖",
    "鹿鼎记", "连城诀", "白马啸西风", "碧血剑",
    "飞狐外传", "雪山飞狐", "侠客行", "鸳鸯刀",
    "越女剑", "书剑恩仇录", "三十三剑客图"
]

# 加载停用词
with open(stopwords_file, encoding='utf-8') as f:
    stopwords = set([line.strip() for line in f])

# 构建语料库
corpus = []
for book_title in book_titles_list:
    book_path = os.path.join(data_dir, f"{book_title}.txt")
    try:
        with open(book_path, encoding='utf-8') as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(book_path, encoding='gb18030') as f:
            text = f.read()
    corpus.append(text)

# 分词与预处理
def preprocess(text):
    tokens = jieba.lcut(text)
    return [token for token in tokens if token not in stopwords and token.strip()]

corpus_tokens = [preprocess(text) for text in corpus]

# 随机选取两个段落
def extract_paragraphs(corpus, num_paragraphs=2):
    paragraphs = []
    for text in corpus:
        tokens = preprocess(text)
        paragraphs.extend([' '.join(tokens[i:i + 100]) for i in range(0, len(tokens), 100)])
    return random.sample(paragraphs, num_paragraphs)

paragraph1, paragraph2 = extract_paragraphs(corpus)

# 输出段落内容
print("段落1: ")
print(paragraph1)
print("\n段落2: ")
print(paragraph2)

### 模型训练

# 训练Word2Vec模型
word2vec_model = Word2Vec(sentences=corpus_tokens, vector_size=100, window=5, min_count=1, sg=0)
word2vec_model.save("word2vec.model")

### 模型验证

def calculate_similarity(model, word1, word2):
    if word1 in model.wv and word2 in model.wv:
        vector_1 = model.wv[word1]
        vector_2 = model.wv[word2]
        similarity = cosine_similarity([vector_1], [vector_2])[0][0]
        return similarity
    else:
        return None

def get_average_vector(model, text):
    tokens = preprocess(text)
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# 验证Word2Vec模型
print("验证Word2Vec模型")
word1, word2 = "张无忌", "赵敏"
similarity = calculate_similarity(word2vec_model, word1, word2)
if similarity is not None:
    print(f"'{word1}' 和 '{word2}' 的相似度: {similarity}")
else:
    print(f"词 '{word1}' 或 '{word2}' 不在词汇表中")

all_words = list(word2vec_model.wv.index_to_key)
all_vectors = word2vec_model.wv[all_words]
kmeans = KMeans(n_clusters=10, random_state=0).fit(all_vectors)
labels = kmeans.labels_

plt.figure(figsize=(10, 5))
plt.scatter(all_vectors[:, 0], all_vectors[:, 1], c=labels)
plt.title("Word2Vec 词向量聚类")
plt.savefig('word2vec_clustering.png')

vector1 = get_average_vector(word2vec_model, paragraph1)
vector2 = get_average_vector(word2vec_model, paragraph2)
similarity = cosine_similarity([vector1], [vector2])[0][0]
print(f"段落1和段落2的相似度: {similarity}")

# 展示图形
plt.show()
