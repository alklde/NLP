import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
from preprocesser import preprocess_corpus
from utils import *
from ngram import *

def prepare_sentences(words, window_size=20):
    """
    将单词列表转换为句子列表，每个句子包含window_size个单词
    """
    sentences = []

    for i in range(0, len(words), window_size):
        sentence = words[i:i+window_size]
        # 过滤掉长度为1的句子
        if len(sentence) > 1:
            sentences.append(sentence)
    return sentences

def train_word2vec(sentences, vector_size=100, window=5, min_count=5, workers=4):
    """
    训练Word2Vec模型
    """
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=workers, sg=1)
    model.save("word2vec.model")
    return model

def visualilze_word_vectors(model, top_n=50, title="词向量可视化"):
    """
    可视化Word2Vec模型的词向量
    """
    top_word = model.wv.index_to_key[:top_n]
    word_vectors = np.array([model.wv[word] for word in top_word])

    # 可视化词向量
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(word_vectors)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(12, 8))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
    
    for i, word in enumerate(top_word):
        plt.annotate(word, (pca_result[i, 0], pca_result[i, 1])) 
    plt.title(title)
    plt.show()

def analyze_word_similarity(model, target_word, top_n=10):
    """
    分析目标词的相似度
    """
    for word in target_word:
        if word in model.wv.key_to_index:
            similar_words = model.wv.most_similar(word, topn=top_n)
            print(f"与 {word} 最相似的 {top_n} 个词:")
            for sim_word, sim_score in similar_words:
                print(f"{sim_word}: {sim_score:.4f}")
        else:
            print(f"词 {word} 不在模型词汇表中")
    return similar_words

if __name__ == "__main__":
    corpus = preprocess_corpus("./data/199801.txt")

    # 训练Word2Vec模型
    seg_lines = corpus[1]
    words = [word for line in seg_lines for word in line]
    sentences = prepare_sentences(words)
    model = train_word2vec(sentences)

    # 可视化词向量
    visualilze_word_vectors(model)

    # 分析相似度
    target_word = ['新', '华', '社', '女', '士', '人']
    analyze_word_similarity(model, target_word)