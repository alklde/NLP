import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocesser import preprocess_corpus
from train import train_word2vec, prepare_sentences

def bow_w2v_compare(sentences, w2v_model, corpus_name):
    """
    对比BOW模型和Word2Vec模型
    """
    texts = [' '.join(sent) for sent in sentences[:100]]
    bow_vectorizer = CountVectorizer()
    bow_matrix = bow_vectorizer.fit_transform(texts)

    print(f"{corpus_name} BOW模型：")
    print(f"  - 矩阵维度：{bow_matrix.shape}（{bow_matrix.shape[0]}个句子 × {bow_matrix.shape[1]}个词）")
    print(f"  - 词汇表大小：{len(bow_vectorizer.vocabulary_)}")
    
    print(f"\n{corpus_name} Word2Vec模型：")
    print(f"  - 词向量维度：{w2v_model.wv.vector_size}")
    print(f"  - 词汇表大小：{len(w2v_model.wv.index_to_key)}")
    
    # 句子相似度
    if len(sentences) >= 2:
        sent1 = sentences[0]
        sent2 = sentences[1]
        bow_sim = cosine_similarity(bow_matrix[0:1], bow_matrix[1:2])[0][0]
        
        def get_sentence_vector(sent, model):
            """
            计算句子的向量
            """
            vec = [model.wv[word] for word in sent if word in model.wv.key_to_index]
            return np.mean(vec, axis=0) if vec else np.zeros(model.wv.vector_size)
        
        sent1_vec = get_sentence_vector(sent1, w2v_model)
        sent2_vec = get_sentence_vector(sent2, w2v_model)
        w2v_sim = cosine_similarity([sent1_vec], [sent2_vec])[0][0]
        
        print(f"\n 句子相似度对比：")
        print(f" BOW模型相似度：{bow_sim:.4f}")
        print(f" Word2Vec模型相似度：{w2v_sim:.4f}")


if __name__ == "__main__":
    # 加载语料
    corpus = preprocess_corpus("./data/Ci.txt")

    # 训练Word2Vec模型
    seg_lines = corpus[1]
    words = [word for line in seg_lines for word in line]
    sentences = prepare_sentences(words)
    
    # 训练Word2Vec模型
    w2v_model = train_word2vec(sentences)
    
    # 对比BOW模型和Word2Vec模型
    bow_w2v_compare(sentences, w2v_model, "宋词")
    