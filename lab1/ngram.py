from matplotlib import pyplot as plt
from collections import Counter
from utils import save_dict
from preprocesser import preprocess_corpus

def count_word_freq(words):
    """
    统计词频
    """
    word_freq = Counter(words)
    return word_freq

def build_ngram_model(words, n=2):
    """
    构建n-gram模型
    """
    ngram_model = []
    for i in range(len(words) - n + 1):
        ngram = ''.join(words[i:i+n])
        ngram_model.append(ngram)
    return Counter(ngram_model)

def plot_word_freq(word_freq, title='词频分布', top_n=20):
    """
    可视化词频分布
    """
    word, freqs = zip(*word_freq.most_common(top_n))

    # plt中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(10, 6))
    plt.bar(word, freqs)
    plt.title(title)
    plt.xlabel('词')
    plt.ylabel('频率')
    plt.xticks(rotation=45)
    plt.show()

if __name__ == "__main__":
    corpus = preprocess_corpus("./data/Ci.txt")

    # 统计词频
    seg_lines = corpus[1]
    words = [word for line in seg_lines for word in line]
    word_freq = count_word_freq(words)
    save_dict(word_freq, "word_freq.json")
    plot_word_freq(word_freq)

    # 构建N-gram模型
    ngram_model = build_ngram_model(words, n=2)
    save_dict(ngram_model, "ngram_model.json")

    # 显示前10个N-gram
    for bigram, freq in ngram_model.most_common(10):
        print(f"{bigram}: {freq}")
