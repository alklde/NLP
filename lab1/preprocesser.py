from utils import load_corpus, clean_text, segment_text
from collections import Counter

def preprocess_corpus(corpus_path):
    """
    对语料库进行预处理，包括加载、清理、分词
    """
    # 加载语料库
    lines = load_corpus(corpus_path)

    # 清理文本
    cleaned_lines = [clean_text(line) for line in lines]

    # 分词
    seg_lines = [segment_text(line) for line in cleaned_lines]

    res = (cleaned_lines, seg_lines)
    return res

if __name__ == '__main__':
    corpus_path = 'data/Ci.txt'
    preprocess_corpus(corpus_path)