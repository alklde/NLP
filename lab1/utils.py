import os
import re
import jieba

def load_corpus(corpus_path):
    """
    加载语料库
    """
    try:
        with open(corpus_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'big5']
        for encoding in encodings:
            try:
                with open(corpus_path, 'r', encoding=encoding) as f:
                    lines = f.readlines()
                break
            except UnicodeDecodeError:
                continue
        return lines

def clean_text(text):
    """
    清理文本中的空格和特殊字符，保留中文、英文和数字
    """
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def segment_text(text):
    """
    文本中文分词，返回分词后的词列表
    """
    words = jieba.lcut(text)
    words = [word for word in words if word.strip()]
    return words

def save_dict(dict, dict_path, count=None):
    """
    保存字典到文件
    """
    with open(dict_path, 'w', encoding='utf-8') as f:
        # 保存count个项
        for key, value in dict.most_common(count):
            f.write(f'{key}\t{value}\n')
