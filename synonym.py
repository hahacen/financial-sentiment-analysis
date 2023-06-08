# First, you're going to need to import wordnet:
from nltk.corpus import wordnet
# import pandas_datareader as pdr
import datetime
import meta_parameters
import synonyms
import jieba.analyse
import jieba.finalseg


def daily_return():
    pass


def get_synonym(text: str):
    synonyms = []
    for synset in meta_parameters.wordnet.synsets(text):
        for lemma in synset.lemmas():
            synonyms.append(lemma.name())
    return synonyms


def get_synonyms_ph(phrase):
    words = phrase.split()
    synonyms = set()
    for word in words:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
    return synonyms


def combine_syn(sentence:str):
    words = synonyms.seg(sentence)[0]
    synonyms_list = ['' for _ in range(10)]
    for i, word in enumerate(words):
        syns = synonyms.nearby(word, 10)[0]
        for j, syn in enumerate(syns):
            if j < 10:
                synonyms_list[j] += ' '.join(syn)
    return synonyms_list

phrase = 'exceed expectation'
# synonymsww = get_synonyms_ph(phrase)
# print(synonymsww)
# print(combine_syn("蛋糕真好吃"))

print(get_synonym("story"))
# print(get_synonym("expectation"))
synlist = synonyms.nearby('成本', 20)



# print(synonyms.seg("她提出了一个无法拒绝的要求——她希望有一个会说话的孩子。"))
# text = "Jieba是一个强大的中文分词库，可以用于中文文本的处理。它支持关键词提取、词性标注等功能。"

# Extract top keywords using TF-IDF algorithm
# keywords = jieba.analyse.extract_tags("她提出了一个无法拒绝的要求——她希望有一个会说话的孩子。", topK=8)
# print(keywords)
