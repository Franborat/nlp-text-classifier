import json
import numpy as np
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pickle
import click


def read_corpus_to_list(fname='/data/raw'):
    jsonFile = []
    with open(fname) as f:
        for line in f:
            jsonFile.append(json.loads(line))

    corpus = []
    for line in jsonFile:
        corpus.append(line['content'])

    return corpus


def read_category_to_np(fname='/data/raw/'):
    jsonFile = []
    with open(fname) as f:
        for line in f:
            jsonFile.append(json.loads(line))

    category = []
    for line in jsonFile:
        category.append(line['annotation']['label'][0])

    category = np.array(category)

    return category


def read_processed_corpus(fname='data/processed/corpus.pickle'):
    with open(fname, 'rb') as f:
        corpus = pickle.load(f)
    return corpus


def read_processed_category(fname='data/processed/category.pickle'):
    with open(fname, 'rb') as f:
        category = pickle.load(f)
    return category


# Function: To stem and erase numbers

# Input: List of texts(arrays)
# Output: List of texts(arrays)
def stem(corpus):
    stemmer = PorterStemmer()
    stem_corpus = [' '.join([stemmer.stem(word) for word in text.split(' ')])
                   for text in corpus]
    no_numbers_corpus = [re.sub(r'\d+', '', text) for text in stem_corpus]

    return no_numbers_corpus

# Function:

# 1st: To preprocess, tokenize and filter stopwords
# 2nd: Creates a dictionary and a bag of words

# 2nd --> It assigns a score to a word based on its occurrence in a particular document.
#         It doesn't take into account the fact that the word might also be having a high frequency
#         of occurrence in other documents as well.

# max_df = 0.50 means "ignore terms that appear in more than 50% of the documents".
# min_df = 0.01 means "ignore terms that appear in less than 1% of the documents". 5--> In less than 5 documents

# Input: List of texts(arrays)
# Output: Matrix of token counts


def bow(stem_corpus):
    vectorizer = CountVectorizer(min_df=5, max_df=0.7, max_features=1500, stop_words=stopwords.words('english'),
                                 lowercase=True, ngram_range=(1, 2))

    _bow = vectorizer.fit_transform(stem_corpus)

    # return vectorizer, bow
    return _bow, vectorizer


# Input: bag-of-words
# Output: numpy.ndarray, 7600 texts with its 1500 array of tfidf values of the dictionary

# The TFIDF value for a word in a particular document is higher if the frequency of occurrence of that word
# is higher in that specific document but lower in all the other documents.

def tfidf(_bow):

    tfidfconverter = TfidfTransformer()
    _tfidf = tfidfconverter.fit_transform(_bow)

    # return tfidfconverter, tfidf
    return _tfidf, tfidfconverter


@click.command()
@click.argument('input_folder', type=click.Path(exists=True, readable=True, dir_okay=True))
@click.argument('output_folder1', type=click.Path(writable=True, dir_okay=True))
@click.argument('output_folder2', type=click.Path(writable=True, dir_okay=True))
def main(input_folder, output_folder1, output_folder2):

    print('Preprocessing data')

    category = read_category_to_np(input_folder+'/world-business-sports-scitech-text-classification.json')
    corpus = read_corpus_to_list(input_folder+'/world-business-sports-scitech-text-classification.json')

    corpus = stem(corpus)
    corpus, vectorizer = bow(corpus)
    corpus, tfidfconverter = tfidf(corpus)

    # Save in processed data

    # pickle - saving
    with open(output_folder1+'/corpus.pickle', 'wb') as f:
        pickle.dump(corpus, f)

    with open(output_folder1+'/category.pickle', 'wb') as f:
        pickle.dump(category, f)

    # Save in transformers:

    # pickle - saving
    with open(output_folder2+'/vectorizer.pickle', 'wb') as f:
        pickle.dump(vectorizer, f)

    with open(output_folder2+'/tfidfconverter.pickle', 'wb') as f:
        pickle.dump(tfidfconverter, f)

    # corpus.to_pickle(output_file1)
    # category.to_pickle(output_file1)
    #
    # vectorizer.to_pickle(output_file2)
    # tfidfconverter.to_pickle(output_file2)


if __name__ == '__main__':
    main()