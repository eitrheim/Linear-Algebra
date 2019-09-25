from bs4 import BeautifulSoup
from string import digits
import requests
import urllib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import nltk
# nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import numpy as np
import re


def scrape_to_str(url):
    """
    Scrape all words from a Wikipedia page

    Args:
        url: the url of the page

    Returns:
        wiki_join: a string containing the text from the page
    """
    html = requests.get(url)
    soup = BeautifulSoup(html.content)
    wiki_str = ""
    for entry in soup.find_all(name='p'):
        paragraph = entry.get_text()
        wiki_str += paragraph
    lower_str = wiki_str.lower()
    remove_digits = str.maketrans('', '', digits)
    alpha_string = lower_str.translate(remove_digits)
    cleaned_up = re.sub('[^A-Za-z0-9 ]+', '', alpha_string)
    wiki_split = cleaned_up.split()
    wnl = WordNetLemmatizer()
    wiki_lem_n = [wnl.lemmatize(x, pos='n') for x in wiki_split]
    wiki_lem_nv = [wnl.lemmatize(x, pos='v') for x in wiki_lem_n]
    wiki_lem_nva = [wnl.lemmatize(x, pos='a') for x in wiki_lem_nv]
    delim = " "
    wiki_join = delim.join(wiki_lem_nva)
    return wiki_join


def scrape_to_stop_words(url):
    """
        Scrape all words from a Wikipedia page, remove words with 3 characters or less.

        Args:
            url: the url of the page

        Returns:
            new_stop_words: a list of all stop words
        """
    html = requests.get(url)
    soup = BeautifulSoup(html.content)
    wiki_str = ""
    for entry in soup.find_all(name='p'):
        paragraph = entry.get_text()
        wiki_str += paragraph
    lower_str = wiki_str.lower()
    remove_digits = str.maketrans('', '', digits)
    alpha_string = lower_str.translate(remove_digits)
    cleaned_up = re.sub('[^A-Za-z0-9 ]+', '', alpha_string)
    wiki_split = cleaned_up.split()
    wnl = WordNetLemmatizer()
    wiki_lem_n = [wnl.lemmatize(x, pos='n') for x in wiki_split]
    wiki_lem_nv = [wnl.lemmatize(x, pos='v') for x in wiki_lem_n]
    wiki_lem_nva = [wnl.lemmatize(x, pos='a') for x in wiki_lem_nv]
    unique_stop_words = list(set([word for word in wiki_lem_nva if len(word) <= 3]))
    new_stop_words = text.ENGLISH_STOP_WORDS.union(unique_stop_words)
    return new_stop_words


linear_algebra = "https://en.wikipedia.org/wiki/Linear_algebra"
lin_alg_str = scrape_to_str(linear_algebra)
lin_alg_stop_words = scrape_to_stop_words(linear_algebra)

philosophy = "https://en.wikipedia.org/wiki/Philosophy"
philosophy_str = scrape_to_str(philosophy)
philosophy_stop_words = scrape_to_stop_words(philosophy)

algebra = "https://en.wikipedia.org/wiki/Algebra"
algebra_str = scrape_to_str(algebra)
algebra_stop_words = scrape_to_stop_words(algebra)

tajikistan = "https://en.wikipedia.org/wiki/Tajikistan"
tajikistan_str = scrape_to_str(tajikistan)
tajikistan_stop_words = scrape_to_stop_words(tajikistan)

geometry = "https://en.wikipedia.org/wiki/Geometry"
geometry_str = scrape_to_str(geometry)
geometry_stop_words = scrape_to_stop_words(geometry)

math = "https://en.wikipedia.org/wiki/Mathematics"
math_str = scrape_to_str(math)
math_stop_words = scrape_to_stop_words(math)

lohan = "https://en.wikipedia.org/wiki/Lindsay_Lohan"
lohan_str = scrape_to_str(lohan)
lohan_stop_words = scrape_to_stop_words(lohan)

musk = "https://en.wikipedia.org/wiki/Musk"
musk_str = scrape_to_str(musk)
musk_stop_words = scrape_to_stop_words(musk)

linreg = "https://en.wikipedia.org/wiki/Linear_regression"
linreg_str = scrape_to_str(linreg)
linreg_stop_words = scrape_to_stop_words(linreg)

analytics = "https://en.wikipedia.org/wiki/Analytics"
analytics_str = scrape_to_str(analytics)
analytics_stop_words = scrape_to_stop_words(analytics)

# Combine Lists
combined_list = [lin_alg_str, philosophy_str, algebra_str, tajikistan_str, geometry_str,
                 math_str, lohan_str, musk_str, linreg_str, analytics_str]
rows_df = ['lin_alg', 'philosophy', 'algebra', 'tajikistan', 'geometry', 'math', 'lohan',
           'musk', 'linreg', 'analytics']

# Combine Stop Words
frozenset_list = [lin_alg_stop_words, philosophy_stop_words, algebra_stop_words, tajikistan_stop_words,
                  geometry_stop_words, math_stop_words, lohan_stop_words, musk_stop_words,
                  linreg_stop_words, analytics_stop_words]
final_stop_words = frozenset().union(*frozenset_list)

# Bag of Words Vectorizer
bow = CountVectorizer(stop_words=final_stop_words)
wiki_bow = bow.fit_transform(combined_list)
wiki_bow.shape
bow_features = bow.get_feature_names()

# Change to DataFrame and Save
bow_df = pd.DataFrame(wiki_bow.toarray(), columns=bow.get_feature_names(), index=rows_df)
bow_df.to_csv('~/Documents/MSCA/Linear_Algebra_and_Matrix_Analysis/Group Project/bow.csv', encoding='utf-8')

# TFIDF Vectorizer
tfidf = TfidfVectorizer(stop_words=final_stop_words)
wiki_tfidf = tfidf.fit_transform(combined_list)
tfidf_features = tfidf.get_feature_names()
tfidf_df = pd.DataFrame(wiki_tfidf.toarray(), columns=tfidf.get_feature_names(), index=rows_df)
tfidf_df.to_csv('~/Documents/MSCA/Linear_Algebra_and_Matrix_Analysis/Group Project/tfidf.csv', encoding='utf-8')

# Transforming Vector Via LSA (n_components = 10)
tfidf_matrix = tfidf.fit_transform(combined_list).toarray()
lsa10 = TruncatedSVD(n_components=10, random_state=42)
wiki_lsa10 = lsa10.fit_transform(tfidf_matrix)
lsa_df10 = pd.DataFrame(wiki_lsa10, index=rows_df)
lsa_df10.to_csv('~/Documents/MSCA/Linear_Algebra_and_Matrix_Analysis/Group Project/lsa10.csv', encoding='utf-8')

# Transforming Vector Via LSA (n_components = 5)
tfidf_matrix = tfidf.fit_transform(combined_list).toarray()
lsa5 = TruncatedSVD(n_components=5, random_state=42)
wiki_lsa5 = lsa5.fit_transform(tfidf_matrix)
lsa_df5 = pd.DataFrame(wiki_lsa5, index=rows_df)
lsa_df5.to_csv('~/Documents/MSCA/Linear_Algebra_and_Matrix_Analysis/Group Project/lsa5.csv', encoding='utf-8')

# Euclidean Distance
# Using Bag of Words Frequencies
distances_bow = pdist(bow_df.values, metric='euclidean')
bow_distance_matrix = squareform(distances_bow)
bow_distance_matrix_df = pd.DataFrame(bow_distance_matrix, columns=rows_df, index=rows_df)
bow_distance_matrix_df.to_csv('~/Documents/MSCA/Linear_Algebra_and_Matrix_Analysis/Group Project/bow_dist.csv',
                              encoding='utf-8')

# Using TFIDF Frequencies
distances_tfidf = pdist(tfidf_df.values, metric='euclidean')
tfidf_distance_matrix = squareform(distances_tfidf)
tfidf_distance_matrix_df = pd.DataFrame(tfidf_distance_matrix, columns=rows_df, index=rows_df)
tfidf_distance_matrix_df.to_csv('~/Documents/MSCA/Linear_Algebra_and_Matrix_Analysis/Group Project/tfidf_dist.csv',
                                encoding='utf-8')

# Using TFIDF Frequencies Transformed with LSA10
distances_lsa10 = pdist(lsa_df10.values, metric='euclidean')
lsa_distance_matrix10 = squareform(distances_lsa10)
lsa_distance_matrix_df10 = pd.DataFrame(lsa_distance_matrix10, columns=rows_df, index=rows_df)
lsa_distance_matrix_df10.to_csv('~/Documents/MSCA/Linear_Algebra_and_Matrix_Analysis/Group Project/lsa_dist10.csv',
                                encoding='utf-8')

# Using TFIDF Frequencies Transformed with LSA5
distances_lsa5 = pdist(lsa_df5.values, metric='euclidean')
lsa_distance_matrix5 = squareform(distances_lsa5)
lsa_distance_matrix_df5 = pd.DataFrame(lsa_distance_matrix5, columns=rows_df, index=rows_df)
lsa_distance_matrix_df5.to_csv('~/Documents/MSCA/Linear_Algebra_and_Matrix_Analysis/Group Project/lsa_dist5.csv',
                               encoding='utf-8')

# Cosine Similarity
# Using Bag of Word Frequencies
bow_angle_matrix = cosine_similarity(wiki_bow)
bow_angle_matrix_df = pd.DataFrame(bow_angle_matrix, columns=rows_df, index=rows_df)
bow_angle_matrix_df.to_csv('~/Documents/MSCA/Linear_Algebra_and_Matrix_Analysis/Group Project/bow_angle.csv',
                           encoding='utf-8')

# Using TFIDF Frequencies
tfidf_angle_matrix = cosine_similarity(wiki_tfidf)
tfidf_angle_matrix_df = pd.DataFrame(tfidf_angle_matrix, columns=rows_df, index=rows_df)
tfidf_angle_matrix_df.to_csv('~/Documents/MSCA/Linear_Algebra_and_Matrix_Analysis/Group Project/tfidf_angle.csv',
                             encoding='utf-8')

# Using TFIDF Frequencies Transformed with LSA10
lsa_angle_matrix10 = cosine_similarity(wiki_lsa10)
lsa_angle_matrix_df10 = pd.DataFrame(lsa_angle_matrix10, columns=rows_df, index=rows_df)
lsa_angle_matrix_df10.to_csv('~/Documents/MSCA/Linear_Algebra_and_Matrix_Analysis/Group Project/lsa_angle10.csv',
                             encoding='utf-8')

# Using TFIDF Frequencies Transformed with LSA5
lsa_angle_matrix5 = cosine_similarity(wiki_lsa5)
lsa_angle_matrix_df5 = pd.DataFrame(lsa_angle_matrix5, columns=rows_df, index=rows_df)
lsa_angle_matrix_df5.to_csv('~/Documents/MSCA/Linear_Algebra_and_Matrix_Analysis/Group Project/lsa_angle5.csv',
                            encoding='utf-8')

# Jaccard Distance
lin_algebra_list = set(lin_alg_str.split()).difference(lin_alg_stop_words)
philosophy_list = set(philosophy_str.split()).difference(philosophy_stop_words)
algebra_list = set(algebra_str.split()).difference(algebra_stop_words)
tajikistan_list = set(tajikistan_str.split()).difference(tajikistan_stop_words)
geometry_list = set(geometry_str.split()).difference(geometry_stop_words)
math_list = set(math_str.split()).difference(math_stop_words)
lohan_list = set(lohan_str.split()).difference(lohan_stop_words)
musk_list = set(musk_str.split()).difference(musk_stop_words)
linreg_list = set(linreg_str.split()).difference(linreg_stop_words)
analytics_list = set(analytics_str.split()).difference(analytics_stop_words)


def jaccard_similarity(a, b):
    intersection = a.intersection(b)
    union = a.union(b)
    return len(intersection) / len(union)


sample = [philosophy_list, algebra_list, tajikistan_list, geometry_list,
          math_list, lohan_list, musk_list, linreg_list, analytics_list]
score = list(map(lambda x: jaccard_similarity(lin_algebra_list, x), sample))
labels = ['philosophy', 'algebra', 'tajikistan', 'geometry', 'math', 'lohan',
          'musk', 'linear regression', 'analytics']
jaccard_similarity_score = pd.DataFrame(score, labels)
