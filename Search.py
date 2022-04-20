import os
import urllib.request
import tarfile
import json
import toolz
import re
import nltk
import gensim
import pickle
import numpy as np
# Ensure pyLDAvis version is 3.2.2
import pyLDAvis
import pyLDAvis.gensim
import sys
import pandas as pd
import trectools
from sklearn.neighbors import NearestNeighbors
from functools import reduce
from trectools import TrecRun
from gensim import corpora
from gensim.summarization import bm25
from gensim.parsing.porter import PorterStemmer
from gensim.utils import simple_preprocess
from gensim import models
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict

p = PorterStemmer()
# For Mac OS only
# When using python on MacOS, ssl certificate appears to be missing
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
# Certificate issue done

#Classes and Global variable definition
class Name(object):
    def __init__(self, first, middle, last):
        self.first = first
        self.middle = middle
        self.last = last


class Paper(object):
    def __init__(self, id, title, authors, abstracts, body_text):
        self.id = id
        self.title = title
        self.authors = authors
        self.abstracts = abstracts
        self.body_text = body_text
        
class MyCorpus(object):
    def __init__(self, all_papers):
        self.all_papers = all_papers
    def __iter__(self):
        
        covid_body_dictionary = corpora.Dictionary.load('/home/jovyan/a2v2-team-7-1/tmp/streaming_body_covid.dict')
        
        for paper in all_papers:
            concat_body = []
            for i in range(len(paper.body_text)):
                if paper.body_text[i] != []:
                    #print(type(paper.abstracts[i]))
                    # Removing punctuations in title using regex
                    paper.body_text[i] = re.sub(r'[^\w\s]', '', paper.body_text[i])  
                    # Convert the titles to lowercase
                    paper.body_text[i] = paper.body_text[i].lower()
                    # Porter Stem
                    paper.body_text[i] = p.stem_sentence(paper.body_text[i])

                    #Tokenize
                    body_tokens = word_tokenize(paper.body_text[i])
                    #print(abstract_tokens)
                    filtered_body = [w for w in body_tokens if not w in stop_words and len(w) > 2]
                    paper.body_text[i] = filtered_body
                    concat_body.extend(paper.body_text[i])
            yield covid_body_dictionary.doc2bow(concat_body)
            
#Global Variables
papers = []
stop_words = set(stopwords.words('english'))


#Download and unpack the collection
def getData():
    urls = ['https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-03-27/comm_use_subset.tar.gz', 'https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-03-27/noncomm_use_subset.tar.gz', 'https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-03-27/custom_license.tar.gz', 'https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-03-27/biorxiv_medrxiv.tar.gz']

    # Create data directory
    try:
        os.mkdir('./data')
        print('Directory created')
    except FileExistsError:
        print('Directory already exists')

    #Download all files
    for i in range(len(urls)):
        urllib.request.urlretrieve(urls[i], './data/file'+str(i)+'.tar.gz')
        print('Downloaded file '+str(i+1)+'/'+str(len(urls)))
        tar = tarfile.open('./data/file'+str(i)+'.tar.gz')
        tar.extractall('./data')
        tar.close()
        print('Extracted file '+str(i+1)+'/'+str(len(urls)))
        os.remove('./data/file'+str(i)+'.tar.gz')

#Iterate through the collection and extract key information from each article (Task 1)
def extract():
    #Iterate through all files in the data directory
    for subdir, dirs, files in os.walk('./data'):
        for file in files:
            if os.path.splitext(file)[-1]=='.json':
                with open(os.path.join(subdir, file)) as f:
                    temp = json.load(f)
                    id = temp['paper_id']
                    title = temp['metadata']['title']
                    authors = []
                    for author in temp['metadata']['authors']:
                        name = Name(author['first'], author['middle'], author['last'])
                        authors.append(name)

                    abstracts = []
                    for abstract in temp['abstract']:
                        abstracts.append(abstract['text'])

                    body_text = []
                    for paragraph in temp['body_text']:
                        body_text.append(paragraph['text'])

                    paper = Paper(id, title, authors, abstracts, body_text)
                    papers.append(paper)                  
    all_papers = remove_paper_duplicates() 
    return all_papers

#Return a corpus dictionary of all title words
def create_corpus_dictionary_titles(all_papers):
    
    title_of_all_docs = []
    id_of_all_docs = []
    
    for paper in all_papers:
        #print(type(paper.title))
        #print(paper.title)
        # Removing punctuations in title using regex
        paper.title = re.sub(r'[^\w\s]', '', paper.title)
        # Convert the titles to lowercase
        paper.title = paper.title.lower()
        paper.title = p.stem_sentence(paper.title)
        #Tokenize
        title_tokens = word_tokenize(paper.title) 
        filtered_title = [w for w in title_tokens if not w in stop_words and len(w) > 2]
        paper.title = filtered_title
        title_of_all_docs.append(paper.title)
        id_of_all_docs.append(paper.id)
          
    # Removing words that appear only once
    frequency = defaultdict(int)
    for text in title_of_all_docs:
        for token in text:
            frequency[token] += 1
    title_of_all_docs = [
    [token for token in text if frequency[token] > 1]
    for text in title_of_all_docs
    ]
    
    covid_dictionary = corpora.Dictionary(title_of_all_docs)
    # store the dictionary, for future reference
    covid_dictionary.save('/home/jovyan/a2v2-team-7-1/tmp/covid.dict')
    bow_corpus = [covid_dictionary.doc2bow(text) for text in title_of_all_docs]
    #print(bow_corpus)
    corpora.MmCorpus.serialize('/home/jovyan/a2v2-team-7-1/tmp/BoW_corpus.mm', bow_corpus)
    bm25_instance_titles = bm25.BM25(bow_corpus)
    with open('/home/jovyan/a2v2-team-7-1/tmp/bm25_instance_titles.pickle', 'wb') as file:
        pickle.dump(bm25_instance_titles, file)
    with open('/home/jovyan/a2v2-team-7-1/tmp/titles_of_all_docs_list.pickle', 'wb') as file:
        pickle.dump(title_of_all_docs, file)
    with open('/home/jovyan/a2v2-team-7-1/tmp/ids_of_all_docs_list.pickle', 'wb') as file:
        pickle.dump(id_of_all_docs, file) 
    return

#Return a corpus dictionary of all abstract words
def create_corpus_dictionary_abstracts(all_papers):
    
    abstracts_of_all_docs = []
    
    for paper in all_papers:
        concat_abstract = []
        for i in range(len(paper.abstracts)):
            if paper.abstracts[i] != []:
                #print(type(paper.abstracts[i]))
                # Removing punctuations in title using regex
                paper.abstracts[i] = re.sub(r'[^\w\s]', '', paper.abstracts[i])  
                # Convert the titles to lowercase
                paper.abstracts[i] = paper.abstracts[i].lower()
                # Porter Stem
                paper.abstracts[i] = p.stem_sentence(paper.abstracts[i])
                
                #Tokenize
                abstract_tokens = word_tokenize(paper.abstracts[i])
                #print(abstract_tokens)
                filtered_abstract = [w for w in abstract_tokens if not w in stop_words and len(w) > 2]
                paper.abstracts[i] = filtered_abstract
                concat_abstract.extend(paper.abstracts[i])
        abstracts_of_all_docs.append(concat_abstract) 
    #print("Abstract of all docs: ", abstracts_of_all_docs)
    #print(len(abstracts_of_all_docs))
    
    # Removing words that appear only once
    frequency = defaultdict(int)
    for text in abstracts_of_all_docs:
        for token in text:
            frequency[token] += 1
    abstracts_of_all_docs = [
    [token for token in text if frequency[token] > 1]
    for text in abstracts_of_all_docs
    ]
       
    covid_abstract_dictionary = corpora.Dictionary(abstracts_of_all_docs)
    #print(covid_abstract_dictionary)
    # store the dictionary, for future reference
    covid_abstract_dictionary.save('/home/jovyan/a2v2-team-7-1/tmp/abstract_covid.dict')
    bow_abstract_corpus = [covid_abstract_dictionary.doc2bow(text) for text in abstracts_of_all_docs]
    #print(bow_abstract_corpus)
    corpora.MmCorpus.serialize('/home/jovyan/a2v2-team-7-1/tmp/abstract_BoW_corpus.mm', bow_abstract_corpus)
    bm25_instance_abstracts = bm25.BM25(bow_abstract_corpus)
    with open('/home/jovyan/a2v2-team-7-1/tmp/bm25_instance_abstracts.pickle', 'wb') as file:
        pickle.dump(bm25_instance_abstracts, file)
    
    return False
                       
class Read_files(object):
    def __init__(self, directoryname):
        self.directoryname = directoryname
    def __iter__(self):
        for fname in os.listdir(self.directoryname):
            for line in open(os.path.join(self.directoryname, fname), encoding='latin'):
                yield simple_preprocess(line)
                
class Read_body_text_from_files(object):
    def __init__(self, directoryname):
        self.directoryname = directoryname
    def __iter__(self):
        for subdir, dirs, files in os.walk('./data'):
            for file in files:
                if os.path.splitext(file)[-1]=='.json':
                    with open(os.path.join(subdir, file)) as f:
                        temp = json.load(f)
                        id = temp['paper_id']
                        title = temp['metadata']['title']
                        authors = []
                        for author in temp['metadata']['authors']:
                            name = Name(author['first'], author['middle'], author['last'])
                            authors.append(name)

                        abstracts = []
                        for abstract in temp['abstract']:
                            abstracts.append(abstract['text'])

                        body_text = []
                        for paragraph in temp['body_text']:
                            body_text.append(paragraph['text'])

                        paper = Paper(id, title, authors, abstracts, body_text)
                        
                        concat_body = []
                        for i in range(len(paper.body_text)):
                            if paper.body_text[i] != []:
                                #print(type(paper.abstracts[i]))
                                # Removing punctuations in title using regex
                                paper.body_text[i] = re.sub(r'[^\w\s]', '', paper.body_text[i])  
                                # Convert the titles to lowercase
                                paper.body_text[i] = paper.body_text[i].lower()
                                # Porter Stem
                                paper.body_text[i] = p.stem_sentence(paper.body_text[i])

                                #Tokenize
                                body_tokens = word_tokenize(paper.body_text[i])
                                #print(abstract_tokens)
                                filtered_body = [w for w in body_tokens if not w in stop_words and len(w) > 2]
                                paper.body_text[i] = filtered_body
                                concat_body.extend(paper.body_text[i])
                        yield concat_body

#Return a corpus dictionary of all body text words streaming corpus one doc at a time
def stream_corpus_dictionary_body_text(all_papers):
    
    body_text_of_all_docs = []
    
    for paper in all_papers:
        concat_body = []
        for i in range(len(paper.body_text)):
            if paper.body_text[i] != []:
                #print(type(paper.abstracts[i]))
                # Removing punctuations in title using regex
                paper.body_text[i] = re.sub(r'[^\w\s]', '', paper.body_text[i])  
                # Convert the titles to lowercase
                paper.body_text[i] = paper.body_text[i].lower()
                # Porter Stem
                paper.body_text[i] = p.stem_sentence(paper.body_text[i])
                
                #Tokenize
                body_tokens = word_tokenize(paper.body_text[i])
                #print(abstract_tokens)
                filtered_body = [w for w in body_tokens if not w in stop_words and len(w) > 2]
                paper.body_text[i] = filtered_body
                concat_body.extend(paper.body_text[i])
        body_text_of_all_docs.append(concat_body) 
    
    #print(len(body_text_of_all_docs))
        
    # Removing words that appear only once
    frequency = defaultdict(int)
    for text in body_text_of_all_docs:
        for token in text:
            frequency[token] += 1
    body_text_of_all_docs = [
    [token for token in text if frequency[token] > 1]
    for text in body_text_of_all_docs
    ]
       
    covid_body_dictionary = corpora.Dictionary(body_text_of_all_docs)
    # store the dictionary, for future reference
    covid_body_dictionary.save('/home/jovyan/a2v2-team-7-1/tmp/body_covid.dict')   
    bow_body_corpus = [covid_body_dictionary.doc2bow(text) for text in body_text_of_all_docs]
    corpora.MmCorpus.serialize('/home/jovyan/a2v2-team-7-1/tmp/body_BoW_corpus.mm', bow_body_corpus)
    bm25_instance_body = bm25.BM25(bow_body_corpus)
    with open('/home/jovyan/a2v2-team-7-1/tmp/bm25_instance_body.pickle', 'wb') as file:
        pickle.dump(bm25_instance_body, file)
    
    return all_papers
            
#Return a corpus dictionary of all body text words
# Warning jupyter notebook keeps returning killed due to max memory
def create_corpus_dictionary_body_text(all_papers):
    
    body_text_of_all_docs = []
    
    for paper in all_papers:
        concat_body = []
        for i in range(len(paper.body_text)):
            if paper.body_text[i] != []:
                #print(type(paper.abstracts[i]))
                # Removing punctuations in title using regex
                paper.body_text[i] = re.sub(r'[^\w\s]', '', paper.body_text[i])  
                # Convert the titles to lowercase
                paper.body_text[i] = paper.body_text[i].lower()
                # Porter Stem
                paper.body_text[i] = p.stem_sentence(paper.body_text[i])
                
                #Tokenize
                body_tokens = word_tokenize(paper.body_text[i])
                #print(abstract_tokens)
                filtered_body = [w for w in body_tokens if not w in stop_words and len(w) > 2]
                paper.body_text[i] = filtered_body
                concat_body.extend(paper.body_text[i])
        body_text_of_all_docs.append(concat_body) 
        
    # Removing words that appear only once
    frequency = defaultdict(int)
    for text in body_text_of_all_docs:
        for token in text:
            frequency[token] += 1
    body_text_of_all_docs = [
    [token for token in text if frequency[token] > 1]
    for text in body_text_of_all_docs
    ]

    covid_body_dictionary = corpora.Dictionary(body_text_of_all_docs)
    # store the dictionary, for future reference
    covid_body_dictionary.save('/home/jovyan/a2v2-team-7-1/tmp/body_covid.dict')
  
    bow_body_corpus = [covid_body_dictionary.doc2bow(text) for text in body_text_of_all_docs]
    corpora.MmCorpus.serialize('/home/jovyan/a2v2-team-7-1/tmp/body_BoW_corpus.mm', bow_body_corpus)
    bm25_instance_body = bm25.BM25(bow_body_corpus)
    with open('/home/jovyan/a2v2-team-7-1/tmp/bm25_instance_body.pickle', 'wb') as file:
        pickle.dump(bm25_instance_body, file)
    
    return False

#Return a query all words
def create_corpus_query(query):
    # Removing punctuations in title using regex
    query = re.sub(r'[^\w\s]', '', query)
    # Convert the titles to lowercase
    query = query.lower()
    # Stem query
    query = p.stem_sentence(query)
    #Tokenize
    query_tokens = word_tokenize(query) 
    filtered_query = [w for w in query_tokens if not w in stop_words]
    query = filtered_query
    return query


# BM25 function specifically for abstract text - returns ids, titles, and scores for query results
def bm25_ranking_abstracts(query, num_docs):
    #Cleans and tokenizes query
    query = create_corpus_query(query)
 
    # Use abstract text to build out dictionary (could also be title or body text)
    covid_dictionary = corpora.Dictionary.load('/home/jovyan/a2v2-team-7-1/tmp/abstract_covid.dict')
    query_bow = covid_dictionary.doc2bow(query)

    # Open BM25 instance
    with open('/home/jovyan/a2v2-team-7-1/tmp/bm25_instance_abstracts.pickle', 'rb') as file:
        bm25_instance = pickle.load(file)
    
    #Open associated titles and docs
    with open('/home/jovyan/a2v2-team-7-1/tmp/titles_of_all_docs_list.pickle', 'rb') as file:
        title_of_all_docs = pickle.load(file)
    with open('/home/jovyan/a2v2-team-7-1/tmp/ids_of_all_docs_list.pickle', 'rb') as file:
        ids_of_all_docs = pickle.load(file)
   
    scores = bm25_instance.get_scores(query_bow)
    best_docs = np.argsort(scores)[-num_docs:]
    # Currently top docs but in ascending order so we reverse here
    best_docs = list(best_docs)
    best_docs.reverse()
    best_scores = [scores[i] for i in best_docs]
    best_ids = [ids_of_all_docs[i] for i in best_docs]
    best_titles = [title_of_all_docs[i] for i in best_docs] 
    return best_ids, best_titles, best_scores

# BM25 function specifically for title text - returns ids, titles, and scores for query results
def bm25_ranking_titles(query, num_docs):
    #Cleans and tokenizes query
    query = create_corpus_query(query)
 
    # Use abstract text to build out dictionary (could also be title or body text)
    covid_dictionary = corpora.Dictionary.load('/home/jovyan/a2v2-team-7-1/tmp/title_covid.dict')
    query_bow = covid_dictionary.doc2bow(query)

    # Open BM25 instance
    with open('/home/jovyan/a2v2-team-7-1/tmp/bm25_instance_titles.pickle', 'rb') as file:
        bm25_instance = pickle.load(file)
    
    #Open associated titles and docs
    with open('/home/jovyan/a2v2-team-7-1/tmp/titles_of_all_docs_list.pickle', 'rb') as file:
        title_of_all_docs = pickle.load(file)
    with open('/home/jovyan/a2v2-team-7-1/tmp/ids_of_all_docs_list.pickle', 'rb') as file:
        ids_of_all_docs = pickle.load(file)   
    
    scores = bm25_instance.get_scores(query_bow)    
    best_docs = np.argsort(scores)[-num_docs:]
    # Currently top docs but in ascending order so we reverse here
    best_docs = list(best_docs)
    best_docs.reverse()
    best_scores = [scores[i] for i in best_docs]
    best_ids = [ids_of_all_docs[i] for i in best_docs]
    best_titles = [title_of_all_docs[i] for i in best_docs]  
    return best_ids, best_titles, best_scores

# BM25 function specifically for body text - returns ids, titles, and scores for query results
def bm25_ranking_body_text(query, num_docs):
    #Cleans and tokenizes query
    query = create_corpus_query(query)
 
    # Use abstract text to build out dictionary (could also be title or body text)
    covid_dictionary = corpora.Dictionary.load('/home/jovyan/a2v2-team-7-1/tmp/streaming_body_covid.dict')
    query_bow = covid_dictionary.doc2bow(query)

    # Open BM25 instance
    with open('/home/jovyan/a2v2-team-7-1/tmp/bm25_instance_body.pickle', 'rb') as file:
        bm25_instance = pickle.load(file)
    
    #Open associated titles and docs
    with open('/home/jovyan/a2v2-team-7-1/tmp/titles_of_all_docs_list.pickle', 'rb') as file:
        title_of_all_docs = pickle.load(file)
    with open('/home/jovyan/a2v2-team-7-1/tmp/ids_of_all_docs_list.pickle', 'rb') as file:
        ids_of_all_docs = pickle.load(file)   
    
    scores = bm25_instance.get_scores(query_bow)    
    best_docs = np.argsort(scores)[-num_docs:]
    # Currently top docs but in ascending order so we reverse here
    best_docs = list(best_docs)
    best_docs.reverse()
    best_scores = [scores[i] for i in best_docs]
    best_ids = [ids_of_all_docs[i] for i in best_docs]
    best_titles = [title_of_all_docs[i] for i in best_docs]  
    return best_ids, best_titles, best_scores


# Function to either sum or multiply BM25 scores across multiple attributes (title, abstract, body_text, etc.)
def fusion_scores(query, num_docs):
    query = create_corpus_query(query)
    # Use abstract text to build out dictionary (could also be title or body text)
    covid_dictionary_title = corpora.Dictionary.load('/home/jovyan/a2v2-team-7-1/tmp/title_covid.dict')
    query_bow = covid_dictionary_title.doc2bow(query)
    # Open BM25 instance
    with open('/home/jovyan/a2v2-team-7-1/tmp/bm25_instance_titles.pickle', 'rb') as file:
        bm25_instance = pickle.load(file)
    
    #Open associated titles and docs
    with open('/home/jovyan/a2v2-team-7-1/tmp/titles_of_all_docs_list.pickle', 'rb') as file:
        title_of_all_docs = pickle.load(file)
    with open('/home/jovyan/a2v2-team-7-1/tmp/ids_of_all_docs_list.pickle', 'rb') as file:
        ids_of_all_docs = pickle.load(file)
    title_scores = bm25_instance.get_scores(query_bow)
    
    # Use abstract text to build out dictionary (could also be title or body text)
    covid_dictionary_abstract = corpora.Dictionary.load('/home/jovyan/a2v2-team-7-1/tmp/abstract_covid.dict')
    query_bow = covid_dictionary_abstract.doc2bow(query)
    # Open BM25 instance
    with open('/home/jovyan/a2v2-team-7-1/tmp/bm25_instance_abstracts.pickle', 'rb') as file:
        bm25_instance = pickle.load(file)
    abstract_scores = bm25_instance.get_scores(query_bow)
    
    # Open BM25 instance
    with open('/home/jovyan/a2v2-team-7-1/tmp/bm25_instance_body.pickle', 'rb') as file:
        bm25_instance = pickle.load(file)
    body_scores = bm25_instance.get_scores(query_bow)
    
    #Use when summing titles, abstracts, and body 
    #sum_scores = [title_scores[i] + abstract_scores[i] + body_scores[i] for i in range(len(title_scores))]
    
    #Use when summing titles, abstracts 
    sum_scores = [title_scores[i] + abstract_scores[i] for i in range(len(title_scores))]
    
    #Use when multipling titles, abstracts, and body
    mult_scores = []
#     for i in range(len(title_scores)):
#         non_zero = [x for x in [title_scores[i], abstract_scores[i], body_scores[i]] if x != 0]
#         if len(non_zero) == 0:
#             mult_scores.append(0)
#         else:
#             mult_scores.append(np.prod(non_zero))
    
### Old looping for just abstract and title scores
    for i in range(len(title_scores)):
        if title_scores[i] != 0 and abstract_scores[i] != 0:
            mult_scores.append(title_scores[i]*abstract_scores[i])
        else:
            mult_scores.append(max(title_scores[i],abstract_scores[i]))
    
    best_docs = np.argsort(mult_scores)[-num_docs:]
    # Currently top docs but in ascending order so we reverse here
    best_docs = list(best_docs)
    best_docs.reverse()
    best_scores = [mult_scores[i] for i in best_docs]
    best_ids = [ids_of_all_docs[i] for i in best_docs]
    best_titles = [title_of_all_docs[i] for i in best_docs]
    
    return best_ids, best_titles, best_scores

# Function for task 2 in data visualization using LDA.  Please open the tmp folder to see topic modeling performed on abstract and title
# Open title_LDA_vis.html & abstract_LDA_vis.html and press trust html to visual results 
def build_lda_model():
    
    #Build LDA Model For Title Text
    covid_dictionary = corpora.Dictionary.load('/home/jovyan/a2v2-team-7-1/tmp/title_covid.dict')
    bow_corpus = corpora.MmCorpus('/home/jovyan/a2v2-team-7-1/tmp/title_BoW_corpus.mm')
    lda_model = gensim.models.LdaMulticore(corpus=bow_corpus, id2word=covid_dictionary, num_topics=7, iterations=200)
    #print(lda_model.print_topics()) 
    lda_visualization = pyLDAvis.gensim.prepare(lda_model, bow_corpus, covid_dictionary)
    #lda_visualization = vis_gensim.prepare(lda_model, bow_corpus, covid_dictionary)
    pyLDAvis.save_html(lda_visualization, '/home/jovyan/a2v2-team-7-1/tmp/title_LDA_vis.html' )
    #pyLDAvis.display(lda_visualization)
    
    #Build LDA Model For Abstract Text
    covid_dictionary = corpora.Dictionary.load('/home/jovyan/a2v2-team-7-1/tmp/abstract_covid.dict')
    bow_corpus = corpora.MmCorpus('/home/jovyan/a2v2-team-7-1/tmp/abstract_BoW_corpus.mm')
    lda_model = gensim.models.LdaMulticore(corpus=bow_corpus, id2word=covid_dictionary, num_topics=7, iterations=200)
    #print(lda_model.print_topics()) 
    lda_visualization = pyLDAvis.gensim.prepare(lda_model, bow_corpus, covid_dictionary)
    #lda_visualization = vis_gensim.prepare(lda_model, bow_corpus, covid_dictionary)
    pyLDAvis.save_html(lda_visualization, '/home/jovyan/a2v2-team-7-1/tmp/abstract_LDA_vis.html' )
    #pyLDAvis.display(lda_visualization)
    
    #Build LDA Model For Body Text
#     covid_dictionary = corpora.Dictionary.load('/home/jovyan/a2v2-team-7-1/tmp/streaming_body_covid.dict')
#     bow_corpus = corpora.MmCorpus('/home/jovyan/a2v2-team-7-1/tmp/streaming_body_BoW_corpus.mm')
#     lda_model = gensim.models.LdaMulticore(corpus=bow_corpus, id2word=covid_dictionary, num_topics=6)
#     #print(lda_model.print_topics()) 
#     lda_visualization = pyLDAvis.gensim.prepare(lda_model, bow_corpus, covid_dictionary)
#     #lda_visualization = vis_gensim.prepare(lda_model, bow_corpus, covid_dictionary)
#     pyLDAvis.save_html(lda_visualization, '/home/jovyan/a2v2-team-7-1/tmp/body_LDA_vis.html' )
#     #pyLDAvis.display(lda_visualization)
    
    return lda_model


#Organize the collection (Task 2)
def organize(all_papers):
    
    #Tokenize title, remove stopwords, punctionation, and lowercase all letters
    for paper in all_papers:
        # Removing punctuations in title using regex
        paper.title = re.sub(r'[^\w\s]', '', paper.title)
        # Convert the titles to lowercase
        paper.title = paper.title.lower()
        title_tokens = word_tokenize(paper.title) 
        filtered_title = [w for w in title_tokens if not w in stop_words]
        paper.title = filtered_title
        #print(paper.title)

    #Build LDA model will load pre-created BoW and corpus dictionary to visualize results with Gensim LDA functions
    build_lda_model()  
    return

#Answer a set of textual queries (Task 3)
def retrieve(queries, num_results=100):
    
    #body_results = [bm25_ranking_body_text(q,num_results)[0] for q in queries]
    #abstract_results = [bm25_ranking_abstracts(q,num_results)[0] for q in queries]
    #title_results = [bm25_ranking_titles(q,num_results)[0] for q in queries]
    mult_results = [fusion_scores(q,num_results)[0] for q in queries]
    
    #body_titles = [bm25_ranking_body_text(q,num_results)[1] for q in queries]
    #abstract_titles = [bm25_ranking_abstracts(q,num_results)[1] for q in queries]
    #title_titles = [bm25_ranking_titles(q,num_results)[1] for q in queries]
    mult_titles = [fusion_scores(q,num_results)[1] for q in queries]
    
    #body_scores = [bm25_ranking_body_text(q,num_results)[2] for q in queries]
    #abstract_scores = [bm25_ranking_abstracts(q,num_results)[2] for q in queries]
    #title_scores = [bm25_ranking_titles(q,num_results)[2] for q in queries]
    mult_scores = [fusion_scores(q,num_results)[2] for q in queries]
       
#     Obsolete work on fusion implementation:
#
#     fusion([abstract_results,title_results],[abstract_scores,title_scores])
#     same_results = []
#     for query in range(len(abstract_results)):
#         for rank in range(len(abstract_results[query])):
#             if abstract_results[query][rank] in title_results[query]:
#                 same_results.append(abstract_results[query][rank])           
#     print(same_results)
#     print(len(same_results))
   
    #Output results
    save_sum = open('/home/jovyan/a2v2-team-7-1/tmp/mult_fusion_results.txt','w')
    #title_sum  = open('/home/jovyan/a2v2-team-7-1/tmp/title_results.txt','w')
    for query in range(len(mult_results)):
        for rank in range(len(mult_results[query])):
            print(str(query+1)+'\t'+str(rank+1)+'\t'+str(mult_results[query][rank])+'\t'+str(mult_scores[query][rank]))
            save_sum.write(str(query+1)+'\t'+str(rank+1)+'\t'+str(mult_results[query][rank])+'\t'+str(mult_scores[query][rank])+'\n')
            save_sum.write(' '.join(mult_titles[query][rank])+'\n')
    
    # This was a loop to compare fusion scores against BM25 title results 
#     for query in range(len(title_results)):
#         for rank in range(len(title_results[query])):
#             #print(str(query+1)+'\t'+str(rank+1)+'\t'+str(title_results[query][rank])+'\t'+str(title_scores[query][rank]))
#             title_sum.write(str(query+1)+'\t'+str(rank+1)+'\t'+str(title_results[query][rank])+'\t'+str(title_scores[query][rank])+'\n')
#             title_sum.write(' '.join(title_titles[query][rank])+'\n')
    
# Function to experiment with pre-built fusion functions.           
# def fusion(result_list, score_list):
#     store_r = []
#     for i,results in enumerate(result_list):
#         r = open('/home/jovyan/a2v2-team-7-1/tmp/r'+str(i)+'.txt','w')
#         for query in range(len(results)):
#             for rank in range(len(results[query])):
#                 # This should be query ID, 0, doc_num, relevance score
#                 q_rels.write((str(query+1)+' 0 '+str(results[query][rank])+' '+str(score_list[i][query][rank])+'\n'))
#                 # This should be in format: query   q0          docid  rank     score     system
#                 r.write((str(query+1)+'\tQ0\t'+str(results[query][rank])+ '\t' + str(rank) + '\t' + str(score_list[i][query][rank]) + '\t' + 'sys_'+str(i)+'\n'))
#                 store_r.append(trectools.TrecRun('/home/jovyan/a2v2-team-7-1/tmp/r'+str(i)+'.txt'))
#     fused_run = reciprocal_rank_fusion([r1,r2])
#     qrels = trectools.TrecQrel('/home/jovyan/a2v2-team-7-1/tmp/q_rels.txt')
#     print(qrels)
#     return 

# Remove duplicate research papers by checking for repeated ID or titles
def remove_paper_duplicates():
    no_duplicate_papers = toolz.unique(papers, key=lambda x: x.id)
    no_duplicate_papers = toolz.unique(papers, key=lambda x: x.title)
    return no_duplicate_papers


# Do this once to build out the data folder
# Using data_small to experiment with functions with reduced run-time
#getData()

# Extracts all data into objects of paper class
#all_papers = extract()

# Builds corpus dictionary and BoW for each metadata attribute
# Note: Only uncomment out one function below at a time w/ the extract function above to build models (saved in tmp)
# Title:
#corpus_dict_titles = create_corpus_dictionary_titles(docs)

# Abstract:
#corpus_dict_abstracts = create_corpus_dictionary_abstracts(docs)

# Body Text:
# Memory issues were noticed creating body text BoW and Dictionary
# The goal is to stream each corpus one at a time (This is still running out of memory in Jupyter notebook)
# dict_MUL = corpora.Dictionary(Read_body_text_from_files('./data'))
# dict_MUL.save('/home/jovyan/a2v2-team-7-1/tmp/streaming_body_covid.dict')
# corpus_memory_friendly = MyCorpus(all_papers)
# corpora.MmCorpus.serialize('/home/jovyan/a2v2-team-7-1/tmp/streaming_body_BoW_corpus.mm', corpus_memory_friendly)

# Organizes and visualizes documents using LDA model
#organize(all_papers)

q = ['coronavirus origin',
     'coronavirus response to weather changes',
     'coronavirus immunity']

# Retrieve text based queries
# Query results are also saved to tmp/sum_fusion_results.txt with tokenized title text for eval
retrieve(q)
