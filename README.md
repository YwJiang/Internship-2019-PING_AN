## Introduction 

This repo records my full-time internship in PingAn Financial (Group) Company of China 



## Intelligent FAQ robot

It's a software that can find answers for frequently asked law questions

### Framework

faq_demo consists of Analysis, Retrieval, Match and Rank parts:

- **Analysis**: conduct data cleaning and text tokenization, generate sentence embedding vector via Skip-thought algorithm
- **Retrieval:** retrieve similar questions via ElasticSearch retrieval and Annoy semantic retrieval
- **Match: ** compare the similarity  between similar questions and original question. BM25 similarity, edit distance, ABCNN and jaccard similarity algorithm are used to calculated the score. XGBoost is used to ..
- **Rank: **rank and return top N similar questions and answers



## Text classification

It's a classifier for Law judicial adjudicative document classification.



