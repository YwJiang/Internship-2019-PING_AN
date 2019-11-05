## Introduction 

This repo records my full-time internship in PingAn Financial (Group) Company of China 



## Intelligent FAQ robot

To develop a Question Answering Retrieving System which can provide legal consultancy services.

### Framework

faq_demo consists of Analysis, Retrieval, Match and Rank parts:

- **Analysis**: conduct data cleaning and text tokenization, generate sentence embedding vector via Skip-thought algorithm
- **Retrieval:** retrieve similar questions via ElasticSearch retrieval and Annoy semantic retrieval
- **Match:** compare the similarity  between similar questions and original question. BM25 similarity, edit distance, ABCNN and jaccard similarity algorithm are used to calculated the score. XGBoost is used to calculate the final similarity score.
- **Rank:** rerank the similar questions precisely and return top N similar questions and their answers in corpus, as recommended answers





## Text classification

To build law judicial adjudicative document portrait by deconstructing documents and implementing
multilabel classification.

- **Preprocessing:** deconstruct the law judical documents by regular expressions, and conducted text tokenization
- **Implement text classification algorithms:** FastText, Long Short-Term Memory (LSTM) CNN , Attention-CNN



