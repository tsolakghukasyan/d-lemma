# Lemmatisation using Deep Learning

Author: Tsolak Ghukasyan\
Project advisor: Adam Mathias Bittlingmayer

## Introduction

Lemmatization tools often require linguistic expertise and are usually based on rules and lookup tables. To break the linguistic barrier, this project aims to utilise annotated text datasets and word embeddings and develop a universal model for lemmatization. Since there are lemma-annotated UD treebanks and fastText embeddings publicly available for over 60 different languages, the proposed project can be used to train lemmatization models for dozens of languages.

## Models

In this project, 5 different models were considered:

- _Identity baseline_\
To check the efficiency of developed models, 2 baselines are used. Identity function, i.e. returning the input token as its lemma, serves as a weak baseline for main models.

- _Most common lemma with identity backoff_\
Returning most common lemma serves as a stronger baseline for developed models. This baseline backs off to identity for unknown words.

- _Linear regression_\
One of the main models that were developed was a linear regressor with cosine proximity loss that for each input token tries to produce its lemma's embedding.  

- _Seq2seq_\
A word level sequence-to-sequence model using LSTM cells. This model receives sequence of tokens as input and produces the sequence of their lemmas.

- _Transformer_\
An encoder-decoder model based on self-attention mechanism, introduced by Google in [Attention Is All You Need](https://arxiv.org/abs/1706.03762). Similar to seq2seq, it processes a sequence of input tokens to output the sequence of their lemmas.

Other model ideas were also considered such as LSTM networks with softmax layers, however these were rejected because of memory and performance requirements.

## Training & Evaluation

Two languages were selected for training and evaluation of aforementioned models: English as a relatively low-morphology language and Finnish as a language with high morphology. Since one of this project's goals is developing a lemmatization model for resource-scarce languages, only a 10000-token subset of respective UD treebanks was used for training the models.

_Results for English:_

| Model       | Accuracy | BLEU |
|-------------|:--------:|-----:|
| identity    |  78.55%  |  0.579  |
| most common |  91.63%  |  0.764  |
| linear reg. |  88.71%  | -    |
| transformer |    -     |  0.439  |

_Results for Finnish:_

| Model       | Accuracy | BLEU |
|-------------|:--------:|-----:|
| identity    |  52.77%  |      |
| most common |  74.01%  |      |
| linear reg. |  80.12%  |      |
| transformer |    -     |   -  |

_*Word-level seq2seq without attention did not produce any meaningful results._

## Conclusion and Avenues for Further Research

It can be clearly seen that advanced deep learning models do not perform well in this task, with the main reason being limited training data. However, a simple linear regression model demonstrates results very close to the strong baseline, and for Finnish even surpass its performance. The regressor learns to lemmatize not only very common words such as 'are', 'got', 'was' etc, but also learns certain relations (e.g. 'killed'-'kill', 'said'-'say'  'years'-'year'). The model also demonstrates capability to lemmatize unseen wordforms (e.g. 'submitted'-'submit', 'replacing'-'replace'). With adjustments and improvements that will enable use of context information, there is a good chance to beat both baselines.

To further research efficiency of advanced deep learning models, it could be useful to experiment with the following models:
- Word-level seq2seq with attention
- Char-level seq2seq with attention
- DeepMind's Relation networks

## Resources

For training and evaluation:

UD treebanks: http://universaldependencies.org/

Word embeddings: https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md 

## Related Work

[Abhisek Chakrabarty et al. A Neural Lemmatizer for Bengali](https://pdfs.semanticscholar.org/12c6/1ee4f804d4007fc12cfd0d13ba260c051e48.pdf)

[Abhisek Chakrabarty et al. Context Sensitive Lemmatization Using Two Successive Bidirectional Gated Recurrent Networks](http://www.aclweb.org/anthology/P17-1136)

