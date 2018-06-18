# Lemmatization using Deep Learning

Author: Tsolak Ghukasyan\
Project advisor: Adam Mathias Bittlingmayer

## Introduction

Lemmatization tools are still usually [implemented with rules and lookup tables even in today's top libraries](https://spacy.io/usage/adding-languages#lemmatizer), which require linguistic knowledge of each language to build.

**d-lemma** is developing simple universal models for *learning* lemmatization, using only annotated text datasets and word embeddings.

d-lemma models support a growing set of languages - lemma-annotated UD treebanks and fastText embeddings are publicly available for over 60 different languages.

## Approaches

In this project, 6 different approaches were considered.

To understand the evaluation of the developed learning models, 2 baseline approaches are used:

- _Identity baseline_\
Identity function, i.e. returning the input token as its lemma, serves as a weak baseline for main models.

- _Most common lemma with identity backoff_\
Returning the most common lemma serves as a stronger baseline for developed models. This baseline backs off to identity for unknown words.

The 4 learning models are:

- _Linear regression_\
A linear regressor with cosine proximity loss that for each input token tries to produce its lemma's embedding.  

- _Regression with LSTM_\
A recurrent neural network with single LSTM unit that receives the sequence of input tokens' embeddings and produces the embeddings of their lemmas.  

- _Seq2seq_\
A word level sequence-to-sequence model using LSTM cells. This model receives a sequence of tokens as input and produces the sequence of their lemmas.

- _Transformer_\
An encoder-decoder model based on self-attention mechanism, introduced by Google in [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762). Similar to seq2seq, it processes a sequence of input tokens to output the sequence of their lemmas.

Other model ideas were also considered such as LSTM networks with softmax layers, however these were rejected because of memory and performance requirements.

## Training and Evaluation

Two languages were selected for training and evaluation of the aforementioned models: English as a relatively low-morphology language and Finnish as a high-morphology language.

Since one of this project's goals is developing a lemmatization model for low-resource languages, the models were trained with only a 10000-token subset of the respective UD treebanks.

_Results for English:_

| Model       | Accuracy | BLEU |
|-------------|:--------:|:-----:|
| identity    |  78.55%  |  0.579  |
| most common |  91.63%  |  0.764  |
| linear reg. |  88.71%  |  0.685  |
| LSTM |  92.7%  |  -  |
| transformer |    -     |  0.439  |

_Results for Finnish:_

| Model       | Accuracy | BLEU |
|-------------|:--------:|:-----:|
| identity    |  47.35%  |  0.128  |
| most common |  66.50%  |  0.285  |
| linear reg. |  73.15%  |  0.389  |
| LSTM |  75.07%  |  -  |
| transformer |    -     |  -   |

_*Word-level seq2seq without attention did not produce any meaningful results._

Because the output of transformer and seq2seq models is of variable length, it may contain a different number or order of tokens than the input, so it is not possible to give a token-level accuracy score.

## Conclusion

It can be clearly seen that advanced deep learning models do not perform well in this task, with the main reason being limited training data.

However, a simple linear regression model demonstrates results very close to the strong baseline, and for Finnish even outperforms it.

The regressor learns to lemmatize not only very common words such as 'are', 'got', 'was' etc, but also learns certain relations (e.g. 'killed'-'kill', 'said'-'say'  'years'-'year').  The model also demonstrates capability to lemmatize unseen wordforms (e.g. 'submitted'-'submit', 'replacing'-'replace').  With adjustments and improvements that will enable use of context information, there is a good chance to beat both baselines.

## Future Work

For further research of advanced deep learning approaches' efficiency, it could be useful to experiment with the following models:
- word-level seq2seq with attention
- char-level seq2seq with attention
- DeepMind's relation networks

It could also be useful to slice the evaluation metrics by word frequency or length, to understand how the approaches differ.

## Datasets

For training and evaluation:

UD treebanks: [universaldependencies.org](http://universaldependencies.org/)

Word embeddings: [fasttext.cc/docs/en/pretrained-vectors.html](https://fasttext.cc/docs/en/pretrained-vectors.html)

## Related Work

[*A Neural Lemmatizer for Bengali*](https://pdfs.semanticscholar.org/12c6/1ee4f804d4007fc12cfd0d13ba260c051e48.pdf)  
Abhisek Chakrabarty et al. 

[*Context Sensitive Lemmatization Using Two Successive Bidirectional Gated Recurrent Networks*](http://www.aclweb.org/anthology/P17-1136)  
Abhisek Chakrabarty et al.

