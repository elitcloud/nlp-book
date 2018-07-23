---
description: Work in progress
---

# Document Matrix

A document can be represented as a matrix $$D \in \mathcal{R}^{n \times d}$$ where $$n$$ is the number of words and $$d$$ is the dimension of word embeddings such that the $$i$$'th row in $$D$$ represents the $$i$$'th word in the document.  Word embeddings can be retrieved by a [vector space model](../vector-space-model/).  Currently, ELIT provides APIs for three vector space models, [Word2Vec](../vector-space-model/word2vec.md), [FastText](../vector-space-model/fasttext.md), and [GloVe](../vector-space-model/glove.md).



