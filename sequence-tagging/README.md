---
description: How to develop NLP components in ELIT.
---

# Sequence Tagging

This chapter gives a comprehensive tutorial for the development of NLP components using APIs in ELIT.  Towards the end of this chapter, we will develop components for two NLP tasks, [part-of-speech tagging](../nlp-tasks/part-of-speech-tagging.md) and [named entity recognition](../nlp-tasks/named-entity-recognition.md), that can be approached as a sequence tagging problem.  The objective of **sequence tagging** in NLP is to label each token in a sequence with a certain tag \(e.g., part-of-speech tag\) such that it takes a sequence of tokens and generates a sequence of tags corresponding to those tokens.  

In general, an NLP component defines a [decoding strategy](decoding-strategy.md) that guides it to process through the input text and an [inference model](inference-model.md) that makes predictions for each state during the process.  The following sections first explain how the APIs work in details then show how to implement a part-of-speech tagger and a named entity recognizer with the APIs, that are simple yet show state-of-the-art performance.

{% hint style="info" %}
* APIs: [component.py](https://github.com/elitcloud/elit/blob/master/elit/component.py).
* Part-of-Speech Tagger: [pos.py](https://github.com/elitcloud/elit/blob/master/elit/pos.py).
* Named Entity Recognizer: [ner.py](https://github.com/elitcloud/elit/blob/master/elit/pos.py).
{% endhint %}

## Terminologies

* \*\*\*\*[**Token**](../nlp-tasks/tokenization.md): a basic linguistic unit that has a meaning of its own.  Typical words \(e.g., `girl`, `pretty`\), abbreviations \(e.g., `Mr.`, `'s`\), as well as symbols \(e.g., `$`, `:-)`\) are considered individual tokens.
* \*\*\*\*[**Sentence**](../utilities/structure.md#sentence): a list of tokens.
* \*\*\*\*[**Document**](../utilities/structure.md#document): a list of sentences.

