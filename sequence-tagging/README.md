# Sequence Tagging

This chapter gives a detailed tutorial for the development of NLP components using APIs in ELIT.  We will develop components for two NLP tasks, [part-of-speech tagging](../theory-and-method/part-of-speech-tagging.md) and [named entity recognition](../theory-and-method/named-entity-recognition.md), that can be approached as a sequence tagging problem.  Given a sequence of tokens, the goal of **sequence tagging** is to label each token with a certain tag \(e.g., part-of-speech tag\) such that it generates a sequence of tags corresponding to those tokens.

An [NLP component](component.md) is an object that takes input text, makes predictions on the input text for a target task, and generates output inferred by those predictions.  An NLP component defines a [decoding strategy](decoding-strategy.md) that guides it to process through the input text and an [inference model](inference-model.md) that makes predictions for each state during the process.  The following sections first explain the component APIs in ELIT, then describe how to implement a part-of-speech tagger and a named entity recognizer that are simple yet give state-of-the-art performance.

{% hint style="info" %}
See the actual implementations of the [component APIs](https://github.com/elitcloud/elit/blob/master/elit/component.py), the [part-of-speech tagger](https://github.com/elitcloud/elit/blob/master/elit/pos.py), and the [named entity recognizer](https://github.com/elitcloud/elit/blob/master/elit/pos.py) described in this chapter.
{% endhint %}

## Terminology

* \*\*\*\*[**Token**](../theory-and-method/tokenization.md): a basic linguistic unit that has a meaning of its own.  Typical words \(e.g., `girl`, `pretty`\), abbreviations \(e.g., `Mr.`, `'s`\), as well as symbols \(e.g., `$`, `:-)`\) are considered individual tokens.
* \*\*\*\*[**Sentence**](../appendix/structure.md#sentence): a list of tokens.
* \*\*\*\*[**Document**](../appendix/structure.md#document): a list of sentences.

