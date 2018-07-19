---
description: Development of NLP components in ELIT.
---

# NLP Components

An [NLP component](./#nlp-component) can be viewed as a function that 1\) takes input text, 2\) makes predictions on the input text for an NLP task \(e.g., [part-of-speech tagging](../nlp-tasks/part-of-speech-tagging.md), [dependency parsing](../nlp-tasks/dependency-parsing.md)\), and 3\) generates output inferred by those predictions.  Generally, an NLP component needs to define a [decoding strategy](./#decoding-strategies) that processes through the input text and an [inference model](./#inference-models) that makes predictions for each state during the decoding.

## Terminologies

* **Token**: a basic linguistic unit that has a meaning of its own.  Typical words \(e.g., girl, pretty\), abbreviations \(e.g., Mr., 's\), as well as symbols \(e.g., $, :-\)\) are considered individual tokens.  See [tokenization](../nlp-tasks/tokenization.md) for more details.
* **Sentence**: a list of tokens.
* **Document**: a list of sentences.

## Decoding Strategies

Given a document, a decoding strategy guides the component to visit every state so it can make predictions for the task.  For example, a part-of-speech tagger visits every token in a document and predicts the part-of-speech tag of that token.  In this case, the decoding strategy determines which token to be visited first, how to move onto the next token after making the prediction, and when to stop.  At any step, the component needs to create a state that contains information such as the currently visited token, its surrounding tokens, and previously predicted part-of-speech tags to extract features for the prediction.

### NLPState

[`NLPState`](https://github.com/elitcloud/elit/blob/master/elit/component.py) provides a generic template to define a decoding strategy.

```python
class NLPState(abc.ABC):
    def __init__(self, document: elit.util.Document):
        self.document = document
        self.outputs = None

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def process(self, output: numpy.array):
        pass

    @abc.abstractmethod
    def has_next(self) -> bool:
        return

    @abc.abstractmethod
    def finalize(self):
        pass

    @abc.abstractmethod
    def eval(self, metric: elit.util.EvalMetric):
        pass

    @property
    @abc.abstractmethod
    def labels(self):
        return

    @property
    @abc.abstractmethod
    def x(self) -> numpy.array:
        return

    @property
    @abc.abstractmethod
    def y(self) -> int:
        return
```

* `__init__()` takes an input document and sets the outputs to `None`.
* `reset()` resets to the initial state.
* `process()` applies predicted output to the current state, and moves onto the next state.
* `has_next()` returns `True` if there is a next state to be processed; otherwise, `False`.
* `finalize()` saves all predicted outputs \(`self.outputs`\) and inferred labels \(`self.labels`\) to the input document \(`self.document`\).
* `eval()` updates the evaluation metric by comparing the gold-standard labels and the inferred labels \(`self.labels`\).
* `labels` returns the labels for the input document inferred from `self.outputs`.
* `x` returns the feature vector \(or matrix\) extracted from the current state.
* `y` returns the class ID of the gold-standard label for the current state \(training only\).

### ForwardState

ForwardState defines the one-pass, left-to-right decoding strategy.  Many NLP tasks such as part-of-speech tagging or named entity recognition use a simple decoding strategy known as the one-pass, left-to-right tagging. For these common tasks, ELIT provides another abstract class, ForwardState, that visits and makes a prediction for every word in a document from the top-left to the bottom-right.

```python
class ForwardState(NLPState):
    def __init__(self, 
                 document: elit.util.Document,
                 label_map: elit.lexicon.LabelMap,
                 zero_output: numpy.array,
                 key: str, key_out: str=None):
        super().__init__(document)
        
        self.label_map = label_map
        self.zero_output = zero_output
        
        self.key = key
        self.key_out = key_out if key_out else key + '-out'

        self.sen_id = 0
        self.tok_id = 0
        self.reset()
```

* document
* * * * In addition to the input document, ForwardState takes two more parameters, label\_map and zero\_output . The label\_map gives the mapping between class labels and their unique IDs and zero\_output is numpy.zeros\(num\_class\), where num\_class is the total number of classes, and used to create a placeholder for self.output \(we will see the use case of this in the part-of-speech tagging section\). It also initializes two pointers, sen\_id and tok\_id, that indicate which token in which sentence to be processed.
* The reset method sets the pointers to 0 , implying that it starts with the first token in the first sentence, and reinitializes the output to zero vectors.
* The process method applies the prediction result, output , to the current state, and moves onto the next token if exists; otherwise, to the first token in the following sentence.
* The has\_next method returns True if the current sentence is valid; otherwise, False.
* The labels method returns the predicted labels for all words in the document.

## Inference Models

## NLP Component

