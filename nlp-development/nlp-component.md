# NLP Component

An [NLP component](development.md#nlp-component) can be viewed as a function that 1\) takes input text, 2\) makes predictions on the input text for an NLP task \(e.g., part-of-speech tagging, dependency parsing\), and 3\) generates output inferred by those predictions.  Generally, an NLP component needs to define a [decoding strategy](development.md#decoding-strategies) that processes through the input text and an [inference model](development.md#inference-models) that makes predictions for each state during the decoding.

## Terminologies

* **Token**: a basic linguistic unit that has a full semantic of its own.  Typical words \(e.g., girl, pretty\), abbreviations \(e.g., Mr., 's\), as well as symbols \(e.g., $, :-\)\) are all considered individual tokens.  See [Tokenization](../nlp-tasks/tokenization.md) for more details.
* **Sentence**: a list of tokens.
* **Document**: a list of sentences.

## Decoding Strategies

Given a document that contains a list of sentences where each sentence comprises a list of tokens, a decoding strategy guides the component to visit every state and make predictions.  For example, a part-of-speech tagger visits every token in a document and predicts the part-of-speech tag of that token.  In this case, the decoding strategy determines which token to be visited first, how to move onto the next token, and when to stop.  At any step, the component needs to create a state providing necessary information for feature extraction such as the currently visited token, surrounding tokens, previously predicted part-of-speech tags, etc.

### NLPState

The abstract class `NLPState` provides a template to define a decoding strategy.

```python
class NLPState(abc.ABC):
    def __init__(self, document):
        self.document = document
        self.output = None

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def process(self, output):
        pass

    @abc.abstractmethod
    def has_next(self):
        return

    @property
    @abc.abstractmethod
    def labels(self):
        return

    @abc.abstractmethod
    def supply(self):
        pass

    @abc.abstractmethod
    def eval(self, metric):
        pass

    @property
    @abc.abstractmethod
    def x(self):
        pass

    @property
    @abc.abstractmethod
    def y(self):
        pass
```

* The NLPState takes a document, and creates a placeholder for self.output that stores the prediction results.
* The reset method sets it to the initial state.
* The process method takes the prediction output, applies it to the current state, and moves onto the next state.
* The has\_next method returns True if the next state is available; otherwise, False.
* The labels method returns the predicted labels for the input document inferred from the self\_output.
* The supply method saves the predicted labels as well as other information \(e.g., self.output\) to the input document.
* The eval method updates the evaluation metric, metric, using the gold-standard labels in the input document and the predicted labels from the self.labels method.
* The x method returns the feature vector \(or matrix\) for the current state, that is a numpy array.
* The y method returns the ID of the gold-standard label for the current state.

### ForwardState

Many NLP tasks such as part-of-speech tagging or named entity recognition use a simple decoding strategy known as the one-pass, left-to-right tagging. For these common tasks, ELIT provides another abstract class, ForwardState, that visits and makes a prediction for every word in a document from the top-left to the bottom-right.

```text

```

* In addition to the input document, ForwardState takes two more parameters, label\_map and zero\_output . The label\_map gives the mapping between class labels and their unique IDs and zero\_output is numpy.zeros\(num\_class\), where num\_class is the total number of classes, and used to create a placeholder for self.output \(we will see the use case of this in the part-of-speech tagging section\). It also initializes two pointers, sen\_id and tok\_id, that indicate which token in which sentence to be processed.
* The reset method sets the pointers to 0 , implying that it starts with the first token in the first sentence, and reinitializes the output to zero vectors.
* The process method applies the prediction result, output , to the current state, and moves onto the next token if exists; otherwise, to the first token in the following sentence.
* The has\_next method returns True if the current sentence is valid; otherwise, False.
* The labels method returns the predicted labels for all words in the document.

## Inference Models

## NLP Component

