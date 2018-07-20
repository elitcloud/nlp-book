---
description: Development of NLP components in ELIT.
---

# NLP Components

An [NLP component](./#component) can be viewed as a function that 1\) takes input text, 2\) makes predictions on the input text for an NLP task \(e.g., [part-of-speech tagging](../nlp-tasks/part-of-speech-tagging.md), [dependency parsing](../nlp-tasks/dependency-parsing.md)\), and 3\) generates output inferred by those predictions.  Generally, an NLP component needs to define a [decoding strategy](./#decoding-strategy) that processes through the input text and an [inference model](./#inference-model) that makes predictions for each state during the decoding.

{% hint style="info" %}
See [component.py](https://github.com/elitcloud/elit/blob/master/elit/component.py) for the implementations of classes and methods described in this section.
{% endhint %}

## Terminologies

* \*\*\*\*[**Token**](../nlp-tasks/tokenization.md): a basic linguistic unit that has a meaning of its own.  Typical words \(e.g., girl, pretty\), abbreviations \(e.g., Mr., 's\), as well as symbols \(e.g., $, :-\)\) are considered individual tokens.
* \*\*\*\*[**Sentence**](../utilities/structure.md#sentence): a list of tokens.
* \*\*\*\*[**Document**](../utilities/structure.md#document): a list of sentences.

## Decoding Strategy

Given a document, a decoding strategy guides the component to visit every state so it can make predictions for the task.  For example, a [part-of-speech tagger](part-of-speech-tagger.md) visits every token in a document and predicts the part-of-speech tag of that token.  In this case, the decoding strategy determines which token to be visited first, how to move onto the next token after making the prediction, and when to stop.  At any step, the component needs to create a state that contains information such as the currently visited token, its surrounding tokens, and previously predicted part-of-speech tags to extract features for the prediction.

### NLPState

`NLPState` provides a generic template to define a decoding strategy.  We will see an example of how this class is inherited to define the [one-pass left-to-right decoding strategy](./#oplrstate) in the following section.

```python
class NLPState(abc.ABC):
    def __init__(self, document: elit.struct.Document):
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
* `finalize()` saves all predicted outputs, `self.outputs`, and inferred labels, `self.labels`, to the input document, `self.document`.
* `eval()` updates the evaluation metric by comparing the gold-standard labels and the inferred labels, `self.labels`.
* `labels()` returns the labels for the input document inferred from `self.outputs`.
* `x()` returns the feature vector \(or matrix\) extracted from the current state.
* `y()` returns the class ID of the gold-standard label for the current state \(training only\).

### OPLRState

`OPLRState` defines the one-pass left-to-right decoding strategy \(OPLR\), one of the most commonly used decoding strategies in NLP.  Given a document, it visits the first token in the first document, moves onto the next token in the same document if exists; otherwise, the first token in the next document, and so on.  This strategy is adapted by popular tasks such as [part-of-speech tagging](../nlp-tasks/part-of-speech-tagging.md) or [named entity recognition](../nlp-tasks/named-entity-recognition.md).

```python
class OPLRState(NLPState):
    def __init__(self, 
                 document: elit.struct.Document,
                 label_map: elit.lexicon.LabelMap,
                 zero_output: numpy.array,
                 key: str, key_out: str=None):
        super().__init__(document)
        
        self.label_map = label_map
        self.zero_output = zero_output
        
        self.key = key
        self.key_out = key_out if key_out else key + '-out'

        self.sen_id = 0    # sentence ID
        self.tok_id = 0    # token ID
        self.reset()
```

* `document` is an input document.
* [`label_map`](../utilities/lexicons.md#labelmap) collects class labels during training and maps them to unique IDs.
* `zero_output` is a vector whose dimension is the number of class labels, where all values are `0`.  This is used to initialize `self.outputs`.
* `key` is the key to each sentence in the input document where the inferred labels are  to be saved. 
* `key_out` is the key to each sentence in the input document where the predicted outputs are to be saved.
* `self.sen_id` and `self.tok_id` are initialized to `0` \(L15-16\), indicating that the current state is set to the first token of the first sentence in the input document.
*  `reset()` is called to set all member instances to the initial state \(L17\).

The abstract methods [`reset()`](./#reset), [`process()`](./#process), [`has_next()`](./#has_next), [`finalize()`](./#finalize), [`labels()`](./#labels), and [`y()`](./#y) from `NLPState` are defined in `OPLRState` whereas `eval()` and `x()` are not, which are rather task-specific.  Thus, `OPLRState` is still an abstract class that gets inherited by components such as a [part-of-speech tagger](part-of-speech-tagger.md) or a [named entity recognizer](named-entity-recognizer.md). 

#### reset\(\)

```python
def reset(self):
    if self.outputs is None:
        self.outputs = [[self.zero_output] * len(s) for s in self.document]
    else:
        for i, s in enumerate(self.document):
            self.outputs[i] = [self.zero_output] * len(s)

    self.sen_id = 0
    self.tok_id = 0
```

When `reset()` is called for the first time, it creates a 3D matrix $$M \in \mathcal{R}^{x \times y \times z}$$, where _`x`_ is the number of class labels, _`y`_ is the number of tokens in each sentence, and `z` is the number of sentences \(L2-3\).  For later uses, it resets each output to the 2D zero matrix \(L4-6\).  Notice that the memory usage of `self.outputs` is low since it does not actually create zero vectors but just points to `zero_output`.  It is important to ensure an efficient memory allocation because thousands and millions of states may be created to process big data.

#### process\(\)

```python
def process(self, output):
    # apply the output to the current state
    self.outputs[self.sen_id][self.tok_id] = output

    # move onto the next state
    self.tok_id += 1
    if self.tok_id == len(self.document.get_sentence(self.sen_id)):
        self.sen_id += 1
        self.tok_id = 0
```

#### has\_next\(\)

```python
def has_next(self):
    return 0 <= self.sen_id < len(self.document)
```

#### finalize\(\)

```python
def finalize(self):
    """
    Finalizes the predicted labels and the prediction scores for all tokens in the input document.
    """
    for i, labels in enumerate(self.labels):
        d = self.document.get_sentence(i)
        d[self.key] = labels
        d[self.key_out] = self.outputs[i]
```

#### labels\(\)

```python
@property
def labels(self):
    """
    :rtype: list of (list of str)
    """
    def aux(scores):
        if size < len(scores): scores = scores[:size]
        return self.label_map.get(np.argmax(scores))

    size = len(self.label_map)
    return [[aux(o) for o in output] for output in self.outputs]
```

#### 

#### y\(\)

```python
@property
def y(self):
    label = self.document.get_sentence(self.sen_id)[self.key][self.tok_id]
    return self.label_map.add(label)
```



## Inference Model

### FFNNModel

Feed-Forward Neural Network \(FFNN\)

### LSTMModel

_To be filled._

## Component

### NLPComponent

### TokenTagger

