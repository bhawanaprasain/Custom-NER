# Custom-NER
Custom NER trained using CoNLL data

Spacy incorporates optimized implementations for a lot of the common NLP tasks including Named Entity Recognition. Spacy has pipeline like `tok2vec`, `tagger`, `parser`, `ner`, `lemmatizer` etc. However we can build our own custom NER as per our requirement.

This repository shows the  steps to train a custom NER using  BERT Transformer.

## BERT Transformer
Bidirectional Encoder Representations from Transformers (BERT) is a transformer-based machine learning technique for natural language processing (NLP). BERT applies bidirectional training of Transformer, a popular attention model. Unlike directional models, which read the text input sequentially either just from left-to-right or right-to-left, BERT allows the model to learn the context of a word based on all of its surroundings i.e both from left and right of the word.

## About the dataset
[CoNLL data](https://deepai.org/dataset/conll-2003-english) contain data in IOB format. The data present cannot be directly used for training. We need data to be in spacy format. Before converting data into spacy format, we need to convert it to JSON format.
So general steps required in data preprocessing are:
* Convert IOB format data  to SpaCy JSON Format
* Convert thus obtained JSON data to spacy format.

In the case where data annotation is not done, annotation tools can be used.

## Training custom NER with CoNLL data

Start a Google colab session with GPU as hardware accelerator.

Install spacy using command

` pip3 install spacy==3.0.6 `

We will be using a config file `config.cfg`  from SpaCy 3. In spacy [training section](https://spacy.io/usage/training) we can choose options for:
* Language
* Components 
* Hardware
* Optimization

For this NER, English language and GPU is chosen. After chosing the options, we get a template of config file :

```
  # This is an auto-generated partial config. To use it with 'spacy train'
  # you can run spacy init fill-config to auto-fill all default settings:
  # python -m spacy init fill-config ./base_config.cfg ./config.cfg
  [paths]
  train = null
  dev = null
  vectors = null
  [system]
  gpu_allocator = "pytorch"

  [nlp]
  lang = "en"
  pipeline = ["transformer","ner"]
  batch_size = 128

  [components]

  [components.transformer]
  factory = "transformer"

  [components.transformer.model]
  @architectures = "spacy-transformers.TransformerModel.v3"
  name = "roberta-base"
  tokenizer_config = {"use_fast": true}

  [components.transformer.model.get_spans]
  @span_getters = "spacy-transformers.strided_spans.v1"
  window = 128
  stride = 96

  [components.ner]
  factory = "ner"

  [components.ner.model]
  @architectures = "spacy.TransitionBasedParser.v2"
  state_type = "ner"
  extra_state_tokens = false
  hidden_width = 64
  maxout_pieces = 2
  use_upper = false
  nO = null

  [components.ner.model.tok2vec]
  @architectures = "spacy-transformers.TransformerListener.v1"
  grad_factor = 1.0

  [components.ner.model.tok2vec.pooling]
  @layers = "reduce_mean.v1"

  [corpora]

  [corpora.train]
  @readers = "spacy.Corpus.v1"
  path = ${paths.train}
  max_length = 0

  [corpora.dev]
  @readers = "spacy.Corpus.v1"
  path = ${paths.dev}
  max_length = 0

  [training]
  accumulate_gradient = 3
  dev_corpus = "corpora.dev"
  train_corpus = "corpora.train"

  [training.optimizer]
  @optimizers = "Adam.v1"

  [training.optimizer.learn_rate]
  @schedules = "warmup_linear.v1"
  warmup_steps = 250
  total_steps = 20000
  initial_rate = 5e-5

  [training.batcher]
  @batchers = "spacy.batch_by_padded.v1"
  discard_oversize = true
  size = 2000
  buffer = 256

  [initialize]
  vectors = ${paths.vectors}
  ```
  
  

  In the next step we need to autofill the config file with command:
  
  ```!python -m spacy init fill-config ./yourconfigfolder/config.cfg ./yourconfigfolder/config_train_spacy.cfg  ```
  
Now we can start training with the following command:


```!python -m spacy train -g 0 ./yourconfigfolder/config_train_spacy.cfg — output ./outputfolder/ ```


You can obtain the  model with name `model-best` once training is over. You can load the model with spacy using command:


```custom_ner = spacy.load("/outputfolder/model-best") ```

### Usage example:

```
doc = custom_ner(' Prime Minister Costas Simitis is going to make an official announcement ')
```


```
for ent in doc.ents:
    if ent.label_[-3:] == "PER":
      print(ent,ent.label_)
```
      
#### Output

`Costas B-PER
 Simitis I-PER`








  
  
  
