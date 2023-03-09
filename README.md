# GenSen

Learning General Purpose Distributed Sentence Representations via Large Scale Multi-task Learning

Sandeep Subramanian, Adam Trischler, Yoshua Bengio & Christopher Pal

ICLR 2018


### About

GenSen is a technique to learn general purpose, fixed-length representations of sentences via multi-task training. These representations are useful for transfer and low-resource learning. For details please refer to ICLR [paper](https://openreview.net/forum?id=B18WgG-CZ&noteId=B18WgG-CZ).

### Usage


#### Pre-training with GenSen from scratch

##### 1. Training SentencePiece tokenizer 
For building the vocabulary process and tokenization process, we are using [sentencepiece](https://github.com/google/sentencepiece) model.

You can train the sentencepiece model by running, based on datasets available in `config.json` under `tasks` key.

```bash
python train_sentencepiece.py
```
*Note*: you can replace the configurations of sentencepiece model in the provided `config.json` under `vocab` key.

##### 2. Preparing datasets
Use `tokenization.py` to create a pre-training dataset from a dump of raw text. It has the following arguments:

* `--model_path`: File defining the sentencepiece tokenizer.
* `--num_workers`: If >1 parallelize across multiple processes (4 by default).

Usage example:
```bash
python tokenization.py \
--model_path=./tokenizer.model \
--num_workers=6
```
*Note*: this script reading the data files from the provided `config.json`.

##### 3. Training GenSen Multi-Task model

To train a model from scratch, simply run `train.py` with an appropriate JSON config file. An example config is provided in `config.json`.

```bash
python train.py
```


##### 4. Creating a GenSen encoder from a trained multi-task model

Once you have a trained model, we can throw away all of the decoders and just retain the encoder used to compute sentence representations.

You can do this by running

```bash
python extract_encoder.py -t <path_to_trained_model> -s <path_to_save_encoder>
```

Once you have done this, you can load this model just like any of the pre-trained models by specifying the encoder and tokenizer folder as `path_to_encoder`.

```python
from encoder import GenSen

your_gensen = GenSen(
    model_folder='encoder_and_tokenizer_directory',
    trainable=True,    # (for fine-tuning the encoder)
    use_cuda=True
)
```

#### Using a pre-trained model to extract sentence representations.

You can use our pre-trained models to extract the last hidden state or all hidden states of our multi-task GRU. Additionally, you can concatenate the output of multiple models to replicate the numbers in our paper.

```python
from encoder import GenSen, GenSenEnsemble

sentences = [
        'hello world .',
        'the quick brown fox jumped over the lazy dog .',
        'this is a sentence .'
    ]

gensen_1 = GenSen(
    model_folder='/path/to/pretrained_model_1',
    trainable=True,    # (for fine-tuning the encoder)
    use_cuda=True
)
reps_h, reps_h_t = gensen_1.get_representation(
    sentences, pool='last', language='ar', normalize=True, add_start_end=True, return_numpy=True
)
print(reps_h.shape, reps_h_t.shape)
```

##### `get_representation` parameters:
- `sentences`, which should be a list of strings. If your strings are not pre-tokenized, then set `normalize=True` to normalize and tokenize text before computing representations.
- `pool` specifies the pooling operation to get the sentence representation from the encoder hidden states (`last` or `max`).
- `language` specifies the language of the passed sentences, to choose the appropriate tokenization and normalization functions (`ar`, `en` or `bi` in case of bilingual sentences, uses `langdetect` to detect langauge, which is slower than setting `language` beforehand).
- `normalize` whether to normalize sentences before encoding (recommended).
- `add_start_end` whether to add `<s>` and `</s>` tokens at the start and end of each sentence before encoding (recommended). 

##### `get_representation` output:
- `reps_h` (batch_size x seq_len x dim_src) contains the hidden states for all words in all sentences (padded to the max length of sentences)
- `reps_h_t` (batch_size x dim_src) contains only the last hidden state for all sentences in the minibatch 

GenSen will return the output of a single model. You can concatenate the output of multiple models by creating a GenSen instance with multiple GenSen instances, as follows:

```python
gensen_2 = GenSen(
    model_folder='/path/to/pretrained_model_2',
    trainable=True,    # (for fine-tuning the encoder)
    use_cuda=True
)
gensen = GenSenEnsemble(gensen_1, gensen_2)
reps_h, reps_h_t = gensen.get_representation(
    sentences, pool='last', language='ar', normalize=True, add_start_end=True, return_numpy=True
)
```

1) `reps_h` (batch_size x seq_len x dim_src * 2) contains the hidden states for all words in all sentences (padded to the max length of sentences)
2) `reps_h_t` (batch_size x dim_src * 2) contains only the last hidden state for all sentences in the minibatch 

The model will produce a fixed-length vector for each sentence as well as the hidden states corresponding to each word in every sentence (padded to max sentence length). You can also return a numpy array instead of a `torch.FloatTensor` by setting `return_numpy=True`. 

##### Vocabulary Expansion

If you have a specific domain for which you want to compute representations, you can call `vocab_expansion` on instances of the GenSen or GenSenEnsemble class simply by `gensen.vocab_expansion(vocab, pretrained_embeddings_path)` where vocab is a list of unique words in the new domain. This will learn a linear mapping from the provided pretrained embeddings (which have a significantly larger vocabulary) provided to the space of gensen's word vectors. For an example of how this is used in an actual setting, please refer to `gensen_senteval.py`.


### Downstream Benchmarking & Transfer Learning

#### English

We use the [SentEval](https://github.com/facebookresearch/SentEval) toolkit to run most of our transfer learning experiments.
 To replicate these numbers, clone their repository and follow setup instructions. Once complete, copy `gensen/downstream/senteval.py` and `gensen/encoder.py` into their examples folder and run the following commands to reproduce different rows in Table 2 of our paper. Note: Please set the path to the pretrained glove embeddings (`glove.840B.300d.h5`) and model folder as appropriate.

#### Arabic

We use the Arabic Language Understanding Evaluation (ALUE) benchmark to compare our pretrained encoders to other SOTA models (e.g. BERT).

Use `gensen/downstream/alue_finetuning.py` to run ALUE benchmark. It has the following arguments:

* `--model_dir`: Folder defining the model encoder and the sentencepiece tokenizer.
* `--pretrained_embeddings`: (Optional) If Not None, use vocab expansion technique (None by default).
* `--finetuning_output`: (Optional) If Not None, Folder path to save the fine-tuning models (None by default).
* `--tasks`: (Optional) List of ALUE benchmark tasks (All tasks by default).
* `--seed`: (Optional) Random seed value (90 by default).
* `--print_summary`: (Optional) whether print summary report or not (True by default).
* `--report_path`: (Optional) If not None, File path to save the summary report as excel sheet (None by default). 

Usage example:
```bash
python -m gensen.downstream.alue_finetuning \
--model_dir=<path_of_encoder_tokenizer> \
--pretrained_embeddings=<path_of_pretrained_embeddings> \
--finetuning_output=<path_to_save_finetuning_models> \
--tasks=Diagnostic,IDAT,Madar \
--seed=32 \
--print_summary=True \
--report_path=<path_to_save_report>

```


#### Bi-Lingual

We assess the pretrained GenSen encoder bilingual capabilities using two tasks that we have built:
 __Bilingual Natural Language Inference__ and __Bilingual Semantic Sentence Similarity__. 
### Reference

```
@article{subramanian2018learning,
title={Learning general purpose distributed sentence representations via large scale multi-task learning},
author={Subramanian, Sandeep and Trischler, Adam and Bengio, Yoshua and Pal, Christopher J},
journal={arXiv preprint arXiv:1804.00079},
year={2018}
}
```
