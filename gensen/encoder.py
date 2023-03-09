"""GenSen Encoder"""
import os
import copy
import logging

import h5py
from sklearn.linear_model import LinearRegression
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from gensen.utils import normalize_text
from gensen.tokenization import SentencePieceTokenizer


class Encoder(nn.Module):
    """GenSen Encoder."""

    def __init__(
            self, vocab_size, embedding_dim,
            hidden_dim, num_layers, device,
            trainable=False
    ):
        """Initialize params."""
        super(Encoder, self).__init__()
        self.src_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )

        self.encoder = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        if not trainable:
            for param in self.parameters():
                param.requires_grad = False

        self.device = device

    def set_pretrained_embeddings(self, embedding_matrix):
        """Set embedding weights."""
        if (
                embedding_matrix.shape[0] != self.src_embedding.weight.size(0) or
                embedding_matrix.shape[1] != self.src_embedding.weight.size(1)
        ):
            logging.info('''
                Warning pretrained embedding shape mismatch %d x %d
                expected %d x %d''' % (
                embedding_matrix.shape[0], embedding_matrix.shape[1],
                self.src_embedding.weight.size(0), self.src_embedding.weight.size(1)
            ))
            self.src_embedding = nn.Embedding(
                embedding_matrix.shape[0],
                embedding_matrix.shape[1]
            )
            self.src_vocab_size = embedding_matrix.shape[0]
            self.src_emb_dim = embedding_matrix.shape[1]

        with torch.no_grad():
            self.src_embedding.weight.set_(torch.from_numpy(embedding_matrix))

        self.src_embedding.to(self.device)

    def forward(self, input, lengths, return_all=False, pool='last', dropout=0.0):
        """Propagate input through the encoder."""
        embedding = self.src_embedding(input)
        src_emb = pack_padded_sequence(embedding, lengths, enforce_sorted=False, batch_first=True)
        h, h_t = self.encoder(src_emb)

        # Get hidden state via max-pooling or h_t
        if pool == 'last':
            h_t = torch.cat((h_t[-1], h_t[-2]), 1).to(self.device)

        elif pool == 'max':
            h_out, _ = pad_packed_sequence(h, padding_value=float('-inf'), batch_first=True)
            h_t = torch.max(h_out, 1)[0].squeeze().to(self.device)

        elif pool == 'mean':
            # Apply mean pooling over all hidden states with ignore padding tokens (zeros)
            h_out, seq_lengths = pad_packed_sequence(h, batch_first=True)
            h_sum = torch.sum(h_out, axis=1).to(self.device)
            h_t = h_sum / seq_lengths.reshape(-1, 1).to(self.device)

        else:
            raise ValueError("Pool %s is not valid " % pool)

        h_t = nn.Dropout(dropout)(h_t)

        # Return all or only the last hidden state
        if return_all:
            h, _ = pad_packed_sequence(h, batch_first=True)
            return h, h_t
        else:
            return h_t


class GenSenEnsemble(nn.Module):
    """Concat Gensen."""

    def __init__(self, *args, **kwargs):
        """A wrapper class for multiple GenSen models."""
        super(GenSenEnsemble, self).__init__()
        self.gensen_models = args

    def vocab_expansion(self, task_vocab, pretrained_embedding):
        """Expand the model's vocabulary with pretrained word embeddings."""
        for model in self.gensen_models:
            model.vocab_expansion(task_vocab, pretrained_embedding)

    def get_representation(
            self, sentences, pool='last',
            normalize=False, return_numpy=True, add_start_end=True
    ):
        """Get model representations."""
        representations = [
            model.get_representation(
                sentences, pool=pool, normalize=normalize,
                return_numpy=return_numpy, add_start_end=add_start_end
            )
            for model in self.gensen_models
        ]
        if return_numpy:
            return np.concatenate([x[0] for x in representations], axis=2), \
                   np.concatenate([x[1] for x in representations], axis=1)
        else:
            return torch.cat([x[0] for x in representations], 2), \
                   torch.cat([x[1] for x in representations], 1)


class GenSen(nn.Module):
    """GenSen Wrapper."""

    def __init__(self,
                 model_folder,
                 device,
                 trainable=False
    ):
        """Initialize params."""
        super(GenSen, self).__init__()
        self.model_folder = model_folder
        self.device = device
        self.trainable = trainable
        self.vocab_expanded = False

        self.tokenizer = SentencePieceTokenizer(model_file=os.path.join(self.model_folder, 'tokenizer.model'))
        self.tokenize = lambda text: self.tokenizer.tokenize(text)
        
        self._load_params()

    def _load_params(self):
        """Load pretrained params."""
        # Word to index mappings
        self.word2id, self.id2word = self.tokenizer.create_word2id_id2word()
        self.task_word2id = self.word2id
        self.id2word = self.id2word

        encoder_model = torch.load(os.path.join(
            self.model_folder,
            'model_params.model'
        ))

        # Initialize encoders
        self.encoder = Encoder(
            vocab_size=encoder_model['src_embedding.weight'].size(0),
            embedding_dim=encoder_model['src_embedding.weight'].size(1),
            hidden_dim=encoder_model['encoder.weight_hh_l0'].size(1),
            num_layers=1 if len(encoder_model) < 10 else 2,
            trainable=self.trainable,
            device=self.device
        )

        # Load pretrained sentence encoder weights
        self.encoder.load_state_dict(encoder_model)

        # Set encoders in eval model.
        self.encoder.eval()

        # Store the initial word embeddings somewhere to re-train vocab expansion multiple times.
        self.model_embedding_matrix = \
            copy.deepcopy(self.encoder.src_embedding.weight.data.cpu().numpy())

        # Move encoder to device
        self.encoder = self.encoder.to(self.device)

    def first_expansion(self, pretrained_embedding):
        """Train linear regression model for the first time."""
        # Read pre-trained word embedding h5 file
        logging.info('Loading pretrained word embeddings')
        pretrained_embeddings = h5py.File(pretrained_embedding)
        pretrained_embedding_matrix = pretrained_embeddings['embedding'].value
        pretrain_vocab = \
            pretrained_embeddings['words_flatten'].value.split('\n')
        pretrain_word2id = {
            word: ind for ind, word in enumerate(pretrain_vocab)
        }

        # Set up training data for vocabulary expansion
        model_train = []
        pretrain_train = []

        for word in pretrain_word2id:
            if word in self.word2id:
                model_train.append(
                    self.model_embedding_matrix[self.word2id[word]]
                )
                pretrain_train.append(
                    pretrained_embedding_matrix[pretrain_word2id[word]]
                )

        logging.info('Training vocab expansion on model')
        lreg = LinearRegression()
        lreg.fit(pretrain_train, model_train)
        self.lreg = lreg
        self.pretrain_word2id = pretrain_word2id
        self.pretrained_embedding_matrix = pretrained_embedding_matrix

    def vocab_expansion(self, task_vocab, pretrained_embedding):
        """Expand the model's vocabulary with pretrained word embeddings."""
        self.task_word2id = {
            '<s>': 0,
            '<pad>': 1,
            '</s>': 2,
            '<unk>': 3,
        }

        self.task_id2word = {
            0: '<s>',
            1: '<pad>',
            2: '</s>',
            3: '<unk>',
        }

        ctr = 4
        for idx, word in enumerate(task_vocab):
            if word not in self.task_word2id:
                self.task_word2id[word] = ctr
                self.task_id2word[ctr] = word
                ctr += 1

        if not self.vocab_expanded:
            self.first_expansion(pretrained_embedding)

        # Expand vocabulary using the linear regression model
        task_embeddings = []
        oov_pretrain = 0
        oov_task = 0

        for word in self.task_id2word.values():
            if word in self.word2id:
                task_embeddings.append(
                    self.model_embedding_matrix[self.word2id[word]]
                )
            elif word in self.pretrain_word2id:
                oov_task += 1
                task_embeddings.append(self.lreg.predict(
                    self.pretrained_embedding_matrix[self.pretrain_word2id[word]].reshape(1, -1)
                ).squeeze().astype(np.float32))
            else:
                oov_pretrain += 1
                oov_task += 1
                task_embeddings.append(
                    self.model_embedding_matrix[self.word2id['<unk>']]
                )

        logging.info('Found %d task OOVs ' % oov_task)
        logging.info('Found %d pretrain OOVs ' % oov_pretrain)
        task_embeddings = np.stack(task_embeddings)
        self.encoder.set_pretrained_embeddings(task_embeddings)
        self.vocab_expanded = True

        # Move encoder to device
        self.encoder = self.encoder.to(self.device)

    def get_minibatch(self, sentences, normalize=False, add_start_end=True, language='ar'):
        """Prepare minibatch."""
        if normalize:
            sentences = [normalize_text(sentence, lang=language) for sentence in sentences]

        sentences = [self.tokenize(sentence) for sentence in sentences]

        if add_start_end:
            sentences = [["<s>"] + sentence + ["</s>"] for sentence in sentences]

        lens = [len(sentence) for sentence in sentences]
        sorted_idx = np.argsort(lens)[::-1]
        sorted_sentences = [sentences[idx] for idx in sorted_idx]
        rev = np.argsort(sorted_idx)
        sorted_lens = [len(sentence) for sentence in sorted_sentences]
        max_len = max(sorted_lens)

        sentences = [
            [self.task_word2id[w] if w in self.task_word2id else self.task_word2id['<unk>'] for w in sentence] +
            [self.task_word2id['<pad>']] * (max_len - len(sentence))
            for sentence in sorted_sentences
        ]

        sentences = torch.tensor(sentences).to(self.device)
        rev = torch.tensor(rev).to(self.device)
        lengths = sorted_lens

        return {
            'sentences': sentences,
            'lengths': lengths,
            'rev': rev
        }

    def get_representation(
        self, sentences, pool='last', language='ar',
        normalize=False, return_numpy=True, add_start_end=True, dropout=0.0
    ):
        """Get model representations."""
        minibatch = self.get_minibatch(
            sentences, normalize=normalize, add_start_end=add_start_end, language=language
        )
        h, h_t = self.encoder(
            input=minibatch['sentences'], lengths=minibatch['lengths'],
            return_all=True, pool=pool, dropout=dropout
        )
        h = h.index_select(0, minibatch['rev'])
        h_t = h_t.index_select(0, minibatch['rev'])
        if return_numpy:
            return h.data.cpu().numpy(), h_t.data.cpu().numpy()
        else:
            return h, h_t


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Sentences need to be lower-cased.
    sentences = [
        'hello world .',
        'the quick brown fox jumped over the lazy dog .',
        'this is a sentence .'
    ]

    gensen_1 = GenSen(
        model_folder='/data/GenSen-Ar/models/encoders',
        device=device
    )
    reps_h, reps_h_t = gensen_1.get_representation(
        sentences, pool='last', return_numpy=True
    )
    # reps_h contains the hidden states for all words in all sentences
    # (padded to the max length of sentences) (batch_size x seq_len x 2048)
    # reps_h_t contains only the last hidden state for all sentences in the minibatch (batch_size x 2048)
    logging.info(reps_h.shape, reps_h_t.shape)

    gensen_2 = GenSen(
        model_folder='/data/GenSen-Ar/models/encoders',
        device=device
    )
    gensen = GenSenEnsemble(gensen_1, gensen_2)
    reps_h, reps_h_t = gensen.get_representation(
        sentences, pool='last', return_numpy=True
    )
    # reps_h contains the hidden states for all words in all sentences
    # (padded to the max length of sentences) (batch_size x seq_len x 2048)
    # reps_h_t contains only the last hidden state for all sentences in the minibatch (batch_size x 4096)
    logging.info(reps_h.shape, reps_h_t.shape)
