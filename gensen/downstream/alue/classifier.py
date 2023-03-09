import os
import copy
import logging

import gensim
import torch
from torch.nn import CosineSimilarity
import numpy as np
from sklearn.linear_model import LinearRegression

from gensen.encoder import Encoder
from gensen.utils import normalize_text
from gensen.tokenization import SentencePieceTokenizer


class GenSenTextClassification(torch.nn.Module):
    def __init__(
        self,
        model_folder,
        pretrained_embedding,
        num_classes,
        dropout,
        task_type,
        clean_text,
        device,
        add_start_end_token,
        task_vocab=None,
        finetune=True,
        vocab_expand=False
    ):

        super(GenSenTextClassification, self).__init__()
        self.model_folder = model_folder
        self.task_type = task_type
        self.clean_text = clean_text
        self.device = device
        self.add_start_end = add_start_end_token
        self.finetune = finetune

        self.tokenizer = SentencePieceTokenizer(
            model_file=os.path.join(self.model_folder, 'tokenizer.model'))
        self._load_params()

        if vocab_expand:
            self.vocab_expansion(task_vocab, pretrained_embedding)

        if self.task_type in ["dual", "dual-regression"]:
            self.l2 = torch.nn.Dropout(dropout)
            self.l3 = torch.nn.Linear(4 * 2048, 512)
            self.l4 = torch.nn.ReLU()
            self.l5 = torch.nn.Linear(512, num_classes)
        elif self.task_type == "sts":
            self.l2 = CosineSimilarity(dim=1)
        else:
            self.l2 = torch.nn.Dropout(dropout)
            self.l3 = torch.nn.Linear(2048, num_classes)

    def _load_params(self):
        """Load pretrained params."""

        # Word to index mappings
        self.word2id, self.id2word = self.tokenizer.create_word2id_id2word()

        encoder_model = torch.load(
            os.path.join(self.model_folder, "model_params.model")
        )

        # Initialize encoders
        self.encoder = Encoder(
            vocab_size=encoder_model["src_embedding.weight"].size(0),
            embedding_dim=encoder_model["src_embedding.weight"].size(1),
            hidden_dim=encoder_model["encoder.weight_hh_l0"].size(1),
            num_layers=1 if len(encoder_model) < 10 else 2,
            trainable=self.finetune,
            device=self.device
        )

        # Load pretrained sentence encoder weights
        self.encoder.load_state_dict(encoder_model)

        # Store the initial word embeddings somewhere to re-train vocab expansion multiple times.
        self.model_embedding_matrix = copy.deepcopy(
            self.encoder.src_embedding.weight.data.numpy()
        )

        # Move encoder to GPU if self.device is GPU
        self.encoder = self.encoder.to(self.device)

    def first_expansion(self, pretrained_embedding):
        """Train linear regression model for the first time."""
        logging.info("Loading pretrained word embeddings")
        pretrained_embeddings = gensim.models.Word2Vec.load(pretrained_embedding)
        pretrained_embedding_matrix = pretrained_embeddings.wv.vectors
        pretrain_vocab = pretrained_embeddings.wv.index2word
        pretrain_word2id = {word: ind for ind, word in enumerate(pretrain_vocab)}

        # Set up training data for vocabulary expansion
        model_train = []
        pretrain_train = []

        for word in pretrain_word2id:
            if word in self.word2id:
                model_train.append(self.model_embedding_matrix[self.word2id[word]])
                pretrain_train.append(
                    pretrained_embedding_matrix[pretrain_word2id[word]]
                )

        logging.info("Training vocab expansion on model")
        lreg = LinearRegression()
        lreg.fit(pretrain_train, model_train)
        self.lreg = lreg
        self.pretrain_word2id = pretrain_word2id
        self.pretrained_embedding_matrix = pretrained_embedding_matrix

    def vocab_expansion(self, task_vocab, pretrained_embedding):
        """Expand the model's vocabulary with pretrained word embeddings."""
        task_word2id = {
            "<s>": 0,
            "<pad>": 1,
            "</s>": 2,
            "<unk>": 3,
        }

        task_id2word = {
            0: "<s>",
            1: "<pad>",
            2: "</s>",
            3: "<unk>",
        }

        ctr = 4
        for idx, word in enumerate(task_vocab):
            if word not in task_word2id:
                task_word2id[word] = ctr
                task_id2word[ctr] = word
                ctr += 1

        self.first_expansion(pretrained_embedding)

        # Expand vocabulary using the linear regression model
        task_embeddings = []
        oov_pretrain = 0
        oov_task = 0

        for word in task_id2word.values():
            if word in self.word2id:
                task_embeddings.append(self.model_embedding_matrix[self.word2id[word]])
            elif word in self.pretrain_word2id:
                oov_task += 1
                task_embeddings.append(
                    self.lreg.predict(
                        self.pretrained_embedding_matrix[
                            self.pretrain_word2id[word]
                        ].reshape(1, -1)
                    )
                    .squeeze()
                    .astype(np.float32)
                )
            else:
                oov_pretrain += 1
                oov_task += 1
                task_embeddings.append(
                    self.model_embedding_matrix[self.word2id["<unk>"]]
                )

        logging.info("Found %d task OOVs " % oov_task)
        logging.info("Found %d pretrain OOVs " % oov_pretrain)
        task_embeddings = np.stack(task_embeddings)
        self.encoder.set_pretrained_embeddings(task_embeddings)

    def save(self, path, epoch, state_dict):
        if not os.path.exists(path):
            os.makedirs(path)

        save_path = os.path.join(path, "finetuned_model_{}.model".format(epoch))
        logging.info('Saving the model to {}'.format(save_path))
        torch.save(
            {"model_state_dict": state_dict},
            open(save_path, "wb"),
        )

    def transform_minibatch(self, sentences, normalize, add_start_end, language="ar"):
        """Prepare minibatch."""
        if normalize:
            sentences = [
                normalize_text(sentence, lang=language) for sentence in sentences
            ]

        sentences = [self.tokenizer.tokenize(sentence) for sentence in sentences]

        if add_start_end:
            sentences = [
                    ["<s>"] + sentence + ["</s>"]
                    for sentence in sentences
            ]

        lens = [len(sentence) for sentence in sentences]
        sorted_idx = np.argsort(lens)[::-1]
        sorted_sentences = [sentences[idx] for idx in sorted_idx]
        rev = np.argsort(sorted_idx)
        sorted_lens = [len(sentence) for sentence in sorted_sentences]
        max_len = max(sorted_lens)

        sentences = [
            [self.tokenizer.token_to_id(w) for w in sentence]
            + [self.tokenizer.token_to_id("<pad>")] * (max_len - len(sentence))
            for sentence in sorted_sentences
        ]

        sentences = torch.tensor(sentences).to(self.device)
        rev = torch.tensor(rev).to(self.device)

        return {"sentences": sentences, "lengths": sorted_lens, "rev": rev}

    def forward(self, minibatch, pool, language="ar"):
        if self.task_type in ["dual", "dual-regression", "sts"]:
            minibatch1 = self.transform_minibatch(
                minibatch[0],
                normalize=self.clean_text,
                add_start_end=self.add_start_end,
                language=language,
            )
            minibatch2 = self.transform_minibatch(
                minibatch[1],
                normalize=self.clean_text,
                add_start_end=self.add_start_end,
                language=language,
            )

            sent1_h = self.encoder(
                input=minibatch1["sentences"],
                lengths=minibatch1["lengths"],
                return_all=False,
                pool=pool,
            )
            sent1_h = sent1_h.index_select(0, minibatch1["rev"])

            sent2_h = self.encoder(
                input=minibatch2["sentences"],
                lengths=minibatch2["lengths"],
                return_all=False,
                pool=pool,
            )
            sent2_h = sent2_h.index_select(0, minibatch2["rev"])

            if self.task_type == "sts":
                return self.l2(sent1_h, sent2_h)

            features = torch.cat(
                (sent1_h, sent2_h, torch.abs(sent1_h - sent2_h), sent1_h * sent2_h), 1
            )

            output = self.l2(features)
            output = self.l3(output)
            output = self.l4(output)
            output = self.l5(output)

        else:
            minibatch = self.transform_minibatch(
                minibatch, normalize=self.clean_text, add_start_end=self.add_start_end
            )
            h_t = self.encoder(
                input=minibatch["sentences"],
                lengths=minibatch["lengths"],
                return_all=False,
                pool=pool,
            )
            h_t = h_t.index_select(0, minibatch["rev"])
            output_2 = self.l2(h_t)
            output = self.l3(output_2)

        return output
