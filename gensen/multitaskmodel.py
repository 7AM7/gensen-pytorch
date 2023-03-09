"""Parent model for Multitask Training."""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from gensen.tasks.tasks_factory import TasksFactory


class MultitaskModel(nn.Module):
    r"""A Multi Task Sequence to Sequence (Seq2Seq) model with GRUs.

    Ref: Multi-Task Sequence to Sequence Learning
    https://arxiv.org/pdf/1511.06114.pdf
    """

    def __init__(
        self, src_emb_dim, trg_emb_dim, src_vocab_size,
        trg_vocab_size, src_hidden_dim, trg_hidden_dim,
        src_pad_token, trg_pad_token, tasks, device,
        pooling_strategy='last', bidirectional=False,
        num_layers_src=1, dropout=0.
    ):
        """Initialize MultitaskModel Model."""
        super(MultitaskModel, self).__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.src_emb_dim = src_emb_dim
        self.trg_emb_dim = trg_emb_dim
        self.trg_hidden_dim = trg_hidden_dim
        self.bidirectional = bidirectional
        self.num_layers_src = num_layers_src
        self.dropout = dropout
        self.pooling_strategy = pooling_strategy
        self.tasks = tasks
        self.device = device
        self.num_directions = 2 if bidirectional else 1
        self.src_pad_token = src_pad_token
        self.trg_pad_token = trg_pad_token
        self.src_hidden_dim = src_hidden_dim // 2 \
            if self.bidirectional else src_hidden_dim

        """Initialize Encoder."""
        self.src_embedding = nn.Embedding(
            self.src_vocab_size,
            self.src_emb_dim,
            self.src_pad_token,
        )
        self.src_embedding.weight.data.uniform_(-0.1, 0.1)

        self.encoder = nn.GRU(
            self.src_emb_dim,
            self.src_hidden_dim,
            self.num_layers_src,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=self.dropout
        )

        self.enc_drp = nn.Dropout(self.dropout)

        """Initialize Decoders."""
        self.tasks_object = nn.ModuleDict()
        self.task_factory = TasksFactory(dropout=self.dropout, device=self.device)
        self.init_decoders()

    def init_decoders(self):
        """Initialize Decoders."""
        for task in self.tasks:
            task_args = None
            if task['tasktype'] == "seq2seq":
                task_args = {
                                "tasktype": task['tasktype'],
                                "trg_vocab_size": self.trg_vocab_size,
                                "trg_emb_dim": self.trg_emb_dim,
                                "trg_hidden_dim": self.trg_hidden_dim,
                                "trg_pad_token": self.trg_pad_token,
                 }

            elif task['tasktype'] in ["pair-classification", "classification"]:
                task_args = {
                                "tasktype": task['tasktype'],
                                "num_of_classes": task['num_of_classes'],
                                "src_hidden_dim": self.src_hidden_dim
                }

            if task_args is not None:
                self.tasks_object[task['taskname']] = self.task_factory.init_decoder(task_args)

    def set_pretrained_embeddings(self, embedding_matrix):
        """Set embedding weights."""
        if (
            embedding_matrix.shape[0] != self.src_embedding.weight.size(0) or
            embedding_matrix.shape[1] != self.src_embedding.weight.size(1)
        ):
            self.src_embedding = nn.Embedding(
                embedding_matrix.shape[0],
                embedding_matrix.shape[1]
            )
            self.src_vocab_size = embedding_matrix.shape[0]
            self.src_emb_dim = embedding_matrix.shape[1]

        try:
            self.src_embedding.weight.data.set_(
                torch.from_numpy(embedding_matrix)
            )
        except:
            self.src_embedding.weight.data.set_(
                torch.from_numpy(embedding_matrix).to(self.device)
            )

        self.src_embedding.to(self.device)

    def forward(self, minibatch, task_name):
        """
        Calculate logits and loss.
        :param task_name: (str) of the current task name.

        Seq2Seq:
        inputs: minibatch['seq2seq']
        input_src       - batch size x source sequence length
        src_lengths     - batch size (tensor)

        Pair-Classification:
        inputs: minibatch['pair-classification']
        sent1           - batch size x source sequence length
        sent2           - batch size x target sequence length
        sent1_lengths   - batch size (tensor)
        sent2_lengths   - batch size (tensor)

        Classification:
        inputs: minibatch['classification']
        sent           - batch size x source sequence length
        sent_lengths   - batch size (tensor)

        returns: (tuple) decoder loss, decoder logits
        """
        decoder = self.tasks_object[task_name]

        if minibatch['task_type'] == 'pair_classification':
            sent1_emb = self.src_embedding(minibatch['sent1'])
            sent2_emb = self.src_embedding(minibatch['sent2'])

            sent1_lengths = minibatch['sent1_lens'].data.view(-1).tolist()
            sent1_emb = pack_padded_sequence(
                sent1_emb, sent1_lengths, batch_first=True, enforce_sorted=False
            )
            sent1, sent1_h = self.encoder(sent1_emb)

            sent2_lengths = minibatch['sent2_lens'].data.view(-1).tolist()
            sent2_emb = pack_padded_sequence(
                sent2_emb, sent2_lengths, batch_first=True, enforce_sorted=False
            )
            sent2, sent2_h = self.encoder(sent2_emb)

            sent1_h = self.pooling(sent1, sent1_h)
            sent2_h = self.pooling(sent2, sent2_h)

            """Decoder forward/loss"""
            logits = decoder(sent1_h, sent2_h)
            loss = decoder.loss(logits, minibatch['labels'])

            return loss, logits

        elif minibatch['task_type'] == 'classification':
            sent_emb = self.src_embedding(minibatch['sent'])

            sent_lengths = minibatch['sent_lens'].data.view(-1).tolist()
            sent_emb = pack_padded_sequence(
                sent_emb, sent_lengths, batch_first=True, enforce_sorted=False
            )
            sent, sent_h = self.encoder(sent_emb)
            sent_h = self.pooling(sent, sent_h)

            """Decoder forward/loss"""
            logits = decoder(sent_h)
            loss = decoder.loss(logits, minibatch['labels'])

            return loss, logits

        elif minibatch['task_type'] == 'seq2seq':
            src_emb = self.src_embedding(minibatch['input_src'])
            src_lengths = minibatch['src_lens'].data.view(-1).tolist()
            src_emb = pack_padded_sequence(
                src_emb, src_lengths, batch_first=True, enforce_sorted=False
            )

            src_h, src_h_t = self.encoder(src_emb)
            h_t = self.pooling(src_h, src_h_t, )

            h_t = h_t.unsqueeze(0)
            h_t = self.enc_drp(h_t)

            """Decoder forward/loss"""
            logits = decoder(h_t, minibatch['input_trg'])
            loss = decoder.loss(logits, minibatch['output_trg'])

            return loss, logits

    def pooling(self, src_h, src_h_t):
        """
        Applying specific pooling strategy (max, mean, last) on the output hidden state.
        :param src_h: (batch_sum_seq_len X hidden_dim) of all hidden states.
        :param src_h_t: (1, max_seq_len X hidden_dim) of last hidden state.
        """
        if self.pooling_strategy == 'max':
            src_out, _ = pad_packed_sequence(src_h, padding_value=float('-inf'), batch_first=True)
            src_h_t = torch.max(src_out, 1)[0].squeeze().to(self.device)

        elif self.pooling_strategy == 'mean':
            # Apply mean pooling over all hidden states with ignore padding tokens (zeros)
            src_out, seq_lengths = pad_packed_sequence(src_h, batch_first=True)
            h_sum = torch.sum(src_out, axis=1).to(self.device)
            src_h_t = h_sum / seq_lengths.reshape(-1, 1).to(self.device)

        elif self.pooling_strategy == 'last':
            if self.bidirectional:
                src_h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1).to(self.device)
            else:
                src_h_t = src_h_t[-1].to(self.device)

        return src_h_t
