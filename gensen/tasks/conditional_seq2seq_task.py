import math

import torch
import torch.nn as nn

from gensen.tasks.behaviours.seq2seq import Seq2SeqBehaviour


class ConditionalGRU(nn.Module):
    """A Gated Recurrent Unit (GRU) cell with peepholes."""

    def __init__(self, input_dim, hidden_dim, dropout=0.):
        """Initialize params."""
        super(ConditionalGRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_weights = nn.Linear(self.input_dim, 3 * self.hidden_dim)
        self.hidden_weights = nn.Linear(self.hidden_dim, 2 * self.hidden_dim)
        self.peep_weights = nn.Linear(self.hidden_dim, 3 * self.hidden_dim)
        self.candidate_weights = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.reset_parameters()

    def reset_parameters(self):
        """Set params."""
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hidden, ctx):
        """Propagate input through the layer.

        inputs:
        input   - batch size x target sequence length  x embedding dimension
        hidden  - batch size x hidden dimension
        ctx     - batch size x hidden dimension

        returns: output, hidden
        output  - batch size x target sequence length  x hidden dimension
        hidden  - batch size x hidden dimension
        """
        def recurrence(input, hidden, ctx):
            """Recurrence helper."""
            input_gate = self.input_weights(input)
            hidden_gate = self.hidden_weights(hidden)
            peep_gate = self.peep_weights(ctx)

            i_r, i_i, i_n = input_gate.chunk(3, 1)
            h_r, h_i = hidden_gate.chunk(2, 1)
            p_r, p_i, p_n = peep_gate.chunk(3, 1)

            resetgate = torch.sigmoid(i_r + h_r + p_r)
            inputgate = torch.sigmoid(i_i + h_i + p_i)
            h_n = self.candidate_weights(resetgate * hidden)
            newgate = torch.tanh(i_n + h_n + p_n)
            hy = hidden + inputgate * (newgate - hidden)
            return hy

        input = input.transpose(0, 1)

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = recurrence(input[i], hidden, ctx)
            if isinstance(hidden, tuple):
                output.append(hidden[0])
            else:
                output.append(hidden)

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())
        output = output.transpose(0, 1)
        return output, hidden


class ConditionalSeq2SeqTask(Seq2SeqBehaviour):
    def __init__(self,
                 trg_vocab_size,
                 trg_emb_dim,
                 trg_hidden_dim,
                 trg_pad_token,
                 dropout,
                 device):
        """Initialize Conditional Seq2Seq Decoder Model."""
        Seq2SeqBehaviour.__init__(self, trg_vocab_size, trg_emb_dim,
                                  trg_hidden_dim, trg_pad_token)

        self.trg_vocab_size = trg_vocab_size
        self.trg_hidden_dim = trg_hidden_dim
        self.trg_pad_token = trg_pad_token
        self.trg_emb_dim = trg_emb_dim
        self.dropout = dropout
        self.device = device

        weight_mask = torch.ones(self.trg_vocab_size).to(self.device)
        weight_mask[self.trg_pad_token] = 0
        self.loss_criterion = nn.CrossEntropyLoss(weight=weight_mask).to(self.device)

        self.decoder = ConditionalGRU(
            self.trg_emb_dim,
            self.trg_hidden_dim,
            dropout=self.dropout
        )

        self.trg_embedding = nn.Embedding(
                self.trg_vocab_size,
                self.trg_emb_dim,
                self.trg_pad_token,
            )

        self.decoder2vocab = nn.Linear(self.trg_hidden_dim, self.trg_vocab_size)

        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        self.trg_embedding.weight.data.uniform_(-0.1, 0.1)
        self.decoder2vocab.bias.data.fill_(0)

    def forward(self, encoder_hidden_state, input_trg):
        """
        Propogate input through the decoder network.
        :param encoder_hidden_state: (batch_size x encoder_hidden_size) of encoder hidden state.
        :param input_trg: (batch size x target sequence length) of the input target.
        :return: predictions: (batch size x target sequence length x target vocab size)
                            of (pre-softmax over words).
        """
        trg_emb = self.trg_embedding(input_trg)
        trg_h, _ = self.decoder(
            trg_emb, encoder_hidden_state.squeeze(), encoder_hidden_state.squeeze()
        )

        trg_h_reshape = trg_h.contiguous().view(
            trg_h.size(0) * trg_h.size(1), trg_h.size(2)
        )

        decoder_logit = self.decoder2vocab(trg_h_reshape)
        predictions = decoder_logit.view(
            trg_h.size(0), trg_h.size(1), decoder_logit.size(1)
        )
        return predictions

    def loss(self, predictions, labels):
        """
        Calculate decoder network loss.
        :param predictions: (batch size x target sequence length x target vocab size)
                            of (pre-softmax over words).
        :param labels: (batch size x target sequence length x target vocab size)
                            of (pre-softmax over words).
        :return: backward loss: (scalar) of decoder loss.
        """
        loss = self.loss_criterion(
            predictions.contiguous().view(-1, predictions.size(2)),
            labels.contiguous().view(-1),
        )
        return loss
