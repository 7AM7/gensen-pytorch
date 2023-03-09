import torch.nn as nn


class Seq2SeqBehaviour(nn.Module):
    """"
    Interface class for seq2seq decoders
    """
    def __init__(self, trg_vocab_size, trg_emb_dim, trg_hidden_dim, trg_pad_token):
        super(Seq2SeqBehaviour, self).__init__()

    def forward(self, encoder_hidden_state, input_target):
        raise NotImplementedError

    def loss(self, predictions, labels):
        raise NotImplementedError