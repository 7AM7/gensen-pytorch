import torch.nn as nn


class PairedClassificationBehaviour(nn.Module):
    """"
    Interface class for paired classification decoders
    """
    def __init__(self, num_classes):
        super(PairedClassificationBehaviour, self).__init__()

    def forward(self, s1_hidden_state, s2_hidden_state):
        raise NotImplementedError

    def loss(self, predictions, labels):
        raise NotImplementedError