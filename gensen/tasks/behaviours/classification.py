import torch.nn as nn


class ClassificationBehaviour(nn.Module):
    """"
    Interface class for classification decoders
    """
    def __init__(self, num_classes):
        super(ClassificationBehaviour, self).__init__()

    def forward(self, encoder_hidden_state):
        raise NotImplementedError

    def loss(self, predictions, labels):
        raise NotImplementedError