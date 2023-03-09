import torch
import torch.nn as nn

from gensen.tasks.behaviours.classification import ClassificationBehaviour


class OneLayerClassificationTask(ClassificationBehaviour):
    def __init__(self, num_classes, src_hidden_dim, dropout, device):
        ClassificationBehaviour.__init__(self, num_classes)

        self.num_classes = num_classes
        self.dropout = dropout
        self.device = device
        self.src_hidden_dim = src_hidden_dim
        self.loss_criterion = nn.CrossEntropyLoss().to(self.device)

        self.decoder = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.src_hidden_dim, self.num_classes)
        )

    def forward(self, encoder_hidden_state):
        """
        Propogate input through the decoder network.
        :param encoder_hidden_state: (batch_size x encoder_hidden_size)
                                    of encoder hidden state.
        :return: predictions: (batch size x num classes)
                            of (pre-softmax over task classes).
        """
        predictions = self.decoder(encoder_hidden_state)
        return predictions

    def loss(self, predictions, labels):
        """
        Calculate decoder network loss.
        :param predictions: (batch size x num classes)
                            of (pre-softmax over over task classes).
        :param labels: (batch size x num classes)
                            of (pre-softmax over over task classes).
        :return: backward loss: (scalar) of decoder loss.
        """
        loss = self.loss_criterion(predictions, labels)
        return loss
