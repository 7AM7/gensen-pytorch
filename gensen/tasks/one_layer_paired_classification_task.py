import torch
import torch.nn as nn

from gensen.tasks.behaviours.paired_classification import PairedClassificationBehaviour


class OneLayerPairedClassificationTask(PairedClassificationBehaviour):
    def __init__(self, num_classes, src_hidden_dim, dropout, device):
        PairedClassificationBehaviour.__init__(self, num_classes)

        self.num_classes = num_classes
        self.dropout = dropout
        self.device = device
        self.src_hidden_dim = src_hidden_dim
        self.loss_criterion = nn.CrossEntropyLoss().to(self.device)

        self.decoder = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(4 * self.src_hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_classes)
        )

    def forward(self, s1_hidden_state, s2_hidden_state):
        """
        Propogate input through the decoder network.
        :param s1_hidden_state: (batch_size x encoder_hidden1_size)
                                of encoder first sentence hidden state.
        :param s2_hidden_state: (batch_size x encoder_hidden2_size)
                                of encoder second sentence hidden state.
        :return: predictions: (batch size x num classes)
                            of (pre-softmax over task classes).
        """
        features = torch.cat((
            s1_hidden_state, s2_hidden_state,
            torch.abs(s1_hidden_state - s2_hidden_state),
            s1_hidden_state * s2_hidden_state
        ), 1)
        predictions = self.decoder(features)
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
        loss = self.loss_criterion(
            predictions.contiguous().view(-1, predictions.size(1)),
            labels.contiguous().view(-1),
        )
        return loss
