from gensen.tasks.one_layer_paired_classification_task import OneLayerPairedClassificationTask
from gensen.tasks.conditional_seq2seq_task import ConditionalSeq2SeqTask
from gensen.tasks.one_layer_classification_task import OneLayerClassificationTask

class TasksFactory:
    def __init__(self, dropout, device):
        """Initialize TasksFactory."""
        self.dropout = dropout
        self.device = device

    def init_decoder(self, task):
        """
        Initialize Decoders.
        :param task: (dict) of task arguments/options
        """
        if task['tasktype'] == "seq2seq":
            return ConditionalSeq2SeqTask(
                    trg_vocab_size=task['trg_vocab_size'],
                    trg_emb_dim=task['trg_emb_dim'],
                    trg_hidden_dim=task['trg_hidden_dim'],
                    trg_pad_token=task['trg_pad_token'],
                    dropout=self.dropout,
                    device=self.device
            ).to(self.device)

        elif task['tasktype'] == "pair-classification":
            return OneLayerPairedClassificationTask(
                    num_classes=task['num_of_classes'],
                    src_hidden_dim=task['src_hidden_dim'] * 2,
                    dropout=self.dropout,
                    device=self.device
            ).to(self.device)

        elif task['tasktype'] == "classification":
            return OneLayerClassificationTask(
                    num_classes=task['num_of_classes'],
                    src_hidden_dim=task['src_hidden_dim'] * 2,
                    dropout=self.dropout,
                    device=self.device
            ).to(self.device)