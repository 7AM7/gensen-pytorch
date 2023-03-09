import pandas as pd

from gensen.downstream.alue.tasks.behaviour.tasks_behaviour import FineTuningBehaviour


class Madar(FineTuningBehaviour):
    def __init__(self, training_data_path, test_data_path):
        super(FineTuningBehaviour, self).__init__()
        self.training_data_path = training_data_path
        self.test_data_path = test_data_path
        self.num_epochs = 10
        self.train_batch_size = 32
        self.eval_batch_size = 16
        self.learning_rate = 0.001
        self.dropout = 0.0
        self.pooling_strategy = "mean"
        self.num_classes = 26
        self.task_type = "multiclass"  # multiclass | multilabel | regression | dual (e.g: xnli, q2q)
        self.finetune = True
        self.expand_vocab = False
        self.clean_text = True
        self.add_start_end_token = True
        self.language = "ar"  # ar (default) | en | bi (detects language, slow)
        self.eval_metric = "F1-macro"  # F1-macro | Jaccard | Pearson | Matthews

    def prepare(self):
        train_data = pd.read_csv(self.training_data_path, sep='\t', names=['text', 'dialect'])
        test_data = pd.read_csv(self.test_data_path, sep='\t', names=['text', 'dialect'])
        train_data.dialect = train_data.dialect.astype('category').cat.codes
        test_data.dialect = test_data.dialect.astype('category').cat.codes
        train_x, train_y = train_data.text, train_data.dialect
        test_x, test_y = test_data.text, test_data.dialect

        return train_x, train_y, test_x, test_y