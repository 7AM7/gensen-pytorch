import pandas as pd

from gensen.downstream.alue.tasks.behaviour.tasks_behaviour import FineTuningBehaviour


class BiSTS(FineTuningBehaviour):
    def __init__(self, training_data_path, test_data_path):
        super(FineTuningBehaviour, self).__init__()
        self.training_data_path = training_data_path
        self.test_data_path = test_data_path
        self.num_epochs = 10
        self.train_batch_size = 32
        self.eval_batch_size = 16
        self.learning_rate = 0.001
        self.dropout = 0.1
        self.pooling_strategy = "mean"
        self.num_classes = 1
        self.task_type = "sts"  # multiclass | multilabel | regression | dual (e.g: xnli, q2q)
        self.finetune = False
        self.expand_vocab = False
        self.clean_text = True
        self.add_start_end_token = True
        self.language = "bi"    # ar | en | bi (detects language, slow) (default)
        self.eval_metric = "Pearson"    # F1-macro | Jaccard | Pearson | Matthews

    def prepare(self):
        train_data = pd.read_csv(self.training_data_path, sep="\t")
        test_data = pd.read_csv(self.test_data_path, sep="\t")
        train_x_1, train_x_2, train_y = (
            train_data.sentence1,
            train_data.sentence2,
            train_data.gold_label,
        )
        test_x_1, test_x_2, test_y = (
            test_data.sentence1,
            test_data.sentence2,
            test_data.gold_label,
        )
        train_y /= 5
        test_y /= 5

        return (train_x_1, train_x_2), train_y, (test_x_1, test_x_2), test_y