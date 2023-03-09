import pandas as pd

from gensen.downstream.alue.tasks.behaviour.tasks_behaviour import FineTuningBehaviour


class HateSpeech(FineTuningBehaviour):
    def __init__(self, training_data_path, test_data_path, test_data_label_path=None):
        super(FineTuningBehaviour, self).__init__()
        self.training_data_path = training_data_path
        self.test_data_path = test_data_path
        self.test_data_label_path = test_data_label_path
        self.train_batch_size = 32
        self.eval_batch_size = 16
        self.num_epochs = 10
        self.learning_rate = 0.001
        self.dropout = 0.25
        self.pooling_strategy = "mean"
        self.num_classes = 2
        self.task_type = "multiclass"    # multiclass | multilabel | regression | dual (e.g: xnli, q2q)
        self.finetune = True
        self.expand_vocab = False
        self.clean_text = True
        self.add_start_end_token = True
        self.language = "ar"    # ar (default) | en | bi (detects language, slow)
        self.eval_metric = "F1-macro"  # F1-macro | Jaccard | Pearson | Matthews

    def prepare(self):
        if not self.test_data_label_path:
            raise ValueError("Could not find HateSpeech test_data label file : {}".format(
                self.test_data_label_path))

        train_data = pd.read_csv(
            self.training_data_path,
            quotechar="▁",
            sep="\t",
            names=['text', 'offensive', 'hate'],
        )
        test_data_sentences = pd.read_csv(
            self.test_data_path,
            quotechar="▁",
            sep="\t",
            header=None,
            names=["text"],
        )

        test_data_labels = pd.read_csv(
            self.test_data_label_path,
            quotechar="▁",
            sep="\t",
            header=None,
            names=["hate"],
        )

        train_data.hate = train_data.hate.astype("category").cat.codes
        test_data_labels.hate = test_data_labels.hate.astype("category").cat.codes
        train_x, train_y = train_data.text, train_data.hate
        test_x, test_y = test_data_sentences.text, test_data_labels.hate

        return train_x, train_y, test_x, test_y