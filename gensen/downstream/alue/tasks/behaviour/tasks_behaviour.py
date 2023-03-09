class FineTuningBehaviour:
    """"
    Interface class for Fine-Tuning tasks
    """
    def __init__(self, training_data_path, test_data_path):
        super(FineTuningBehaviour, self).__init__()
        self.training_data_path = training_data_path
        self.test_data_path = test_data_path
        self.num_epochs = None
        self.train_batch_size = None
        self.eval_batch_size = None
        self.learning_rate = None
        self.dropout = None
        self.pooling_strategy = None    # mean | max | last
        self.num_classes = None
        self.task_type = None   # multiclass | multilabel | regression | dual (e.g: xnli, q2q)
        self.finetune = None
        self.expand_vocab = None
        self.clean_text = None
        self.add_start_end_token = None
        self.language = None    # ar | en | bi (detects language, slow) (default)
        self.eval_metric = None    # F1-macro | Jaccard | Pearson | Matthews

    def prepare(self):
        raise NotImplementedError
