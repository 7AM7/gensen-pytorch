from gensen.iterators.seq2seq_iterator import Seq2SeqIterator
from gensen.iterators.pair_classification_iterator import PairClassificationIterator
from gensen.iterators.classification_iterator import ClassificationIterator


class IteratorsFactory:
    def __init__(self, tasks, tokenizer, device, buffer_size,
                 shuffle=False):
        """Initialize IteratorsFactory."""
        self.tasks = tasks
        self.tasktype = [(task["taskname"], task["tasktype"]) for task in tasks]
        self.tokenizer = tokenizer
        self.device = device
        self.buffer_size = buffer_size
        self.shuffle = shuffle

        self.build_iterators()

    def build_iterators(self):
        """
        Initialize Iterators.
        """

        # seq2seq iterator parameters
        self.seq2seq_tasks = [
            task for task in self.tasks if task["tasktype"] == "seq2seq"
        ]
        self.seq2seq_tasknames = [task["taskname"] for task in self.seq2seq_tasks]
        self.seq2seq_tasknames_dev = [
            task["taskname"] for task in self.seq2seq_tasks if "val_src" in task
        ]
        seq2seq_train_src = [task["train_src"] for task in self.seq2seq_tasks]
        seq2seq_train_trg = [task["train_trg"] for task in self.seq2seq_tasks]
        seq2seq_dev_src = [
            task["val_src"] for task in self.seq2seq_tasks if "val_src" in task
        ]
        seq2seq_dev_trg = [
            task["val_trg"] for task in self.seq2seq_tasks if "val_src" in task
        ]

        # instantiate seq2seq iterator
        self.seq2seq_iterator = Seq2SeqIterator(
            seq2seq_train_src,
            seq2seq_train_trg,
            seq2seq_dev_src,
            seq2seq_dev_trg,
            self.seq2seq_tasknames,
            self.seq2seq_tasknames_dev,
            self.tokenizer,
            self.device,
            self.buffer_size,
            shuffle=self.shuffle
        )

        # paired classification iterator parameters
        self.pair_classification_tasks = [
            task for task in self.tasks if task["tasktype"] == "pair-classification"
        ]
        self.pair_tasknames = [task["taskname"] for task in self.pair_classification_tasks]
        self.pair_tasknames_dev = [
            task["taskname"]
            for task in self.pair_classification_tasks
            if "val_src" in task
        ]
        pair_train_src = [task["train_src"] for task in self.pair_classification_tasks]
        pair_val_src = [
            task["val_src"]
            for task in self.pair_classification_tasks
            if "val_src" in task
        ]
        pair_text2labels = [
            task["labels_mapping"] for task in self.pair_classification_tasks
        ]

        # instantiate paired classification iterator
        self.pair_classification_iterator = PairClassificationIterator(
            pair_train_src,
            pair_val_src,
            pair_text2labels,
            self.pair_tasknames,
            self.pair_tasknames_dev,
            self.tokenizer,
            self.device,
            self.buffer_size,
            shuffle=self.shuffle
        )

        # classification iterator parameters
        self.classification_tasks = [
            task for task in self.tasks if task["tasktype"] == "classification"
        ]
        self.classification_tasknames = [task["taskname"] for task in self.classification_tasks]
        self.classification_tasknames_dev = [
            task["taskname"]
            for task in self.classification_tasks
            if "val_src" in task
        ]
        classification_train_src = [task["train_src"] for task in self.classification_tasks]
        classification_val_src = [
            task["val_src"]
            for task in self.classification_tasks
            if "val_src" in task
        ]
        classification_text2labels = [
            task["labels_mapping"] for task in self.classification_tasks
        ]

        # instantiate classification iterator
        self.classification_iterator = ClassificationIterator(
            classification_train_src,
            classification_val_src,
            classification_text2labels,
            self.classification_tasknames,
            self.classification_tasknames_dev,
            self.tokenizer,
            self.device,
            self.buffer_size,
            shuffle=self.shuffle
        )

    def get_minibatch(
        self,
        task_name,
        task_idx,
        batch_size,
        max_len_src=None,
        max_len_trg=None,
        minibatch_type="train",
    ):
        """
        Fetch minibatch for a given task

        :param task_name: (str) task name
        :param task_idx: (int) start index to fetch data from
        :param batch_size: (int) minibatch size
        :param max_len_src: (int) optional for seq2seq tasks (source sentence length)
        :param max_len_trg: (int) optional for seq2seq tasks (target sentence length)
        :param minibatch_type: (string) type of minibatch (train/dev)
        :return: (dict) a ِminibatch of corresponding task type
        """
        if task_name in self.seq2seq_tasknames:
            minibatch = self.seq2seq_iterator.get_parallel_minibatch(
                task_name,
                task_idx,
                batch_size,
                max_len_src,
                max_len_trg,
                minibatch_type=minibatch_type,
            )

        elif task_name in self.pair_tasknames:
            minibatch = self.pair_classification_iterator.get_parallel_minibatch(
                task_name,
                task_idx,
                batch_size,
                max_len_src,
                minibatch_type=minibatch_type,
            )

        elif task_name in self.classification_tasknames:
            minibatch = self.classification_iterator.get_parallel_minibatch(
                task_name,
                task_idx,
                batch_size,
                max_len_src,
                minibatch_type=minibatch_type,
            )

        return minibatch

    def fetch_buffer(self, task_name):
        """
        Fetch new buffer for a given task

        :param task_name: (str) task name
        """
        if task_name in self.seq2seq_tasknames:
            self.seq2seq_iterator.fetch_buffer(task_name)

        elif task_name in self.pair_tasknames:
            self.pair_classification_iterator.fetch_buffer(task_name)

        elif task_name in self.classification_tasknames:
            self.classification_iterator.fetch_buffer(task_name)