import json
import random
import logging

import torch
from sklearn.utils import shuffle
import numpy as np


class PairClassificationIterator:
    """Data iterator for pair classification datasets (e.g.: NLI)"""

    def __init__(
            self,
            data,
            data_dev,
            text_to_labels,
            tasknames,
            tasknames_dev,
            tokenizer,
            device,
            buffer_size=1e6,
            shuffle=False
    ):
        """
        Prepare minibatch for Paired classification
        :param data : (list) of training corpus file names
        :param data_dev : (list) of evaluation corpus file names
        :param text_to_labels: (list) of text files of labels mappning
        :param tasknames: (list) of training tasknames
        :param tasknames_dev: (list) of evaluation tasknames
        :param tokenizer: (sentencepiece)  sentencepiece model object
        :param device: (torch.device) CUDA or CPU
        :param buffer_size: (int) how many samples to fetch into memory
        :param shuffle: (bool) whether to shuffle data after each buffer or not

        Each of dataset is a tab-separate file of the form
        sentence1 \t sentence2 \t label
        """
        self.fname_src = data
        self.fname_src_dev = data_dev
        self.tasknames = tasknames
        self.device = device
        self.buffer_size = buffer_size
        self.shuffle = shuffle
        self.tokenizer = tokenizer

        # Open a list of file pointers to all the files.
        self.f_src = dict((taskname, open(fname, "r")) for taskname, fname in zip(tasknames, self.fname_src))
        self.f_src_dev = dict((taskname, open(fname, "r")) for taskname, fname in zip(tasknames_dev, self.fname_src_dev))

        # Initialize dictionaries that contain sentences & word mapping dicts
        self.src = dict((taskname, {"data": []}) for taskname in tasknames)
        self.src_dev = dict((taskname, {"data": []}) for taskname in tasknames_dev)

        # Initialize dictionary that contains text label to numerical mapping for each task
        self.text2labels = dict((taskname, json.load(open(fname, "r"))) for taskname, fname in zip(tasknames, text_to_labels))

        logging.info("Reading paired classification training data")
        for taskname in self.src:
            self.fetch_buffer(taskname)

        logging.info("Reading and tokenizing paired classification validation data")
        for taskname in self.src_dev:
            for src in self.f_src_dev[taskname]:
                self.src_dev[taskname]["data"].append(src.split('\t'))

            # Sort sentences by decreasing length (hacky bucketing)
            self.src_dev[taskname]["data"] = sorted(
                    self.src_dev[taskname]["data"],
                    # sorting by sentences
                    key=lambda x: len(x[1]),
                    reverse=True,
            )

    def _shuffle_dataset(self, task_idx):
        rand_state = random.randrange(0, 100)
        lines = open(self.fname_src[task_idx], 'r').readlines()
        lines = shuffle(lines, random_state=rand_state)
        open(self.fname_src[task_idx], 'w').writelines(lines)
        del lines

    def _reset_filepointer(self, taskname, shuffle_file=False):
        task_idx = self.tasknames.index(taskname)
        if shuffle_file:
            self._shuffle_dataset(task_idx)

        self.f_src[taskname] = open(self.fname_src[task_idx], "r")

    def fetch_buffer(self, taskname, reset=True):
        """
        Fetch sentences from the file into the buffer.
        :param taskname: (str) name of the current task.
        :param reset: (bool) reset the contents of the current buffer.
        """
        logging.info("Fetching and tokenizing sentences ...")
        logging.info("Processing task: {}".format(taskname))

        # Reset the contents of the current buffer.
        if reset:
            self.src[taskname]["data"] = []

        # shuffle dataset before fetching each buffer
        self._reset_filepointer(taskname, shuffle_file=self.shuffle)

        # Populate buffer
        for src in self.f_src[taskname]:
            if len(self.src[taskname]["data"]) == self.buffer_size:
                break
            self.src[taskname]["data"].append(src.split('\t'))

        # Sort sentences by decreasing length (hacky bucketing)
        self.src[taskname]["data"] = sorted(
                self.src[taskname]["data"],
                # sorting by first sentence
                key=lambda x: len(x[1]),
                reverse=True,
        )

        """If buffer isn't full after reading the contents of the file,
        cycle around. """
        if len(self.src[taskname]["data"]) < self.buffer_size:
            logging.info("Reached end of dataset, resetting file pointer ...")
            # Cast things to list to avoid issue with calling .append above
            self.src[taskname]["data"] = list(self.src[taskname]["data"])
            self.fetch_buffer(taskname, reset=False)

        logging.info("Fetched {} sentences".format(len(self.src[taskname]["data"])))

    def get_parallel_minibatch(self, taskname, index, batch_size, max_len_src, minibatch_type="train"):
        """
        Prepare minibatch for paired classification tasks
        :param taskname: (str) task specific name
        :param index: (int) start index to fetch data from
        :param batch_size: (int) how many samples to fetch
        :param max_len_src: (int) max src sentence length
        :param minibatch_type: (string) type of minibatch (train/dev)
        :return: (dict) a ِminibatch of pair of sentences and their labels
        """
        if minibatch_type == "train":
            src_data = self.src
        else:
            src_data = self.src_dev

        sent1 = [
            ["<s>"] + line[1].split()[: max_len_src - 2] + ["</s>"]
            for line in src_data[taskname]["data"][index: index + batch_size]
        ]

        sent2 = [
            ["<s>"] + line[2].split()[: max_len_src - 2] + ["</s>"]
            for line in src_data[taskname]["data"][index: index + batch_size]
        ]

        labels = [
            self.text2labels[taskname][line[0]] for line in src_data[taskname]["data"][index: index + batch_size]
        ]

        sent1_lens = [len(line) for line in sent1]
        max_sent1_len = max(sent1_lens)

        sent2_lens = [len(line) for line in sent2]
        max_sent2_len = max(sent2_lens)

        sent1 = [
            [
                self.tokenizer.token_to_id(w)
                for w in line
            ]
            + [self.tokenizer.token_to_id("<pad>")] * (max_sent1_len - len(line))
            for line in sent1
        ]

        sent2 = [
            [
                self.tokenizer.token_to_id(w)
                for w in line
            ]
            + [self.tokenizer.token_to_id("<pad>")] * (max_sent2_len - len(line))
            for line in sent2
        ]

        sent1 = torch.tensor(sent1).to(self.device)
        sent2 = torch.tensor(sent2).to(self.device)
        labels = torch.tensor(labels).to(self.device)
        sent1_lens = torch.tensor(sent1_lens, requires_grad=False).squeeze().to(self.device)
        sent2_lens = torch.tensor(sent2_lens, requires_grad=False).squeeze().to(self.device)

        return {
            "sent1": sent1,
            "sent2": sent2,
            "sent1_lens": sent1_lens,
            "sent2_lens": sent2_lens,
            "labels": labels,
            "task_type": "pair_classification",
            "batch_type": minibatch_type
        }
