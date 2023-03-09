import random
import logging

import torch
from sklearn.utils import shuffle


class Seq2SeqIterator:
    """Data iterator for seq2seq tasks (e.g: machine translation)"""

    def __init__(
        self,
        src,
        trg,
        src_dev,
        trg_dev,
        tasknames,
        tasknames_dev,
        tokenizer,
        device,
        buffer_size=1e6,
        shuffle=False
    ):
        """
        Prepare minibatch for Seq2Seq
        :param src : (list) of training src corpus file names
        :param trg : (list) of training target corpus file names
        :param src_dev : (list) of evaluation src corpus file names
        :param trg_dev : (list) of evaluation target corpus file names
        :param tasknames: (list) of training seq2seq tasknames
        :param tasknames_dev: (list) of evaluation seq2seq tasknames
        :param tokenizer: (sentencepiece)  sentencepiece model object
        :param device: (torch.device) CUDA or CPU
        :param buffer_size: (int) how many samples to fetch into memory
        :param shuffle: (bool) whether to shuffle data after each buffer or not
        """
        self.fname_src = src
        self.fname_trg = trg
        self.fname_src_dev = src_dev
        self.fname_trg_dev = trg_dev
        self.tasknames = tasknames
        self.device = device
        self.buffer_size = buffer_size
        self.shuffle = shuffle
        self.tokenizer = tokenizer

        # Open a list of file pointers to all the files.
        self.f_src = dict((taskname, open(fname, "r")) for taskname, fname in zip(tasknames, self.fname_src))
        self.f_trg = dict((taskname, open(fname, "r")) for taskname, fname in zip(tasknames, self.fname_trg))

        self.f_src_dev = dict((taskname, open(fname, "r")) for taskname, fname in zip(tasknames_dev, self.fname_src_dev))
        self.f_trg_dev = dict((taskname, open(fname, "r")) for taskname, fname in zip(tasknames_dev, self.fname_trg_dev))

        # Initialize dictionaries that contain sentences & word mapping dicts
        self.src = dict((taskname, {"data": []}) for taskname in tasknames)
        self.trg = dict((taskname, {"data": []}) for taskname in tasknames)

        self.src_dev = dict((taskname, {"data": []}) for taskname in tasknames_dev)
        self.trg_dev = dict((taskname, {"data": []}) for taskname in tasknames_dev)

        logging.info("Reading Seq2Seq training data")
        for taskname in self.src:
            self.fetch_buffer(taskname)

        logging.info("Reading and tokenizing Seq2Seq validation data")
        for taskname in self.src_dev:
            for src, trg in zip(self.f_src_dev[taskname], self.f_trg_dev[taskname]):
                self.src_dev[taskname]["data"].append(src.split())
                self.trg_dev[taskname]["data"].append(trg.split())

            # Sort sentences by decreasing length (hacky bucketing)
            self.src_dev[taskname]["data"], self.trg_dev[taskname]["data"] = zip(
                *sorted(
                    zip(self.src_dev[taskname]["data"], self.trg_dev[taskname]["data"]),
                    key=lambda x: len(x[0]),
                    reverse=True,
                )
            )

    def _shuffle_dataset(self, task_idx):
        rand_state = random.randrange(0, 100)
        src_lines = open(self.fname_src[task_idx], "r").readlines()
        trg_lines = open(self.fname_trg[task_idx], "r").readlines()
        src_lines, trg_lines = shuffle(src_lines, trg_lines, random_state=rand_state)
        open(self.fname_src[task_idx], 'w').writelines(src_lines)
        open(self.fname_trg[task_idx], 'w').writelines(trg_lines)
        del src_lines
        del trg_lines

    def _reset_filepointer(self, taskname, shuffle_file=False):
        task_idx = self.tasknames.index(taskname)
        if shuffle_file:
            self._shuffle_dataset(task_idx)

        self.f_src[taskname] = open(self.fname_src[task_idx], "r")
        self.f_trg[taskname] = open(self.fname_trg[task_idx], "r")

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
            self.trg[taskname]["data"] = []

        # shuffle dataset before fetching each buffer
        self._reset_filepointer(taskname, shuffle_file=self.shuffle)

        # Populate buffer
        for src, trg in zip(self.f_src[taskname], self.f_trg[taskname]):
            if len(self.src[taskname]["data"]) == self.buffer_size:
                break
            self.src[taskname]["data"].append(src.split())
            self.trg[taskname]["data"].append(trg.split())

        # Sort sentences by decreasing length (hacky bucketing)
        self.src[taskname]["data"], self.trg[taskname]["data"] = zip(
            *sorted(
                zip(self.src[taskname]["data"], self.trg[taskname]["data"]),
                key=lambda x: len(x[0]),
                reverse=True,
            )
        )

        """If buffer isn't full after reading the contents of the file,
        cycle around. """
        if len(self.src[taskname]["data"]) < self.buffer_size:
            assert len(self.src[taskname]["data"]) == len(self.trg[taskname]["data"])
            logging.info("Reached end of dataset, resetting file pointer ...")
            # Cast things to list to avoid issue with calling .append above
            self.src[taskname]["data"] = list(self.src[taskname]["data"])
            self.trg[taskname]["data"] = list(self.trg[taskname]["data"])
            self.fetch_buffer(taskname, reset=False)

        logging.info("Fetched {} sentences".format(len(self.src[taskname]["data"])))

    def get_parallel_minibatch(
        self,
        taskname,
        index,
        batch_size,
        max_len_src,
        max_len_trg,
        minibatch_type="train",
    ):
        """
        Prepare minibatch for Seq2Seq tasks
        :param taskname: (str) task specific name
        :param index: (int) start index to fetch data from
        :param batch_size: (int) how many samples to fetch
        :param max_len_src: (int) max src sentence length
        :param max_len_trg: (int) max target sentence length
        :param minibatch_type: (string) type of minibatch (train/dev)
        :return: (dict) a ِminibatch of src/target sentences
        """
        if minibatch_type == "train":
            src_data = self.src
            trg_data = self.trg
        else:
            src_data = self.src_dev
            trg_data = self.trg_dev

        src_lines = [
            ["<s>"] + line[: max_len_src - 2] + ["</s>"]
            for line in src_data[taskname]["data"][index: index + batch_size]
        ]

        trg_lines = [
            ["<s>"] + line[: max_len_trg - 2] + ["</s>"]
            for line in trg_data[taskname]["data"][index: index + batch_size]
        ]

        src_lens = [len(line) for line in src_lines]
        trg_lens = [len(line) for line in trg_lines]
        max_src_len = max(src_lens)
        max_trg_len = max(trg_lens)

        # Map words to indices
        input_lines_src = [
            [
                self.tokenizer.token_to_id(w)
                for w in line
            ]
            + [self.tokenizer.token_to_id("<pad>")] * (max_src_len - len(line))
            for line in src_lines
        ]

        input_lines_trg = [
            [
                self.tokenizer.token_to_id(w)
                for w in line[:-1]
            ]
            + [self.tokenizer.token_to_id("<pad>")] * (max_trg_len - len(line))
            for line in trg_lines
        ]

        output_lines_trg = [
            [
                self.tokenizer.token_to_id(w)
                for w in line[1:]
            ]
            + [self.tokenizer.token_to_id("<pad>")] * (max_trg_len - len(line))
            for line in trg_lines
        ]

        # Cast lists to torch tensors
        input_lines_src = torch.tensor(input_lines_src).to(self.device)
        input_lines_trg = torch.tensor(input_lines_trg).to(self.device)
        output_lines_trg = torch.tensor(output_lines_trg).to(self.device)
        src_lens = torch.tensor(src_lens, requires_grad=False).squeeze().to(self.device)

        # Return minibatch of src-trg pairs
        return {
            "input_src": input_lines_src,
            "input_trg": input_lines_trg,
            "output_trg": output_lines_trg,
            "src_lens": src_lens,
            "task_type": "seq2seq",
            "batch_type": minibatch_type,
        }
