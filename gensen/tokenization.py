import os
import argparse
import json
from multiprocessing import Pool
import logging
import time

import sentencepiece as spm

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s",
    level=logging.INFO,
    filename="log/tokenize_dataset_run_1.log",
)


class SentencePieceTokenizer:
    def __init__(self, model_file, lowercase=False):
        self.lowercase = lowercase
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(model_file)

    def tokenize(self, text):
        if self.lowercase:
            text = text.lower()

        return self.tokenizer.EncodeAsPieces(text)

    def detokenize(self, tokens):
        return "".join(tokens).replace("▁", " ")

    def id_to_token(self, id):
        return self.tokenizer.id_to_piece(int(id))

    def token_to_id(self, token):
        return self.tokenizer.piece_to_id(token)

    def create_word2id_id2word(self):
        word2id = {
            self.tokenizer.id_to_piece(id): id
            for id in range(self.tokenizer.get_piece_size())
        }
        id2word = dict(zip(word2id.values(), word2id.keys()))

        return word2id, id2word


def create_tokenizing_instances(params):
    """
    tokenizing and lowercase's the text sentences from the input file
     and saving them to the output file with '.token' suffix.
    :param index: (int) the current task/file index.
    :param input_file: (str) the current input file path.
    :param tokenizer: (sentencepiece) the sentencepiece tokenizer object.
    :param task_type: (str) the current task/file task type (e.g seq2seq, ..).
    :param lowercase: (bool) whether lower case the input tokens or not.
    :param key: (str) the current task file type either train or val (e.g train_trg, val_src).
    :return: A tuple of : index, output_path, key
    """
    (index, input_file, tokenizer,
     task_type, lowercase, key) = params
    output_path = input_file + '.token'

    if not os.path.exists(input_file):
        logging.info("Input file {} does not exist. skipped ...".format(input_file))
        return

    result = []
    logging.info('Reading from {} input file ...'.format(input_file))
    # Read and Tokenize the input file
    with open(input_file, 'r') as reader:
        lines = reader.read().splitlines()
        for line in lines:
            tokens = ''
            if task_type == 'seq2seq':
                line = line.lower() if lowercase else line
                tokens = ' '.join(tokenizer.tokenize(line))

            elif task_type == 'pair-classification':
                line = line.lower().split('\t') if lowercase else line.split('\t')
                labels = line[0]
                sent1 = ' '.join(tokenizer.tokenize(line[1]))
                sent2 = ' '.join(tokenizer.tokenize(line[2]))
                tokens = "{}\t{}\t{}".format(labels, sent1, sent2)

            elif task_type == 'classification':
                line = line.lower().split('\t') if lowercase else line.split('\t')
                labels = line[0]
                sent = ' '.join(tokenizer.tokenize(line[1]))
                tokens = "{}\t{}".format(labels, sent)
            else:
                logging.info("Unknown task type")

            result.append(tokens)

    logging.info('Writing to {} output file ...'.format(output_path))
    # Write the Tokenized text to the output file
    with open(output_path, 'w') as writer:
        for line in result:
            writer.write("{0}\n".format(line))

    return index, output_path, key


def update_config(config, results, config_file_path):
    """
    Updating the config with new data path for each task,
    after adding the '.token' suffix.
    :param config: (json) the current config file.
    :param results: (list) of indexes, keys and new paths.
    :param config_file_path: (str) the config path.
    """
    for res in results:
        if not res:
            continue
        index = res[0]
        new_path = res[1]
        key = res[2]
        config["tasks"][index][key] = new_path

    with open(config_file_path + ".temp", "w") as jsonFile:
        json.dump(config, jsonFile, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenizing dataset")

    parser.add_argument(
        "-m",
        "--model_path",
        help="The model file for sentencepiece tokenizer",
        required=True,
        type=str,
    )

    parser.add_argument(
        "-w",
        "--num_workers",
        help="Number of workers for parallel processing, where each generates an output file.",
        required=False,
        default=4,
        type=int,
    )
    args = parser.parse_args()
    time_start = time.time()

    logging.info("Loading the configurations and sentencepiece tokenizer ...")
    config_file_path = "config.json"
    config = json.load(open(config_file_path, "r"))
    tasks = config["tasks"]
    lowercase = config["vocab"]["lowercase"]
    if not os.path.exists(args.model_path):
        raise ValueError("Could not find sentencepiece model file : {}".format(args.model_path))
    else:
        tokenizer = SentencePieceTokenizer(model_file=args.model_path,
                                           lowercase=lowercase)

    logging.info('*** Tokenization process starting ***')
    process_args = []
    for index, task in enumerate(tasks):
        task_type = task['tasktype']
        if task_type == 'seq2seq':
            process_args.append((index, task["train_trg"], tokenizer,
                                 task_type, lowercase, 'train_trg'))
            if "val_trg" in task:
                process_args.append((index, task["val_trg"], tokenizer,
                                     task_type, lowercase, 'val_trg'))

        if "val_src" in task:
            process_args.append((index, task["val_src"], tokenizer,
                                 task_type, lowercase, 'val_src'))
        process_args.append((index, task["train_src"], tokenizer,
                             task_type, lowercase, 'train_src'))

    # For duplicated files
    process_args = list(set(process_args))
    logging.info('Number of files : {}'.format(len(process_args)))

    results = []
    if args.num_workers > 1:
        pool = Pool(processes=args.num_workers)
        results = pool.map(create_tokenizing_instances, process_args)
        pool.close()
        pool.join()
    else:
        for process_arg in process_args:
            res = create_tokenizing_instances(process_arg)
            results.append(res)

    logging.info('*** Tokenization process finished Successfully. ***')

    logging.info('*** Updating the config file ...  ***')
    update_config(config, results, config_file_path)

    time_end = time.time()
    logging.info('Time cost=%.1f', time_end - time_start)
