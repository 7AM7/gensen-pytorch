import os
import json
import logging

import pandas as pd
from tqdm import tqdm
import sentencepiece as spm

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s",
    level=logging.INFO,
    filename="log/train_sentencepiece_run_1.log",
)


class TrainSentencePiece:
    def __init__(
        self,
        tasks,
        save_dir,
        model_prefix="tokenizer",
        vocab_size=20000,
        model_type="bpe",
        max_num_sentences=10000000,
        special_tokens=["<s>", "</s>", "<pad>"],
    ):
        """
        Initialize TrainSentencePiece Model.
        :param tasks: (list) of training tasks configs
        :param save_dir: (str) of directory to save vocabularies
        :param vocab_size: (int) of vocab size
        :param model_prefix: (str) of model prefix/name
        :param model_type: (str) of sentencepiece model type (e.g.: BPE, unigram)
        :param max_num_sentences: (int) of maximum size of sentences the trainer loads from dataset.
        :param special_tokens: (list of str) of sentencepiece special tokens (e.g.: [<pad>,<s>])
        """
        self.tasks = tasks
        self.vocab_size = vocab_size
        self.model_prefix = model_prefix
        self.model_type = model_type
        self.max_num_sentences = max_num_sentences
        self.save_dir = save_dir
        self.special_tokens = special_tokens

        self.train_model()

    def train_model(self):
        """ Training SentencePiece Model."""
        # Check if save directory exists.
        if not os.path.exists(self.save_dir):
            raise ValueError("Could not find save dir : {}".format(self.save_dir))

        logging.info("Training SentencePiece model/vocab ...")
        model_file = os.path.join(self.save_dir, self.model_prefix)
        # Check if a cached model file exists.
        if os.path.exists(model_file + ".model"):
            logging.info("Found existing SentencePiece model/vocab ...")
        else:
            logging.info(
                "Could not find existing SentencePiece model/vocab. Reconstructing files ..."
            )

            input_files = self._reconstruct_files()
            logging.info("Training SentencePiece model/vocab ...")
            spm.SentencePieceTrainer.train(
                input=input_files,
                input_sentence_size=self.max_num_sentences,
                model_prefix=model_file,
                unk_surface="<unk>",
                user_defined_symbols=self.special_tokens,
                bos_id=0,
                pad_id=1,
                eos_id=2,
                unk_id=3,
                # amount of characters covered by the model.
                character_coverage=1.0,
                vocab_size=self.vocab_size,
                model_type=self.model_type,
            )
            # Remove reconstructed files after training
            for f in input_files:
                if '.recon' in f:
                    os.remove(f)

    def _reconstruct_files(self, sep='\t'):
        """
        Reconstructing corpus files (e.g. Bilingual, Monolingual)
        """
        logging.info("Reconstructing corpus files ...")
        output_paths = []
        for task in tqdm(self.tasks):
            input_path = task["train_src"]
            output_path = input_path + '.recon'
            if task['tasktype'] == 'seq2seq':
                output_paths.extend((input_path,
                                    task["train_trg"]))
                continue

            df = pd.read_csv(input_path, sep=sep)
            # drop label column
            df.drop(df.columns[0], axis=1, inplace=True)

            # Bilingual
            if len(df.columns) == 2:
                df.to_csv(output_path, sep='\t', index=False)
            # Monolingual
            elif len(df.columns) == 1:
                df.to_csv(output_path, index=False)

            output_paths.append(output_path)

        return output_paths


if __name__ == "__main__":
    config_file_path = "../config.json"
    if not os.path.exists(config_file_path):
        raise ValueError("Could not find configuration file : {}".format(config_file_path))
    else:
        logging.info("Loading configuration file ...")
        config = json.load(open(config_file_path, "r"))

    TrainSentencePiece(
        tasks=config["tasks"],
        save_dir=config["model"]["vocab_dir"],
        model_prefix=config["vocab"]["model_prefix"],
        vocab_size=config["vocab"]["vocab_size"],
        model_type=config["vocab"]["model_type"],
        max_num_sentences=config["vocab"]["max_num_sentences"],
        special_tokens=config["vocab"]["special_tokens"],
    )
