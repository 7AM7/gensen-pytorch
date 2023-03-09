import os
import random
import argparse
import logging

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s",
    level=logging.INFO,
    filename="log/alue_1.log",
)

import pandas as pd
from tabulate import tabulate
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from gensen.downstream.alue.trainer import Trainer, CustomDataset
from gensen.downstream.alue.tasks.Diagnostic import Diagnostic
from gensen.downstream.alue.tasks.Emotion_classification import EmotionClassification
from gensen.downstream.alue.tasks.Hate_speech import HateSpeech
from gensen.downstream.alue.tasks.IDAT import IDAT
from gensen.downstream.alue.tasks.Madar import Madar
from gensen.downstream.alue.tasks.Offensive import Offensive
from gensen.downstream.alue.tasks.Q2Q import Q2Q
from gensen.downstream.alue.tasks.V_Reg import VReg
from gensen.downstream.alue.tasks.XNLI import XNLI
from gensen.downstream.alue.tasks.biSTS import BiSTS
from gensen.downstream.alue.tasks.biNLI import BiNLI

cudnn.benchmark = True


class AlueBenchmarking:
    def __init__(
        self,
        model_dir,
        pretrained_embeddings,
        finetuning_output,
        device,
        print_summary,
        report_path,
    ):
        self.model_dir = model_dir
        self.pretrained_embeddings = pretrained_embeddings
        self.finetuning_output = finetuning_output
        self.device = device
        self.print_summary = print_summary
        self.report_path = report_path

    def run(self, task_names):
        results = []
        for task_name in task_names:
            if task_name == "IDAT":
                task = IDAT(
                    training_data_path="alue/data/idat/IDAT_training_text.csv",
                    test_data_path="alue/data/idat/IDAT_test_text.csv",
                )

            elif task_name == "biSTS":
                task = BiSTS(
                    training_data_path="alue/bilingual/biSTS/train.csv",
                    test_data_path="alue/bilingual/biSTS/test.csv",
                )

            elif task_name == "biNLI":
                task = BiNLI(
                    training_data_path="alue/bilingual/xnli/train_xnli.csv",
                    test_data_path="alue/bilingual/xnli/test_xnli.csv",
                )

            elif task_name == "Madar":
                task = Madar(
                    training_data_path="alue/data/madar/MADAR-Corpus-26-train.tsv",
                    test_data_path="alue/data/madar/MADAR-Corpus-26-test.tsv",
                )

            elif task_name == "Offensive":
                task = Offensive(
                    training_data_path="alue/data/osact4/OSACT2020-sharedTask-train.txt",
                    test_data_path="alue/private/private_datasets/offensive/tweets_v1.0.txt",
                    test_data_label_path="alue/private/private_datasets/offensive/offensive_labels_v1.0.txt",
                )

            elif task_name == "Hate_speech":
                task = HateSpeech(
                    training_data_path="alue/data/osact4/OSACT2020-sharedTask-train.txt",
                    test_data_path="alue/private/private_datasets/offensive/tweets_v1.0.txt",
                    test_data_label_path="alue/private/private_datasets/offensive/hatespeech_labels_v1.0.txt",
                )
            elif task_name == "Emotion_classification":
                task = EmotionClassification(
                    training_data_path="alue/data/affect-in-tweets/emotion-c/2018-E-c-Ar-train.txt",
                    test_data_path="alue/data/affect-in-tweets/emotion-c/emotion_with_labels_v1.0.tsv",
                )

            elif task_name == "V_Reg":
                task = VReg(
                    training_data_path="alue/data/affect-in-tweets/V-reg/2018-Valence-reg-Ar-train.txt",
                    test_data_path="alue/data/affect-in-tweets/V-reg/vreg_with_labels_v1.0.tsv",
                )

            elif task_name == "Q2Q":
                task = Q2Q(
                    training_data_path="alue/data/q2q/q2q_similarity_workshop_v2.1.tsv",
                    test_data_path="alue/data/q2q/q2q_with_labels_v1.0.tsv",
                )

            elif task_name == "XNLI":
                task = XNLI(
                    training_data_path="alue/data/xnli/arabic_test.tsv",
                    test_data_path="alue/data/xnli/arabic_dev.tsv",
                )

            elif task_name == "Diagnostic":
                task = Diagnostic(
                    training_data_path="alue/data/xnli/arabic_test.tsv",
                    test_data_path="alue/data/xnli/diagnostic.tsv",
                )
            else:
                logging.info("Unknown task name {}".format(task_name))
                break

            logging.info("========== {} ==========".format(task_name))
            logging.info("Preparing the dataset ...")
            train_x, train_y, test_x, test_y = task.prepare()

            tensor_type = (
                torch.float
                if task.task_type in ["regression", "multilabel", "sts"]
                else torch.long
            )
            dataset_train = CustomDataset(
                train_x, train_y, y_type=tensor_type, device=self.device
            )
            dataset_test = CustomDataset(
                test_x, test_y, y_type=tensor_type, device=self.device
            )

            dataloader_train = DataLoader(
                dataset_train,
                sampler=RandomSampler(dataset_train),
                batch_size=task.train_batch_size,
            )

            dataloader_test = DataLoader(
                dataset_test,
                sampler=SequentialSampler(dataset_test),
                batch_size=task.eval_batch_size,
            )

            if self.finetuning_output:
                self.finetuning_output = os.path.join(self.finetuning_output, task_name)

            logging.info("Training and evaluating ...")
            trainer = Trainer(
                samples=train_x,
                model_folder=self.model_dir,
                pretrained_embeddings=self.pretrained_embeddings,
                saved_finetuned_model_path=self.finetuning_output,
                num_epochs=task.num_epochs,
                learning_rate=task.learning_rate,
                dropout=task.dropout,
                language=task.language,
                pooling_strategy=task.pooling_strategy,
                num_classes=task.num_classes,
                task_type=task.task_type,
                task_name=task_name,
                task_metric_name=task.eval_metric,
                finetune=task.finetune,
                expand_vocab=task.expand_vocab,
                clean_text=task.clean_text,
                add_start_end_token=task.add_start_end_token,
                device=self.device,
            )

            if task_name == "biSTS":
                training_result = trainer.run(dataloader_train, None, "training_")

                testing_result = trainer.run(dataloader_test, None, "testing_")
                results.extend((training_result, testing_result))
            else:
                result = trainer.run(dataloader_train, dataloader_test)
                results.append(result)

        if self.print_summary or self.report_path:
            self.summary_report(results)

        logging.info("=========================")

    def summary_report(self, results):
        logging.info("========== Fine-tuning summary report ==========")
        df = pd.DataFrame(
            results,
            columns=[
                "Task Name",
                "Accuracy",
                "Metric Value",
                "Metric Name",
                "Best Epoch",
                "Training Time",
            ],
        )

        if self.print_summary:
            logging.info(tabulate(df, headers="keys", tablefmt="psql"))

        if self.report_path:
            logging.info(
                "Saving fine-tuning summary report to {}".format(self.report_path)
            )
            df.to_excel(self.report_path, index=False)

        logging.info("========== Models Averages (without Bi-lingual tasks) ==========")
        df = df.loc[
            (df["Task Name"] != "training_biSTS")
            & (df["Task Name"] != "testing_biSTS")
            & (df["Task Name"] != "biNLI")
        ]
        avg = df["Metric Value"].mean()
        if not np.isnan(avg):
            logging.info("Models Average: {}".format(round(avg, 3)))

        if (df["Task Name"] == "Diagnostic").any():
            avg_df = df.loc[(df["Task Name"] != "Diagnostic")]
            logging.info(
                "Models without Diagnostic Average: {}".format(
                    round(avg_df["Metric Value"].mean(), 3)
                )
            )

        if ((df['Task Name'] == 'Diagnostic').any() and
                (df['Task Name'] == 'XNLI').any()):
            avg_df = df.loc[
                (df["Task Name"] != "Diagnostic") & (df["Task Name"] != "XNLI")
            ]
            logging.info(
                "Models without XNLI and Diagnostic Average: {}".format(
                    round(avg_df["Metric Value"].mean(), 3)
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ALUE BenchMarking")

    parser.add_argument(
        "-m",
        "--model_dir",
        help="The model folder path",
        required=True,
        default=None,
        type=str,
    )

    parser.add_argument(
        "-p",
        "--pretrained_embeddings",
        help="The pretrained embeddings file path",
        required=False,
        default=None,
        type=str,
    )

    parser.add_argument(
        "-s",
        "--finetuning_output",
        help="The path of fine-tuning output folder",
        required=False,
        default=None,
        type=str,
    )

    parser.add_argument(
        "-t",
        "--tasks",
        help="List of fine-tuning tasks names",
        required=False,
        default="Diagnostic,IDAT,Madar,Offensive,Hate_speech,Emotion_classification,V_Reg,Q2Q,XNLI,biSTS,biNLI",
        type=lambda s: [str(item).strip() for item in s.split(",")],
    )

    parser.add_argument(
        "-d",
        "--seed",
        help="Random seed value",
        required=False,
        default=90,
        type=int,
    )

    parser.add_argument(
        "-q",
        "--print_summary",
        help="whether print summary report or not",
        required=False,
        default=True,
        type=bool,
    )

    parser.add_argument(
        "-r",
        "--report_path",
        help="The path to save the summary report",
        required=False,
        default=None,
        type=str,
    )
    args = parser.parse_args()

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    alue = AlueBenchmarking(
        model_dir=args.model_dir,
        pretrained_embeddings=args.pretrained_embeddings,
        finetuning_output=args.finetuning_output,
        device=device,
        print_summary=args.print_summary,
        report_path=args.report_path,
    )
    logging.info("Fine-tuning tasks  {}".format(" ".join(args.tasks)))
    alue.run(args.tasks)
