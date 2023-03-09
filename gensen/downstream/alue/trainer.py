import time
import logging

import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, matthews_corrcoef

from gensen.downstream.alue.classifier import GenSenTextClassification
from gensen.utils import normalize_text


class CustomDataset(Dataset):
    def __init__(self, x, y, y_type, device):
        if type(x) is tuple:
            self.x1 = x[0]
            self.x2 = x[1]
            self.dual = True
        else:
            self.x = x
            self.dual = False

        self.y = y
        self.y_type = y_type
        self.device = device

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return {
            "x": (self.x1[index], self.x2[index]) if self.dual else self.x[index],
            "y": torch.tensor(
                self.y[index],
                dtype=self.y_type,
                device=self.device,
            ),
        }


class Trainer:
    def __init__(
        self,
        samples,
        model_folder,
        pretrained_embeddings,
        num_epochs,
        learning_rate,
        dropout,
        pooling_strategy,
        num_classes,
        task_type,
        task_metric_name,
        task_name,
        device,
        language="ar",
        saved_finetuned_model_path=None,
        finetune=True,
        expand_vocab=False,
        clean_text=True,
        add_start_end_token=True,
    ):

        if expand_vocab:
            if type(samples) is tuple:
                samples = samples[0].tolist() + samples[1].tolist()
            else:
                samples = samples.tolist()

            if task_type in ["dual", "dual-regression"]:
                task_vocab = self.get_task_vocab(
                    samples,
                    clean=clean_text,
                    language=language,
                )
            else:
                task_vocab = self.get_task_vocab(
                    samples, clean=clean_text, language=language
                )
        else:
            task_vocab = None

        self.model = GenSenTextClassification(
            model_folder=model_folder,
            pretrained_embedding=pretrained_embeddings,
            num_classes=num_classes,
            dropout=dropout,
            task_vocab=task_vocab,
            vocab_expand=expand_vocab,
            clean_text=clean_text,
            add_start_end_token=add_start_end_token,
            task_type=task_type,
            finetune=finetune,
            device=device,
        ).to(device)

        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=learning_rate
        )

        self.task_type = task_type
        if self.task_type in ["multiclass", "dual"]:
            self.loss_fn = CrossEntropyLoss().to(device)
        elif self.task_type == "multilabel":
            self.loss_fn = BCEWithLogitsLoss().to(device)
        elif self.task_type in ["regression", "dual-regression", "sts"]:
            self.loss_fn = MSELoss().to(device)

        self.num_epochs = num_epochs
        self.pooling_strategy = pooling_strategy
        self.language = language
        self.saved_finetuned_path = saved_finetuned_model_path
        self.task_name = task_name
        self.task_metric_name = task_metric_name

    def run(self, train_loader, test_loader, prefix=None):
        if self.task_name == 'biSTS':
            start_time = time.time()
            preds, labels, _ = self.eval(train_loader)
            end_time = time.time()
            eval_time = round((end_time - start_time), 3)

            metric_val, accuracy = self.metrics(preds, labels)
            results = {
                'Training Time': eval_time,
                'Best Epoch': None,
                'Accuracy': None,
                'Metric Value': round(metric_val, 3),
                'Metric Name': self.task_metric_name,
                'Task Name': prefix + self.task_name,
            }
            return results

        results = {'Metric Value': float('-inf')}
        for epoch in tqdm(range(0, self.num_epochs)):
            start_time = time.time()
            train_loss = self.train(train_loader)
            end_time = time.time()
            training_time = round((end_time - start_time), 3)
            logging.info(
                "Training Epoch {} loss: {} time: {} seconds".format(
                    epoch + 1, round(np.mean(train_loss), 3),
                    training_time
                )
            )

            preds, labels, test_loss = self.eval(test_loader)
            test_loss = np.mean(test_loss)
            logging.info("Testing loss: {}".format(round(test_loss, 3)))

            metric_val, accuracy = self.metrics(preds, labels)
            if results['Metric Value'] < metric_val:
                results = {
                    'Training Time': training_time,
                    'Best Epoch': epoch + 1,
                    'Accuracy': round(accuracy, 3) if accuracy is not None else None,
                    'Metric Value': round(metric_val, 3),
                    'Metric Name': self.task_metric_name,
                    'Task Name': self.task_name,
                    'best_state_dict': self.model.state_dict(),
                }

        if self.saved_finetuned_path:
            self.model.save(path=self.saved_finetuned_path, epoch=results['Best Epoch'],
                            state_dict=results['best_state_dict'])

        del results['best_state_dict']
        return results

    def train(self, train_loader):
        self.model.train()
        train_loss = []
        for batch in tqdm(train_loader):
            sentences, labels = batch["x"], batch["y"]
            predictions = self.model(
                sentences, self.pooling_strategy, self.language
            )

            if self.task_type in ["regression", "dual-regression", "sts"]:
                loss = self.loss_fn(predictions.flatten(), labels)
            else:
                loss = self.loss_fn(predictions, labels)

            self.optimizer.zero_grad()
            train_loss.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        return train_loss

    def eval(self, test_loader):
        self.model.eval()
        final_predictions = []
        final_labels = []
        total_test_loss = []

        for batch in test_loader:
            with torch.no_grad():
                sentences, labels = batch["x"], batch["y"]
                final_labels.extend(labels.tolist())
                predictions = self.model(
                    sentences, self.pooling_strategy, self.language
                )

                if self.task_type in ["regression", "dual-regression", "sts"]:
                    loss = self.loss_fn(predictions.flatten(), labels)
                else:
                    loss = self.loss_fn(predictions, labels)
                total_test_loss.append(loss.item())

                if self.task_type == "multilabel":
                    predictions = torch.sigmoid(predictions).data
                    predictions = predictions > 0.10
                    final_predictions.extend(predictions.tolist())
                elif self.task_type in ["regression", "dual-regression", "sts"]:
                    final_predictions.extend(predictions.flatten().tolist())
                else:
                    final_predictions.extend(torch.argmax(predictions, dim=1).tolist())

        return final_predictions, final_labels, total_test_loss

    def metrics(self, preds, labels):
        if self.task_metric_name == 'Pearson':
            pearson_corr = pearsonr(labels, preds)[0]
            logging.info("Testing Pearson: {}".format(100 * round(pearson_corr, 3)))
            return pearson_corr, None

        else:
            metric_val = f1_score(labels, preds, average="macro")
            logging.info("Testing F1 macro: {}".format(round(metric_val, 3)))

            accuracy = accuracy_score(labels, preds)
            logging.info("Testing accuracy: {}".format(round(accuracy, 3)))

            if self.task_metric_name == "Jaccard":
                metric_val = jaccard_score(labels, preds, average="macro")
                logging.info("Testing Jaccard Score: {}".format(round(metric_val, 3)))

            elif self.task_metric_name == "Matthews":
                metric_val = matthews_corrcoef(labels, preds)
                logging.info("Testing Mattews: {}".format(round(metric_val, 3)))

            return metric_val, accuracy

    @staticmethod
    def get_task_vocab(samples, clean, language="ar"):
        vocab = []
        vocab.append("<s>")
        vocab.append("<pad>")
        vocab.append("<unk>")
        vocab.append("</s>")
        for sample in samples:
            if clean:
                sample = normalize_text(sample, lang=language).split()
            else:
                sample = sample.split()
            for word in sample:
                if word not in vocab:
                    vocab.append(word)
        return vocab