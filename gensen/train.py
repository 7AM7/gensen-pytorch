import os
import json
import time
import logging

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s",
    level=logging.INFO,
    filename="log/ar_training_run_10.log",
)

import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from sklearn import metrics

from gensen.iterators.iterators_factory import IteratorsFactory
from gensen.multitaskmodel import MultitaskModel
from gensen.tokenization import SentencePieceTokenizer

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

config_file_path = "config.json"
config = json.load(open(config_file_path, "r"))
model_dir = config["model"]["model_dir"]

if not os.path.exists(model_dir):
    os.mkdir(model_dir)

# load pre-trained model (if any)
pretrained_path = None
checkpoint = torch.load(pretrained_path, map_location=lambda storage, loc: storage) if pretrained_path else None

batch_size = config["training"]["batch_size"]
max_len_src = config["model"]["max_src_length"]
max_len_trg = config["model"]["max_trg_length"]
buffer_size = config["management"]["buffer_size"]

tasks = config["tasks"]
tasknames = [task["taskname"] for task in tasks]
tasktypes = dict((task["taskname"], task["tasktype"]) for task in tasks)

# Load SentencePiece Tokenizer
logging.info("Loading sentencepiece tokenizer ...")
model_path = os.path.join(config["model"]["vocab_dir"], config["vocab"]["model_prefix"] + '.model')
lowercase = config["vocab"]["lowercase"]
if not os.path.exists(model_path):
    raise ValueError("Could not find sentencepiece model file : {}".format(model_path))
tokenizer = SentencePieceTokenizer(model_file=model_path, lowercase=lowercase)

# Build iterators
logging.info("Creating iterators ...")
iterators = IteratorsFactory(tasks=tasks,
                             tokenizer=tokenizer,
                             device=device,
                             buffer_size=buffer_size,
                             shuffle=True)

logging.info("Model Parameters : ")
logging.info("Source Word Embedding Dim  : {}".format(config["model"]["dim_word_src"]))
logging.info("Target Word Embedding Dim  : {}".format(config["model"]["dim_word_trg"]))
logging.info("Source RNN Hidden Dim  : {}".format(config["model"]["dim_src"]))
logging.info("Target RNN Hidden Dim  : {}".format(config["model"]["dim_trg"]))
logging.info("Source RNN Bidirectional  : {}".format(config["model"]["bidirectional"]))
logging.info("Batch Size : {}".format(config["training"]["batch_size"]))
logging.info("Optimizer : {}".format(config["training"]["optimizer"]))
logging.info("Learning Rate : {}".format(config["training"]["lrate"]))


model = MultitaskModel(
    src_emb_dim=config["model"]["dim_word_src"],
    trg_emb_dim=config["model"]["dim_word_trg"],
    src_vocab_size=config["vocab"]["vocab_size"] + 4,  # add 4 for special tokens,
    trg_vocab_size=config["vocab"]["vocab_size"] + 4,  # add 4 for special tokens,
    src_hidden_dim=config["model"]["dim_src"],
    trg_hidden_dim=config["model"]["dim_trg"],
    tasks=tasks,
    bidirectional=config["model"]["bidirectional"],
    src_pad_token=1,  # pad token id (pre-defined in vocab)
    trg_pad_token=1,
    num_layers_src=config["model"]["n_layers_src"],
    dropout=config["model"]["dropout"],
    pooling_strategy=config["model"]["pooling_strategy"],
    device=device,
).to(device)

logging.info("========== MultiTask Model ==========")
logging.info(model)
logging.info("Total number of trainable parameters: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

n_gpus = config["training"]["n_gpus"]
# model = torch.nn.DataParallel(model, device_ids=range(n_gpus))

lr = config["training"]["lrate"]
optimizer = optim.Adam(model.parameters(), lr=lr)

task_losses = dict((taskname, []) for taskname in tasknames)
task_idxs = dict((taskname, 0) for taskname in tasknames)
examples_processed = dict((taskname, 0) for taskname in tasknames)

updates = 0
epoch = 0
mbatch_times = []

if pretrained_path:
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    del checkpoint
    torch.cuda.empty_cache()
    model.train()

logging.info("Commencing Training ...")
while True:
    start = time.time()

    sampled_task_idx = np.random.randint(low=0, high=len(tasknames))
    sampled_task_name = tasknames[sampled_task_idx]
    sampled_task_type = tasktypes[sampled_task_name]

    # Get a minibatch corresponding to the sampled task
    minibatch = iterators.get_minibatch(
        sampled_task_name,
        task_idxs[sampled_task_name],
        batch_size * n_gpus,
        max_len_src,
        max_len_trg,
        minibatch_type="train",
    )
    task_idxs[sampled_task_name] += batch_size * n_gpus
    examples_processed[sampled_task_name] += batch_size * n_gpus

    # If current task's buffer is exhausted, fetch new one
    if task_idxs[sampled_task_name] >= buffer_size:
        iterators.fetch_buffer(sampled_task_name)
        task_idxs[sampled_task_name] = 0

    optimizer.zero_grad()
    loss, _ = model(minibatch, sampled_task_name)
    task_losses[sampled_task_name].append(loss.item())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    end = time.time()
    mbatch_times.append(end - start)

    if updates % config["management"]["monitor_loss"] == 0 and updates != 0:
        for task in tasknames:
            logging.info(
                "{0} Examples Processed : {1} Loss : {2} minibatches : {3}".format(
                    task,
                    examples_processed[task],
                    round(np.mean(task_losses[task]), 3),
                    len(task_losses[task]),
                )
            )

        logging.info(
            "Average time per mininbatch : {}".format(round(np.mean(mbatch_times), 3))
        )

        logging.info("******************************************************")
        task_losses = dict((taskname, []) for taskname in tasknames)
        mbatch_times = []

    if (
        updates % config["management"]["checkpoint_freq"] == 0
        and updates != 0
    ):
        logging.info("Saving model ...")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            open(os.path.join(model_dir, "model_{}".format(epoch)), "wb"),
        )

    if updates % config["management"]["eval_freq"] == 0:
        with torch.no_grad():
            logging.info("############################")
            logging.info("##### Evaluating model #####")
            logging.info("############################")
            model.eval()

            # Evaluate Seq2Seq tasks
            for taskname in iterators.seq2seq_tasknames_dev:
                validation_loss = []
                for j in range(
                    0,
                    len(iterators.seq2seq_iterator.src_dev[taskname]["data"]),
                    batch_size * n_gpus,
                ):
                    validation_minibatch = iterators.seq2seq_iterator.get_parallel_minibatch(
                        taskname,
                        j,
                        batch_size * n_gpus,
                        max_len_src,
                        max_len_trg,
                        minibatch_type="dev",
                    )
                    loss, _ = model(validation_minibatch, taskname)
                    validation_loss.append(loss.item())

                logging.info(
                    "{0} validation loss : {1}".format(
                        taskname, round(np.mean(validation_loss), 3)
                    )
                )

            # Evaluate pair-classification tasks
            for taskname in iterators.pair_tasknames_dev:
                preds_labels = []
                golden_labels = []
                for j in range(
                    0,
                    len(iterators.pair_classification_iterator.src_dev[taskname]["data"]),
                    batch_size * n_gpus,
                ):
                    validation_minibatch = iterators.pair_classification_iterator.get_parallel_minibatch(
                        taskname, j, batch_size * n_gpus, max_len_src, minibatch_type="dev"
                    )
                    _, logits = model(validation_minibatch, taskname)

                    class_preds = torch.argmax(logits, dim=1).tolist()
                    labels = validation_minibatch["labels"].data.cpu().numpy()

                    preds_labels.extend(class_preds)
                    golden_labels.extend(labels)

                logging.info('{} F1 Macro Score: {:.3f}'.format(taskname,
                                                          metrics.f1_score(golden_labels,
                                                          preds_labels, average="macro")))

            # Evaluate classification tasks
            for taskname in iterators.classification_tasknames_dev:
                preds_labels = []
                golden_labels = []
                for j in range(
                    0,
                    len(iterators.classification_iterator.src_dev[taskname]["data"]),
                    batch_size * n_gpus,
                ):
                    validation_minibatch = iterators.classification_iterator.get_parallel_minibatch(
                        taskname, j, batch_size * n_gpus, max_len_src, minibatch_type="dev"
                    )
                    _, logits = model(validation_minibatch, taskname)

                    class_preds = torch.argmax(logits, dim=1).tolist()
                    labels = validation_minibatch["labels"].data.cpu().numpy()

                    preds_labels.extend(class_preds)
                    golden_labels.extend(labels)

                logging.info('{} F1 Macro Score: {:.3f}'.format(taskname,
                                                          metrics.f1_score(golden_labels,
                                                          preds_labels, average="macro")))

            logging.info("******************************************************")
            model.train()

    updates += batch_size * n_gpus
    epoch += 1
