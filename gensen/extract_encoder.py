"""Creates a GenSen model from a MultiTask model."""
import os
import pickle
import argparse

import torch

parser = argparse.ArgumentParser()
parser.add_argument(
    "-t",
    "--trained_model_path",
    help="Path containing a saved model",
    required=True,
    type=str
)

parser.add_argument(
    "-s",
    "--save_folder",
    help="Path to save the encoder",
    required=True,
    type=str
)

args = parser.parse_args()

checkpoint = torch.load(args.trained_model_path)
model = checkpoint["model_state_dict"]
print(list(model))

for item in list(model):
    if (
        not item.startswith('encoder') and
        not item.startswith('src_embedding')
    ):
        del model[item]

torch.save(
    model,
    os.path.join(args.save_folder, 'model_params.model')
)