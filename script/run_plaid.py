from pathlib import Path
import wandb
from plaid.esmfold import esmfold_v1
from plaid.compression.hourglass_vq import HourglassVQLightningModule
import argparse
import os
import sys
import numpy as np
from torchdrug.utils import comm

from torchdrug import datasets, transforms, tasks, core
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from peer import ours


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compression_model_id", type=str, default="kyytc8i9")
    parser.add_argument("--dataset", type=str, default="BetaLactamase")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_epoch", type=int, default=500)
    parser.add_argument("--dataset_location", type=str, default="/homefs/home/lux70/storage/data/torchdrug")
    parser.add_argument("--hourglass_weights_dir", type=str, default="/homefs/home/lux70/storage/plaid/checkpoints/hourglass_vq")
    return parser.parse_args()



def set_seed(seed):
    torch.manual_seed(seed + comm.get_rank())
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return



def main(args):
    model = ours.PLAID(args.hourglass_weights_dir, args.compression_model_id)
    truncate_transform = transforms.TruncateProtein(max_length=args.max_length, random=False)
    protein_view_transform = transforms.ProteinView(view="residue")
    transform = transforms.Compose([truncate_transform, protein_view_transform])

    dataset_fn = getattr(datasets, args.dataset)
    dataset = dataset_fn(
        args.dataset_location,
        atom_feature=None,
        bond_feature=None,
        residue_feature="default",
        transform=transform
    )
    train_set, valid_set, test_set = dataset.split()

    task = tasks.PropertyPrediction(model, task=dataset.tasks,
                                    criterion="mse", metric=("mae", "rmse", "spearmanr"),
                                    normalization=False, num_mlp_layer=2)

    optimizer = torch.optim.Adam(task.parameters(), lr=5e-5)
    solver = core.Engine(task, train_set, valid_set, test_set, optimizer, 
                            gpus=[0], batch_size=args.batch_size, logger="wandb")
    

    solver.train(num_epoch=args.num_epoch)
    # update_cfg_dict = {
    # "shorten_factor": hourglass.enc.shorten_factor,
    # "downproj_factor": hourglass.enc.downproj_factor
    # }
    # wandb.config.update(update_cfg_dict)
    solver.evaluate("valid")
    solver.evaluate("test")


if __name__ == "__main__":
    main(parse_args())



import os

# List all directories in the current directory
directories = [d for d in os.listdir(".") if os.path.isdir(d)]

# Iterate over each directory
for directory in directories:
    # Get the list of YAML files in the current directory
    yaml_files = [f for f in os.listdir(directory) if f.endswith(".yaml")]

    # Iterate over each YAML file
    for yaml_file in yaml_files:
        file_path = os.path.join(directory, yaml_file)

        # Read the content of the YAML file
        with open(file_path, "r") as file:
            content = file.readlines()

        # Flag to track if the file was modified
        modified = False

        # Iterate over each line in the content
        for i, line in enumerate(content):
            if line.startswith("    path: ~/scratch/esm-model-weights/"):
                content[i] = f"    path: ~/scratch/esm-model-weights/\n"
                modified = True

        # If the file was modified, write the updated content back to the file
        if modified:
            with open(file_path, "w") as file:
                file.writelines(content)

print("YAML files modified successfully.")