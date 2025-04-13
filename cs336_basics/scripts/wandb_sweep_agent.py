import wandb
import argparse
from cs336_basics.scripts.train_llm import experiment

parser = argparse.ArgumentParser()
parser.add_argument('--sweep_id', type=str, default=None)
parser.add_argument('--project', type=str)
args = parser.parse_args()
wandb.agent(args.sweep_id, function=experiment, project=args.project, count=6)
