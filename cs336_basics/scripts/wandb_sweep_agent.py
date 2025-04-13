import wandb
from cs336_basics.scripts.train_llm import experiment
sweep_id='rwica1wk'

wandb.agent(sweep_id, function=experiment, project="cs336-lr-sweep", count=6)
