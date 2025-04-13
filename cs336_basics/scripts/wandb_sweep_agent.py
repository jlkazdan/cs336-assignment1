import wandb
from cs336_basics.scripts.train_llm import experiment
sweep_id='9qkzufpp'

wandb.agent(sweep_id, function=experiment, project="cs336-bs-sweep", count=6)
