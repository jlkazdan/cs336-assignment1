import wandb 
from cs336_basics.scripts.train_llm import experiment

sweep_config = {
    'method': 'grid',  # Grid, random, or bayes
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'lr': {
            'values': [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
        },
        }
    }


sweep_id = wandb.sweep(sweep=sweep_config, project="cs336-lr-sweep")
print(sweep_id)
