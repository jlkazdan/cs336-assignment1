from dataclasses import dataclass, field

@dataclass
class llm_train_config:
    vocab_size: int = field(default=32000)
    context_length: int = field(default=256)
    d_model: int = field(default=512)
    d_ff: int = field(default=1344)
    rope_theta: int = field(default=10000)
    num_layers: int = field(default=4)
    num_heads: int = field(default=16)
    total_tokens_processed: int = field(default=50000)  # this should be set much higher for actual training
    training_data: str = field(default='data/tokenized_data/owt_train.bin')
    validation_data: str = field(default='data/tokenized_data/owt_valid.bin')#'data/tokenized_data/TinyStoriesV2-GPT4-valid.bin')
    device: str = field(default='cuda:0')  # change this after moving to the server
    batch_size: int = field(default=64)  # Note: this one doesn't use field()
    # optimizer params
    steps: int = field(default=100000000)
    lr_max: float = field(default=5e-4)
    lr_min: float = field(default=0)
    betas: tuple = field(default=(0.9, 0.999))
    eps: float = field(default=1e-8)
    weight_decay: float = field(default=0.001)
    warmup_steps: int = field(default=50)
    terminal_steps_start: int = field(default=1000000000)  # field(0.75*steps)no terminal steps for now
    gradient_norm: float = field(default=1)
    #logging parameters
    save_every: int = field(default = 5000)
    validation_every: int = field(default = 50)
    save_location: str = field(default = 'cs336_basics/checkpoints/owt/')
    #wandb params
    wandb_project: str = field(default = 'cs336-assignment1')
    experiment_name: str = field(default = 'owt')
