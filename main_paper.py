"""
Train the full BioBERTa model with all modalities.
Uses default hyperparameters (batch_size=128) to match the paper exactly.

main.py in the original repo overrides batch_size to 32 —
this script uses the paper defaults.
"""
from src import Trainer, TrainingArguments, BioBERTaForMaskedLM, BioBertaConfig, Tokenizer

tokenizer = Tokenizer.from_pretrained('pretrained_tokenizers/tokenizer/')

model_config = BioBertaConfig(
    num_hidden_layers=2,
    hidden_size=312,
    intermediate_size=1200,
    vocab_size=len(tokenizer.encoder),
    type_vocab_size=len(tokenizer.type_encoder),
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    mask_token_id=tokenizer.mask_token_id,
    max_position_embeddings=2048,
    pool_method='attention',
    pool_self=True,
)

model = BioBERTaForMaskedLM(model_config)
print('num_params', sum(p.numel() for p in model.parameters()))

args = TrainingArguments(
    tokenizer_path='pretrained_tokenizers/tokenizer',
    dataset_path='data/processed/dataset',
    wandb_mode='online',
    # Paper defaults — do NOT override batch_size (stays at 128)
    dataloader_num_workers=16,
    device='cuda:0',
)

trainer = Trainer(model, tokenizer, args)

trainer.train()
