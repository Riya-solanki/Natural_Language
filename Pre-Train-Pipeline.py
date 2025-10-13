import torch
from TrainConfig import TrainConfig
from Trainer import Trainer

ADAM_W = 'AdamW'

# =========================
# CONFIG CREATION FUNCTION
# =========================
def create_train_config(
    vocab_size,
    model_name_suffix,
    tokens_batch_size,
    batch_size,
    warmup_iters,
    max_lr,
    min_lr,
    weight_decay=1e-2,
    checkpoint_save_iter=10000,
    num_iters=300005,
    val_eval_iters=100,
    val_eval_interval=1000,
    train_shard_names=None,
    val_name=None,
    merge_info_name=None,
    vocab_name=None
):
    dec_context_size = 512
    
    return TrainConfig(
        # Model
        tokens_batch_size=tokens_batch_size,
        batch_size=batch_size,
        dec_context_size=dec_context_size,
        batch_overlap=0,
        betas=(0.9, 0.95),
        vocab_size=vocab_size,
        d_model=786,
        num_heads=12,
        num_decoder_blocks=12,
        pos_enc_dropout=0,
        drop_prob=0,
        weight_decay=weight_decay,
        d_feedfwd=512 * 4,
        mask_attention=True,
        pre_norm=True,

        # Data Loader and Checkpointing
        x_data_loader_dtype=torch.int32,
        y_data_loader_dtype=torch.int64,
        load_check_point=False,
        checkpoint_path='./CheckPoints/',
        checkpoint_name='',
        checkpoint_save_iter=checkpoint_save_iter,
        num_iters=num_iters,
        eval_val_set=True,
        val_eval_iters=val_eval_iters,
        val_eval_interval=val_eval_interval,

        # Optimization
        optimizer_name=ADAM_W,
        max_lr=max_lr,
        min_lr=min_lr,
        model_name=model_name_suffix,
        warmup_iters=warmup_iters,
        clip_grad_norm=1.0,

        # Training Files
        replacements={},
        file_name="All Files",
        file_path="/content/Natural_Language/Cleaned Corpus",
        vocab_path="/content/Natural_Language/TokenizerFast",
        load_merge_info_name=merge_info_name,
        load_vocab_name=vocab_name,

        data_path='/content/Natural_Language/Data',
        train_shard_names=train_shard_names,
        val_name=val_name,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )

# =========================
# EXPERIMENT CONFIG LIST
# =========================
configs = []

# Format: (vocab_size, batch_size, tokens_batch_size, warmup_iters, max_lr, min_lr, train_shards, val_shard, merge_info, vocab_file)
experiments = [
    # Main model
    (12000, 8, 512*8*1, 5000, 2e-4, 2e-5,
     ['Final-12000-Sanskrit-Train2024-09-10 23-06-00'],
     'Final-12000-Sanskrit-Val2024-09-10 23-06-00',
     '/content/Natural_Language/TokenizerFast/merges.txt',
     '/content/Natural_Language/TokenizerFast/vocab.json'),

    # Batch size experiments
    (12000, 64, 512*64*2, 300, 9e-4, 2e-5,
     ['Final-12000-Sanskrit-Train2024-09-10 23-06-00'],
     'Final-12000-Sanskrit-Val2024-09-10 23-06-00',
     '/content/Natural_Language/TokenizerFast/merges.txt',
     '/content/Natural_Language/TokenizerFast/vocab.json'),

    (12000, 64, 512*64*1, 625, 6e-4, 2e-5,
     ['Final-12000-Sanskrit-Train2024-09-10 23-06-00'],
     'Final-12000-Sanskrit-Val2024-09-10 23-06-00',
     '/content/Natural_Language/TokenizerFast/merges.txt',
     '/content/Natural_Language/TokenizerFast/vocab.json'),

    (12000, 32, 512*32*1, 1250, 3.5e-4, 2e-5,
     ['Final-12000-Sanskrit-Train2024-09-10 23-06-00'],
     'Final-12000-Sanskrit-Val2024-09-10 23-06-00',
     '/content/Natural_Language/TokenizerFast/merges.txt',
     '/content/Natural_Language/TokenizerFast/vocab.json'),

    (12000, 16, 512*16*1, 2500, 2e-4, 2e-5,
     ['Final-12000-Sanskrit-Train2024-09-10 23-06-00'],
     'Final-12000-Sanskrit-Val2024-09-10 23-06-00',
     '/content/Natural_Language/TokenizerFast/merges.txt',
     '/content/Natural_Language/TokenizerFast/vocab.json'),

    # Vocab size experiments
    (16000, 8, 512*8*1, 5000, 2e-4, 2e-5,
     ['Final-16000-Sanskrit-Train2024-09-09 21-21-21'],
     'Final-16000-Sanskrit-Val2024-09-09 21-21-21',
     'Final-Corpus-Tokenizer-Merge-Info-NL-16000-2024-09-01 00-21-55',
     'Final-Corpus-Tokenizer-Vocab-NL-16000-2024-09-01 00-21-55'),

    (24000, 8, 512*8*1, 5000, 2e-4, 2e-5,
     ['Final-24000-Sanskrit-Train2024-09-09 20-38-10'],
     'Final-24000-Sanskrit-Val2024-09-09 20-38-10',
     'Final-Corpus-Tokenizer-Merge-Info-NL-24000-2024-09-02 17-00-41',
     'Final-Corpus-Tokenizer-Vocab-NL-24000-2024-09-02 17-00-41'),

    (33000, 8, 512*8*1, 5000, 2e-4, 2e-5,
     ['Final-33000-Sanskrit-Train2024-09-04 17-43-46'],
     'Final-33000-Sanskrit-Val2024-09-04 17-43-46',
     'Final-Corpus-Tokenizer-Merge-Info-NL-33000-2024-09-04 14-21-34',
     'Final-Corpus-Tokenizer-Vocab-NL-33000-2024-09-04 14-21-34'),

    (43008, 8, 512*8*1, 5000, 2e-4, 2e-5,
     ['Final-43008-Sanskrit-Train2024-09-07 08-00-01'],
     'Final-43008-Sanskrit-Val2024-09-07 08-00-01',
     'Final-Corpus-Tokenizer-Merge-Info-NL-43008-2024-09-06 18-19-18',
     'Final-Corpus-Tokenizer-Vocab-NL-43008-2024-09-06 18-19-18')
]
# =========================
# CREATE CONFIG OBJECTS
# =========================
pipe_indx = 0
for idx, exp in enumerate(experiments):
    vocab_size, batch_size, tokens_batch_size, warmup_iters, max_lr, min_lr, train_shards, val_shard, merge_info, vocab_file = exp
    model_name_suffix = f'Pipe-{pipe_indx}-{vocab_size}-CFG-{idx}-Complete-Sanskrit'
    cfg = create_train_config(
        vocab_size=vocab_size,
        model_name_suffix=model_name_suffix,
        tokens_batch_size=tokens_batch_size,
        batch_size=batch_size,
        warmup_iters=warmup_iters,
        max_lr=max_lr,
        min_lr=min_lr,
        train_shard_names=train_shards,
        val_name=val_shard,
        merge_info_name=merge_info,
        vocab_name=vocab_file
    )
    configs.append(cfg)

# =========================
# TRAINING LOOP
# =========================
for i, cfg in enumerate(configs):
    print(f"🚀 Starting training for config {i+1}/{len(configs)}: {cfg.model_name}")
    trainer = Trainer(cfg)
    trainer.train()
    del trainer
    torch.cuda.empty_cache()  # Free GPU memory between experiments