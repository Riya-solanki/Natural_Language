# pre-train-pipeline.py
import os
import random
from datetime import datetime
import torch
from Transformer import Decoder
from TrainConfig import TrainConfig
from Trainer import Trainer
from BPETokenizer import BPETokenizer

# ----------------- CONFIGURE PATHS -----------------
DATA_DIR = "/content/Natural_Language/Data"
RAW_TXT_DIR = "/content/Natural_Language/RawTxts"
VOCAB_DIR = "/content/Natural_Language/TokenizerFast"
CHECKPOINT_DIR = "./CheckPoints"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ----------------- Utility: build shards -----------------
def build_shards_from_txts(tokenizer, raw_txt_dir, out_dir, train_ratio=0.95):
    files = [os.path.join(raw_txt_dir, f) for f in os.listdir(raw_txt_dir) if f.endswith(".txt")]
    if not files:
        raise FileNotFoundError(f"No .txt files found in {raw_txt_dir}")
    random.shuffle(files)
    split_idx = int(len(files) * train_ratio)
    train_files, val_files = files[:split_idx], files[split_idx:]

    def encode_concat(file_list):
        all_ids = []
        for f in file_list:
            with open(f, "r", encoding="utf-8") as fh:
                txt = fh.read().strip()
            if not txt:
                continue
            ids = tokenizer.encode(txt)
            eos_id = getattr(tokenizer, "eos_token_id", tokenizer.pad_token_id)
            all_ids.extend(ids + [eos_id])
        return torch.tensor(all_ids, dtype=torch.long)

    train_ids = encode_concat(train_files)
    val_ids = encode_concat(val_files)

    train_path = os.path.join(out_dir, f"Train-{datetime.now().strftime('%Y%m%d%H%M%S')}.pt")
    val_path = os.path.join(out_dir, f"Val-{datetime.now().strftime('%Y%m%d%H%M%S')}.pt")
    torch.save(train_ids, train_path)
    torch.save(val_ids, val_path)
    print(f"Saved train shard: {train_path}, val shard: {val_path}")
    return train_path, val_path

# ----------------- Load tokenizer -----------------
tokenizer = BPETokenizer()

# ----------------- Build shards if missing -----------------
existing_train = [f for f in os.listdir(DATA_DIR) if f.startswith("Train-") and f.endswith(".pt")]
existing_val = [f for f in os.listdir(DATA_DIR) if f.startswith("Val-") and f.endswith(".pt")]
if not existing_train or not existing_val:
    print("No shards found — building from raw .txt files...")
    train_path, val_path = build_shards_from_txts(tokenizer, RAW_TXT_DIR, DATA_DIR)
else:
    train_path = os.path.join(DATA_DIR, sorted(existing_train)[-1])
    val_path = os.path.join(DATA_DIR, sorted(existing_val)[-1])
    print(f"Using existing shards: {train_path}, {val_path}")

# ----------------- Helper: create TrainConfig -----------------
def create_train_config(
    vocab_size,
    model_name,
    tokens_batch_size,
    batch_size,
    warmup_iters,
    max_lr,
    min_lr,
    num_iters=2000,
    train_shard_names=None,
    val_name=None
):
    return TrainConfig(
        tokens_batch_size=tokens_batch_size,
        batch_size=batch_size,
        dec_context_size=512,
        batch_overlap=0,
        betas=(0.9, 0.95),
        vocab_size=vocab_size,
        d_model=786,
        num_heads=12,
        num_decoder_blocks=12,
        pos_enc_dropout=0.0,
        drop_prob=0.1,
        weight_decay=0.01,
        d_feedfwd=786 * 4,
        mask_attention=True,
        pre_norm=True,
        x_data_loader_dtype=torch.int32,
        y_data_loader_dtype=torch.int64,
        load_check_point=False,
        checkpoint_path=CHECKPOINT_DIR,
        checkpoint_name='',
        checkpoint_save_iter=1000,
        num_iters=num_iters,
        eval_val_set=True,
        val_eval_iters=50,
        val_eval_interval=50,
        optimizer_name="AdamW",
        max_lr=max_lr,
        min_lr=min_lr,
        model_name=model_name,
        warmup_iters=warmup_iters,
        clip_grad_norm=1.0,
        replacements={},
        file_name="All Files",
        file_path=RAW_TXT_DIR,
        vocab_path=VOCAB_DIR,
        load_merge_info_name=None,
        load_vocab_name=None,
        data_path=DATA_DIR,
        train_shard_names=train_shard_names,
        val_name=val_name,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        temperature=0.8,
        top_k=50,
        top_p=0.0,
        generation_max_len=128,
        grammar_check=False,
        meter=None
    )

# ----------------- Experiments -----------------
experiments = [
    (12000, 512, 500, 2e-4, 2e-5),
    (12000, 1024, 300, 9e-4, 2e-5),
    (12000, 2048, 625, 6e-4, 2e-5)
]

configs = []
for idx, exp in enumerate(experiments):
    vocab_size, tokens_batch_size, warmup_iters, max_lr, min_lr = exp
    model_name = f"Pipe-{idx}-Sanskrit"
    cfg = create_train_config(
        vocab_size=vocab_size,
        model_name=model_name,
        tokens_batch_size=tokens_batch_size,
        batch_size=1,
        warmup_iters=warmup_iters,
        max_lr=max_lr,
        min_lr=min_lr,
        num_iters=500,  # debug
        train_shard_names=[train_path],
        val_name=val_path
    )
    configs.append(cfg)

# ----------------- Training loop -----------------
for i, cfg in enumerate(configs):
    print(f"🚀 Training config {i+1}/{len(configs)}: {cfg.model_name}")
    trainer = Trainer(cfg)
    trainer.train()
    ckpt_path = os.path.join(cfg.checkpoint_path, f"{cfg.model_name}.pt")
    torch.save(trainer.GPT.state_dict(), ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")
    del trainer
    torch.cuda.empty_cache()

# ----------------- Generate sample from last checkpoint -----------------
last_cfg = configs[-1]
trainer = Trainer(last_cfg)
ckpt_path = os.path.join(last_cfg.checkpoint_path, f"{last_cfg.model_name}.pt")
from Transformer import Decoder
trainer.GPT = Decoder(
    vocab_size=last_cfg.vocab_size,
    d_model=last_cfg.d_model,
    context_size=last_cfg.dec_context_size,
    pos_enc_dropout=last_cfg.pos_enc_dropout,
    num_decoder_blocks=last_cfg.num_decoder_blocks,
    num_heads=last_cfg.num_heads,
    drop_prob=last_cfg.drop_prob,
    d_feedfwd=last_cfg.d_feedfwd,
    pre_norm=last_cfg.pre_norm,
    mask_attention=last_cfg.mask_attention
).to(last_cfg.device)
trainer.GPT.load_state_dict(torch.load(ckpt_path, map_location=last_cfg.device))
print("Loaded last checkpoint for generation.")

prompt_text = "रामः"
prompt_ids = tokenizer.encode(prompt_text)
generated_ids = trainer.generate_text(
    tokenizer=tokenizer,
    prompt_ids=prompt_ids,
    max_len=80
)
generated_text = tokenizer.decode(generated_ids)
print("\n--- Generated Text ---\n", generated_text)
