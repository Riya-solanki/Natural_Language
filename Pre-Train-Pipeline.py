# ===============================================================
# Pre-Training Pipeline (Single Checkpoint Version)
# ===============================================================

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
CHECKPOINT_DIR = "/content/Natural_Language/CheckPoints"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ----------------- UTILITY: BUILD SHARDS -----------------
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
    print(f"✅ Saved train shard: {train_path}, val shard: {val_path}")
    return train_path, val_path

# ----------------- LOAD TOKENIZER -----------------
tokenizer = BPETokenizer()

# ----------------- LOAD OR BUILD SHARDS -----------------
existing_train = [f for f in os.listdir(DATA_DIR) if f.startswith("Train-") and f.endswith(".pt")]
existing_val = [f for f in os.listdir(DATA_DIR) if f.startswith("Val-") and f.endswith(".pt")]

if not existing_train or not existing_val:
    print("No shards found — building from raw .txt files...")
    train_path, val_path = build_shards_from_txts(tokenizer, RAW_TXT_DIR, DATA_DIR)
else:
    train_path = os.path.join(DATA_DIR, sorted(existing_train)[-1])
    val_path = os.path.join(DATA_DIR, sorted(existing_val)[-1])
    print(f"✅ Using existing shards: {train_path}, {val_path}")

# ----------------- CREATE SINGLE TRAIN CONFIG -----------------
def create_single_config():
    vocab_size = 12000
    tokens_batch_size = 512 * 8 * 1
    batch_size = 8
    warmup_iters = 5000
    max_lr = 2e-4
    min_lr = 2e-5
    num_iters = 300005
    checkpoint_save_iter = 50000
    val_eval_interval = 1000

    assert tokens_batch_size % (batch_size * 512) == 0, \
        "Tokens Batch Size must be a multiple of Batch Size × Context Size"
    grad_accum_iters = tokens_batch_size // (batch_size * 512)

    model_name = f"Final-Sanskrit-{vocab_size}"

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
        weight_decay=1e-2,
        d_feedfwd=786 * 4,
        mask_attention=True,
        pre_norm=True,
        x_data_loader_dtype=torch.int32,
        y_data_loader_dtype=torch.int64,
        load_check_point=False,
        checkpoint_path=CHECKPOINT_DIR,
        checkpoint_name='',
        checkpoint_save_iter=checkpoint_save_iter,
        num_iters=num_iters,
        eval_val_set=True,
        val_eval_iters=100,
        val_eval_interval=val_eval_interval,
        optimizer_name='AdamW',
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
        train_shard_names=[train_path],
        val_name=val_path,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        temperature=0.8,
        top_k=50,
        top_p=0.0,
        generation_max_len=128,
        grammar_check=False,
        meter=None
    )

cfg = create_single_config()

# ----------------- TRAIN SINGLE CONFIG -----------------
print(f"\n🚀 Training: {cfg.model_name}")
trainer = Trainer(cfg)
trainer.train()

# ----------------- SAVE SINGLE CHECKPOINT -----------------
ckpt_path = os.path.join(cfg.checkpoint_path, f"{cfg.model_name}.pt")
torch.save(trainer.GPT.state_dict(), ckpt_path)
print(f"💾 Saved single checkpoint: {ckpt_path}")

# ----------------- GENERATE SAMPLE -----------------
trainer.GPT.load_state_dict(torch.load(ckpt_path, map_location=cfg.device))
prompt_text = "रामः"
prompt_ids = tokenizer.encode(prompt_text)
generated_ids = trainer.generate_text(
    tokenizer=tokenizer,
    prompt_ids=prompt_ids,
    max_len=80
)
generated_text = tokenizer.decode(generated_ids)
print("\n--- Generated Text ---\n", generated_text)
