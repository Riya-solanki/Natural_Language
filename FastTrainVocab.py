from tokenizers import ByteLevelBPETokenizer
import os

# === CONFIG ===
CORPUS_DIR = "/content/Natural_Language/Cleaned Corpus"         # your cleaned dataset folder
SAVE_DIR = "/content/Natural_Language/TokenizerFast"            # output folder for the new tokenizer
VOCAB_SIZE = 10000                      # can adjust (e.g., 20k, 30k)

# === PREPARE OUTPUT DIR ===
os.makedirs(SAVE_DIR, exist_ok=True)

# === GATHER ALL FILES ===
files = [os.path.join(CORPUS_DIR, f) for f in os.listdir(CORPUS_DIR) if f.endswith(".txt")]
print(f"📂 Found {len(files)} text files in '{CORPUS_DIR}'")

# === TRAIN TOKENIZER ===
print("🚀 Training fast tokenizer...")
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=files,
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
)

# === SAVE ===
tokenizer.save_model(SAVE_DIR)
print(f"✅ Tokenizer saved to '{SAVE_DIR}'")
