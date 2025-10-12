import os
from BPETokenizer import BPETokenizer

# === CONFIGURATION ===
CORPUS_FOLDER = "/content/Natural_Language/Cleaned Corpus"
VOCAB_SIZE = 10000
WITHOUT_NEWLINE = True
SKIP_FIRST_CHUNK_IN_LINE = False
REPLACEMENTS = {}
REMOVE_SPECIAL_TOK = True
SAVE_DIR = "/content/Natural_Language/Tokenizer"

# === SETUP ===
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

tokenizer = BPETokenizer()

# === LOAD ALL .TXT FILES ===
txt_files = [f for f in os.listdir(CORPUS_FOLDER) if f.endswith(".txt")]
print(f"📁 Found {len(txt_files)} text files in '{CORPUS_FOLDER}'")

combined_text = ""
for i, file in enumerate(txt_files, 1):
    file_path = os.path.join(CORPUS_FOLDER, file)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            combined_text += "\n" + content
        if i % 50 == 0 or i == len(txt_files):
            print(f"✅ Loaded {i}/{len(txt_files)} files...")
    except Exception as e:
        print(f"⚠️ Could not read {file}: {e}")

print("\n🚀 Starting vocabulary training...\n")

# === TRAIN TOKENIZER ON COMBINED TEXT ===
try:
    TextTokens, Vocab = tokenizer.TrainVocab_fromText(
        combined_text,
        VOCAB_SIZE,
        PrintStat=True,
        PrintStatsEvery_Token=500,
        WithoutNewLine=WITHOUT_NEWLINE,
        SkipFirstChunkInLine=SKIP_FIRST_CHUNK_IN_LINE,
        Replacements=REPLACEMENTS
    )

except ValueError as e:
    if "max() arg is an empty sequence" in str(e):
        print("⚠️ No more bigrams to merge — training stopped early.")
    else:
        raise e

# === SAVE TOKENIZER ===
tokenizer.save(
    SAVE_DIR,
    "/content/Natural_Language/Final-Corpus-Tokenizer-Merge-Info-NL-",
    "/content/Natural_Language/Final-Corpus-Tokenizer-Vocab-NL-",
    Save_SpecialTok=False
)

print("\n🎉 Tokenizer training completed successfully!")
print(f"📦 Final Vocabulary Size: {len(Vocab)}")
print(f"💾 Saved to: {SAVE_DIR}")
