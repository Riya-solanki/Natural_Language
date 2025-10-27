import torch
from tokenizers import ByteLevelBPETokenizer
from datetime import datetime
import pandas as pd
import os
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# ===========================
# Task Constants
# ===========================
PRE_TRAIN = "Pre-Train"
SEM_ANLG = "SemAnlg"
SIMILE = "Simile"


class DataLoader:
    def __init__(self, save_path: str):
        self.save_path = save_path
        self.train_batch_index = 0
        self.val_batch_index = 0

    # =================== PRE-TRAIN DATA ===================
    def extractData_MultipleDocs(self, file_path, train_split_percentage, vocab_path, merge_info_name, vocab_name):
        tokenizer = ByteLevelBPETokenizer(
            os.path.join(vocab_path, vocab_name),
            os.path.join(vocab_path, merge_info_name)
        )

        print("🚀 Starting Pre-Training data extraction...")
        all_tokens = []

        for filename in os.listdir(file_path):
            path = os.path.join(file_path, filename)
            if not filename.endswith(".txt"):
                continue
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()

            tokens = tokenizer.encode(text).ids
            all_tokens += tokens + [tokenizer.token_to_id("<pad>")]  # separator
            print(f"✅ Tokens in {filename}: {len(tokens)}")

        print(f"📊 Total tokens collected: {len(all_tokens)}")
        data = torch.tensor(all_tokens)

        split_idx = int(train_split_percentage * len(data))
        train_data = data[:split_idx]
        val_data = data[split_idx:]

        dt = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        torch.save(train_data, os.path.join(self.save_path, f"Train-{dt}.pt"))
        torch.save(val_data, os.path.join(self.save_path, f"Val-{dt}.pt"))
        print(f"✅ Pre-Training data saved at {self.save_path}")

    # =================== SEMANTIC ANALOGIES ===================
    def extract_Analogies_data(self, csv_path, vocab_path, merge_info_name, vocab_name,
                               shuffle=True, train_split_percentage=0.9, train_name="Train", val_name="Val",
                               without_new_line=False, skip_first_chunk_in_line=False, max_pad_len=512, truncation=False):
        tokenizer = ByteLevelBPETokenizer(
            os.path.join(vocab_path, vocab_name),
            os.path.join(vocab_path, merge_info_name)
        )

        print(f"🚀 Loading Semantic Analogies data from: {csv_path}")
        csv_data = pd.read_csv(csv_path)
        data = []

        for _, row in csv_data.iterrows():
            tokens = []
            words = [transliterate(row[i], sanscript.SLP1, sanscript.IAST) for i in range(1, 5)]
            tokens.append(tokenizer.token_to_id("<s>"))  # start token

            for w in words:
                tokens += tokenizer.encode(w).ids
                tokens.append(tokenizer.token_to_id("<pad>"))  # separator

            tokens.append(tokenizer.token_to_id("</s>"))  # end token
            data.append(tokens)

        mask = torch.zeros((len(data), max_pad_len), dtype=torch.int8)
        x_data = torch.ones((len(data), max_pad_len), dtype=torch.int32) * tokenizer.token_to_id("<pad>")
        y_data = torch.ones(x_data.shape, dtype=torch.int32) * tokenizer.token_to_id("<pad>")

        for indx, line in enumerate(data):
            mask[indx, : len(line)] = 1
            if not truncation and len(line) > max_pad_len:
                raise ValueError(f"Length at index {indx} exceeds 'max_pad_len'")
            x_data[indx, : len(line)] = torch.tensor(line, dtype=torch.int32)
            y_data[indx, : len(line) - 1] = torch.tensor(line[1:], dtype=torch.int32)

        if shuffle:
            shuffle_indices = torch.randperm(x_data.size(0))
            x_data = x_data.index_select(0, shuffle_indices)
            y_data = y_data.index_select(0, shuffle_indices)
            mask = mask.index_select(0, shuffle_indices)

        split_idx = int(train_split_percentage * len(x_data))
        x_train, x_val = x_data[:split_idx], x_data[split_idx:]
        y_train, y_val = y_data[:split_idx], y_data[split_idx:]
        mask_train, mask_val = mask[:split_idx], mask[split_idx:]

        dt = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        torch.save(x_train, os.path.join(self.save_path, f"X-{train_name}-{dt}.pt"))
        torch.save(y_train, os.path.join(self.save_path, f"Y-{train_name}-{dt}.pt"))
        torch.save(mask_train, os.path.join(self.save_path, f"Mask-{train_name}-{dt}.pt"))
        torch.save(x_val, os.path.join(self.save_path, f"X-{val_name}-{dt}.pt"))
        torch.save(y_val, os.path.join(self.save_path, f"Y-{val_name}-{dt}.pt"))
        torch.save(mask_val, os.path.join(self.save_path, f"Mask-{val_name}-{dt}.pt"))
        print(f"✅ Semantic Analogies data saved successfully in {self.save_path}")

    # =================== SIMILE DATA ===================
    def readAndAugmentSimileData(self, file_path, vocab_path, merge_info_name, vocab_name,
                                 max_len=512, truncation=False, shuffle=True, train_split_percentage=0.9,
                                 to_lower_case=True):
        tokenizer = ByteLevelBPETokenizer(
            os.path.join(vocab_path, vocab_name),
            os.path.join(vocab_path, merge_info_name)
        )

        print(f"🚀 Starting data extraction for SIMILE task ...")
        print(f"📖 Loading Simile data from: {file_path}")
        data = pd.read_excel(file_path)

        # ✅ Match your actual Excel columns
        required_cols = [
            'Input String',
            'Word Indicating Similarity',
            'Object of Comparison(Upameya)',
            'Standard of Comparison (Upamāna)'
        ]

        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise KeyError(f"Missing columns in Excel: {missing_cols}")

        if to_lower_case:
            data = data.applymap(lambda x: x.lower() if isinstance(x, str) else x)

        data = data.dropna(subset=required_cols)

        input_augmented = []
        output_augmented = []

        for _, row in data.iterrows():
            text = row['Input String']
            sim_word = row['Word Indicating Similarity']
            input_augmented.append(text)
            output_augmented.append(sim_word)

        all_tokens = []
        for inp, out in zip(input_augmented, output_augmented):
            tokens = [tokenizer.token_to_id("<s>")]
            tokens += tokenizer.encode(inp).ids
            tokens.append(tokenizer.token_to_id("<pad>"))
            tokens += tokenizer.encode(out).ids
            tokens.append(tokenizer.token_to_id("</s>"))
            all_tokens.append(tokens)

        mask = torch.zeros((len(all_tokens), max_len), dtype=torch.int8)
        x_data = torch.ones((len(all_tokens), max_len), dtype=torch.int32) * tokenizer.token_to_id("<pad>")
        y_data = torch.ones(x_data.shape, dtype=torch.int32) * tokenizer.token_to_id("<pad>")

        for indx, line in enumerate(all_tokens):
            mask[indx, : len(line)] = 1
            if not truncation and len(line) > max_len:
                raise ValueError(f"Length at index {indx} exceeds max_len")
            x_data[indx, : len(line)] = torch.tensor(line, dtype=torch.int32)
            y_data[indx, : len(line) - 1] = torch.tensor(line[1:], dtype=torch.int32)

        if shuffle:
            shuffle_indices = torch.randperm(x_data.size(0))
            x_data = x_data.index_select(0, shuffle_indices)
            y_data = y_data.index_select(0, shuffle_indices)
            mask = mask.index_select(0, shuffle_indices)

        split_idx = int(train_split_percentage * len(x_data))
        x_train, x_val = x_data[:split_idx], x_data[split_idx:]
        y_train, y_val = y_data[:split_idx], y_data[split_idx:]
        mask_train, mask_val = mask[:split_idx], mask[split_idx:]

        dt = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        torch.save(x_train, os.path.join(self.save_path, f"X-Train-{dt}.pt"))
        torch.save(y_train, os.path.join(self.save_path, f"Y-Train-{dt}.pt"))
        torch.save(mask_train, os.path.join(self.save_path, f"Mask-Train-{dt}.pt"))
        torch.save(x_val, os.path.join(self.save_path, f"X-Val-{dt}.pt"))
        torch.save(y_val, os.path.join(self.save_path, f"Y-Val-{dt}.pt"))
        torch.save(mask_val, os.path.join(self.save_path, f"Mask-Val-{dt}.pt"))

        print(f"✅ Simile Data processed and saved successfully in {self.save_path}")


# =================== MAIN ===================
if __name__ == "__main__":
    # 👇 change this to "Pre-Train", "SemAnlg", or "Simile" as needed
    data_extraction_task = SIMILE

    save_path = '/content/Natural_Language/Data/'
    os.makedirs(save_path, exist_ok=True)
    vocab_path = '/content/Natural_Language/TokenizerFast'
    merge_info_name = 'merges.txt'
    vocab_name = 'vocab.json'

    data_loader = DataLoader(save_path=save_path)

    if data_extraction_task == PRE_TRAIN:
        file_path = '/content/Natural_Language/Cleaned Corpus/'
        data_loader.extractData_MultipleDocs(
            file_path=file_path,
            train_split_percentage=0.98,
            vocab_path=vocab_path,
            merge_info_name=merge_info_name,
            vocab_name=vocab_name
        )

    elif data_extraction_task == SEM_ANLG:
        csv_path = '/content/Natural_Language/Eval_Tasks/Semantic Analogies.csv'
        data_loader.extract_Analogies_data(
            csv_path=csv_path,
            vocab_path=vocab_path,
            merge_info_name=merge_info_name,
            vocab_name=vocab_name
        )

    elif data_extraction_task == SIMILE:
        file_path = '/content/Natural_Language/Simile_Data.xlsx'
        data_loader.readAndAugmentSimileData(
            file_path=file_path,
            vocab_path=vocab_path,
            merge_info_name=merge_info_name,
            vocab_name=vocab_name
        )
