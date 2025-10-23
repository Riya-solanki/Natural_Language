# Trainer.py
import torch
import torch.nn.functional as F
import glob
import os
import time
from Transformer import Decoder
from TrainConfig import TrainConfig

# define optimizer name here instead of importing
ADAM_W = 'AdamW'

class Trainer:
    def __init__(self, trainer_config: TrainConfig):
        self.config = trainer_config
        self.device = trainer_config.device

    # ================= Load PT Files =================
    def load_pt_data(self, prefix="Train"):
        files = glob.glob(os.path.join(self.config.data_path, f"{prefix}-*.pt"))
        if not files:
            raise FileNotFoundError(f"Missing {prefix} PT files in {self.config.data_path}")
        file_path = sorted(files)[-1]
        print(f"Loading {prefix} PT data from {file_path}")
        data = torch.load(file_path)
        return data.long()

    # ================= Batch Preparation =================
    def prepare_batches(self, data):
        context_size = self.config.dec_context_size
        batch_size = self.config.batch_size
        total_len = (len(data) // (batch_size * context_size)) * (batch_size * context_size)
        data = data[:total_len]
        data = data.view(batch_size, -1).transpose(0, 1).contiguous()
        return data

    # ================= Validation =================
    @torch.no_grad()
    def estimate_val_loss(self):
        self.GPT.eval()
        total_loss = 0.0
        context_size = self.config.dec_context_size
        for i in range(0, self.X_val.size(0) - context_size, context_size):
            X_batch = self.X_val[i:i+context_size].to(self.device)
            Y_batch = self.Y_val[i:i+context_size].to(self.device)
            preds = self.GPT(X_batch)
            B, T, C = preds.shape
            preds = preds.view(B*T, C)
            Y_batch = Y_batch.view(B*T)
            loss = F.cross_entropy(preds, Y_batch)
            total_loss += loss.item()
        self.GPT.train()
        return total_loss / max(1, (self.X_val.size(0) // context_size))

    # ================= Training =================
    def train(self):
        print("Initializing Training")
        torch.cuda.empty_cache()

        # Initialize GPT / Decoder
        self.GPT = Decoder(
            vocab_size=self.config.vocab_size,
            d_model=self.config.d_model,
            context_size=self.config.dec_context_size,
            pos_enc_dropout=self.config.pos_enc_dropout,
            num_decoder_blocks=self.config.num_decoder_blocks,
            num_heads=self.config.num_heads,
            drop_prob=self.config.drop_prob,
            d_feedfwd=self.config.d_feedfwd,
            pre_norm=self.config.pre_norm,
            mask_attention=self.config.mask_attention
        ).to(self.device)

        print(f"No. of Parameters: {sum(p.numel() for p in self.GPT.parameters()) / 1e6:.3f} M\n")

        # Load PT data
        X_train = self.load_pt_data("Train")
        Y_train = self.load_pt_data("Val")
        self.X_train = self.prepare_batches(X_train)
        self.Y_train = self.prepare_batches(X_train[1:])
        self.X_val = self.prepare_batches(Y_train)
        self.Y_val = self.prepare_batches(Y_train[1:])
        print(f"Training samples: {self.X_train.size(0)}, Validation samples: {self.X_val.size(0)}\n")

        # Optimizer
        decay_params = [p for p in self.GPT.parameters() if p.requires_grad and p.dim() >= 2]
        no_decay_params = [p for p in self.GPT.parameters() if p.requires_grad and p.dim() < 2]

        optimizer = torch.optim.AdamW(
            [{'params': decay_params, 'weight_decay': self.config.weight_decay},
             {'params': no_decay_params, 'weight_decay': 0.0}],
            lr=self.config.max_lr,
            betas=self.config.betas,
            eps=1e-8
        )

        context_size = self.config.dec_context_size
        for iter in range(self.config.num_iters):
            st_time = time.time()
            optimizer.zero_grad()

            start_idx = (iter * context_size) % (self.X_train.size(0) - context_size)
            X_batch = self.X_train[start_idx:start_idx+context_size].to(self.device)
            Y_batch = self.Y_train[start_idx:start_idx+context_size].to(self.device)

            preds = self.GPT(X_batch)
            B, T, C = preds.shape
            preds = preds.view(B*T, C)
            Y_batch = Y_batch.view(B*T)
            loss = F.cross_entropy(preds, Y_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.GPT.parameters(), self.config.clip_grad_norm)
            optimizer.step()

            if (iter % self.config.val_eval_interval == 0) or (iter == self.config.num_iters - 1):
                val_loss = self.estimate_val_loss()
                print(f"Iter {iter}: Training loss {loss.item():.4f}, Val loss {val_loss:.4f}")

            time_taken = time.time() - st_time
            print(f"Iter {iter}: Loss={loss.item():.4f}, Time={time_taken:.2f}s")

        print("Training Complete")

    # ================= Text Generation =================
    @torch.no_grad()
    def generate_text(self, tokenizer, prompt_ids, max_len=None, temperature=None, top_k=None, top_p=None, meter=None, grammar_check=False):
        self.GPT.eval()
        device = next(self.GPT.parameters()).device
        if max_len is None:
            max_len = self.config.generation_max_len
        if temperature is None:
            temperature = self.config.temperature
        if top_k is None:
            top_k = self.config.top_k
        if top_p is None:
            top_p = self.config.top_p
        if meter is None:
            meter = self.config.meter

        if isinstance(prompt_ids, list):
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        elif isinstance(prompt_ids, torch.Tensor):
            input_ids = prompt_ids.unsqueeze(0).to(device)
        else:
            raise ValueError("prompt_ids must be list or 1D torch.Tensor")

        for _ in range(max_len):
            logits = self.GPT(input_ids)
            next_token_logits = logits[0, -1, :] / max(1e-8, temperature)
            filtered_logits = self._top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)

        return input_ids[0].cpu().tolist()

    def _top_k_top_p_filtering(self, logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
        logits = logits.clone()
        if top_k > 0:
            topk_vals, _ = torch.topk(logits, top_k)
            min_topk = topk_vals[-1]
            logits[logits < min_topk] = filter_value

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            if sorted_indices_to_remove.any():
                first_idx = torch.nonzero(sorted_indices_to_remove)[0].item()
                sorted_indices_to_remove[first_idx] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value
        return logits
