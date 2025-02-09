'''
This script demonstrates a non-attention-based LLM for sequence-to-sequence tasks capable of handling very long sequences (>1M tokens).
The model is based on the "S4" architecture, which uses a combination of local convolutions, recurrent supervision, and optional memory 
retrieval mechanisms to process long sequences in a chunk-based manner. The model is trained on a subset of the CNN/DailyMail dataset for 
summarization tasks as a demonstration. The script includes data loading, training, and evaluation steps.

Below is a step-by-step flowchart for a forward pass on a very long text (1M tokens or more):
	1.	Chunk the Input into manageable blocks (e.g., 64 tokens each).
	2.	Local S4: Within each block, run an S4 layer that processes the entire chunk in near-linear time. Output: chunk embeddings.
	3.	Multi-Res Convolution: Refine local chunk embeddings with a dilated convolutional mixer to capture short- and medium-range dependencies.
	4.	Memory Retrieval (Optional step, repeated periodically):
        •	Generate a chunk-level query.
        •	Retrieve relevant vectors/facts from a large, external memory.
        •	Fuse them into the chunk representation via a small MLP or gating mechanism.
	5.	Recurrent Supervisor: Update a global hidden state with a small RNN or specialized aggregator. This state is passed forward to the next chunk.
	6.	Hierarchical S4: Optionally, stack multiple levels—like “supra-chunk” S4 layers that operate on chunk embeddings to unify context across multiple blocks in a single pass.
	7.	Decoder Head: The final representation of each token (and/or chunk) is fed into a projection layer to predict the next token probabilities.

No standard “QK dot-product attention” is used at any stage. Instead:
	•	Global context is handled by S4’s long-range capabilities + the RNN “supervisor.”
	•	Local context is augmented by convolution blocks.
	•	External knowledge or extended context is integrated via a retrieval-based memory mechanism.
 
 Andrew Kiruluta, Feb 2025
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.optim.lr_scheduler import StepLR 
from torch import optim
import random
import argparse


class S4Block(nn.Module):
    """
    Simplified S4-like layer using depthwise conv + gating.
    """
    def __init__(self, embed_dim: int, kernel_size: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            groups=embed_dim,
            padding='same'  # uses PyTorch >=1.9 "same" conv
        )
        self.pointwise = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (B, T, E)
        """
        x_permuted = x.transpose(1, 2)  # (B, E, T)
        x_conv = self.conv(x_permuted)
        x_conv = F.gelu(x_conv)
        x_conv = self.pointwise(x_conv)
        x_conv = self.dropout(x_conv)
        # Residual
        x_out = x_permuted + x_conv
        x_out = x_out.transpose(1, 2)  # (B, T, E)
        return x_out

    
class MultiResConvBlock(nn.Module):
    """
    Multi-dilation convolution for local pattern capturing.
    """
    def __init__(self, embed_dim: int, kernel_size: int, dilation_factors, dropout):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=kernel_size,    
                dilation=d,
                padding=d
            ) for d in dilation_factors
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, E)
        """
        x_permuted = x.transpose(1, 2)  # (B, E, T)
        outputs = []
        for conv in self.convs:
            y = conv(x_permuted)
            y = F.gelu(y)
            outputs.append(y)
        combined = torch.stack(outputs, dim=0).sum(dim=0)
        combined = self.dropout(combined)
        out = x_permuted + combined
        out = out.transpose(1, 2)
        return out


class RecurrentSupervisor(nn.Module):
    """
    A global recurrent cell (GRUCell) for cross-chunk memory.
    """
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.gru_cell = nn.GRUCell(embed_dim, hidden_dim)

    def forward(self, chunk_rep: torch.Tensor, global_state: torch.Tensor):
        new_state = self.gru_cell(chunk_rep, global_state)
        return new_state


class SimpleRetrieverMemory(nn.Module):
    """
    Naive key-value memory for demonstration. Could be replaced with more advanced memory modules such as 
    advanced LSH-based retrievers, FAISS, Annoy, Scann, HNSW, etc.            
    """ 
    def __init__(self, key_dim: int, val_dim: int, max_entries: int):
        super().__init__()
        self.key_dim = key_dim
        self.val_dim = val_dim
        self.max_entries = max_entries
        self.register_buffer("keys", torch.zeros(max_entries, key_dim))
        self.register_buffer("vals", torch.zeros(max_entries, val_dim))
        self.memory_index = 0

    def store(self, keys: torch.Tensor, vals: torch.Tensor):
        B = keys.size(0)
        for i in range(B):
            idx = (self.memory_index + i) % self.max_entries
            self.keys[idx] = keys[i]
            self.vals[idx] = vals[i]
        self.memory_index = (self.memory_index + B) % self.max_entries

    def retrieve(self, query: torch.Tensor, top_k: int = 1):
        B = query.size(0)
        # naive L2 distance
        query_expanded = query.unsqueeze(1)  # (B,1,key_dim)
        keys_expanded = self.keys.unsqueeze(0)  # (1, max_entries, key_dim)
        dist = torch.norm(query_expanded - keys_expanded, dim=-1)  # (B, max_entries)
        _, indices = torch.topk(dist, k=top_k, largest=False, dim=1)
        retrieved_vals = []
        for b in range(B):
            vals_for_b = self.vals[indices[b]]  # (top_k, val_dim)
            retrieved_vals.append(vals_for_b.unsqueeze(0))
        return torch.cat(retrieved_vals, dim=0)  # (B, top_k, val_dim)


class ProposedNonAttentionLLM(nn.Module):
    """
    Non-attention-based LLM adapted for seq2seq tasks (such as summarization).
    - We chunk the input (prompt+partial target) for forward pass.
    - Produces next-token logits for the entire sequence,
      then we slice out the portion for the target.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        chunk_size: int,
        max_memory_entries: int,
        dropout: float,
        s4_kernel_size: int,
        conv_dilation: list,
        conv_kernel_size: int      # new param for MultiResConvBlock
    ):
        """
        Args:
            vocab_size (int): size of the vocabulary
            embed_dim (int): embedding dimension
            hidden_dim (int): hidden size for the recurrent supervisor
            chunk_size (int): how many tokens per chunk
            max_memory_entries (int): capacity for retriever memory
            dropout (float): dropout probability
            s4_kernel_size (int): kernel size for S4Block
            conv_dilation (list): list of dilation factors (e.g. [1,2,4])
            conv_kernel_size (int): kernel size for MultiResConvBlock
        """
        super().__init__()
        self.chunk_size = chunk_size
        self.embed_dim = embed_dim
        
        self.embed = nn.Embedding(vocab_size, embed_dim)
        
        self.s4_block = S4Block(embed_dim, kernel_size=s4_kernel_size, dropout=dropout)
        self.conv_block = MultiResConvBlock(embed_dim, kernel_size=conv_kernel_size, dilation_factors=conv_dilation, dropout=dropout)
        
        self.supervisor = RecurrentSupervisor(embed_dim, hidden_dim)
        self.init_global = nn.Parameter(torch.zeros(hidden_dim))
        
        self.memory = SimpleRetrieverMemory(embed_dim, embed_dim, max_memory_entries)
        self.fuse_retrieved = nn.Linear(embed_dim * 2, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor, target_ids: torch.Tensor, retrieve=True, top_k=1):
        """
        input_ids: (B, T_in)
        target_ids: (B, T_out)
        We'll do a simple approach: combine input_ids + target_ids[:, :-1]
        for an autoregressive style, then produce next-token logits.
        Then we slice the portion that corresponds to target output
        and compute cross-entropy.
        """
        device = input_ids.device
        B, T_in = input_ids.shape
        T_out = target_ids.shape[1]
        
        # Combined sequence
        # (Note: some prefer input_ids + target_ids, or other shifting strategies.
        #  We'll keep the partial shift approach.)
        combined = torch.cat([input_ids, target_ids[:, :-1]], dim=1) if T_out > 1 else input_ids
        x = self.embed(combined)  # (B, T_in + T_out - 1, E)
        T_combined = x.size(1)

        # Initialize global state
        global_state = self.init_global.unsqueeze(0).expand(B, -1)
        
        # chunk-based forward
        outputs = []
        for start in range(0, T_combined, self.chunk_size):
            end = min(start+self.chunk_size, T_combined)
            chunk = x[:, start:end, :]  # (B, chunk_len, E)

            c_s4 = self.s4_block(chunk)
            c_conv = self.conv_block(c_s4)

            # chunk-level rep
            chunk_rep = c_conv.mean(dim=1)
            if retrieve:
                retrieved_vals = self.memory.retrieve(chunk_rep, top_k=top_k)
                retrieved_agg = retrieved_vals.mean(dim=1)
                fused = torch.cat([chunk_rep, retrieved_agg], dim=-1)
                chunk_rep = torch.tanh(self.fuse_retrieved(fused))

            global_state = self.supervisor(chunk_rep, global_state)

            # store in memory
            if retrieve:
                self.memory.store(chunk_rep.detach(), chunk_rep.detach())

            # produce logits
            chunk_logits = self.lm_head(self.dropout(c_conv))  # (B, chunk_len, vocab_size)
            outputs.append(chunk_logits)
        
        logits = torch.cat(outputs, dim=1)  # (B, T_combined, vocab_size)

        # slice out the portion for target
        # The target portion starts at T_in and spans T_out tokens
        if T_out > 1:
            predicted_for_target = logits[:, T_in : T_in + T_out, :]
        else:
            # if T_out == 1
            predicted_for_target = logits[:, T_in:T_in+1, :]

        # If for some reason the model didn't produce enough tokens, clamp
        min_len = min(predicted_for_target.size(1), target_ids.size(1))
        predicted_for_target = predicted_for_target[:, :min_len, :]
        target_ids = target_ids[:, :min_len]

        ce_loss = nn.CrossEntropyLoss()
        predicted_for_target = predicted_for_target.contiguous().reshape(-1, predicted_for_target.size(-1))
        target_ids = target_ids.contiguous().reshape(-1)
    
        loss = ce_loss(predicted_for_target, target_ids)
        
        #loss = ce_loss(
        #    predicted_for_target.reshape(-1, predicted_for_target.size(-1)),
        #    target_ids.reshape(-1))
       
        #loss = ce_loss(
        #    predicted_for_target.contiguous().reshape(-1, predicted_for_target.size(-1)), 
        #    target_ids.contiguous().reshape(-1)
        #    )
        return loss
        
###################################################
# (B) DATA LOADING: CNN/DAILYMAIL w/ BPE TOKENIZER
###################################################

def load_cnn_dailymail_subset(split="train", sample_size=100, version="3.0.0"):
    """
    Loads a subset of CNN/DailyMail dataset from Hugging Face.
    """
    dataset = load_dataset("cnn_dailymail", version, split=split)
    if sample_size < len(dataset):
        dataset = dataset.shuffle(seed=42).select(range(sample_size))
    # Return (document, summary) pairs
    samples = []
    for item in dataset:
        doc = item["article"]
        summ = item["highlights"]
        samples.append((doc, summ))
    return samples

class CNNDailyMailSummDataset(Dataset):
    """
    Summarization dataset: (doc, summary) => (input_ids, target_ids)
    We'll do the subword tokenization in __getitem__.
    """
    def __init__(self, samples, tokenizer, max_length=256):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        document, summary = self.samples[idx]
        # We'll build a prompt: "Summarize:\n{document}\nSummary:"
        prompt = f"Summarize:\n{document}\nSummary:"
        
        # tokenize
        input_enc = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt"
        )
        target_enc = self.tokenizer(
            summary,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = input_enc["input_ids"].squeeze(0)   # (seq_len,)
        target_ids = target_enc["input_ids"].squeeze(0) # (seq_len,)

        return input_ids, target_ids


def collate_fn_pad_bpe(batch):
    """
    Pad subword tokenized inputs & targets to the max length in the batch.
    We'll assume pad_token_id is set properly in the tokenizer.
    """
    input_lens = [len(x[0]) for x in batch]
    target_lens = [len(x[1]) for x in batch]
    max_in = max(input_lens)
    max_tg = max(target_lens)

    # For GPT-2, we typically set pad_token = eos_token
    # Let's fetch from the first sample or define a default
    pad_token_id = 50256  # GPT-2 default
    padded_in, padded_tg = [], []
    
    for (inp, tgt) in batch:
        if len(inp) < max_in:
            pad_len = max_in - len(inp)
            inp = torch.cat([inp, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        if len(tgt) < max_tg:
            pad_len = max_tg - len(tgt)
            tgt = torch.cat([tgt, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        padded_in.append(inp.unsqueeze(0))
        padded_tg.append(tgt.unsqueeze(0))
    
    batch_in = torch.cat(padded_in, dim=0)   # (B, max_in)
    batch_tg = torch.cat(padded_tg, dim=0)   # (B, max_tg)
    return batch_in, batch_tg

#####################################################
# (C) TRAIN/VAL/TEST + PERPLEXITY (SINGLE-CELL DEMO)
#####################################################

def run_epoch(model, loader, optimizer=None, device="cuda"):
    """
    Runs one epoch. If optimizer is None, we do eval mode.
    Returns average loss across the dataset.
    """
    is_train = (optimizer is not None)
    if is_train:
        model.train()
    else:
        model.eval()
    
    total_loss = 0.0
    count = 0
    for (input_ids, target_ids) in loader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        
        if is_train:
            optimizer.zero_grad()
    
        loss = model(input_ids, target_ids)
        loss = loss.contiguous() 
        
        if is_train:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        count += 1
    
    avg_loss = total_loss / count if count > 0 else 0.0
    return avg_loss

def perplexity_from_loss(loss):
    """
    Perplexity = exp(loss), where `loss` is cross-entropy in nats (or typically log base e).
    """
    return math.exp(loss)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='nonAttentionLLM model')
    parser.add_argument('--learning_rate', type=int, default=1e-6, help='learning rate') 
    parser.add_argument('--peak_lr', type=int, default=1e-5, help='peak learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='number of warmup epochs')
    parser.add_argument('--decay_epochs', type=int, default=5, help='number of decay epochs')
    parser.add_argument('--num_epochs', type=int, default=30, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--embed_dim', type=int, default=512, help='embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=1024, help='hidden size for the recurrent supervisor')
    parser.add_argument('--chunk_size', type=int, default=256, help='how many tokens per chunk')
    parser.add_argument('--max_memory_entries', type=int, default=100, help='capacity for retriever memory')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout probability')
    parser.add_argument('--s4_kernel_size', type=int, default=16, help='kernel size for S4Block')
    parser.add_argument('--conv_dilation', type=list, default=[1, 2, 4], help='list of dilation factors (e.g. [1,2,4])')
    parser.add_argument('--conv_kernel_size', type=int, default=3, help='kernel size for MultiResConvBlock')
    args = parser.parse_args()
    
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available(): # apple mps
            device = torch.device("mps")
    else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    print(f'Using device: {device}')
    
    # 1) Load subsets
    # We'll do 300 train, 50 val, 50 test for demonstration (tiny).
    train_samples = load_cnn_dailymail_subset(split="train", sample_size=150000)     # Training Split: ~287,227 examples
    val_samples   = load_cnn_dailymail_subset(split="validation", sample_size=2500)  # Validation Split: ~13,368 examples
    test_samples  = load_cnn_dailymail_subset(split="test", sample_size=50)        # Test Split: ~11,490 examples

    # 2) BPE Tokenizer (GPT-2)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # needed for padding

    # 3) Build Datasets & DataLoaders
    train_dataset = CNNDailyMailSummDataset(train_samples, tokenizer, max_length=256)
    val_dataset   = CNNDailyMailSummDataset(val_samples,   tokenizer, max_length=256)
    test_dataset  = CNNDailyMailSummDataset(test_samples,  tokenizer, max_length=256)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  collate_fn=collate_fn_pad_bpe)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_pad_bpe)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_pad_bpe)

    # 4) Initialize the Non-Attention Model
    model = ProposedNonAttentionLLM(
        vocab_size=tokenizer.vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        chunk_size=args.chunk_size,
        max_memory_entries=args.max_memory_entries,
        dropout=args.dropout,
        s4_kernel_size=args.s4_kernel_size,
        conv_dilation=args.conv_dilation,
        conv_kernel_size=args.conv_kernel_size 
    ).to(device)

    # 5) Optimizer
    #optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    # 2) Define a scheduler: reduce LR by factor of 0.1 every 3 epochs
    #scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

    initial_lr = args.learning_rate
    peak_lr = args.peak_lr
    warmup_epochs = args.warmup_epochs
    decay_epochs = args.decay_epochs
    def lr_schedule(epoch):
        if epoch < warmup_epochs:
            # linear ramp up
            return initial_lr + (peak_lr - initial_lr)*(epoch / warmup_epochs)
        else:
            # step decay after warmup
            return peak_lr * (0.1 ** ((epoch - warmup_epochs) // 5))
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
    
    # 6) Train for a few epochs, measure val loss & perplexity
    num_epochs = args.num_epochs
    best_val_loss = float('inf')  # keep track of the lowest val loss
    best_model_path = "best_nonatt_llm.pt"
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        
        # set LR
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_schedule(epoch)
        
        # ------------------------ TRAIN ------------------------
        for batch in train_loader:
            # your usual forward/backward code
            optimizer.zero_grad()
            input_ids, target_ids = batch
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            loss = model(input_ids, target_ids)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        train_loss = total_train_loss / len(train_loader)
        # Print stats, track best
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
                f"Train Loss: {train_loss:.4f} ")
        # ---------------------- VALIDATION ----------------------
        model.eval()
        total_val_loss = 0.0
        if epoch % 5 == 0:
            with torch.no_grad():
                for batch in val_loader:
                    input_ids, target_ids = batch
                    input_ids = input_ids.to(device)
                    target_ids = target_ids.to(device)
                    
                    val_loss = model(input_ids, target_ids)
                    total_val_loss += val_loss.item()
            val_loss = total_val_loss / len(val_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}] | "
                f"Val Loss: {val_loss:.4f} ")
            
        # 4) Step the scheduler AFTER val to update LR
        #scheduler.step()

        print(f"Epoch {epoch+1}, LR = {optimizer.param_groups[0]['lr']}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, "best_model.path")
            print("=> New best model saved.")
    print("Training complete.")
'''
    # 7) Final Test: Evaluate on held-out set, measure test perplexity
    test_loss = run_epoch(model, test_loader, optimizer=None, device=device)
    test_ppl  = perplexity_from_loss(test_loss)
    print("\n[TEST RESULTS]")
    print(f"Test Loss: {test_loss:.4f}, Test Perplexity: {test_ppl:.2f}")

    print("\nAll done! Model trained & tested in a single cell.")
'''