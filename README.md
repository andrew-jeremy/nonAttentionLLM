# nonAttentionLLM
non-attention-based LLM
# Non-Attention LLM

A **non-attention-based large language model (LLM)** architecture adapted for seq2seq tasks (like summarization). It avoids the typical `QK^T V` self-attention in Transformers, instead leveraging **state-space** blocks, **multi-resolution convolution**, **recurrent chunk-level bridging**, and an **external retriever memory**.

## Features

- **State-Space Inspired** (`S4Block`) for near-linear mixing of tokens.  
- **Multi-Resolution Convolution** (`MultiResConvBlock`) capturing local context at multiple dilations.  
- **RecurrentSupervisor** for preserving hidden state across chunks.  
- **SimpleRetrieverMemory** or advanced ANN-based memory for retrieval augmentation.  
- **Chunked** processing of input + partial shift of target for autoregressive seq2seq modeling.

## Repo Structure
non_attention_llm/
├─ README.md
├─ requirements.txt
├─ models/
│   ├─ s4_block.py
│   ├─ multi_res_conv_block.py
│   ├─ retriever.py
│   ├─ recurrent_supervisor.py
│   ├─ proposed_llm.py
│   └─ init.py
├─ train.py
├─ example_inference.py
└─ LICENSE

- **`models/s4_block.py`**: Implements a simplified `S4Block`.  
- **`models/multi_res_conv_block.py`**: Dilation-based local mixing.  
- **`models/retriever.py`**: Simple vs. advanced retrieval memory logic.  
- **`models/recurrent_supervisor.py`**: GRU-based global state bridging.  
- **`models/proposed_llm.py`**: The main `ProposedNonAttentionLLM` class tying it all together.  
- **`train.py`**: Demonstration script for training on a toy or real dataset.  
- **`example_inference.py`**: Example of loading a checkpoint and running inference.  
- **`requirements.txt`**: Python dependencies.  

## Installation

1. **Clone** the repository:
   ```bash
   git clone https://github.com/<YourUsername>/non_attention_llm.git
   cd non_attention_llm

conda create -n nonattention python=3.9
conda activate nonattention

pip install -r requirements.txt

## Training
python train.py \
  --vocab_size 32000 \
  --embed_dim 256 \
  --hidden_dim 512 \
  --chunk_size 64 \
  --max_memory 100 \
  --s4_kernel_size 16 \
  --conv_kernel_size 3 \
  --conv_dilation "1,2,4" \
  --dropout 0.1 \
  --epochs 5 \
  --batch_size 4

  ## Inference
  python example_inference.py \
  --checkpoint best_model.pt \
  --prompt "Summarize:\nThe cat is a domestic species of small carnivorous mammal..."

  It will:
	1.	Load the model + checkpoint.
	2.	Tokenize the prompt.
	3.	Generate or compute next-token predictions.

You can customize generation logic (greedy, sampling, partial chunk decoding, etc.) as needed.

Model Components
	1.	S4Block
	•	Depthwise 1D convolution with a gating mechanism, inspired by structured state-space.
	2.	MultiResConvBlock
	•	Multiple parallel 1D convolutions with different dilation factors, then combined with a residual.
	3.	RecurrentSupervisor
	•	Maintains a global hidden state across chunk boundaries using a GRUCell or LSTMCell.
	4.	SimpleRetrieverMemory
	•	Basic key-value store for chunk embeddings. For larger scale, consider using FAISS or Annoy.
	5.	ProposedNonAttentionLLM
	•	The top-level seq2seq style model that chunk-processes the input, retrieves augmentations, and outputs token logits.

Advanced Features
	•	Replace SimpleRetrieverMemory with FAISS or any advanced ANN library if you have large-scale retrieval needs.
	•	Configure kernel sizes, dilation factors, or S4 param.
	•	Chunk Size can be adapted for memory constraints or longer contexts.
	•	Attention-Free design for near-linear time with respect to sequence length.

Requirements

See requirements.txt for typical packages:

torch>=2.0
numpy
tqdm
transformers
datasets

**Explanation**:

- **README.md** is thorough, covering:
  - Intro / features
  - Repo structure
  - Setup and usage
  - Summaries of each module
  - Potential advanced usage
- You can adjust the text (e.g. referencing your actual GitHub URL or your contact info).

---

## 3. Example: `requirements.txt`

A minimal list might be:

Add more if your code uses them (`faiss`, `annoy`, etc.).  

---

## 4. Example: `train.py` and `example_inference.py`

You can provide minimal scripts that parse command-line args, instantiate `ProposedNonAttentionLLM`, and demonstrate training or inference. The **README** references them, so users can see how to run.

---

With these files in place, commit and push to your GitHub repo. Your **non-attention-based LLM** is now publicly available with a **detailed README** for others to install and experiment with!
