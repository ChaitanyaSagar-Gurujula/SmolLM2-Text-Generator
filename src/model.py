# Solving for residual std scaling issue
import os
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchtune.modules import RotaryPositionalEmbeddings
import logging
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = 1536  # Expand dimension to 1536
        self.gate_proj = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.act_fn = nn.SiLU()  # Activation function
        self.down_proj.NANOGPT_SCALE_INIT = 1
        
    def forward(self, x):
        gate = self.gate_proj(x)  # Gate projection
        up = self.up_proj(x)     # Up projection
        return self.down_proj(self.act_fn(gate) * up)  # Apply activation and down-project


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = CausalSelfAttention(config)  # Self-attention block
        self.input_layernorm = nn.RMSNorm(config.n_embd, eps=1e-5)  # RMSNorm for inputs
        self.post_attention_layernorm = nn.RMSNorm(config.n_embd, eps=1e-5)  # RMSNorm post-attention
        self.mlp = LlamaMLP(config)  # Llama-style MLP

    def forward(self, x, attention_mask):
        # Use checkpointing for memory-intensive layers
        return checkpoint(self._forward_impl, x, attention_mask, use_reentrant=False)
        # return checkpoint.checkpoint(self._forward_impl, x, attention_mask, use_reentrant=False)
    
    def _forward_impl(self, x, attention_mask):
        # Apply self-attention with normalization
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, attention_mask) + residual

        # Apply MLP with post-attention normalization
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x) + residual
        return x
    

@dataclass
class GPTConfig:
    block_size: int = 2048 # max sequence length
    vocab_size: int = 49152 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 30 # number of layers
    n_head: int = 9 # number of heads
    n_embd: int = 576 # embedding dimension
    num_key_value_heads: int = 3


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert config.n_embd % config.num_key_value_heads == 0

        # Query projection for all heads
        self.cq_attn = nn.Linear(config.n_embd, config.n_embd, bias=False)  # For queries
        # Key-Value projection for grouped heads
        self.ckv_attn = nn.Linear(config.n_embd, 2 * (config.n_embd // config.num_key_value_heads), bias=False)  # For keys and values
        
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.n_head = config.n_head
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.n_embd // config.n_head

        # Rotary Positional Embedding
        self.rope = RotaryPositionalEmbeddings(dim=self.head_dim, max_seq_len=config.block_size)


        # Bias for causal mask
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, attention_mask=None):
        B, T, C = x.size()  # Batch size, sequence length, embedding dimension (n_embd)
        
        # Compute queries
        q = self.cq_attn(x)  # (B, T, C)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        
        # Compute keys and values (shared across grouped heads)
        kv = self.ckv_attn(x)  # (B, T, 2 * (C / num_key_value_heads))
        kv_dim = C // self.num_key_value_heads
        k, v = kv.split(kv_dim, dim=2)  # Split into keys and values
        k = k.view(B, T, self.num_key_value_heads, kv_dim // self.num_key_value_heads).transpose(1, 2)  # (B, kvh, T, hs)
        v = v.view(B, T, self.num_key_value_heads, kv_dim // self.num_key_value_heads).transpose(1, 2)  # (B, kvh, T, hs)
    
        k = torch.repeat_interleave(k, repeats=self.n_head // self.num_key_value_heads, dim=1)
        v = torch.repeat_interleave(v, repeats=self.n_head // self.num_key_value_heads, dim=1)
        
        # Apply RoPE to queries and keys
        q = self.rope(q)
        k = self.rope(k)
    
        # Handle attention mask
        if attention_mask is not None:
            # Expand attention_mask to (B, 1, 1, T)
            attention_mask = attention_mask[:, None, None, :].to(dtype=torch.bool)
            
            # Create causal mask (lower triangular) and convert to bool
            causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool)).view(1, 1, T, T)
            
            # Combine causal mask and padding mask
            attention_mask = causal_mask & attention_mask  # ✅ Now both are torch.bool


        #print(f"q.shape: {q.shape}, k.shape: {k.shape}, v.shape: {v.shape}, attention_mask.shape: {attention_mask.shape}")
        # Replace with Flash Attention (memory efficient)
        y = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attention_mask,  # Combines padding mask
            #is_causal=True,  # Auto-applies causal mask
            dropout_p=0.0
        )

        # Reshape and combine heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
    
        # Output projection
        y = self.c_proj(y)
        return y


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)

        # Transformer layers
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.n_layer)])
        self.final_norm = nn.RMSNorm(config.n_embd, eps=1e-5)

        # Output head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Share weights between input embedding and output head
        self.token_embedding.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.041666666666666664
        if isinstance(module, nn.Linear):
            if hasattr(module, 'NANGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean = 0.0, std = std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std = std)

    def forward(self, idx, attention_mask=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block size {self.config.block_size}"

        # Token and positional embeddings
        token_embeddings = self.token_embedding(idx)

        # Combine embeddings
        x = token_embeddings 

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask)

        # Final layer normalization
        x = self.final_norm(x)

        # Compute logits
        logits = self.lm_head(x)
        
        return logits
    

    def generate(self, input_ids, max_length=50,eos_token_id=None):
        generated_tokens = []
        current_ids = input_ids
        
        # 🔥 Infer device from input_ids
        device = input_ids.device

        for _ in range(max_length):
            # Forward pass to get logits
            logits = self.forward(current_ids)  # Shape: (batch_size, seq_len, vocab_size)
    
            # 🔥 Only take the last token's logits
            logits = logits[:, -1, :]  # Shape: (batch_size, vocab_size)

            next_token =logits.argmax(dim=-1).cpu().item()
            
            # Store token (avoid GPU-CPU issues)
            generated_tokens.append(next_token)
    
            # Append token to input
            current_ids = torch.cat([current_ids, torch.tensor([[next_token]]).to(device)], dim=1)

            # Stop if EOS token is generated
            if eos_token_id is not None and next_token == eos_token_id:
                break
    
        return generated_tokens
    

# Configuration Class
class OptimizerConfig:
    accumulate_grad_in_fp32 = True
    clip_grad = 1.0
    learning_rate = 0.003
    lr_decay_starting_step = 1600000
    lr_decay_steps = 400000
    lr_decay_style = "linear"
    lr_warmup_steps = 2000
    lr_warmup_style = "linear"
    min_decay_lr = 0.0
    adam_beta1 = 0.9
    adam_beta2 = 0.95
    adam_eps = 1.0e-08
    weight_decay = 0.01
    zero_stage = 0
    name = "adamW"
    torch_adam_is_fused = True


if __name__ == "__main__":
    logging.basicConfig(filename='/kaggle/working/training_log.txt', level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s', force=True)
    # Device setup
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    torch.set_float32_matmul_precision('high')
    
    # Seed setup
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)
    
    # Model initialization
    model = GPT(GPTConfig())
    model.to(device)
    #model = torch.compile(model)

    # Load checkpoint if exists
    best_model_path = '/kaggle/working/best_model.pth'
    checkpoint_model_path = '/kaggle/working/checkpoint_model.pth'
    start_epoch = 0
    start_step = 0
    best_loss = float('inf')
    
    if os.path.exists(checkpoint_model_path):
        model_checkpoint = torch.load(checkpoint_model_path, map_location=device, weights_only=True)
        model.load_state_dict(model_checkpoint['model_state_dict'])
        start_epoch = model_checkpoint['epoch']
        start_step = model_checkpoint['step']+1
        best_loss = model_checkpoint['loss']
        logging.info(f"Resuming from epoch {start_epoch}, step {start_step}, best loss {best_loss:.6f}")
        
    # Model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total Parameters: {total_params:,}")
    logging.info(f"Trainable Parameters: {trainable_params:,}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load streaming dataset
    dataset = load_dataset(
        "HuggingFaceTB/smollm-corpus",
        "cosmopedia-v2",
        streaming=True
    )['train']  # Access only the "train" split
    
    # Define the encode function
    def encode(examples):
        # Tokenize the text
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=2048,return_tensors=None)

    # Stream mapping
    dataset = dataset.map(encode, batched=True,remove_columns=dataset.column_names)

    def collate_fn(batch):
        input_ids = torch.tensor([example['input_ids'] for example in batch], dtype=torch.long)
        attention_mask = torch.tensor([example['attention_mask'] for example in batch], dtype=torch.long)
    
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    from torch.utils.data import DataLoader, IterableDataset
    train_loader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)

    # Optimizer setup
    optimizer_config = OptimizerConfig()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        betas=(optimizer_config.adam_beta1, optimizer_config.adam_beta2),
        eps=optimizer_config.adam_eps,
        weight_decay=optimizer_config.weight_decay
    )

    # Training loop
    target_loss = 0.099999
    max_iterations = 6000
    optimizer.zero_grad()

    scaler = torch.GradScaler()  # ✅ Use AMP GradScaler
    autocast_device = "cuda" if "cuda" in device else "cpu"  # ✅ Ensure valid autocast device

    
    if os.path.exists(checkpoint_model_path):
        optimizer.load_state_dict(model_checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(model_checkpoint['scaler_state_dict'])
    
    sample_text = "Once upon a time"  # Text for tracking improvements

    sample_tokens = tokenizer(sample_text, return_tensors='pt').input_ids.to(device)
    #sample_tokens = torch.tensor(sample_tokens).unsqueeze(0)  # Add batch dimension
    
    
    for epoch in range(start_epoch, 100):
        for i, batch in enumerate(train_loader, start=start_step):
            x = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            # PROPER TARGET SETUP
            y = torch.cat([x.clone()[:, 1:], torch.full((x.size(0), 1), tokenizer.eos_token_id, device=device)], dim=1)


            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                  logits = model(x, attention_mask=attention_mask)
                  loss = F.cross_entropy(
                      logits.view(-1, logits.size(-1)),
                      y.view(-1),
                      ignore_index=tokenizer.eos_token_id  # Exclude padding
                  )

            scaler.scale(loss).backward()  # ✅ Apply scaled gradient
    
            # Gradient accumulation (effective batch size = 4)
            if (i+1) % 16 == 0:  # ✅ Ensure last batch updates
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            # Save best model
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save({
                    'epoch': epoch,
                    'step': i,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'loss': best_loss,
                }, best_model_path)
                

            logging.info(f"Epoch {epoch}, Step {i}, Loss: {loss.item():.6f}, Best Loss: {best_loss:.6f}")

            # Perform prediction every 500 steps
            if (i + 1) % 500 == 0:
                model.eval()
                with torch.no_grad():
            
                    generated_tokens = model.generate(sample_tokens, max_length=50,eos_token_id = tokenizer.eos_token_id)
                    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
                    logging.info(f"Step {i + 1} Prompt: {sample_text} \n Generated Token: {generated_tokens} \n Prediction: {generated_text}")
            
                model.train()
                
            if loss.item() <= target_loss:
                logging.info(f"Target loss reached at step {i}. Training completed!")
                break

            if i >= max_iterations:
                torch.save({
                    'epoch': epoch,
                    'step': i,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'loss': best_loss,
                }, checkpoint_model_path)
                logging.info("Max iterations reached. Training stopped.")
                break

        else:
            continue
        break

    logging.info("Training completed!")
    logging.info(f"Final Loss: {loss.item():.6f}")
    logging.info(f"Best Loss Achieved: {best_loss:.6f}")
    logging.info(f"Best Model Saved To: {best_model_path}")
    logging.info(f"Checpoint Model Saved To: {checkpoint_model_path}")