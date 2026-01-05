import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """
    Layer Normalization: Normalizes input across the feature dimension.
    This helps stabilize training by keeping activation distributions consistent.
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit (GELU) activation function.
    A smoother alternative to ReLU, This smoothness provides a "cleaner" landscape for the optimizer (Gradient Descent) to navigate, which is critical when training networks with billions of parameters.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) acting as the position-wise feed-forward network.
    It expands the dimensionality by 4x and then projects it back.
    """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention Mechanism.
    
    1. Splits the input into multiple 'heads' so the model can attend to different 
       parts of the sequence simultaneously.
    2. Computes Scaled Dot-Product Attention.
    3. Uses a causal mask (upper triangular) to prevent the model from seeing future tokens.
    """
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce the projection dim to match desired output dim

        # 1. The Linear Projections
        # We learn 3 different ways to view the input: Query, Key, and Value.
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        
        # 2. The Buffer (The Mask)
        # register_buffer saves this tensor to the model state_dict (so it saves with the model), but tells PyTorch: "Do NOT update this with Gradient Descent."
        # torch.triu creates an Upper Triangular matrix (ones above the diagonal).
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x) # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        # Queries: (b, num_heads, num_tokens, head_dim)
        # Keys Transposed: (b, num_heads, head_dim, num_tokens)
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head
        
        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Fill the "future" spots with negative infinity, When we apply Softmax later, e^{-inf} becomes 0.
        # This ensures the model gives zero attention to future words.
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        # Scale by sqrt(head_dim) to keep gradients stable
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Multiply weights by Values:
        # (b, num_heads, num_tokens, num_tokens) @ (b, num_heads, num_tokens, head_dim) -> (b, num_heads, num_tokens, head_dim)
        # Transpose: (b, num_heads, num_tokens, head_dim) -> (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)
        
        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec



class TransformerBlock(nn.Module):
    """
    A single Transformer block consisting of:
    1. Multi-Head Attention
    2. FeedForward Network
    This block is applied to the input sequence along with Residual Connection, Layer Normalization & Dropout.
    The input and output shapes are identical, so we can stack this block many times.
    """
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    # If you stack 100 layers, the signal (gradients) usually vanishes or explodes by the time it reaches the bottom.
    # The shortcut creates a direct highway for the gradient to flow backwards from the output to the input untouched.
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)   # Shape: [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


class GPTModel(nn.Module):
    """
    The full GPT Architecture:
    Embeddings -> Transformer Blocks x N -> Final Norm -> Output Head
    """
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        # The Softmax function (which comes next) is translation invariant. Adding a constant bias to all logits doesn't change the probabilities. Removing it saves parameters and memory.
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape

        # Shape: (batch_size, seq_len) -> (batch_size, seq_len, emb_dim)
        tok_embeds = self.tok_emb(in_idx)
        
        # Shape: (seq_len) -> (seq_len, emb_dim)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        
        # Shape: (batch_size, seq_len, emb_dim) + (seq_len, emb_dim) -> (batch_size, seq_len, emb_dim)
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        
        # Shape: (batch_size, seq_len, emb_dim) -> (batch_size, seq_len, vocab_size)
        logits = self.out_head(x)
        
        return logits