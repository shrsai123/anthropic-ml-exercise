import os
from typing import Optional, List
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_attention(
    attn_weights: torch.Tensor,
    input_ids: torch.Tensor,
    tokenizer,
    output_dir: str = "attention_plots",
    layer: int = 0,
    example_idx: int = 0,
    head_idx: int = 0,
    epoch: Optional[int] = None
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    
    attn = attn_weights[example_idx, head_idx].cpu().numpy()
    ids = input_ids[example_idx].cpu()
    
 
    tokens = tokenizer.convert_ids_to_tokens(ids)
    if tokenizer.pad_token in tokens:
        pad_start = tokens.index(tokenizer.pad_token)
        tokens = tokens[:pad_start]
        attn = attn[:pad_start, :pad_start]
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot heatmap
    sns.heatmap(
        attn,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="viridis",
        ax=ax,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    title = f"Layer {layer} - Head {head_idx}"
    if epoch is not None:
        title += f" (Epoch {epoch})"
    ax.set_title(title, fontsize=12)
    
    ax.set_xlabel("Key Tokens")
    ax.set_ylabel("Query Tokens")
    
    # Rotate labels for readability
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    # Save figure
    filename = f"layer{layer}_head{head_idx}"
    if epoch is not None:
        filename += f"_epoch{epoch}"
    filename += ".png"
    
    path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return path


def visualize_all_heads(
    attn_weights: torch.Tensor,
    input_ids: torch.Tensor,
    tokenizer,
    output_dir: str = "attention_plots",
    layer: int = 0,
    example_idx: int = 0,
    epoch: Optional[int] = None
) -> str:
    
   
    os.makedirs(output_dir, exist_ok=True)
    
    num_heads = attn_weights.shape[1]
    attn = attn_weights[example_idx].cpu().numpy()  # [num_heads, seq_len, seq_len]
    ids = input_ids[example_idx].cpu()
    
    # Get tokens and trim padding
    tokens = tokenizer.convert_ids_to_tokens(ids)
    if tokenizer.pad_token in tokens:
        pad_start = tokens.index(tokenizer.pad_token)
        tokens = tokens[:pad_start]
        attn = attn[:, :pad_start, :pad_start]
    
    # Create subplot grid
    cols = 4
    rows = (num_heads + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()
    
    for head_idx in range(num_heads):
        ax = axes[head_idx]
        sns.heatmap(
            attn[head_idx],
            xticklabels=tokens if len(tokens) <= 20 else False,
            yticklabels=tokens if len(tokens) <= 20 else False,
            cmap="viridis",
            ax=ax,
            square=True,
            cbar=False
        )
        ax.set_title(f"Head {head_idx}", fontsize=10)
        ax.tick_params(axis='both', labelsize=6)
    
    # Hide unused subplots
    for idx in range(num_heads, len(axes)):
        axes[idx].axis('off')
    
    title = f"Layer {layer} - All Heads"
    if epoch is not None:
        title += f" (Epoch {epoch})"
    fig.suptitle(title, fontsize=14)
    
    filename = f"layer{layer}_all_heads"
    if epoch is not None:
        filename += f"_epoch{epoch}"
    filename += ".png"
    
    path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return path


def visualize_cls_attention(
    attention_weights: List[torch.Tensor],
    input_ids: torch.Tensor,
    tokenizer,
    output_dir: str = "attention_plots",
    example_idx: int = 0,
    epoch: Optional[int] = None
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    
    ids = input_ids[example_idx].cpu()
    tokens = tokenizer.convert_ids_to_tokens(ids)
    
    # Trim padding
    if tokenizer.pad_token in tokens:
        pad_start = tokens.index(tokenizer.pad_token)
        tokens = tokens[:pad_start]
    else:
        pad_start = len(tokens)
    
    num_layers = len(attention_weights)
    
    # Get CLS attention (first row) averaged over heads
    cls_attentions = []
    for layer_attn in attention_weights:
        # [batch, heads, seq, seq] -> [seq] (CLS row, averaged over heads)
        cls_attn = layer_attn[example_idx, :, 0, :pad_start].mean(dim=0).cpu().numpy()
        cls_attentions.append(cls_attn)
    
    cls_attentions = np.array(cls_attentions)  # [num_layers, seq_len]
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.heatmap(
        cls_attentions,
        xticklabels=tokens,
        yticklabels=[f"Layer {i}" for i in range(num_layers)],
        cmap="viridis",
        ax=ax
    )
    
    title = "[CLS] Token Attention Across Layers"
    if epoch is not None:
        title += f" (Epoch {epoch})"
    ax.set_title(title)
    ax.set_xlabel("Tokens")
    ax.set_ylabel("Layer")
    
    plt.xticks(rotation=90, fontsize=8)
    
    filename = "cls_attention"
    if epoch is not None:
        filename += f"_epoch{epoch}"
    filename += ".png"
    
    path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return path

