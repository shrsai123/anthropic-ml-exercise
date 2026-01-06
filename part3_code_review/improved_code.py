import math
import logging
from dataclasses import dataclass
from typing import Optional, Iterator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TransformerConfig:
    vocab_size: int
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 512


class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoder = PositionalEncoding(
            config.d_model, config.max_seq_len, config.dropout
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True  
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.num_layers)
        self.fc = nn.Linear(config.d_model, config.vocab_size)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-init_range, init_range)
    
    def generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        seq_len = x.size(1)
        
        if mask is None:
            mask = self.generate_causal_mask(seq_len, x.device)
        
        # Scale embeddings by sqrt(d_model) as per "Attention Is All You Need"
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x, mask=mask)
        x = self.fc(x)
        
        return x


def train_model(
    model: TransformerModel,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 10,
    learning_rate: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    gradient_clip: float = 1.0,
    log_interval: int = 100,
    checkpoint_path: Optional[str] = None,
) -> dict:
    model = model.to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=0.01
    )
    
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100) 
    
    history = {"train_loss": [], "val_loss": []}
    global_step = 0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            
            outputs = model(inputs)
            loss = loss_fn(
                    outputs.view(-1, outputs.size(-1)), 
                    targets.view(-1)
                )
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            if global_step % log_interval == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | Step {global_step} | "
                    f"Loss: {loss.item():.4f}"
                )
        
        avg_train_loss = epoch_loss / num_batches
        history["train_loss"].append(avg_train_loss)
        
        # Validation
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, loss_fn, device)
            history["val_loss"].append(val_loss)
            logger.info(
                f"Epoch {epoch+1} complete | "
                f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )
        scheduler.step()
        if checkpoint_path:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_train_loss,
            }, f"{checkpoint_path}_epoch{epoch+1}.pt")
    
    return history


@torch.no_grad()
def evaluate(
    model: TransformerModel,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: str
) -> float:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    for batch in data_loader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        
        outputs = model(inputs)
        loss = loss_fn(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        total_loss += loss.item()
        num_batches += 1
    
    model.train()
    return total_loss / num_batches



if __name__ == "__main__":
    config = TransformerConfig(
        vocab_size=10000,
        d_model=512,
        nhead=8,
        num_layers=6
    )
    
    model = TransformerModel(config)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # train_model(model, train_loader, val_loader, epochs=10)