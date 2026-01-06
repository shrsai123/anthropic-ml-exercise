from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float):
        super(TransformerEncoderBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout,batch_first=True)
        self.ff=nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )

        self.norm1= nn.LayerNorm(embed_dim)
        self.norm2= nn.LayerNorm(embed_dim)
        self.dropout= nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) ->  Tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_weights = self.self_attn(x, x, x, key_padding_mask=~mask, need_weights=True, average_attn_weights=False)
        x=self.norm1(x + self.dropout(attn_out))
        ff_out=self.ff(x)
        x=self.norm2(x + self.dropout(ff_out))
        return x, attn_weights
    

class SentimentTransformer(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.embed_dim = config["embed_dim"]
        self.max_length = config["max_length"]
        self.token_embedding = nn.Embedding(config["vocab_size"], config["embed_dim"])
        self.position_embedding = nn.Embedding(
            config["max_length"], 
            config["embed_dim"]
        )
        self.embed_dropout = nn.Dropout(config["dropout"])
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                config["embed_dim"],
                config["num_heads"],
                config["ff_dim"],
                config["dropout"]
            )
            for _ in range(config["num_layers"])
        ])
        self.classifier = nn.Sequential(
            nn.Linear(config["embed_dim"], config["embed_dim"]),
            nn.Tanh(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["embed_dim"], config["num_classes"])
        )
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.embed_dropout(x)
        attentions = []
        for layer in self.layers:
            x, attn = layer(x, attention_mask)
            attentions.append(attn)
        cls_representation = x[:, 0]
        logits = self.classifier(cls_representation)
        
        return logits, attentions
    
    def get_attention_weights(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> List[torch.Tensor]:
        self.eval()
        with torch.no_grad():
            _, attentions = self.forward(input_ids, attention_mask)
        return attentions