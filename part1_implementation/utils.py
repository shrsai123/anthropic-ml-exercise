import os
import random
from typing import Dict, Optional, List
import torch
import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def validate_config(config: Dict) -> None:
    required_keys = ["model", "training", "dataset", "logging", "device", "seed"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: '{key}'")
    

    model_keys = ["vocab_size", "embed_dim", "num_heads", "ff_dim", 
                  "num_layers", "num_classes", "dropout", "max_length"]
    for key in model_keys:
        if key not in config["model"]:
            raise ValueError(f"Missing required model config key: '{key}'")

    training_keys = ["epochs", "lr", "weight_decay", "label_smoothing", 
                     "early_stopping_patience"]
    for key in training_keys:
        if key not in config["training"]:
            raise ValueError(f"Missing required training config key: '{key}'")
    
    # Check required dataset keys
    dataset_keys = ["batch_size", "max_length"]
    for key in dataset_keys:
        if key not in config["dataset"]:
            raise ValueError(f"Missing required dataset config key: '{key}'")
    
    # Check required logging keys
    logging_keys = ["log_dir", "checkpoint_dir"]
    for key in logging_keys:
        if key not in config["logging"]:
            raise ValueError(f"Missing required logging config key: '{key}'")
    
    # Type conversions
    config["training"]["lr"] = float(config["training"]["lr"])
    config["training"]["weight_decay"] = float(config["training"]["weight_decay"])
    config["training"]["label_smoothing"] = float(config["training"]["label_smoothing"])
    config["dataset"]["batch_size"] = int(config["dataset"]["batch_size"])
    config["dataset"]["max_length"] = int(config["dataset"]["max_length"])
    config["model"]["max_length"] = int(config["model"]["max_length"])
    config["seed"] = int(config["seed"])
    
    # Validate values
    if config["model"]["embed_dim"] % config["model"]["num_heads"] != 0:
        raise ValueError(
            f"embed_dim ({config['model']['embed_dim']}) must be divisible by "
            f"num_heads ({config['model']['num_heads']})"
        )
    
    if config["model"]["dropout"] < 0 or config["model"]["dropout"] > 1:
        raise ValueError(f"dropout must be between 0 and 1")
    
    print("Configuration validated successfully!")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: Dict,
    is_best: bool = False,
    metrics: Optional[Dict] = None
) -> str:
    os.makedirs(config["logging"]["checkpoint_dir"], exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }
    
    if metrics is not None:
        checkpoint["metrics"] = metrics
    
    if is_best:
        path = os.path.join(config["logging"]["checkpoint_dir"], "best_model.pt")
    else:
        path = os.path.join(
            config["logging"]["checkpoint_dir"],
            f"checkpoint_epoch_{epoch}.pt"
        )
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at {path}")
    
    return path


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu"
) -> Dict:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    print(f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint.get('epoch', 'unknown')})")
    
    return {
        "epoch": checkpoint.get("epoch", 0),
        "metrics": checkpoint.get("metrics", {}),
        "config": checkpoint.get("config", {})
    }


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_device(prefer_gpu: bool = True) -> str:
    if prefer_gpu:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
    return "cpu"