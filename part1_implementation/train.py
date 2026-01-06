import torch
import yaml
from torch.utils.data import DataLoader
from model import SentimentTransformer
from dataset import SST2Dataset
from torch.utils.tensorboard import SummaryWriter
from utils import set_seed, save_checkpoint
from evaluate import evaluate
from utils import validate_config
from visualize_attention import visualize_attention
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import os


def train():
    config=yaml.safe_load(open("config.yaml"))
    set_seed(config["seed"])
    validate_config(config)
    train_ds = SST2Dataset(split="train", max_length=config["dataset"]["max_length"])
    val_ds = SST2Dataset(split="validation", max_length=config["dataset"]["max_length"])
    train_loader = DataLoader(train_ds, batch_size=config["dataset"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["dataset"]["batch_size"])
    model=SentimentTransformer(config["model"]).to(config["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"], weight_decay=config["training"]["weight_decay"])
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=config["training"].get("scheduler_t0", 5),
        T_mult=1
    )
    
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=config["training"]["label_smoothing"])

    writer = SummaryWriter(config["logging"]["log_dir"])
    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0
    global_step = 0
    for epoch in range(config["training"]["epochs"]):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        pbar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch + 1}/{config['training']['epochs']}"
        )
        
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(config["device"])
            attention_mask = batch['attention_mask'].to(config["device"])
            labels = batch['labels'].to(config["device"])
            
            # Forward pass
            optimizer.zero_grad()
            logits, _ = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                max_norm=config["training"].get("max_grad_norm", 1.0)
            )
            
            optimizer.step()
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{correct/total:.4f}"
            })
            if global_step % config["logging"].get("log_every", 100) == 0:
                writer.add_scalar("Train/loss_step", loss.item(), global_step)
            
            global_step += 1
        
        # Update learning rate
        scheduler.step()
        
        # Calculate epoch metrics
        avg_train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"\nEpoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, LR: {current_lr:.2e}")
        
        # Validation
        print("Running validation...")
        val_result = evaluate(
            model, 
            val_loader, 
            criterion, 
            config, 
            return_attention=True,
            return_errors=False
        )
        val_loss = val_result["loss"]
        val_acc = val_result["accuracy"]
        attention_weights = val_result["attention_weights"]
        sample_batch = val_result["sample_batch"]
        
        # Log to TensorBoard
        writer.add_scalar("Train/loss_epoch", avg_train_loss, epoch)
        writer.add_scalar("Train/accuracy", train_acc, epoch)
        writer.add_scalar("Val/loss", val_loss, epoch)
        writer.add_scalar("Val/accuracy", val_acc, epoch)
        writer.add_scalar("LR", current_lr, epoch)
        if attention_weights is not None and epoch % config["logging"].get("visualize_every", 1) == 0:
            os.makedirs(config["logging"].get("attention_dir", "attention_plots"), exist_ok=True)
            
            for layer_idx, layer_attn in enumerate(attention_weights):
                visualize_attention(
                    attn_weights=layer_attn,
                    input_ids=sample_batch["input_ids"],
                    tokenizer=val_ds.tokenizer,
                    output_dir=config["logging"].get("attention_dir", "attention_plots"),
                    layer=layer_idx,
                    example_idx=0,
                    head_idx=0,
                    epoch=epoch
                )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, config, is_best=True)
            patience_counter = 0
            print(f"New best model! Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{config['training']['early_stopping_patience']}")
            
            if patience_counter >= config["training"]["early_stopping_patience"]:
                print("Early stopping triggered!")
                break
        
        if (epoch + 1) % config["logging"].get("save_every", 5) == 0:
            save_checkpoint(model, optimizer, epoch, config, is_best=False)
    print("\n" + "=" * 50)
    print("Training Complete!")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Best Val Accuracy: {best_val_acc:.4f}")
    print("=" * 50)
    
    writer.close()


if __name__ == "__main__":
    train()