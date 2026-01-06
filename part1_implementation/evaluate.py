from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from tqdm import tqdm

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    config: Dict,
    return_attention: bool = False,
    return_errors: bool = False
) -> Tuple[float, torch.Tensor, List[dict]]:
    model.eval()
    device = config["device"]
    total_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []
    errors: List[Dict] = []
    attention_weights: Optional[List[torch.Tensor]] = None
    sample_batch: Optional[Dict[str, torch.Tensor]] = None

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass
        logits, attn_weights_list = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        
        # Get predictions
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        if return_attention and attention_weights is None:
            attention_weights = [attn.detach().cpu() for attn in attn_weights_list]
            sample_batch = {
                "input_ids": batch["input_ids"].cpu(),
                "attention_mask": batch["attention_mask"].cpu(),
                "labels": batch["labels"].cpu()
            }
        
        # Track errors
        if return_errors:
            mismatches = preds != labels
            for idx in torch.where(mismatches)[0]:
                errors.append({
                    "input_ids": input_ids[idx].cpu(),
                    "attention_mask": attention_mask[idx].cpu(),
                    "pred": preds[idx].item(),
                    "label": labels[idx].item(),
                    "confidence": torch.softmax(logits[idx], dim=-1).max().item()
                })
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, 
        all_preds, 
        average='binary',
        zero_division=0
    )
    
    # Print results
    print(f"\nEvaluation Results:")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4, target_names=["Negative", "Positive"]))
    
    # Build result dictionary (consistent structure!)
    result = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predictions": all_preds,
        "labels": all_labels
    }
    
    if return_attention:
        result["attention_weights"] = attention_weights
        result["sample_batch"] = sample_batch
    
    if return_errors:
        result["errors"] = errors
    
    return result


def analyze_errors(
    errors: List[Dict],
    tokenizer,
    n_examples: int = 10
) -> None:
    print("\n" + "=" * 50)
    print("ERROR ANALYSIS")
    print("=" * 50)
    
    label_names = ["Negative", "Positive"]
    sorted_errors = sorted(errors, key=lambda x: x["confidence"], reverse=True)
    
    print(f"\nTop {n_examples} most confident errors:")
    print("-" * 50)
    
    for i, error in enumerate(sorted_errors[:n_examples]):
        text = tokenizer.decode(error["input_ids"], skip_special_tokens=True)
        print(f"\n{i+1}. Text: {text[:100]}...")
        print(f"   True: {label_names[error['label']]}, "
              f"Pred: {label_names[error['pred']]} "
              f"(confidence: {error['confidence']:.2%})")
    
    # Statistics
    print("\n" + "-" * 50)
    print(f"Total errors: {len(errors)}")
    
    # False positives vs false negatives
    fp = sum(1 for e in errors if e["pred"] == 1 and e["label"] == 0)
    fn = sum(1 for e in errors if e["pred"] == 0 and e["label"] == 1)
    print(f"False positives: {fp}")
    print(f"False negatives: {fn}")