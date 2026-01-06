from datasets import load_dataset
from transformers import AutoTokenizer
import torch

class SST2Dataset(torch.utils.data.Dataset):
    def __init__(self, split: str,  max_length: int):
        self.dataset = load_dataset("glue", "sst2", split=split)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.tokenizer(
            item['sentence'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()  
        attention_mask = encoding['attention_mask'].squeeze().bool() 
        label = torch.tensor(item['label'], dtype=torch.long)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label
        }