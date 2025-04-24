from transformers import BertTokenizer, BertForSequenceClassification
import torch
from typing import Dict

class EmailClassifier:
    def __init__(self, model_path: str):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.id2label = {
            0: "Incident",
            1: "Request",
            2: "Problem",
            3: "Change"
        }
        
    def classify(self, text: str) -> Dict[str, str]:
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        
        return {
            "category": self.id2label[predicted_class],
            "confidence": torch.softmax(logits, dim=1)[0][predicted_class].item()
        }
