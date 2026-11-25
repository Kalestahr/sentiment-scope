"""Batch inference interface for SentimentScope."""
import torch
from transformers import AutoTokenizer

from .model import DemoGPT


class SentimentPredictor:
    """Load a SentimentScope checkpoint and classify a batch of reviews."""
    LABELS = {0: "negative", 1: "positive"}

    def __init__(self, ckpt_path, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.max_length = ckpt["max_length"]
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt.get("tokenizer", "bert-base-uncased"))
        self.model = DemoGPT(ckpt["config"]).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    @torch.no_grad()
    def predict(self, texts, batch_size=64):
        if isinstance(texts, str):
            texts = [texts]
        results = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            enc = self.tokenizer(chunk, truncation=True, padding="max_length",
                                 max_length=self.max_length, return_tensors="pt")
            probs = torch.softmax(self.model(enc["input_ids"].to(self.device)), dim=-1).cpu()
            for t, p in zip(chunk, probs):
                results.append({"review": t,
                                "label": self.LABELS[int(p.argmax())],
                                "positive_prob": round(float(p[1]), 4),
                                "negative_prob": round(float(p[0]), 4)})
        return results
