"""Дообучение rubert-tiny2 на классификацию токсичных комментариев."""

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def make_loader(path: str, tokenizer, max_length: int, batch_size: int, shuffle: bool) -> DataLoader:
    df = pd.read_csv(path)
    encodings = tokenizer(
        df["text"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    dataset = TensorDataset(
        encodings["input_ids"],
        encodings["attention_mask"],
        torch.tensor(df["label"].values),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(model, loader: DataLoader, optimizer) -> float:
    model.train()
    total_loss = 0.0
    for input_ids, attention_mask, labels in loader:
        optimizer.zero_grad()
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        output.loss.backward()
        optimizer.step()
        total_loss += output.loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader: DataLoader) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for input_ids, attention_mask, labels in loader:
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += output.loss.item()
        all_preds.extend(output.logits.argmax(dim=-1).tolist())
        all_labels.extend(labels.tolist())
    return {
        "val_loss": total_loss / len(loader),
        "val_accuracy": accuracy_score(all_labels, all_preds),
        "val_f1": f1_score(all_labels, all_preds),
    }


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.name, num_labels=cfg.model.num_labels
    )

    train_loader = make_loader(
        cfg.data.train_path, tokenizer, cfg.model.max_length, cfg.training.batch_size, shuffle=True
    )
    val_loader = make_loader(
        cfg.data.val_path, tokenizer, cfg.model.max_length, cfg.training.batch_size, shuffle=False
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )

    for epoch in range(1, cfg.training.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        metrics = evaluate(model, val_loader)
        print(
            f"epoch {epoch}: train_loss={train_loss:.4f} "
            f"val_loss={metrics['val_loss']:.4f} "
            f"val_accuracy={metrics['val_accuracy']:.4f} "
            f"val_f1={metrics['val_f1']:.4f}"
        )

    model.save_pretrained(cfg.data.checkpoint_dir)
    tokenizer.save_pretrained(cfg.data.checkpoint_dir)
    print(f"Чекпоинт сохранён в {cfg.data.checkpoint_dir}")


if __name__ == "__main__":
    main()
