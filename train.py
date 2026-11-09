"""Дообучение rubert-tiny2 на классификацию токсичных комментариев."""

import random

import hydra
import mlflow
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# унифицируем seed для воспроизводимости
def set_seed(seed: int) -> None:
    """Фиксирует seed во всех используемых генераторах случайных чисел."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# для логирования нужен плоский словарь..
def flatten_config(cfg: DictConfig) -> dict:
    """Превращает вложенный конфиг в плоский словарь: {"training.lr": 5e-05, ...}."""
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    return pd.json_normalize(config_dict, sep=".").to_dict(orient="records")[0]


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


# здесь храним вычисление метрик
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
    set_seed(cfg.seed)

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

    # конфиги для mlflow, пишем пути
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    run_name = f"lr={cfg.training.lr}_bs={cfg.training.batch_size}_ep={cfg.training.epochs}"

    # запускаем обучение сразу в mlflow
    with mlflow.start_run(run_name=run_name):
        # гиперпараметры берутся прямо из конфига Hydra
        # логируем гиперпараметры в mlflow 
        mlflow.log_params(flatten_config(cfg))

        # цикл по эпохам
        for epoch in range(1, cfg.training.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer)
            metrics = evaluate(model, val_loader)

            # логирование лосса, метрик, отсюда идут данные для красивых графиков в интерфейсе
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metrics(metrics, step=epoch)

            print(
                f"epoch {epoch}: train_loss={train_loss:.4f} "
                f"val_loss={metrics['val_loss']:.4f} "
                f"val_accuracy={metrics['val_accuracy']:.4f} "
                f"val_f1={metrics['val_f1']:.4f}"
            )

        # логируем артефактом в mlflow финальную версию модели по нашему пути из конфига гидры
        model.save_pretrained(cfg.data.checkpoint_dir)
        tokenizer.save_pretrained(cfg.data.checkpoint_dir)
        mlflow.log_artifacts(cfg.data.checkpoint_dir, artifact_path="checkpoint")
        print(f"Чекпоинт сохранён в {cfg.data.checkpoint_dir} и залогирован в MLflow")


if __name__ == "__main__":
    main()
