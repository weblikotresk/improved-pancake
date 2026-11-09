"""Выгружает небольшое подмножество датасета токсичности в CSV."""

from pathlib import Path

import hydra
from datasets import load_dataset
from omegaconf import DictConfig


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    dataset = load_dataset(cfg.data.dataset_name)

    # train-подмножество берём из train, val-подмножество - из test
    train_df = dataset["train"].to_pandas().sample(n=cfg.data.train_size, random_state=cfg.seed)
    val_df = dataset["test"].to_pandas().sample(n=cfg.data.val_size, random_state=cfg.seed)

    for df, path in [(train_df, cfg.data.train_path), (val_df, cfg.data.val_path)]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"{path}: {len(df)} строк, доля токсичных {df['label'].mean():.2f}")


if __name__ == "__main__":
    main()
