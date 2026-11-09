# ЛР3. Конфиги, DVC и трекинг экспериментов

Дообучение `cointegrated/rubert-tiny2` на бинарную классификацию токсичных комментариев
(датасет `AlexSham/Toxic_Russian_Comments`, подмножество: 2000 train / 500 val).

## Структура

```
config/
  config.yaml              # seed, настройки MLflow, выбор групп
  model/rubert_tiny2.yaml  # модель: имя, число классов, max_length
  training/default.yaml    # обучение: lr, batch_size, epochs, weight_decay
  data/default.yaml        # данные: размеры подмножества, пути к CSV и чекпоинту
prepare_data.py            # выгрузка подмножества с HuggingFace в CSV
train.py                   # обучение + логирование в MLflow
data/*.csv.dvc             # данные под DVC
models/checkpoint.dvc      # чекпоинт лучшей модели под DVC
```

## Установка

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Данные

Данные и чекпоинт лежат в DVC remote `D:\dvc_storage` (см. `.dvc/config`):

```bash
dvc pull
```

Если remote недоступен, подмножество можно выгрузить заново (тот же seed даёт те же строки):

```bash
python prepare_data.py
```

## Обучение

Любой параметр переопределяется из CLI без правки кода:

```bash
python train.py training.lr=1e-4 training.batch_size=32
```

Каждый запуск логируется в MLflow: все параметры конфига, `train_loss`, `val_loss`,
`val_accuracy`, `val_f1` по эпохам и финальный чекпоинт как артефакт. Просмотр:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

и открыть http://localhost:5000, эксперимент `toxic-comments`.

## Результаты

| Запуск | lr | batch_size | val_f1 | val_accuracy |
|---|---|---|---|---|
| 1 | 5e-5 | 32 | 0.777 | 0.930 |
| **2** | **1e-4** | **32** | **0.814** | **0.938** |
| 3 | 2e-5 | 16 | 0.811 | 0.932 |

Метрики после 3-й эпохи, `seed=42`.

## Воспроизведение лучшего запуска

```bash
dvc pull
python train.py training.lr=1e-4
```

Ожидаемый результат на 3-й эпохе: `val_accuracy=0.9380`, `val_f1=0.8144`.
Повторный запуск дал те же метрики до 4-го знака (проверено на CPU).

Для воспроизводимости:

- `seed` фиксируется для `random`, `numpy` и `torch` (функция `set_seed` в `train.py`),
  в том числе до инициализации новой головы классификатора и перемешивания `DataLoader`;
- версии зависимостей зафиксированы в `requirements.txt`;
- версия данных зафиксирована хешами в `data/*.csv.dvc`;
- версия кода и конфигов — коммит в git.
