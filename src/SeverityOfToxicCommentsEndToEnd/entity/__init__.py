from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    test_data_path: Path
    train_data_path: Path
    tokenizer_path: Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    processed_test_data_dir: Path
    processed_train_data_dir: Path
    tokenizer_dir: Path
    epochs: int
    embedding_dim: int
    batch_size: int
    fasttext_model_path: Path
    max_features: int
    maxpadlen: int
    tokenizer_dir: Path