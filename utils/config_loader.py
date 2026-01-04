from pathlib import Path
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(
            f"Config file tidak ditemukan: {config_file.resolve()}"
        )

    with config_file.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Isi config.yaml tidak valid")

    return config
