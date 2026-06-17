import os
from dataclasses import dataclass, field
from typing import Tuple, Dict


@dataclass
class Config:
    project_name: str = "NMT-Residual-Attention-SFEN"
    experiment_name: str = "ucsd_ped2_mgtt"
    seed: int = 42

    device: str = "cuda"
    num_workers: int = 0
    pin_memory: bool = True

    dataset_name: str = "UCSDPed2"
    seq_len: int = 3
    image_size: Tuple[int, int] = (224, 224)

    train_root: str = "data/UCSD_train/Train"
    val_root: str = "data/UCSD_val"
    test_root: str = "data/UCSD_test"

    anomaly_ranges_dict: Dict[str, list] = field(default_factory=lambda: {
        "Test001": [(66, 180)],
        "Test002": [(100, 180)],
        "Test003": [(5, 146)],
        "Test004": [(36, 180)],
    })

    one_based_labels: bool = True

    model_name: str = "RHCNetAutoencoder"

    in_channels: int = 3
    base_channels: int = 32

     rhc_mid_channels: int = 16
    rhc_out_channels: int = 32
    se_reduction: int = 16

    temporal_channels: int = 256
    num_heads: int = 8
    memory_slots: int = 150
    temporal_layers: int = 2
    temporal_dropout: float = 0.1


    epochs: int = 200
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    resume_training: bool = False


    lambda_mse: float = 0.30
    lambda_ssim: float = 0.20
    lambda_temp: float = 0.20
    lambda_grad: float = 0.30

    scheduler_name: str = "StepLR"
    step_size: int = 20
    gamma: float = 0.5


    save_dir: str = "checkpoints/UCSDPed2"
    results_dir: str = "results/UCSDPed2"
    logs_dir: str = "results/UCSDPed2/logs"

    best_model_name: str = "best_model.pth"
    last_model_name: str = "last_model.pth"
    checkpoint_name: str = "checkpoint.pth"


    metric_name: str = "AUC"
    anomaly_score: str = "mse"
    save_predictions: bool = True


    @property
    def best_model_path(self):
        return os.path.join(self.save_dir, self.best_model_name)

    @property
    def last_model_path(self):
        return os.path.join(self.save_dir, self.last_model_name)

    @property
    def checkpoint_path(self):
        return os.path.join(self.save_dir, self.checkpoint_name)

    def create_dirs(self):
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

    def print_config(self):
        print("\n========== Configuration ==========")
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")
        print("===================================\n")


if __name__ == "__main__":
    config = Config()
    config.create_dirs()
    config.print_config()


