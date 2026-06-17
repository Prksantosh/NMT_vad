
import os
import random
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve

from configs.config import Config
from models.autoencoder_skip import RHCNetAutoencoder


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


config = Config()
config.create_dirs()
set_seed(config.seed)

DEVICE = torch.device(config.device if torch.cuda.is_available() else "cpu")

SEQ_LEN = config.seq_len
IMG_SIZE = config.image_size[0]
BATCH_SIZE = 1

GRAYSCALE_DATASET = True

TEST_VIDEO_DIR = os.path.join(config.test_root, "Test004")
CHECKPOINT_PATH = config.best_model_path
RESULTS_DIR = os.path.join(config.results_dir, "Test004")

USE_GROUND_TRUTH = True
GROUND_TRUTH_ONE_BASED = config.one_based_labels
ANOMALY_RANGES = [(29, 180)]


class VideoSequenceDataset(Dataset):
    def __init__(self, root_dir, seq_len=3, img_size=224, grayscale_dataset=True):
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.img_size = img_size
        self.grayscale_dataset = grayscale_dataset

        valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        self.frames = sorted([
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith(valid_ext)
        ])

        if len(self.frames) <= seq_len:
            raise ValueError(
                f"Not enough frames found in {root_dir}. "
                f"Found {len(self.frames)}, need > {seq_len}."
            )

    def __len__(self):
        return len(self.frames) - self.seq_len

    def _read_frame(self, path):
        if self.grayscale_dataset:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                raise ValueError(f"Failed to read image: {path}")

            img = cv2.resize(img, (self.img_size, self.img_size))
            img = img.astype(np.float32) / 255.0
            img = np.stack([img, img, img], axis=0)

        else:
            img = cv2.imread(path, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError(f"Failed to read image: {path}")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))

        return img

    def __getitem__(self, idx):
        seq = []

        for i in range(self.seq_len):
            seq.append(self._read_frame(self.frames[idx + i]))

        seq = np.stack(seq, axis=0)
        target = self._read_frame(self.frames[idx + self.seq_len])

        return {
            "sequence": torch.tensor(seq, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "target_path": self.frames[idx + self.seq_len],
            "target_index": idx + self.seq_len
        }


# =========================================================
# METRICS
# =========================================================
def compute_psnr(pred, target, eps=1e-8):
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    psnr = 10 * torch.log10(1.0 / (mse + eps))
    return psnr


def build_frame_labels(num_frames, anomaly_ranges, one_based=True):
    labels = np.zeros(num_frames, dtype=np.int32)

    for start, end in anomaly_ranges:
        if one_based:
            start -= 1
            end -= 1

        start = max(0, start)
        end = min(num_frames - 1, end)

        if end >= start:
            labels[start:end + 1] = 1

    return labels


def compute_auc_eer(scores, labels):
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)

    auc = roc_auc_score(labels, scores)

    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr

    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    eer_threshold = thresholds[idx]

    return auc, eer, eer_threshold


# =========================================================
# VISUALIZATION HELPERS
# =========================================================
def to_uint8_image(x):
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)


def make_error_map(target, pred):
    err = np.abs(target - pred)

    if err.ndim == 3:
        err = np.mean(err, axis=2)

    err = err - err.min()
    err = err / (err.max() + 1e-8)

    return err


def make_heatmap_overlay(base_img_u8, error_map):
    heat = (error_map * 255.0).astype(np.uint8)
    heat = np.ascontiguousarray(heat)

    heatmap = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

    base_bgr = cv2.cvtColor(base_img_u8, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(base_bgr, 0.65, heatmap, 0.35, 0)

    return heatmap, overlay


def save_visualization(save_path, last_input, target, pred, err_map):
    last_input_u8 = to_uint8_image(last_input)
    target_u8 = to_uint8_image(target)
    pred_u8 = to_uint8_image(pred)

    err_u8 = (err_map * 255.0).astype(np.uint8)
    heatmap, overlay = make_heatmap_overlay(target_u8, err_map)

    canvas_items = [
        cv2.cvtColor(last_input_u8, cv2.COLOR_RGB2BGR),
        cv2.cvtColor(target_u8, cv2.COLOR_RGB2BGR),
        cv2.cvtColor(pred_u8, cv2.COLOR_RGB2BGR),
        cv2.cvtColor(err_u8, cv2.COLOR_GRAY2BGR),
        heatmap,
        overlay
    ]

    title_h = 35
    labels = ["Last Input", "Target", "Prediction", "Error Map", "Heatmap", "Overlay"]
    labeled_items = []

    for img, txt in zip(canvas_items, labels):
        panel = np.full(
            (title_h + img.shape[0], img.shape[1], 3),
            255,
            dtype=np.uint8
        )

        panel[title_h:, :, :] = img

        cv2.putText(
            panel,
            txt,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )

        labeled_items.append(panel)

    canvas = np.concatenate(labeled_items, axis=1)
    cv2.imwrite(save_path, canvas)


# =========================================================
# MAIN
# =========================================================
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    vis_dir = os.path.join(RESULTS_DIR, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    dataset = VideoSequenceDataset(
        root_dir=TEST_VIDEO_DIR,
        seq_len=SEQ_LEN,
        img_size=IMG_SIZE,
        grayscale_dataset=GRAYSCALE_DATASET
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    model = RHCNetAutoencoder(seq_len=SEQ_LEN).to(DEVICE)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    all_scores = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            seq = batch["sequence"].to(DEVICE)
            target = batch["target"].to(DEVICE)

            pred = model(seq)

            psnr = compute_psnr(pred, target)
            score = (-psnr).item()
            all_scores.append(score)

            last_input = seq[:, -1].squeeze(0).cpu().numpy()
            target_np = target.squeeze(0).cpu().numpy()
            pred_np = pred.squeeze(0).cpu().numpy()

            last_input = np.transpose(last_input, (1, 2, 0))
            target_np = np.transpose(target_np, (1, 2, 0))
            pred_np = np.transpose(pred_np, (1, 2, 0))

            err_map = make_error_map(target_np, pred_np)

            save_path = os.path.join(vis_dir, f"{i:04d}.png")

            save_visualization(
                save_path=save_path,
                last_input=last_input,
                target=target_np,
                pred=pred_np,
                err_map=err_map
            )

            print(f"[{i + 1:04d}/{len(loader):04d}] score={score:.6f} saved={save_path}")

    all_scores = np.asarray(all_scores, dtype=np.float32)

    norm_scores = (
        (all_scores - all_scores.min()) /
        (all_scores.max() - all_scores.min() + 1e-8)
    )

    np.save(os.path.join(RESULTS_DIR, "raw_scores.npy"), all_scores)
    np.save(os.path.join(RESULTS_DIR, "normalized_scores.npy"), norm_scores)

    plt.figure(figsize=(12, 4))
    plt.plot(norm_scores, linewidth=1.5, label="Normalized Anomaly Score")
    plt.title("Anomaly Score Over Time")
    plt.xlabel("Sample Index")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "anomaly_score_plot.png"), dpi=200)
    plt.close()

    print(f"\nSaved score plot to: {os.path.join(RESULTS_DIR, 'anomaly_score_plot.png')}")

    if USE_GROUND_TRUTH:
        labels = build_frame_labels(
            num_frames=len(dataset.frames),
            anomaly_ranges=ANOMALY_RANGES,
            one_based=GROUND_TRUTH_ONE_BASED
        )

        eval_labels = labels[SEQ_LEN:]

        if len(eval_labels) != len(norm_scores):
            raise ValueError(
                f"Mismatch between labels ({len(eval_labels)}) "
                f"and scores ({len(norm_scores)})."
            )

        auc, eer, eer_thr = compute_auc_eer(norm_scores, eval_labels)

        print("\nEvaluation Metrics")
        print(f"AUC           : {auc:.4f}")
        print(f"EER           : {eer:.4f}")
        print(f"EER Threshold : {eer_thr:.4f}")

        with open(os.path.join(RESULTS_DIR, "metrics.txt"), "w") as f:
            f.write(f"AUC: {auc:.6f}\n")
            f.write(f"EER: {eer:.6f}\n")
            f.write(f"EER Threshold: {eer_thr:.6f}\n")

    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()
