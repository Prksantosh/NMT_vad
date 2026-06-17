# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 16:53:52 2026

@author: USER
"""

import os
import torch
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from configs.config import Config
from models.autoencoder_skip import RHCNetAutoencoder
from losses.losses import CombinedPredictionLoss
from datasets.ucsd_ped2 import UCSDEPed2, UCSDEPed2val



optimizer = optim.Adam(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=20,
    gamma=0.5
)


# --------------------------------------------------
# Transforms
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
    #transforms.Normalize(
        #mean=[0.5, 0.5, 0.5],
        #std=[0.5, 0.5, 0.5]
    #)
])



train_dataset = UCSDEPed2(
    root_dir="UCSD\train",
    seq_len=3,
    transform=transform
)

anomaly_ranges_dict = {
    "Test001": [(66, 180)],
    "Test002": [(100, 180)],
    "Test003": [(5, 146)],
    "Test004": [(36, 180)]
}

val_dataset = UCSDEPed2val(
    root_dir="UCSD\Val",
    seq_len=3,
    transform=transform,
    anomaly_ranges_dict=anomaly_ranges_dict,
    one_based=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)




start_epoch = 0
best_auc = 0.0

if resume_training and os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_auc = checkpoint["best_auc"]
    print(f"Resumed from epoch {start_epoch} | Best AUC: {best_auc:.4f}")



def validate(model, val_loader, device, criterion):
    model.eval()

    total_loss = 0.0
    total_mse = 0.0
    total_ssim = 0.0
    total_temp = 0.0
    total_grad = 0.0

    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            frames, target, label = batch

            frames = frames.to(device)
            target = target.to(device)
            label = label.to(device)

            pred = model(frames)
            prev_frame = frames[:, -1]

            loss_dict = criterion(pred, target, prev_frame)

            total_loss += loss_dict["total_loss"].item()
            total_mse += loss_dict["mse_loss"].item()
            total_ssim += loss_dict["ssim_loss"].item()
            total_temp += loss_dict["temp_loss"].item()
            total_grad += loss_dict["grad_loss"].item()


            score = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
            all_scores.extend(score.detach().cpu().numpy().tolist())
            all_labels.extend(label.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(val_loader)
    avg_mse = total_mse / len(val_loader)
    avg_ssim = total_ssim / len(val_loader)
    avg_temp = total_temp / len(val_loader)
    avg_grad = total_grad / len(val_loader)

    auc = roc_auc_score(all_labels, all_scores)

    return avg_loss, avg_mse, avg_ssim, avg_temp, avg_grad, auc



epochs = 200

for epoch in range(start_epoch, epochs):
    model.train()

    total_loss = 0.0
    total_mse = 0.0
    total_ssim = 0.0
    total_temp = 0.0
    total_grad = 0.0

    for frames, target in train_loader:
        frames = frames.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        pred = model(frames)
        prev_frame = frames[:, -1]

        loss_dict = criterion(pred, target, prev_frame)
        loss = loss_dict["total_loss"]

        loss.backward()
        optimizer.step()

        total_loss += loss_dict["total_loss"].item()
        total_mse += loss_dict["mse_loss"].item()
        total_ssim += loss_dict["ssim_loss"].item()
        total_temp += loss_dict["temp_loss"].item()
        total_grad += loss_dict["grad_loss"].item()

    scheduler.step()

    avg_train_loss = total_loss / len(train_loader)
    avg_train_mse = total_mse / len(train_loader)
    avg_train_ssim = total_ssim / len(train_loader)
    avg_train_temp = total_temp / len(train_loader)
    avg_train_grad = total_grad / len(train_loader)


    val_loss, val_mse, val_ssim, val_temp, val_grad, val_auc = validate(
        model=model,
        val_loader=val_loader,
        device=device,
        criterion=criterion
    )

    print(
        f"Epoch [{epoch+1}/{epochs}] | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Train MSE: {avg_train_mse:.4f} | "
        f"Train SSIM: {avg_train_ssim:.4f} | "
        f"Train Temp: {avg_train_temp:.4f} | "
        f"Train Grad: {avg_train_grad:.4f} || "
        f"Val Loss: {val_loss:.4f} | "
        f"Val MSE: {val_mse:.4f} | "
        f"Val SSIM: {val_ssim:.4f} | "
        f"Val Temp: {val_temp:.4f} | "
        f"Val Grad: {val_grad:.4f} | "
        f"Val AUC: {val_auc:.4f}"
    )


    torch.save(model.state_dict(), last_model_path)


    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_auc": best_auc,
        },
        checkpoint_path
    )


    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), best_model_path)
        print(f"✅ New best model saved at epoch {epoch+1} | Best Val AUC: {best_auc:.4f}")

print("Training completed.")
print(f"Best validation AUC: {best_auc:.4f}")
