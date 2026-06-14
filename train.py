import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms


from configs.config import Config
from models.emu_autoencoder import RHCNetAutoencoder
#from models.emu_autoencoder import RHCNetAutoencoder
from losses.losses import SSIMLoss, TemporalLoss
#from datasets.avenue_dataset import AvenueDataset
from datasets.ShanghaiTech_dataset import ShanghaiTech
#from datasets.UBnormal_dataset import UBnormal
from torch.utils.data import DataLoader

# --------------------------------------------------
# Config
# --------------------------------------------------
config = Config()
device = torch.device(config.device if torch.cuda.is_available() else "cpu")

model = RHCNetAutoencoder(seq_len=4).to(device)

mse_loss = nn.MSELoss()
ssim_loss = SSIMLoss()
temporal_loss = TemporalLoss()

lambda_mse = 0.80
lambda_ssim = 0.12
lambda_temp = 0.08

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
# Dataset + Dataloader
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5,0.5,0.5],
        std=[0.5,0.5,0.5]
    )
])

train_dataset = ShanghaiTech(
    root_dir=r"C:\Users\USER\Desktop\MVA\eidetic_vad-main\eidetic_vad-main\data\Shanghai_train",
    seq_len=4,
    transform=transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
) 

# --------------------------------------------------
# Training Loop
# --------------------------------------------------
epochs = 100

best_loss = float("inf")   # 🔥 track best loss

for epoch in range(epochs):

    model.train()

    total_loss = 0
    total_mse = 0
    total_ssim = 0
    total_temp = 0

    for frames, target in train_loader:

        frames = frames.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        pred = model(frames)

        prev_frame = frames[:, -1]

        # Individual losses
        loss_mse = mse_loss(pred, target)
        loss_ssim = ssim_loss(pred, target)
        loss_temp = temporal_loss(pred, prev_frame, target)

        # Total loss
        loss = (
            lambda_mse * loss_mse +
            lambda_ssim * loss_ssim +
            lambda_temp * loss_temp
        )

        loss.backward()
        optimizer.step()

        # Accumulate losses
        total_loss += loss.item()
        total_mse += loss_mse.item()
        total_ssim += loss_ssim.item()
        total_temp += loss_temp.item()

    scheduler.step()

    avg_loss = total_loss / len(train_loader)
    avg_mse = total_mse / len(train_loader)
    avg_ssim = total_ssim / len(train_loader)
    avg_temp = total_temp / len(train_loader)

    print(
        f"Epoch [{epoch+1}/{epochs}] | "
        f"Total Loss: {avg_loss:.4f} | "
        f"MSE: {avg_mse:.4f} | "
        f"SSIM: {avg_ssim:.4f} | "
        f"Temporal: {avg_temp:.4f}"
    )

    # --------------------------------------------------
    # 🔥 SAVE ONLY BEST MODEL
    # --------------------------------------------------
    if avg_loss < best_loss:
        best_loss = avg_loss

        torch.save(
            model.state_dict(),
            "best_rhcnet_model.pth"
        )

        print(f"✅ New best model saved at epoch {epoch+1} | Loss: {best_loss:.6f}")










