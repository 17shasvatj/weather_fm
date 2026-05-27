import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from torch.utils.data import DataLoader

from data_loader import WeatherDataset
from weather_vit import WeatherViT

# === Config ===
TRAIN_DATA = 'train_data.npy'
TEST_DATA = 'test_data.npy'
MEAN = 'mean.npy'
STD = 'std.npy'
LAT_WEIGHTS = 'lat_weights.npy'

N_INPUT_STEPS = 2
BATCH_SIZE = 16
NUM_EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-5
WARMUP_STEPS = 500
GRAD_CLIP = 1.0
LOG_EVERY = 50
SAVE_EVERY_EPOCH = 5
NUM_VARS = 21

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# === Data ===
train_set = WeatherDataset(TRAIN_DATA, MEAN, STD, n_input_steps=N_INPUT_STEPS)
test_set = WeatherDataset(TEST_DATA, MEAN, STD, n_input_steps=N_INPUT_STEPS)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

print(f"Train samples: {len(train_set)}")
print(f"Test samples: {len(test_set)}")
print(f"Input channels: {N_INPUT_STEPS * NUM_VARS}")
print(f"Target channels: {NUM_VARS}")

# === Latitude-weighted MSE loss ===
lat_weights = np.load(LAT_WEIGHTS)
lat_weights = torch.tensor(lat_weights, dtype=torch.float32).to(DEVICE)


def weighted_mse_loss(pred, target, lat_w):
    error = (pred - target) ** 2
    weighted = error * lat_w[None, None, :, None]
    return weighted.mean()


# === Model ===
def make_model():
    return WeatherViT(
        in_channels=N_INPUT_STEPS * NUM_VARS,
        out_channels=NUM_VARS,
        img_h=105,
        img_w=281,
        patch_h=3,
        patch_w=3,
        embed_dim=256,
        num_heads=8,
        num_layers=8,
        ff_hidden_dim=1024,
    ).to(DEVICE)


model = make_model()
num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {num_params:,}")

# === Sanity check: overfit one batch ===
print("\n=== Sanity check: overfitting one batch ===", flush=True)
test_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

x, y = next(iter(train_loader))
x, y = x.to(DEVICE), y.to(DEVICE)
print(f"Input shape: {x.shape}")
print(f"Target shape: {y.shape}")

for i in range(100):
    pred = model(x)
    loss = weighted_mse_loss(pred, y, lat_weights)
    test_optimizer.zero_grad()
    loss.backward()
    test_optimizer.step()
    if i % 10 == 0:
        print(f"  Step {i} | Loss: {loss.item():.6f}", flush=True)

# === Reinitialize for full training ===
print("\n=== Reinitializing for full training ===", flush=True)
model = make_model()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

total_steps = len(train_loader) * NUM_EPOCHS


def lr_lambda(step):
    if step < WARMUP_STEPS:
        return step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / (total_steps - WARMUP_STEPS)
    return 0.5 * (1 + np.cos(np.pi * progress))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# === Training ===
os.makedirs('checkpoints', exist_ok=True)
global_step = 0
best_test_loss = float('inf')

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    epoch_steps = 0
    t0 = time.time()

    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        pred = model(x)
        loss = weighted_mse_loss(pred, y, lat_weights)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
        epoch_steps += 1
        global_step += 1

        if global_step % LOG_EVERY == 0:
            avg_loss = epoch_loss / epoch_steps
            lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - t0
            steps_per_sec = epoch_steps / elapsed
            print(f"  Step {global_step} | Loss: {loss.item():.6f} | "
                  f"Avg: {avg_loss:.6f} | LR: {lr:.2e} | "
                  f"Speed: {steps_per_sec:.1f} steps/s", flush=True)

    train_loss = epoch_loss / epoch_steps

    # === Evaluation ===
    model.eval()
    test_loss = 0.0
    test_steps = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            pred = model(x)
            loss = weighted_mse_loss(pred, y, lat_weights)
            test_loss += loss.item()
            test_steps += 1

    test_loss = test_loss / test_steps
    elapsed = time.time() - t0

    print(f"Epoch {epoch}/{NUM_EPOCHS} | "
          f"Train Loss: {train_loss:.6f} | "
          f"Test Loss: {test_loss:.6f} | "
          f"Time: {elapsed:.1f}s", flush=True)

    # Save best model
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(model.state_dict(), 'checkpoints/best_model.pt')
        print(f"  → Saved best model (test loss: {test_loss:.6f})", flush=True)

    # Periodic checkpoint
    if epoch % SAVE_EVERY_EPOCH == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
            'global_step': global_step,
        }, f'checkpoints/epoch_{epoch}.pt')
        print(f"  → Saved checkpoint epoch {epoch}", flush=True)

print(f"\nTraining complete. Best test loss: {best_test_loss:.6f}")