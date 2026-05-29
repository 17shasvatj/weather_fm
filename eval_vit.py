import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from weather_vit import WeatherViT
from data_loader import WeatherDataset

# === Config ===
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_INPUT_STEPS = 2
NUM_VARS = 21

# Variable names and units for reporting
VAR_NAMES = [
    't2m', 'msl', 'u10', 'v10', 'd2m', 'sp',
    'z_500', 'z_700', 'z_850',
    't_500', 't_700', 't_850',
    'u_500', 'u_700', 'u_850',
    'v_500', 'v_700', 'v_850',
    'q_500', 'q_700', 'q_850',
]

VAR_UNITS = [
    'K', 'Pa', 'm/s', 'm/s', 'K', 'Pa',
    'm²/s²', 'm²/s²', 'm²/s²',
    'K', 'K', 'K',
    'm/s', 'm/s', 'm/s',
    'm/s', 'm/s', 'm/s',
    'kg/kg', 'kg/kg', 'kg/kg',
]

# === Load model ===
print("Loading model...", flush=True)
model = WeatherViT(
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

model.load_state_dict(torch.load('checkpoints/best_model.pt', map_location=DEVICE))
model.eval()
print("Model loaded.", flush=True)

# === Load data ===
print("Loading test data...", flush=True)
test_data = np.load('test_data.npy', mmap_mode='r')  # raw, unnormalized (T, 21, H, W)
mean = np.load('mean.npy')   # (1, 21, 1, 1)
std = np.load('std.npy')     # (1, 21, 1, 1)
lat_weights = np.load('lat_weights.npy')  # (H,)

T, C, H, W = test_data.shape
print(f"Test data: {test_data.shape}", flush=True)


def normalize(x):
    return (x - mean) / std


def denormalize(x):
    return x * std + mean


# === 1. Single-step evaluation: Model vs Persistence ===
print("\n=== Single-step (6h) evaluation ===", flush=True)

model_errors = np.zeros(NUM_VARS)
persistence_errors = np.zeros(NUM_VARS)
count = 0

with torch.no_grad():
    for t in range(N_INPUT_STEPS, T - 1):
        # Build normalized input: 2 frames stacked
        frames = normalize(test_data[t - 1:t + 1])  # (2, C, H, W)
        x = np.concatenate(frames, axis=0)[np.newaxis]  # (1, 2*C, H, W)
        x = torch.tensor(x.copy(), dtype=torch.float32).to(DEVICE)

        # Model predicts normalized delta
        pred_delta_norm = model(x).cpu().numpy()[0]  # (C, H, W)

        # Current state (normalized) + predicted delta -> predicted next state (normalized)
        current_norm = normalize(test_data[t:t + 1])[0]  # (C, H, W)
        pred_next_norm = current_norm + pred_delta_norm

        # Denormalize to physical units
        pred_next = denormalize(pred_next_norm[np.newaxis])[0]  # (C, H, W)
        actual_next = test_data[t + 1]  # (C, H, W)
        current_raw = test_data[t]  # (C, H, W) persistence prediction

        # Weighted squared errors per variable
        for c in range(NUM_VARS):
            model_err = ((pred_next[c] - actual_next[c]) ** 2 * lat_weights[:, None]).mean()
            persist_err = ((current_raw[c] - actual_next[c]) ** 2 * lat_weights[:, None]).mean()
            model_errors[c] += model_err
            persistence_errors[c] += persist_err

        count += 1

        if (t - N_INPUT_STEPS) % 100 == 0:
            print(f"  Processed {t - N_INPUT_STEPS}/{T - N_INPUT_STEPS - 1} timesteps", flush=True)

model_rmse = np.sqrt(model_errors / count)
persistence_rmse = np.sqrt(persistence_errors / count)

print("\n6-hour forecast RMSE (physical units):")
print(f"{'Variable':<12} {'Unit':<10} {'Model':<12} {'Persistence':<12} {'Improvement':<12}")
print("-" * 60)
for i in range(NUM_VARS):
    improvement = (1 - model_rmse[i] / persistence_rmse[i]) * 100
    print(f"{VAR_NAMES[i]:<12} {VAR_UNITS[i]:<10} {model_rmse[i]:<12.4f} {persistence_rmse[i]:<12.4f} {improvement:<12.1f}%")


# === 2. Multi-step rollout: 6h to 72h ===
print("\n=== Multi-step rollout evaluation ===", flush=True)

rollout_steps = [1, 2, 4, 8, 12]  # 6h, 12h, 24h, 48h, 72h
max_rollout = max(rollout_steps)

# Pick evenly spaced start points in the test set
n_rollouts = 50
start_indices = np.linspace(N_INPUT_STEPS, T - max_rollout - 1, n_rollouts, dtype=int)

rollout_rmse = {s: np.zeros(NUM_VARS) for s in rollout_steps}
persist_rollout_rmse = {s: np.zeros(NUM_VARS) for s in rollout_steps}
rollout_count = 0

with torch.no_grad():
    for idx, start in enumerate(start_indices):
        # Initial 2 frames (normalized)
        prev_norm = normalize(test_data[start - 1:start])[0]  # (C, H, W)
        curr_norm = normalize(test_data[start:start + 1])[0]  # (C, H, W)
        curr_raw = test_data[start].copy()  # persistence anchor

        for step in range(1, max_rollout + 1):
            # Build input
            x = np.concatenate([prev_norm, curr_norm], axis=0)[np.newaxis]  # (1, 2*C, H, W)
            x = torch.tensor(x.copy(), dtype=torch.float32).to(DEVICE)

            # Predict
            pred_delta_norm = model(x).cpu().numpy()[0]
            next_norm = curr_norm + pred_delta_norm

            # Shift
            prev_norm = curr_norm
            curr_norm = next_norm

            if step in rollout_steps:
                # Denormalize prediction
                pred_raw = denormalize(curr_norm[np.newaxis])[0]
                actual_raw = test_data[start + step]

                for c in range(NUM_VARS):
                    model_err = ((pred_raw[c] - actual_raw[c]) ** 2 * lat_weights[:, None]).mean()
                    persist_err = ((curr_raw[c] - actual_raw[c]) ** 2 * lat_weights[:, None]).mean()
                    rollout_rmse[step][c] += model_err
                    persist_rollout_rmse[step][c] += persist_err

        rollout_count += 1
        if idx % 10 == 0:
            print(f"  Rollout {idx}/{n_rollouts}", flush=True)

# Print rollout results
print("\nMulti-step rollout RMSE:")
lead_hours = {1: '6h', 2: '12h', 4: '24h', 8: '48h', 12: '72h'}

# Print for key variables
key_vars = [0, 1, 2, 6, 9]  # t2m, msl, u10, z_500, t_500

print(f"\n{'Lead time':<10}", end='')
for vi in key_vars:
    print(f"{VAR_NAMES[vi]:<16}", end='')
print()
print("-" * 90)

for step in rollout_steps:
    rmse = np.sqrt(rollout_rmse[step] / rollout_count)
    print(f"{lead_hours[step]:<10}", end='')
    for vi in key_vars:
        print(f"{rmse[vi]:<16.4f}", end='')
    print()

print(f"\n{'Lead time':<10}", end='')
for vi in key_vars:
    print(f"{VAR_NAMES[vi]:<16}", end='')
print("  (persistence)")
print("-" * 90)

for step in rollout_steps:
    rmse = np.sqrt(persist_rollout_rmse[step] / rollout_count)
    print(f"{lead_hours[step]:<10}", end='')
    for vi in key_vars:
        print(f"{rmse[vi]:<16.4f}", end='')
    print()


# === 3. Plot rollout error growth ===
os.makedirs('eval_outputs', exist_ok=True)

fig, axes = plt.subplots(1, len(key_vars), figsize=(4 * len(key_vars), 4))

for i, vi in enumerate(key_vars):
    hours = [int(lead_hours[s].replace('h', '')) for s in rollout_steps]
    model_vals = [np.sqrt(rollout_rmse[s][vi] / rollout_count) for s in rollout_steps]
    persist_vals = [np.sqrt(persist_rollout_rmse[s][vi] / rollout_count) for s in rollout_steps]

    axes[i].plot(hours, model_vals, 'b-o', label='Model')
    axes[i].plot(hours, persist_vals, 'r--o', label='Persistence')
    axes[i].set_title(f"{VAR_NAMES[vi]} ({VAR_UNITS[vi]})")
    axes[i].set_xlabel('Lead time (hours)')
    axes[i].set_ylabel('RMSE')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.suptitle('Forecast RMSE vs Lead Time', fontsize=14)
plt.tight_layout()
plt.savefig('eval_outputs/rollout_rmse.png', dpi=150)
plt.close()
print("\nSaved eval_outputs/rollout_rmse.png", flush=True)


# === 4. ACC (Anomaly Correlation Coefficient) ===
print("\n=== Anomaly Correlation Coefficient (6h) ===", flush=True)

# Compute climatology from test set (mean over time)
climatology = test_data.mean(axis=0)  # (C, H, W)

acc_sum = np.zeros(NUM_VARS)
acc_count = 0

with torch.no_grad():
    for t in range(N_INPUT_STEPS, T - 1):
        frames = normalize(test_data[t - 1:t + 1])
        x = np.concatenate(frames, axis=0)[np.newaxis]
        x = torch.tensor(x.copy(), dtype=torch.float32).to(DEVICE)

        pred_delta_norm = model(x).cpu().numpy()[0]
        current_norm = normalize(test_data[t:t + 1])[0]
        pred_next_norm = current_norm + pred_delta_norm
        pred_next = denormalize(pred_next_norm[np.newaxis])[0]
        actual_next = test_data[t + 1]

        for c in range(NUM_VARS):
            pred_anom = (pred_next[c] - climatology[c]) * np.sqrt(lat_weights[:, None])
            actual_anom = (actual_next[c] - climatology[c]) * np.sqrt(lat_weights[:, None])

            num = (pred_anom * actual_anom).sum()
            den = np.sqrt((pred_anom ** 2).sum() * (actual_anom ** 2).sum())

            if den > 0:
                acc_sum[c] += num / den

        acc_count += 1

        if (t - N_INPUT_STEPS) % 100 == 0:
            print(f"  Processed {t - N_INPUT_STEPS}/{T - N_INPUT_STEPS - 1}", flush=True)

acc = acc_sum / acc_count

print("\n6-hour ACC:")
print(f"{'Variable':<12} {'ACC':<10}")
print("-" * 25)
for i in range(NUM_VARS):
    print(f"{VAR_NAMES[i]:<12} {acc[i]:<10.4f}")

# Save all results
np.savez('eval_outputs/results.npz',
         model_rmse=model_rmse,
         persistence_rmse=persistence_rmse,
         rollout_rmse={str(k): v for k, v in rollout_rmse.items()},
         acc=acc,
         var_names=VAR_NAMES,
         var_units=VAR_UNITS)

print("\nSaved eval_outputs/results.npz")
print("Evaluation complete!", flush=True)