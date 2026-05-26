import xarray as xr
import numpy as np
import glob
import os

DATA_DIR = 'era5_data'

# === Step 1: Load surface data ===
print("Loading surface data...")
sfc_files = sorted(glob.glob(os.path.join(DATA_DIR, 'era5_surface_*.nc')))
print(f"Found {len(sfc_files)} surface files")
ds_sfc = xr.open_mfdataset(sfc_files, combine='by_coords')
print(f"Surface variables: {list(ds_sfc.data_vars)}")

# === Step 2: Load pressure level data ===
print("Loading pressure level data...")
pl_files = sorted(glob.glob(os.path.join(DATA_DIR, 'era5_pressure_*.nc')))
print(f"Found {len(pl_files)} pressure level files")
ds_pl = xr.open_mfdataset(pl_files, combine='by_coords')
print(f"Pressure variables: {list(ds_pl.data_vars)}")
print(f"Pressure levels: {ds_pl.pressure_level.values}")

# === Step 3: Check time alignment ===
# Use whichever time coordinate name exists
time_coord = 'valid_time' if 'valid_time' in ds_sfc.coords else 'time'
print(f"Time coordinate: {time_coord}")
print(f"Surface time range: {ds_sfc[time_coord].values[0]} to {ds_sfc[time_coord].values[-1]}")
print(f"Pressure time range: {ds_pl[time_coord].values[0]} to {ds_pl[time_coord].values[-1]}")

# === Step 4: Build the 21-channel array ===
print("Building data array...")

# Surface variables (6): each is (T, H, W)
surface_var_names = ['t2m', 'msl', 'u10', 'v10', 'd2m', 'sp']
arrays = []
for var in surface_var_names:
    print(f"  Loading surface: {var}")
    arrays.append(ds_sfc[var].values)  # (T, H, W)

# Pressure level variables (5 vars × 3 levels = 15): each is (T, H, W) per level
pressure_var_names = ['z', 't', 'u', 'v', 'q']
levels = [500, 700, 850]
for var in pressure_var_names:
    for level in levels:
        print(f"  Loading pressure: {var} @ {level} hPa")
        arr = ds_pl[var].sel(pressure_level=level).values  # (T, H, W)
        arrays.append(arr)

# Stack into (T, 21, H, W)
data = np.stack(arrays, axis=1).astype(np.float32)
print(f"Full data shape: {data.shape}")  # (T, 21, H, W)

# === Step 5: Train/test split by year ===
print("Splitting train/test...")
years = ds_sfc[time_coord].dt.year.values
train_mask = years < 2020
test_mask = years >= 2020

train_data = data[train_mask]
test_data = data[test_mask]
print(f"Train shape: {train_data.shape}")
print(f"Test shape: {test_data.shape}")

# === Step 6: Compute normalization stats from training set only ===
print("Computing normalization stats...")
mean = train_data.mean(axis=(0, 2, 3), keepdims=True)  # (1, 21, 1, 1)
std = train_data.std(axis=(0, 2, 3), keepdims=True)
print(f"Mean shape: {mean.shape}")
print(f"Std shape: {std.shape}")

# === Step 7: Latitude weights ===
lats = ds_sfc.latitude.values
lat_weights = np.cos(np.deg2rad(lats))
lat_weights = lat_weights / lat_weights.mean()

# === Step 8: Save everything ===
print("Saving...")
np.save('train_data.npy', train_data)
np.save('test_data.npy', test_data)
np.save('mean.npy', mean)
np.save('std.npy', std)
np.save('lat_weights.npy', lat_weights)

# Save variable order for reference
var_order = surface_var_names.copy()
for var in pressure_var_names:
    for level in levels:
        var_order.append(f"{var}_{level}")

with open('variable_order.txt', 'w') as f:
    for i, name in enumerate(var_order):
        f.write(f"{i}: {name}\n")

print(f"\nSaved:")
print(f"  train_data.npy: {train_data.shape}")
print(f"  test_data.npy: {test_data.shape}")
print(f"  mean.npy: {mean.shape}")
print(f"  std.npy: {std.shape}")
print(f"  lat_weights.npy: {lat_weights.shape}")
print(f"  variable_order.txt")
print(f"\nVariables ({len(var_order)}):")
for i, name in enumerate(var_order):
    print(f"  {i}: {name}")
print("\nDone!")