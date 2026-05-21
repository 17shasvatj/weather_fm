import xarray as xr
import numpy as np
import glob

# Load
instant_files = sorted(glob.glob("era5_*/data_stream-oper_stepType-instant.nc"))
ds = xr.open_mfdataset(instant_files, combine='by_coords')

var_names = ['t2m', 'msl', 'u10', 'v10', 'd2m', 'sp']
arrays = [ds[var].values for var in var_names]
data = np.stack(arrays, axis=1).astype(np.float32)

# Split by year
train_mask = ds.valid_time.dt.year < 2020
test_mask = ds.valid_time.dt.year >= 2020

train_data = data[train_mask.values]
test_data = data[test_mask.values]

# Stats from training set only
mean = train_data.mean(axis=(0, 2, 3), keepdims=True)
std = train_data.std(axis=(0, 2, 3), keepdims=True)

# Latitude weights
lats = ds.latitude.values
lat_weights = np.cos(np.deg2rad(lats))
lat_weights = lat_weights / lat_weights.mean()

np.save('train_data.npy', train_data)
np.save('test_data.npy', test_data)
np.save('mean.npy', mean)
np.save('std.npy', std)
np.save('lat_weights.npy', lat_weights)

print(f"Train: {train_data.shape}")
print(f"Test: {test_data.shape}")
print(f"Variables: {var_names}")
print("Saved all files")