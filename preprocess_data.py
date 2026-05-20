import xarray as xr
import numpy as np
import glob

# Load only instantaneous files
instant_files = sorted(glob.glob("era5_*/data_stream-oper_stepType-instant.nc"))

ds = xr.open_mfdataset(instant_files, combine='by_coords')
#print(ds)

# 6 instantaneous variables
var_names = ['t2m', 'msl', 'u10', 'v10', 'd2m', 'sp']

# Stack into (T, C, H, W)
arrays = [ds[var].values for var in var_names]
data = np.stack(arrays, axis=1).astype(np.float32) # (T, 6, H, W)
print(f"Shape: {data.shape}")

# Compute and save normalization stats
mean = data.mean(axis=(0, 2, 3), keepdims=True) # (1, C, 1, 1)
std = data.std(axis=(0, 2, 3), keepdims=True)

np.save('data.npy', data)
np.save('mean.npy', mean)
np.save('std.npy', std)

print(f"Variables: {var_names}")
print(f"Mean shape: {mean.shape}")
print(f"Std shape: {std.shape}")
print("Saved data.npy, mean.npy, std.npy")