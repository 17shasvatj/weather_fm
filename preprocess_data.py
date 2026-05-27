import xarray as xr
import numpy as np
import glob
import os

DATA_DIR = 'era5_data'

surface_var_names = ['t2m', 'msl', 'u10', 'v10', 'd2m', 'sp']
pressure_var_names = ['z', 't', 'u', 'v', 'q']
levels = [500, 700, 850]

# Load and extract surface, then close
print("Loading surface...", flush=True)
sfc_files = sorted(glob.glob(os.path.join(DATA_DIR, 'era5_surface_*.nc')))
ds_sfc = xr.open_mfdataset(sfc_files, combine='by_coords')
time_coord = 'valid_time' if 'valid_time' in ds_sfc.coords else 'time'
years = ds_sfc[time_coord].dt.year.values

arrays = []
for var in surface_var_names:
    print(f"  {var}", flush=True)
    arrays.append(ds_sfc[var].values)

ds_sfc.close()
del ds_sfc
print("  Surface closed, memory freed", flush=True)

# Load and extract pressure, then close
print("Loading pressure...", flush=True)
pl_files = sorted(glob.glob(os.path.join(DATA_DIR, 'era5_pressure_*.nc')))
ds_pl = xr.open_mfdataset(pl_files, combine='by_coords')

for var in pressure_var_names:
    for level in levels:
        print(f"  {var} @ {level}", flush=True)
        arrays.append(ds_pl[var].sel(pressure_level=level).values)

ds_pl.close()
del ds_pl
print("  Pressure closed, memory freed", flush=True)

# Stack
print("Stacking...", flush=True)
data = np.stack(arrays, axis=1).astype(np.float32)
del arrays
print(f"Shape: {data.shape}", flush=True)

# Split
train_data = data[years < 2020]
test_data = data[years >= 2020]
del data
print(f"Train: {train_data.shape}", flush=True)
print(f"Test: {test_data.shape}", flush=True)

# Stats
mean = train_data.mean(axis=(0, 2, 3), keepdims=True)
std = train_data.std(axis=(0, 2, 3), keepdims=True)

# Lat weights
lats = xr.open_dataset(os.path.join(DATA_DIR, 'era5_surface_2001.nc')).latitude.values
lat_weights = np.cos(np.deg2rad(lats))
lat_weights = lat_weights / lat_weights.mean()

# Save
np.save('train_data.npy', train_data)
np.save('test_data.npy', test_data)
np.save('mean.npy', mean)
np.save('std.npy', std)
np.save('lat_weights.npy', lat_weights)

var_order = surface_var_names.copy()
for var in pressure_var_names:
    for level in levels:
        var_order.append(f"{var}_{level}")
with open('variable_order.txt', 'w') as f:
    for i, name in enumerate(var_order):
        f.write(f"{i}: {name}\n")

print("Done!", flush=True)