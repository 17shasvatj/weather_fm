# weather_fm
Mini Weather Foundation model trained on subset of ERA5 data

Surface Level Variables used:

2m_temperature, mean_sea_level_temperature, 10m_u_component_of_wind, 10m_v_component_of_wind, 2m_dewpoint_temperature, surface_pressure


Coordinates: [50, -130, 24, -60] (North, West, South, East)
Region: Continental US
Resolution:  0.25° × 0.25°
Grid size: 105 x 281
Time: 6-hourly
Years: 2001-2020 (2020 used as test set)

# Architecture/Training Details
We used a vanilla ViT (Vision Transformer), with inputs being the lat/long grid of variable values. Each variable was represented as a different channel (21 channels for 6 surface variables and 5 atmospheric variables at 3 pressure levels). The model takes in as input the grids corresponding to the two last 6-hourly time points, stacked on channel dimension, and outputs its predictions for the delta of the next time point. We use a patch size of 3x3, with padding to fit the grid dimensions. The embedding dimension was 256. Number of attention heads was 8, and number of layers was 8. T Model was trained for 20 epochs with a batch size of 16 on an A100, using cosine learning rate schedule with a 500 step linear warmup and gradient clipping (1.0). Since grid points cover less area at higher latitudes, we use cos(latitude) as a latitude weight in the loss function (MSE). Years 2000-2019 were used as training data, and 2020 was used as our test set. Final train loss was 0.029, and final test loss was 0.032.

# Results
We use persistence and climatology as a baseline. We report RMSE and ACC. For multi-step predictions, we ran 200 multi step rollouts for forecasts of 6h, 12h, 24h, 48h, and 72h and evaluated the RMSE. 
**Table 1: Single-Step (6h) Evaluation — RMSE and ACC**

| Variable | Unit | Model RMSE | Persistence RMSE | Improvement | ACC |
|----------|------|------------|------------------|-------------|-----|
| t2m | K | 1.13 | 4.43 | 74.6% | 0.984 |
| msl | Pa | 69.56 | 263.92 | 73.6% | 0.993 |
| u10 | m/s | 0.92 | 2.15 | 57.4% | 0.956 |
| v10 | m/s | 0.96 | 2.39 | 59.7% | 0.961 |
| d2m | K | 1.40 | 2.36 | 40.6% | 0.974 |
| sp | Pa | 76.56 | 224.67 | 65.9% | 0.989 |
| z_500 | m²/s² | 61.91 | 225.09 | 72.5% | 0.998 |
| z_700 | m²/s² | 48.17 | 165.36 | 70.9% | 0.997 |
| z_850 | m²/s² | 47.64 | 170.87 | 72.1% | 0.994 |
| t_500 | K | 0.66 | 1.51 | 56.4% | 0.992 |
| t_700 | K | 0.67 | 1.59 | 58.0% | 0.993 |
| t_850 | K | 0.86 | 2.20 | 60.8% | 0.992 |
| u_500 | m/s | 2.03 | 4.49 | 54.8% | 0.980 |
| u_700 | m/s | 1.68 | 3.60 | 53.3% | 0.970 |
| u_850 | m/s | 1.48 | 3.45 | 57.0% | 0.962 |
| v_500 | m/s | 2.06 | 5.55 | 62.8% | 0.973 |
| v_700 | m/s | 1.70 | 4.18 | 59.3% | 0.964 |
| v_850 | m/s | 1.54 | 4.06 | 62.1% | 0.963 |
| q_500 | kg/kg | 0.0003 | 0.0006 | 54.5% | 0.954 |
| q_700 | kg/kg | 0.0006 | 0.0012 | 53.0% | 0.963 |
| q_850 | kg/kg | 0.0008 | 0.0015 | 49.0% | 0.965 |

**Table 2: Multi-Step Rollout RMSE (Key Variables)**

| Lead time | t2m (K) | msl (Pa) | u10 (m/s) | z_500 (m²/s²) | t_500 (K) |
|-----------|---------|----------|-----------|----------------|-----------|
| 6h | 1.13 | 70.45 | 0.92 | 61.97 | 0.66 |
| 12h | 1.38 | 110.81 | 1.16 | 107.47 | 0.93 |
| 24h | 1.62 | 221.14 | 1.65 | 233.96 | 1.51 |
| 48h | 2.67 | 512.67 | 2.73 | 568.19 | 2.86 |
| 72h | 3.96 | 729.89 | 3.63 | 856.93 | 4.03 |

**Figure 1: Forecast RMSE vs Lead Time**
<img width="3000" height="600" alt="rollout_rmse" src="https://github.com/user-attachments/assets/049394be-db83-440f-959e-6ace5d66f962" />

