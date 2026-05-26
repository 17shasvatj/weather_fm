import cdsapi
import os

c = cdsapi.Client()

save_dir = 'era5_data'
os.makedirs(save_dir, exist_ok=True)

# === Surface variables ===
surface_vars = [
    '2m_temperature',
    'mean_sea_level_pressure',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    '2m_dewpoint_temperature',
    'surface_pressure',
]

# === Pressure level variables ===
pressure_vars = [
    'geopotential',
    'temperature',
    'u_component_of_wind',
    'v_component_of_wind',
    'specific_humidity',
]

pressure_levels = ['500', '700', '850']

# === Common request params ===
common = {
    'product_type': 'reanalysis',
    'day': [f'{d:02d}' for d in range(1, 32)],
    'time': ['00:00', '06:00', '12:00', '18:00'],
    'area': [50, -130, 24, -60],
    'data_format': 'netcdf',
}

for year in range(2001, 2021):

    # --- Surface (full year) ---
    sfc_file = os.path.join(save_dir, f'era5_surface_{year}.nc')
    if os.path.exists(sfc_file):
        print(f'Surface {year} already exists, skipping')
    else:
        print(f'Downloading surface {year}...')
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                **common,
                'variable': surface_vars,
                'year': str(year),
                'month': [f'{m:02d}' for m in range(1, 13)],
            },
            sfc_file,
        )
        print(f'Surface {year} done')

    # --- Pressure levels (quarterly) ---
    quarters = [
        ('q1', range(1, 4)),
        ('q2', range(4, 7)),
        ('q3', range(7, 10)),
        ('q4', range(10, 13)),
    ]
    for q_name, months in quarters:
        pl_file = os.path.join(save_dir, f'era5_pressure_{year}_{q_name}.nc')
        if os.path.exists(pl_file):
            print(f'Pressure {year} {q_name} already exists, skipping')
            continue
        print(f'Downloading pressure levels {year} {q_name}...')
        c.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                **common,
                'variable': pressure_vars,
                'year': str(year),
                'month': [f'{m:02d}' for m in months],
                'pressure_level': pressure_levels,
            },
            pl_file,
        )
        print(f'Pressure {year} {q_name} done')

print('All downloads complete')
