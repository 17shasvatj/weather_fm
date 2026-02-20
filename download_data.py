import cdsapi

c = cdsapi.Client()

variables = [
    '2m_temperature',
    'mean_sea_level_pressure',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    'total_precipitation',
    '2m_dewpoint_temperature',
    'surface_pressure',
]

for year in range(2015, 2021):
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': variables,
            'year': str(year),
            'month': [f'{m:02d}' for m in range(1, 13)],
            'day': [f'{d:02d}' for d in range(1, 32)],
            'time': ['00:00', '06:00', '12:00', '18:00'],
            'area': [50, -130, 24, -60],
            'data_format': 'netcdf',
        },
        f'era5_{year}.nc'
    )
    print(f'{year} done')
