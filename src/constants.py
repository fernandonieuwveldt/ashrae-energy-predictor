import os

INPUT_DIR = '/data'
OUTPUT_DIR = '/data/test'

if os.environ.get('USER', 'notflash') == 'flash':
    INPUT_DIR = '../data'
    OUTPUT_DIR = '../data/train/output'


BUILDING_FEATURES = ['site_id', 'building_id', 'primary_use', 'square_feet', 'year_built', 'floor_count']
WEATHER_FEATURES = ['site_id', 'timestamp', 'air_temperature', 'cloud_coverage', 'dew_temperature', 
                    'precip_depth_1_hr', 'sea_level_pressure' , 'wind_direction', 'wind_speed']
TRAIN_FEATURES = ['building_id', 'meter', 'timestamp', 'meter_reading']

ENERGY_FEATURES = BUILDING_FEATURES+WEATHER_FEATURES+TRAIN_FEATURES

CATEGORICAL_FEATURES = ['building_id', 'meter', 'site_id', 'primary_use', 'hour', 'day', 'week',
                        'meter_primary_use', 'site_primary_use', 'site_meter']
NUMERICAL_FEATURES = ['air_temperature', 'dew_temperature', 'precip_depth_1_hr', 'floor_count',
                      'sea_level_pressure', 'wind_speed', 'square_feet', 'wind_direction']
EMBEDDING_FEATURES = ['primary_use', 'building_id', 'meter', 'site_id', 'hour', 'day', 'week', 'year']

TARGET = 'meter_reading'
