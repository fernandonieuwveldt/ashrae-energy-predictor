import os
import pandas
import numpy
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split
from constants import INPUT_DIR, ENERGY_FEATURES, TARGET


class AshreaEnergyData:

    def __init__(self):
        building = pandas.read_csv(f"{INPUT_DIR}/train/building_metadata.csv")
        weather = pandas.read_csv(f"{INPUT_DIR}/train/weather_train.csv", parse_dates=['timestamp'])

        if os.environ.get('USER', 'notflash') != 'flash':
            train = pandas.read_csv(f"{INPUT_DIR}/train/train.csv", parse_dates=['timestamp'])
        else:
            train = pandas.read_csv(f"{INPUT_DIR}/train/train_mini.csv", parse_dates=['timestamp'])

        self.data = train.merge(building, on='building_id', how='left')
        self.data = self.data.merge(weather, on=['site_id', 'timestamp'], how='left')
        self.data = PreProcess().transform(self.data)

        del train
        del building
        del weather


class AshreaEnergyTestData:

    def __init__(self, train=None, building=None, weather=None):
        building = pandas.read_csv(f"{INPUT_DIR}/train/building_metadata.csv")
        weather = pandas.read_csv(f"{INPUT_DIR}/test/weather_test.csv", parse_dates=['timestamp'])
        if os.environ.get('USER', 'notflash') != 'flash':
            train = pandas.read_csv(f"{INPUT_DIR}/test/test.csv", parse_dates=['timestamp'])
        else:
            train = pandas.read_csv(f"{INPUT_DIR}/test/test_snip.csv", parse_dates=['timestamp'])

        self.data = train.merge(building, on='building_id', how='left')
        self.data = self.data.merge(weather, on=['site_id', 'timestamp'], how='left')
        self.data = PreProcess().transform(self.data)

        del train
        del building
        del weather


class PreProcess:
    """
    pre processing data set
    """

    def __init__(self):
        pass

    @staticmethod
    def _time_mapper(xt=None):
        return xt.assign(hour=xt['timestamp'].dt.hour,
                         day=xt['timestamp'].dt.day,
                         week=xt['timestamp'].dt.week,
                         year=xt['timestamp'].dt.year)

    def transform(self, X):
        X = self._time_mapper(X)#.drop('timestamp', axis=1)
        # X['timestamp'] = X['timestamp'].map(lambda t: t.timestamp())
        X['meter_primary_use'] = X['meter'].map(lambda s:str(s)+'_') + X['primary_use']
        X['site_primary_use'] = X['site_id'].map(lambda s:str(s)+'_') + X['primary_use']
        X['site_meter'] = X['site_id'].map(lambda s:str(s)+'_') + X['meter'].map(str)
        # X['floor_count'] = numpy.log(X['floor_count'])
        # drop features with too many missing values
        # X.drop(['year_built', 'floor_count', 'cloud_coverage'], axis=1, inplace=True)
        return X[X[TARGET]>0].reset_index(drop=True)

class DataSplitter(object):
    """
    Simple data object to store all training data information

    """
    def __init__(self):
        energy = AshreaEnergyData()
        features = energy.data.drop(TARGET, axis=1)
        target = energy.data[TARGET].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(features, target, stratify=features['building_id'], random_state=42)

        # free up memory
        del energy
        del features
        del target

if __name__ == '__main__':
    energy = AshreaEnergyData()
