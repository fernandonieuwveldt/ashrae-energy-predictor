from constants import INPUT_DIR, ENERGY_FEATURES, TARGET
import pandas
from sklearn.base import TransformerMixin


class FeatureExtractor(TransformerMixin):
    """
    Extract relevant features for training

    """
    def __init__(self, features=None):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if 'meter_reading' not in X.columns and self.features == ENERGY_FEATURES:  # hack for test data
            self.features.remove('meter_reading')
        return X[self.features]


class AshreaEnergyData:

    def __init__(self):
        building = pandas.read_csv(f"{INPUT_DIR}/train/building_metadata.csv")
        weather = pandas.read_csv(f"{INPUT_DIR}/train/weather_train.csv", parse_dates=['timestamp'])
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
        train = pandas.read_csv(f"{INPUT_DIR}/test/test.csv", parse_dates=['timestamp'])

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
        X = self._time_mapper(X).drop('timestamp', axis=1)
        # drop features with too many missing values
        # X.drop(['year_built', 'floor_count', 'cloud_coverage'], axis=1, inplace=True)
        return X  # [X[TARGET]>0]


if __name__ == '__main__':
    energy = AshreaEnergyData()
