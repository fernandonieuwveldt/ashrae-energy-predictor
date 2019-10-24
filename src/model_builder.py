from sklearn.metrics import mean_squared_log_error, mean_squared_error 
from constants import TARGET
from regressor import BaggedRFModel, CatBoostModel, XGBModel, LinearModel, ClusteredXGBModel
from energy_data import AshreaEnergyData, AshreaEnergyTestData
from feature_builder import EnergyFeatureBuilder
import numpy
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    energy = AshreaEnergyData()
    target = numpy.log1p(energy.data[TARGET])
    samples = energy.data.drop(TARGET, axis=1)
    transformer = EnergyFeatureBuilder(with_embedding=True)
    transformer.fit(samples)
    # print('data loaded')
    # reg = ClusteredXGBModel(split_feature='building_id')
    # reg.fit(samples, target)
    # print('busy scoring samples...')
    # print('RMSLE: ', mean_squared_error(target, reg.predict(samples))**0.5)
    # # del samples
    # # del energy
    # energy_test = AshreaEnergyTestData()
    # reg.submit(energy_test.data, energy_test.data['row_id'])
