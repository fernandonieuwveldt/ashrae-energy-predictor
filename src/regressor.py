from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor
import pandas
from xgboost import XGBRegressor
from catboost import CatBoostRegressor, Pool
import numpy
from sklearn.model_selection import KFold
from collections import defaultdict
from feature_builder import EnergyFeatureBuilder
from sklearn.linear_model import Lasso, LassoCV
from constants import TARGET, OUTPUT_DIR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

class KaggleSubmitMixin:
    """
    Mixin class for all estimators to have submit file in correct format

    """
    def submit(self, xtest=None, ids=None):
        """
        Saves prediction outputs in kaggle submission file format

        :param xtest: competition test data
        :param trans_ids: transaction ids
        :return: saved kaggle file format
        """
        test_predictions = numpy.expm1(self.predict(xtest))
        my_submission_file = pandas.DataFrame()
        my_submission_file['row_id'] = ids
        my_submission_file[TARGET] = numpy.round_(test_predictions, decimals=4, out=None)
        my_submission_file.to_csv('submission.gz', index=False, compression='gzip')


class BaggedRFModel(RegressorMixin, KaggleSubmitMixin):
    """
    Bagging model with Randomforest as base estimator
    """
    rf_params = {'n_estimators': 5,
                 'max_depth': 3,
                 'n_jobs': -1}

    def __init__(self, n_estimators_bagging=5):
        self.n_estimators_bagging = n_estimators_bagging
        self.energy_transformer = EnergyFeatureBuilder()
        self.regressor = None

    def fit(self, xtrain=None, ytrain=None):
        """
        Fit bagging model with Random Forest based estimator

        :param xtrain: pandas data_frame for training
        :param ytrain: train target variable
        :return: BaggedRFModel objects
        """
        self.energy_transformer.fit(xtrain, ytrain)
        xtrain = self.energy_transformer.transform(xtrain)
        estimator = RandomForestRegressor(**self.rf_params)

        self.regressor = BaggingRegressor(base_estimator=estimator,
                                          n_estimators=self.n_estimators_bagging)
        self.regressor.fit(xtrain, ytrain)
        return self

    def predict(self, xtest=None):
        """
        Apply trained Regressor on test data

        :param xtest: pandas data_frame for test data
        :return: 
        """
        xtest = self.energy_transformer.transform(xtest)
        return self.regressor.predict(xtest)


class CatBoostModel(RegressorMixin, KaggleSubmitMixin):
    """
    Estimator using Catboost
    """
    _SEED = 1

    def __init__(self):
        self.energy_transformer = EnergyFeatureBuilder()
        self.regressor = None

    def fit(self, xtrain=None, ytrain=None, xval=None, yval=None):
        """
        Fit Catboost model

        :param xtrain: pandas data_frame for training
        :param ytrain: train target variable
        :param xval: pandas data_frame for model validation
        :param yval: target variable
        :return: CatBoostModel object
        """
        # If no validation data is specified, use training data
        if not any([xval, yval]):
            xval = xtrain
            yval = ytrain

        self.energy_transformer.fit(xtrain, ytrain)
        xtrain = self.energy_transformer.transform(xtrain)
        xval = self.energy_transformer.transform(xval)

        nr_cat_features = int(numpy.sum(xtrain.max(axis=0) == 1))
        cat_features = list(range(nr_cat_features))
        train_data = Pool(data=xtrain,
                          label=ytrain,
                          cat_features=cat_features)
        valid_data = Pool(data=xval,
                          label=yval,
                          cat_features=cat_features)
        params = {'loss_function': 'RMSE',
                  'eval_metric': 'RMSE',
                  'cat_features': cat_features,
                  'iterations': 10,
                  'verbose': 1,
                  'max_depth': 3,
                  'random_seed': self._SEED,
                  'od_type': "Iter",
                  # 'od_wait': 100,
                  }
        self.regressor = CatBoostRegressor(**params)
        self.regressor.fit(train_data, eval_set=valid_data)
        return self

    def predict(self, xtest=None):
        """
        Apply trained Regressor on test data

        :param xtest:  pandas data_frame for test data
        :return: array of predicted probability values
        """
        xtest = self.energy_transformer.transform(xtest)
        return self.regressor.predict(xtest)


class XGBModel(RegressorMixin, KaggleSubmitMixin):
    def __init__(self):
        self.energy_transformer = EnergyFeatureBuilder()
        self.regressor = None

    def fit(self, xtrain=None, ytrain=None, xval=None, yval=None):
        """
        Fit XGM based model

        :param xtrain: pandas data_frame for training
        :param ytrain: train target variable
        :param xval: pandas data_frame for model validation
        :param yval: target variable
        :return: XGBModel object
        """
        # If no validation data is specified, use training data
        if not any([xval, yval]):
            xval = xtrain.copy()
            yval = ytrain.copy()

        self.energy_transformer.fit(xtrain, ytrain)
        xtrain = self.energy_transformer.transform(xtrain)
        xval = self.energy_transformer.transform(xval)
        eval_set = [(xval, yval)]
        self.regressor = XGBRegressor(n_estimators=20,
                                        max_depth=3,
                                        learning_rate=0.048,
                                        subsample=0.9,
                                        colsample_bytree=0.9,
                                        reg_alpha=0.5,
                                        reg_lamdba=0.5,
                                        n_jobs=-1)
        self.regressor.fit(xtrain, ytrain, eval_metric=["RMSE"],
                           early_stopping_rounds=1, eval_set=eval_set, verbose=True)

    def predict(self, xtest=None):
        """
        Apply trained Regressor on test data

        :param xtest: pandas data_frame for test data
        :return: array of predicted probability values
        """
        xtest = self.energy_transformer.transform(xtest)
        return self.regressor.predict(xtest)


class KFoldModel(RegressorMixin, KaggleSubmitMixin):
    """
    Apply models on different fold splits of the data

    """
    _SEED = 1
    _SPLITS = 5

    def __init__(self, n_estimators=500, split_feature=''):
        self.n_estimators = n_estimators
        self.regressor = None
        self.split_estimators = defaultdict(list)
        self.split_transformers = defaultdict(list)
        self.split_feature = split_feature
        self.energy_transformer = EnergyFeatureBuilder()
        self.estimators = []

    def fit(self, xtrain=None, ytrain=None):
        """
        Fits _SPLITS number of models

        :param xtrain: pandas data_frame for training
        :param ytrain: train target variable
        :param xval: pandas data_frame for model validation
        :param yval: target variable
        :return: KFoldModel object
        """
        self.energy_transformer.fit(xtrain, ytrain)
        xtrain = self.energy_transformer.transform(xtrain)

        folds = KFold(n_splits=self._SPLITS, shuffle=True)
        for fold_n, (train_index, valid_index) in enumerate(folds.split(xtrain)):
            self.regressor = XGBRegressor(n_estimators=10,
                                          max_depth=9,
                                          learning_rate=0.048,
                                          subsample=0.85,
                                          colsample_bytree=0.85,
                                          reg_alpha=0.15,
                                          reg_lamdba=0.85,
                                          n_jobs=-1)
            x_train_, x_valid = xtrain[train_index, :], xtrain[valid_index, :]
            y_train_, y_valid = ytrain[train_index], ytrain[valid_index]
            eval_set = [(x_valid, y_valid)]
            fit_params = {'eval_metric': ["error"],
                          'early_stopping_rounds': 100,
                          'eval_set': eval_set,
                          'verbose': True}
            self.regressor.fit(x_train_, y_train_, **fit_params)
            self.estimators.append(self.Regressor)
        return self

    def predict(self, xtest=None):
        """
        Apply trained Regressor on test data

        :param xtest: Test data
        :return: array of predicted probability values

        """
        xtest = self.energy_transformer.transform(xtest)
        pred = numpy.zeros((xtest.shape[0], 2))
        for reg in self.estimators:
            pred += reg.predict(xtest)/self._SPLITS
        return pred


class ClusteredXGBModel(RegressorMixin, KaggleSubmitMixin):
    """
    Splits data on group and apply model on each group

    """
    _SEED = 1
    _SPLITS = 5

    def __init__(self, split_feature=None):
        self.split_estimators = defaultdict(list)
        self.split_transformers = defaultdict(list)
        self.split_feature = split_feature
        self.energy_transformer = EnergyFeatureBuilder()
        self.regressor = None

    def fit(self, xtrain=None, ytrain=None):
        """
        Combination of cluster Regressor and bagging on samples

        :param xtrain: pandas data_frame containing training data
        :param ytrain: numpy array with model targets
        :return: ClusteredXGBModel object
        """
        xtrain['target'] = ytrain
        iter_ = 0
        for name, group in xtrain.groupby(self.split_feature):
            iter_+=1
            if iter_ % 10 == 0:
                print('feature group: ', iter_)
            group_target = group['target'].values
            group_features = group.drop('target', axis=1)
            group_transformer = EnergyFeatureBuilder()
            group_transformer.fit(group_features, group_target)
            group_features = group_transformer.transform(group_features)

            folds = KFold(n_splits=self._SPLITS, shuffle=True)
            for fold_n, (train_index, valid_index) in enumerate(folds.split(group_features)):
                x_train_, x_valid = group_features[train_index, :], group_features[valid_index, :]
                y_train_, y_valid = group_target[train_index], group_target[valid_index]

                estimator =  RandomForestRegressor(n_estimators=20, max_depth=5, n_jobs=-1)
                estimator.fit(x_train_, y_train_)

                self.split_estimators[name].append(estimator)
                self.split_transformers[name].append(group_transformer)

        return self

    def predict(self, xtest=None):
        """
        Transform and Predict based on splits of the split feature

        :param xtest: pandas dataframe with test data
        :return: model probabilities
        """
        predictions = numpy.zeros((xtest.shape[0], ))
        for name, group in xtest.groupby(self.split_feature):
            for s in range(self._SPLITS):
                group_transformed = self.split_transformers[name][s].transform(group)
                predictions[group.index.values] += self.split_estimators[name][s].predict(group_transformed) / \
                                                   self._SPLITS
        return predictions


class LinearModel(RegressorMixin, KaggleSubmitMixin):
    """
    Bagging model with Randomforest as base estimator
    """

    def __init__(self):
        self.energy_transformer = EnergyFeatureBuilder()
        self.regressor = None

    def fit(self, xtrain=None, ytrain=None):
        """

        :param xtrain: pandas data_frame for training
        :param ytrain: train target variable
        :return:  
        """
        self.energy_transformer.fit(xtrain, ytrain)
        xtrain = self.energy_transformer.transform(xtrain)
        self.regressor = Lasso(alpha=0.5)
        self.regressor.fit(xtrain, ytrain)
        return self

    def predict(self, xtest=None):
        """
        Apply trained Regressor on test data

        :param xtest: pandas data_frame for test data
        :return: 
        """
        xtest = self.energy_transformer.transform(xtest)
        return self.regressor.predict(xtest)
