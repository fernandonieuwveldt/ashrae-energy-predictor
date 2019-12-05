import pandas
import tensorflow
import keras
import numpy
from keras.models import Model, load_model
from keras.layers import Input, Dense, Concatenate, Reshape, Dropout, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from feature_builder import EnergyFeatureBuilder
from keras import backend as K
from keras.losses import mean_squared_error as mse_loss
from constants import EMBEDDING_FEATURES, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET
from sklearn.pipeline import FeatureUnion, Pipeline
from transformers import NumericalTransformer, CategoricalTransformer, GroupSelector, \
    EmbeddingTransformer, MultiColumnLabelEncoder, EmbeddingTransformer, TypeSelector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

tensorflow.logging.set_verbosity(tensorflow.logging.ERROR)
config = tensorflow.ConfigProto(device_count={"CPU": 8})
keras.backend.tensorflow_backend.set_session(tensorflow.Session(config=config))


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=0))


class EmbeddingBasedRegressor:
    """
    Model based on embedding layers for categorical features

    """
    EPOCHS = 10
    OPTIMIZER = optimizers.Adam(lr=0.001)
    EMBEDDING_RATIO = 0.25
    MAX_EMBEDDING = 7
    BATCH_SIZE = 1024
    VERBOSE = 0

    ES_PARAMS = {'monitor': 'val_root_mean_squared_error',
                 'mode': 'min',
                 'verbose': 1,
                 'patience': 3}

    MC_PARAMS = {'filepath': 'best_model.h5',
                 'monitor': 'val_root_mean_squared_error',
                 'mode': 'min',
                 'verbose': 1,
                 'save_best_only': True}
    FOLDS = 2
    SEED = 42

    def __init__(self):
        self.early_stopping = EarlyStopping(**self.ES_PARAMS)
        self.model_checkpoint = ModelCheckpoint(**self.MC_PARAMS)
        self.transformer = None
        self.classifier = None
        self.model = None
        self.feature_category_mapper = None
        self.feature_mode = None
        self.embedding_output_mapper = None
        self.models = []

    @staticmethod
    def preproc_embedding_layer(data_frame=None):
        """
        Creates new unique ordinal mapping to feed to embedding layer and create proper format for Keras model

        :param data_frame: pandas data frame
        :param feature_category_mapper:
        :return: list of preprocessed data frames
        """
        preproc_embedding = [data_frame[c].values for c in EMBEDDING_FEATURES]
        preproc_numerical = data_frame[NUMERICAL_FEATURES].values

        # unpack
        train_data = []
        for pe in preproc_embedding:
            train_data.append(pe)
        train_data.append(preproc_numerical)

        return train_data

    @staticmethod
    def _feature_transformer():
        """
        """
        pipeline_steps = []

        embedding_pipeline = Pipeline(steps=[('feature_selector', GroupSelector(EMBEDDING_FEATURES)),
                                             ('encode_features', MultiColumnLabelEncoder())])
        pipeline_steps.append(('embedding_pipeline', embedding_pipeline))

        numerical_pipeline = Pipeline(steps=[('feature_selector', GroupSelector(NUMERICAL_FEATURES)),
                                             ('numeric_features', NumericalTransformer()),
                                             ('imputer', SimpleImputer(strategy='median')),
                                             ('scaler', StandardScaler()),
                                            ])
        pipeline_steps.append(('numerical_pipeline', numerical_pipeline))

        transformer_pipeline = FeatureUnion(transformer_list=pipeline_steps)
        return transformer_pipeline

    @staticmethod
    def create_embedding_layer(n_unique=None, output_dim=None, input_length=1):
        """
        Creates embedding layers

        :param n_unique: dimension of unique labels
        :param output_dim: dimension of embedding matrix
        :param input_length: default 1
        :return: input data info, embedding layer info
        """
        _input = Input(shape=(1, ))
        _embedding = Embedding(n_unique, output_dim, input_length=input_length)(_input)
        _embedding = Reshape(target_shape=(output_dim, ))(_embedding)
        return _input, _embedding

    def build_network(self, data_frame):
        """
        Build up network with all embedding and other(numeric) layers

        :param feature_category_mapper:
        :param feature_mode:
        :param embedding_output_mapper:
        :return: compiled keras model
        """
        inputs = []
        layers = []

        for feature in EMBEDDING_FEATURES:
            print(feature)
            unique_categories = data_frame[feature].nunique()+1
            embedding_input, embedding_layer = self.create_embedding_layer(unique_categories,
                                                                           min(int(unique_categories*self.EMBEDDING_RATIO)+1, self.MAX_EMBEDDING))
            inputs.append(embedding_input)
            layers.append(embedding_layer)
        
        concat_embedding_layer = Concatenate()(layers)
        concat_embedding_layer = Dropout(0.1)(Dense(64,activation='relu')(concat_embedding_layer))
        concat_embedding_layer = BatchNormalization()(concat_embedding_layer)
        concat_embedding_layer = Dropout(0.1)(Dense(32,activation='relu')(concat_embedding_layer))

        # add layer for other numeric features
        numeric_input = Input(shape=(len(NUMERICAL_FEATURES), ))
        numeric_layer = Dense(len(NUMERICAL_FEATURES))(numeric_input)

        inputs.append(numeric_input)
        layers.append(numeric_layer)
        
        x = Concatenate()([concat_embedding_layer, numeric_layer])
        x = Dense(32, activation='relu')(concat_embedding_layer)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        x = Dense(16, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        output = Dense(1, activation='relu')(x)

        model = Model(inputs, output)
        model.compile(loss=mse_loss, optimizer=self.OPTIMIZER,  metrics=[root_mean_squared_error])

        return model

    def _fit_single_set(self, x_train=None, y_train=None, x_val=None, y_val=None):
        """
        Fit neural net model

        :param x_train: pandas data_frame
        :param y_train: training targets
        :param x_val: pandas data_frame
        :param y_val: validation targets
        :return:
        """
        
        self.transformer = self._feature_transformer()
        self.transformer.fit(x_train)

        x_train = self.transformer.transform(x_train)
        x_val = self.transformer.transform(x_val)

        x_train = pandas.DataFrame(x_train, columns=EMBEDDING_FEATURES+NUMERICAL_FEATURES)
        x_val = pandas.DataFrame(x_val, columns=EMBEDDING_FEATURES+NUMERICAL_FEATURES)

        self.regressor = self.build_network(x_train)

        x_train = self.preproc_embedding_layer(x_train)
        x_val = self.preproc_embedding_layer(x_val)

        params = {'x': x_train,
                  'y': y_train,
                  'validation_data': (x_val, y_val),
                  'batch_size': self.BATCH_SIZE,
                  'epochs': self.EPOCHS,
                  'verbose': self.VERBOSE,
                  'callbacks': [self.early_stopping, self.model_checkpoint]}
        self.regressor.fit(**params)

        del x_train
        del x_val

    def fit(self, x_train=None, y_train=None):
        self.transformers = []
        self.estimators = []

        kf = StratifiedKFold(n_splits=self.FOLDS, shuffle=True, random_state=self.SEED)
        for fold_n, (train_index, valid_index) in enumerate(kf.split(x_train, x_train['building_id'])):
            early_stopping_kwargs = {'monitor': 'val_root_mean_squared_error',
                                     'mode': 'min',
                                     'verbose': 1,
                                     'patience': 3}

            model_checkpoint_kwargs = {'filepath': f"best_model_{fold_n}.h5",
                                       'monitor': 'val_root_mean_squared_error',
                                       'mode': 'min',
                                       'verbose': 1,
                                       'save_best_only': True}

            early_stopping = EarlyStopping(**early_stopping_kwargs)
            model_checkpoint = ModelCheckpoint(**model_checkpoint_kwargs)

            x_train_, x_val = x_train.iloc[train_index], x_train.iloc[valid_index]
            y_train_, y_val = y_train.iloc[train_index], y_train.iloc[valid_index]

            transformer = self._feature_transformer()
            transformer.fit(x_train_)

            x_train_ = transformer.transform(x_train_)
            x_val = transformer.transform(x_val)

            x_train_ = pandas.DataFrame(x_train_, columns=EMBEDDING_FEATURES+NUMERICAL_FEATURES)
            x_val = pandas.DataFrame(x_val, columns=EMBEDDING_FEATURES+NUMERICAL_FEATURES)

            estimator = self.build_network(x_train_)

            x_train_ = self.preproc_embedding_layer(x_train_)
            x_val = self.preproc_embedding_layer(x_val)

            model_kwargs = {'x': x_train_,
                            'y': y_train_,
                            'validation_data': (x_val, y_val),
                            'batch_size': self.BATCH_SIZE,
                            'epochs': self.EPOCHS,
                            'verbose': self.VERBOSE,
                            'callbacks': [early_stopping, model_checkpoint]}
            estimator.fit(**model_kwargs)

            # record transformer and estimator
            self.transformers.append(transformer)
            self.estimators.append(estimator)

        del x_train, x_train_, x_val 


    def predict_single_model(self, x_test=None):
        """
        Loads best model for prediction

        :param x_test:
        :return:
        """
        x_test = self.transformer.transform(x_test)
        x_test = pandas.DataFrame(x_test, columns=EMBEDDING_FEATURES+NUMERICAL_FEATURES)

        x_test = self.preproc_embedding_layer(x_test)
        saved_model = load_model(self.MC_PARAMS['filepath'], custom_objects={'root_mean_squared_error': root_mean_squared_error})
        return saved_model.predict(x_test)

    def predict(self, x_test=None):
        """
        Loads best model for prediction

        :param x_test:
        :return:
        """
        predictions = numpy.zeros((x_test.shape[0], 1))
        for s in range(self.FOLDS):
            x_test_transformed = self.transformers[s].transform(x_test)
            x_test_transformed = pandas.DataFrame(x_test_transformed, columns=EMBEDDING_FEATURES+NUMERICAL_FEATURES)
            x_test_transformed = self.preproc_embedding_layer(x_test_transformed)
            predictions += self.estimators[s].predict(x_test_transformed) / self.FOLDS
        return predictions

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
