from constants import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, EMBEDDING_FEATURES, TARGET
from sklearn.base import TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.impute import SimpleImputer
from transformers import NumericalTransformer, CategoricalTransformer, GroupSelector, \
    EmbeddingTransformer, MultiColumnLabelEncoder, EmbeddingTransformer


class EnergyFeatureBuilder(TransformerMixin):
    def __init__(self, with_embedding=False):
        self.with_embedding = with_embedding
        self.mapper = None
        self.sparse = False if self.with_embedding else True

    def create_pipeline(self):
        """
        Set up transformer pipeline 
        """
        pipeline_steps = []

        categorical_pipeline = Pipeline(steps=[('feature_selector', GroupSelector(CATEGORICAL_FEATURES)),
                                               ('categorical_features', CategoricalTransformer()),
                                               ('onehot', OneHotEncoder(sparse=self.sparse, handle_unknown='ignore'))
                                               ])
        pipeline_steps.append(('categorical_pipeline', categorical_pipeline))

        if self.with_embedding:
            embedding_pipeline = Pipeline(steps=[('feature_selector', GroupSelector(EMBEDDING_FEATURES)),
                                                 #('categorical_features', EmbeddingTransformer()),
                                                 ('encode_features', MultiColumnLabelEncoder())])
            pipeline_steps.append(('embedding_pipeline', embedding_pipeline))

        numerical_pipeline = Pipeline(steps=[('feature_selector', GroupSelector(NUMERICAL_FEATURES)),
                                             ('numeric_features', NumericalTransformer()),
                                             ('imputer', SimpleImputer(strategy='mean')),
                                             ('scaler', StandardScaler()),
                                             ])
        pipeline_steps.append(('numerical_pipeline', numerical_pipeline))

        transformer_pipeline = FeatureUnion(transformer_list=pipeline_steps)
        return transformer_pipeline

    def fit(self, samples, target=None):
        transformer_pipeline = self.create_pipeline()
        self.mapper = transformer_pipeline.fit(samples)
        return self

    def transform(self, x=None):
        return self.mapper.transform(x)


if __name__ == '__main__':
    from energy_data import AshreaEnergyData
    energy = AshreaEnergyData()
    target = energy.data[TARGET]
    samples = energy.data.drop(TARGET, axis=1)
    transformer = EnergyFeatureBuilder()
    transformer.fit(samples)
    samples_transformed = transformer.transform(samples)
