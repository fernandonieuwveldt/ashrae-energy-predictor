from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import numpy


class TypeSelector(TransformerMixin):
    """
    Transformer that select columns based on type or from list of names

    """
    def __init__(self, type=None):
        self.type = type

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """
        Select only features of specified type

        :param X: pandas data_frame
        :param y: None
        :return: pandas data_frame only containing features of specified type
        """
        features = X.dtypes[X.dtypes == self.type].index
        return X[features]
        
class CategoricalTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer extracting categorical features and engineer new ones

    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, xc=None):
        xc = xc.fillna(-999).astype('str')
        return xc


class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer extracting embedding candidate features

    """
    N_UNIQUE = 4

    def __init__(self):
        self.candidates_features = None

    def fit(self, X, y=None):
        """
        Get list of features that are embedding feature candidates

        :param X: pandas data_frame
        :param y: None
        :return: EmbeddingTransformer object
        """

        candidates = X.nunique() > self.N_UNIQUE

        self.candidates_features = list(candidates[candidates].index)
        return self

    def transform(self, xc=None):
        return xc[self.candidates_features].fillna(-999)


class MultiColumnLabelEncoder(TransformerMixin):
    """
    Apply LabelEncoder on multiple variables.

    Sets a default label if unseen labels are supplied during inference
    """
    def __init__(self):
        self.encoders = None
        self.embedding_candidates = []

    def fit(self, X, y=None):
        """
        Fit LabelEncoder's on multiple features

        :param X: pandas data_frame
        :param y: None
        :return: MultiColumnLabelEncoder object
        """
        self.encoders = {}
        for feature, c in X.iteritems():
            encoder = LabelEncoder()
            encoder.fit(c)
            self.encoders[feature] = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
            self.embedding_candidates.append(feature)

        return self

    def transform(self, X):
        """
        Apply LabelEncoder and supply default values for unseen labels during inference

        :param X: pandas data frame
        :return: label encoded pandas data frame
        """
        for feature in self.embedding_candidates:
            X.loc[:, feature] = X.loc[:, feature].apply(lambda x: self.encoders[feature].get(x, 999))
        return X


class NumericalTransformer(TransformerMixin):
    """
    Transformer extracting numerical features and feature engineering

    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """

        :param X: pandas data_frame
        :param y: None
        :return: NumericalTransformer object
        """
        return self

    def transform(self, xn=None):
        return xn


class GroupSelector(TransformerMixin):
    """
    Selects only supplied features.

    This is a helper class to be able select features in sklearn.pipeline
    """
    def __init__(self, features=None):
        self.features = features

    def fit(self, X=None, y=None):
        return self

    def transform(self, xv=None):
        return xv[self.features]


class AutoPCA(TransformerMixin):
    """
    Selects the optimal n_components to use based on the explained_variance threshold.

    """
    def __init__(self, variance_threshold=0.98):
        self.opt_ncomponents = None
        self.variance_threshold = variance_threshold
        self.pca = PCA()

    def fit(self, X, y=None):
        """
        Apply PCA and get optimal number of components based on the variance

        :param X: feature matrix
        :param y: None
        :return: self return object
        """
        self.pca.fit(X)
        self.opt_ncomponents = numpy.argmax(self.pca.explained_variance_ratio_.cumsum() > self.variance_threshold)
        return self

    def transform(self, X, y=None):
        return self.pca.transform(X)[:, :self.opt_ncomponents]


class GroupTransformer(TransformerMixin):
    def __init__(self, feature=None, transformer=None, **kwargs):
        self.feature = feature
        self.transformer = transformer
        self.kwargs = kwargs

    def fit(self, X, y=None):
        """
        """
        self.group_transform_ = {}
        for _id, group in X.groupby(self.feature):
            self.group_transform_[_id] = self.transformer(**self.kwargs).fit(X, y)
        return self

    def transform(self, X, y=None):
        for _id, group in X.groupby(self.feature):
            group_indices = X[self.feature]==_id
            X[group_indices, :] = self.group_transform_[_id].transform(X[group_indices, :])
        return X


class GroupImputer(GroupTransformer):
    def __init__(self, feature=None, imputer=SimpleImputer, **kwargs):
        super().__init__(feature=feature,
                         transformer=imputer,
                         **kwargs
                         )


class GroupScaler(GroupTransformer):
    def __init__(self, feature=None, scaler=StandardScaler, **kwargs):
        super().__init__(feature=feature,
                         transformer=scaler,
                         **kwargs)

