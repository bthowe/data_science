import joblib
import helper_script as helpers
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin


class PipelineWrapper(BaseEstimator, TransformerMixin):
    """A wrapper for the fitted pipeline 'pipeline_obj.' Useful in gridsearching when some actions performed in a
    pipeline should be done using values derived from the entire training dataset and not only on n-1 folds during
    cross-validation."""
    def __init__(self, pipeline_obj):
        self.pipeline_obj = pipeline_obj

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        return joblib.load(self.pipeline_obj).transform(X)


def pipeline_fit(X):
    """Fits a pipeline on data X."""
    pipeline = Pipeline(
        [
            ('step1', FunctionTransformer(helpers.step1, validate=False)),
            ('step2', FunctionTransformer(helpers.step2, validate=False)),
            ('step3', FunctionTransformer(helpers.step3, validate=False)),
            ('step4', FunctionTransformer(helpers.step4, validate=False))
        ]
    )
    pipeline.fit(X)
    joblib.dump(pipeline, '/Users/travis.howe/Downloads/pipeline.pkl')  # could, alternatively, simply return the pipeline object

def model_train(X, y):
    """Puts it all together."""
    pipeline = Pipeline(
        [
            ('feature_create', PipelineWrapper('/Users/travis.howe/Downloads/pipeline.pkl')),
            ('et', ExtraTreesClassifier(n_estimators=100, criterion='gini'))
        ]
    )

    et_parameters = {
        'et__max_depth': [None, 2, 4],
        'et__max_features': ['auto', None, 10]
    }

    grid_search = GridSearchCV(pipeline, et_parameters, scoring='roc_auc', verbose=10, n_jobs=6, cv=3)
    grid_search.fit(X, y)
    print(grid_search.best_params_)
    print(grid_search.best_score_)
    return grid_search