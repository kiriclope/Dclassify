from time import perf_counter
from sklearn.base import clone
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, LeaveOneOut
from sklearn.decomposition import PCA

from mne.decoding import SlidingEstimator, GeneralizingEstimator, cross_val_multiscore
from src.my_mne import my_cross_val_multiscore, my_cross_val_compo_score

def convert_seconds(seconds):
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return h, m, s

class ClassificationCV:
    def __init__(self, model, params, **kwargs):
        # kwargs = {
        #     'n_comp': None, # PCA, int means number of PCs
        #     'scaler': None, # standardization (z score)
        #     'n_boots': 1, # bootstrapping coefs
        #     'n_splits': 3, 'n_repeats': 1, # repeated stratified folds unless -1 is LOOCV
        #     'scoring': 'roc_auc', # scorer
        #     'estimator':'sliding', # sliding or generalizing
        #     'verbose': 0,
        #     'n_jobs': 30,
        # }

        pipeline = []

        # Standardize features, X, z score across trials
        # see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
        self.scaler = kwargs["scaler"]
        if self.scaler is not None and self.scaler != 0:
            pipeline.append(("scaler", StandardScaler()))

        # Reduce features, X, dimensionality with PCA
        # see https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
        self.n_comp = kwargs["n_comp"]
        if (kwargs["n_comp"] is not None) and (kwargs["n_comp"]!=0):
            self.n_comp = kwargs["n_comp"]
            pipeline.append(("pca", PCA(n_components=self.n_comp)))

        # model is the regression/classification based model
        # see e.g. https://scikit-learn.org/stable/api/sklearn.linear_model.html
        # in this implementation one can also train RNNs with skorch
        # see https://skorch.readthedocs.io/en/stable/
        # in particular, perceptrons and MLPs.
        pipeline.append(("model", model))

        # Intermediate steps to be implemented to preprocess the features before fit.
        # see https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
        self.pipe = Pipeline(pipeline)

        # scorer of the classification
        # see https://scikit-learn.org/stable/modules/model_evaluation.html
        # 'accuracy', 'roc_auc', 'f1'
        self.scoring = kwargs["scoring"]

        # Decoding over time (from MNE)
        # if estimator is SlidingEstimator(base_estimator, scoring=None, n_jobs=1, verbose=None)
        # fits a multivariate predictive model on each time instant
        # and evaluates its performance at the same instant on new epochs.
        # if estimator is GeneralizingEstimator(base_estimator, scoring=None, n_jobs=1, verbose=None)
        # fits a multivariate predictive model on each time instant
        # and evaluates its performance at all other instant on new epochs.
        # see https://mne.tools/0.23/auto_tutorials/machine-learning/50_decoding.html
        # 'sliding'or 'generalizing'
        self.estimator = kwargs["estimator"]

        # Defines the type of cross validation
        if kwargs["n_splits"] == -1:
            self.cv = LeaveOneOut()
        else:
            self.cv = RepeatedStratifiedKFold(
                n_splits=kwargs["n_splits"], n_repeats=kwargs["n_repeats"]
            )

        self.n_jobs = kwargs["n_jobs"]

        # Parameter grid for gridsearch of hyperparameters
        # see https://scikit-learn.org/stable/modules/grid_search.html#grid-search
        self.params = params
        self.grid = GridSearchCV(
            self.pipe,
            self.params,
            refit=True,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
        )

        # By default sets best_model to the grid to perform nested CV.
        # This is overwritten when calling the fit method.
        self.best_model = clone(self.grid)

        self.verbose = kwargs["verbose"]

    def fit(self, X, y):
        """Fits the model hyperparameters with GridSearchCV.
        Here, hyperparameters are meant to be fit on a given epoch,
        X (n_samples, n_features), y (n_samples,).
        """

        start = perf_counter()
        if self.verbose:
            print("Fitting hyperparameters on single epoch ...")


        self.grid.fit(X.astype("float32"), y.astype("float32"))
        end = perf_counter()
        if self.verbose:
            print(
                "Elapsed (with compilation) = %dh %dm %ds"
                % convert_seconds(end - start)
            )

        self.best_model = self.grid.best_estimator_
        self.best_params = self.grid.best_params_

        if self.verbose:
            print(self.best_params)

        try:
            # coefs from skorch models
            self.coefs = (
                self.best_model.named_steps["model"]
                .module_.linear.weight.data.cpu()
                .detach()
                .numpy()[0]
            )
            self.bias = (
                self.best_model.named_steps["model"]
                .module_.linear.bias.data.cpu()
                .detach()
                .numpy()[0]
            )
        except:
            # coefs from sklearn models
            self.coefs = self.best_model.named_steps["model"].coef_.T
            self.bias = self.best_model.named_steps["model"].intercept_.T

    def get_bootstrap_coefs(self, X, y, n_boots=10):
        """Bootstrapping model coeficients. Useful when using lasso regularization."""

        start = perf_counter()
        if self.verbose:
            print("Bootstrapping coefficients ...")

        self.bagging_clf = BaggingClassifier(
            base_estimator=self.best_model, n_estimators=n_boots
        )
        self.bagging_clf.fit(X.astype("float32"), y.astype("float32"))
        end = perf_counter()

        if self.verbose:
            print(
                "Elapsed (with compilation) = %dh %dm %ds"
                % convert_seconds(end - start)
            )

        self.coefs, self.bias = get_bagged_coefs(self.bagging_clf, n_estimators=n_boots)

        return self.coefs, self.bias

    def get_overlap(self, pipe, X):
        """Feature projection on model's decision hyperplane"""
        try:
            coefs = (
                pipe.named_steps["model"]
                .module_.linear.weight.data.cpu()
                .detach()
                .numpy()[0]
            )
            bias = (
                pipe.named_steps["model"]
                .module_.linear.bias.data.cpu()
                .detach()
                .numpy()[0]
            )
        except:
            coefs = pipe.named_steps["model"].coef_.T
            bias = pipe.named_steps["model"].intercept_.T

        if self.scaler is not None and self.scaler != 0:
            scaler = pipe.named_steps["scaler"]
            for i in range(X.shape[-1]):
                X[..., i] = scaler.transform(X[..., i])

        if self.n_comp is not None:
            pca = pipe.named_steps["pca"]
            X_pca = np.zeros((X.shape[0], self.n_comp, X.shape[-1]))

            for i in range(X.shape[-1]):
                X_pca[..., i] = pca.transform(X[..., i])

            self.overlaps = (
                np.swapaxes(X_pca, 1, -1) @ coefs + bias
            )  # / np.linalg.norm(coefs, axis=0)
        else:
            self.overlaps = -(
                np.swapaxes(X, 1, -1) @ coefs + bias
            )  # / np.linalg.norm(coefs, axis=0)

        return self.overlaps

    def get_bootstrap_overlaps(self, X):
        """Bootstrapped Feature projections onto models decision boundary."""
        start = perf_counter()
        if self.verbose:
            print("Getting bootstrapped overlaps ...")

        X_copy = np.copy(X)
        overlaps_list = []
        n_boots = len(self.bagging_clf.estimators_)

        for i in range(n_boots):
            model = self.bagging_clf.estimators_[i]
            overlaps = self.get_overlap(model, X_copy)
            overlaps_list.append(overlaps)

        end = perf_counter()
        if self.verbose:
            print(
                "Elapsed (with compilation) = %dh %dm %ds"
                % convert_seconds(end - start)
            )

        return np.array(overlaps_list).mean(0)

    def get_cv_scores(self, X, y, scoring, cv=None, X_test=None, y_test=None, IF_COMPO=0):
        """Cross validated model scores."""

        if cv is None:
            cv = self.cv

        # Here X_test, y_test can be provided for evaluating compositional scores.
        # i.e., training on a set of features X, y and testing on a different one.
        if X_test is None:
            X_test = X
            y_test = y

        start = perf_counter()
        if self.verbose:
            print("Computing cv scores ...")

        if self.estimator == 'sliding':
            estimator = SlidingEstimator(
                clone(self.best_model), n_jobs=1, scoring=scoring, verbose=False
            )
        elif self.estimator == 'generalizing':
            estimator = GeneralizingEstimator(
                clone(self.best_model), n_jobs=1, scoring=scoring, verbose=False
            )

        # self.scores = cross_val_multiscore(estimator, X.astype('float32'), y.astype('float32'),
        #                                    cv=cv, n_jobs=-1, verbose=False)

        if IF_COMPO:
            self.scores = my_cross_val_compo_score(
                estimator,
                X.astype("float32"),
                X_test.astype("float32"),
                y.astype("float32"),
                y_test,
                cv=cv,
                n_jobs=-1,
                verbose=False,
            )
        else:
            self.scores = my_cross_val_multiscore(
                estimator,
                X.astype("float32"),
                X_test.astype("float32"),
                y.astype("float32"),
                y_test.astype("float32"),
                cv=cv,
                n_jobs=-1,
                verbose=False,
            )

        end = perf_counter()
        if self.verbose:
            print(
                "Elapsed (with compilation) = %dh %dm %ds"
                % convert_seconds(end - start)
            )

        return self.scores
