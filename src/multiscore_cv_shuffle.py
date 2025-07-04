import numpy as np
import pandas as pd

from copy import deepcopy

from mne.fixes import _get_check_scoring, is_classifier
from mne.parallel import parallel_func
from mne.fixes import _check_fit_params
from mne.decoding import get_coef
from mne.utils import verbose, warn

from sklearn.base import clone
from sklearn.utils.metaestimators import _safe_split
from sklearn.model_selection._split import check_cv
from sklearn.utils import indexable, _safe_indexing, check_random_state
from sklearn.utils.parallel import Parallel, delayed


def cross_val_perm_multiscore(
        estimator,
        X,
        y=None,
        cv=None,
        groups=None,
        scoring=None,
        n_perm=100,
        n_jobs=None,
        fit_params=None,
        random_state=None
):

    check_scoring = _get_check_scoring()

    X, y, groups = indexable(X, y, groups)
    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    random_state = check_random_state(random_state)

    try:
        scorer = check_scoring(estimator, scoring=scoring)
    except:
        scorer = scoring

    fit_params = fit_params if fit_params is not None else {}

    scores = []
    perm_scores = []
    for train, test in cv.split(X, y):
        X_train, y_train = _safe_split(estimator, X, y, train)
        X_test, y_test = _safe_split(estimator, X, y, test, train)

        estimator.fit(X_train, y_train, **fit_params)
        scores.append(scorer(estimator, X_test, y_test))

        perm_scores.append(_permutation_scores(
            estimator, X_test, y_test, groups, scorer, n_perm=n_perm, n_jobs=n_jobs, random_state=random_state))

    scores = np.array(scores)
    perm_scores = np.array(perm_scores)

    # pvalues = (np.sum(perm_scores >= scores.mean(0), axis=0) + 1.0) / (n_perm + 1)

    return scores, perm_scores #, pvalues


def _permutation_scores(
        estimator,
        X,
        y,
        groups,
        scorer,
        n_perm=100,
        n_jobs=None,
        random_state=None,
        VERBOSE=False,
):
    permutation_scores = Parallel(n_jobs=n_jobs, verbose=VERBOSE)(
        delayed(_score)(
            deepcopy(estimator),
            X,
            _shuffle(y, groups, random_state),
            scorer,
        )
        for _ in range(n_perm)
    )

    return np.array(permutation_scores)

def _score(estimator, X_test, y_test, scorer):
    """Compute the score of an estimator on a given test set.

    This code is the same as sklearn.model_selection._validation._score
    but accepts to output arrays instead of floats.
    """

    if y_test is None:
        score = scorer(estimator, X_test)
    else:
        score = scorer(estimator, X_test, y_test)

    if hasattr(score, "item"):
        try:
            # e.g. unwrap memmapped scalars
            score = score.item()
        except ValueError:
            # non-scalar?
            pass

    return score


def _shuffle(y, groups, random_state):
    """Return a shuffled copy of y eventually shuffle among same groups."""
    if groups is None:
        indices = random_state.permutation(len(y))
    else:
        indices = np.arange(len(groups))
        for group in np.unique(groups):
            this_mask = groups == group
            indices[this_mask] = random_state.permutation(indices[this_mask])
    return _safe_indexing(y, indices)
