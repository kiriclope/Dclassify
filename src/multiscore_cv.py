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
from sklearn.utils import indexable


@verbose
def cross_val_multiscore_A_B(
        estimator,
        X_A,
        y_A=None,
        cv_A=None,
        groups_A=None,
        X_B=None,
        y_B=None,
        cv_B=False,
        groups_B=None,
        scoring=None,
        n_jobs=None,
        verbose=None,
        fit_params=None,
        ret_all=False,
        pre_dispatch="2*n_jobs",
):

    check_scoring = _get_check_scoring()
    cv_dum = cv_A

    X_A, y_A, groups_A = indexable(X_A, y_A, groups_A)
    cv_A = check_cv(cv_A, y_A, classifier=is_classifier(estimator))
    cv_A = list(cv_A.split(X_A, y_A, groups_A))
    y_A = y_A % 2

    # IF_COMPO=1
    # if X_B is None:
    #     IF_COMPO = 0
    #     cv_B = cv_A

    # setting folds for set B
    if cv_B:
        X_B, y_B, groups_B = indexable(X_B, y_B, groups_B)
        cv_B = check_cv(cv_dum, y_B, classifier=is_classifier(estimator))
        cv_B = list(cv_B.split(X_B, y_B, groups_B))

    # if IF_COMPO:
    #     X_B, y_B, groups_B = indexable(X_B, y_B, groups_B)
    #     # setting folds for set B
    #     if cv_B:
    #         cv_B = check_cv(cv_dum, y_B, classifier=is_classifier(estimator))
    #         cv_B = list(cv_B.split(X_B, y_B, groups_B))

    #         while len(cv_B) < len(cv_A):
    #             cv_B.append([[], []])

    #     else:
    #         # no split testing on all X_B
    #         n_samples = len(y_B)
    #         cv_B = []
    #         for (train_A, test_A) in cv_A:
    #             cv_B.append([np.arange(n_samples), np.arange(n_samples)])  # Single tuple for no split

    #     # using exact same folds for A and B for temporal generalization
    #     if X_A.shape == X_B.shape and np.array_equal(X_A, X_B):
    #         cv_B=cv_A

    #     if verbose:
    # print('cv_A', len(cv_A), 'cv_B', len(cv_B))
    # print('X_A', X_A.shape, 'y_A', y_A.shape)
    # print('X_B', X_B.shape, 'y_B', y_B.shape)

    try:
        scorer = check_scoring(estimator, scoring=scoring)
    except:
        scorer = scoring

    parallel, p_func, n_jobs = parallel_func(
        _fit_and_score_A_B, n_jobs, pre_dispatch=pre_dispatch
    )

    # scores, probas, coefs, labels = zip(*parallel(
    #     p_func(
    #         estimator=deepcopy(estimator),
    #         X_A=X_A,
    #         y_A=y_A,
    #         train_A=train_A,
    #         test_A=test_A,
    #         X_B=X_B,
    #         y_B=y_B,
    #         test_B=test_B,
    #         scorer=scorer,
    #         if_compo=IF_COMPO,
    #         fit_params=fit_params,
    #         ret_all=ret_all,
    #         verbose=None,
    #     )
    #     for (train_A, test_A), (_, test_B) in zip(cv_A, cv_B)
    #     # for (train_A, test_A) in cv_A for (_, test_B) in cv_B
    # ))

    fit_params = fit_params if fit_params is not None else {}
    scores, probas, coefs, labels = zip(*parallel(
        p_func(
            estimator=deepcopy(estimator),
            X_A=X_A,
            y_A=y_A,
            train_A=train_A,
            test_A=test_A,
            X_B=None,
            y_B=None,
            test_B=None,
            scorer=scorer,
            if_compo=0,
            fit_params=fit_params,
            ret_all=ret_all,
            verbose=None,
        )
        for (train_A, test_A) in cv_A
    ))

    if cv_B:
        estimator.fit(X_A, y_A, **fit_params)

        parallel, p_func, n_jobs = parallel_func(
            _nofit_and_score_B, n_jobs, pre_dispatch=pre_dispatch
        )

        scores_B = parallel(
            p_func(
                estimator=deepcopy(estimator),
                X_B=X_B,
                y_B=y_B,
                test_B=test_B,
                scorer=scorer,
            )
            for (_, test_B) in cv_B
        )

        scores = [scores, scores_B]

    return scores, probas, coefs, labels


@verbose
def _fit_and_score_A_B(
    estimator,
    X_A,
    y_A,
    train_A,
    test_A,
    X_B,
    y_B,
    test_B,
    scorer,
    if_compo,
    fit_params,
    ret_all,
    verbose,
):
    """Fit estimator and compute scores for a given dataset split."""

    y_A = y_A % 2
    X_train, y_train = _safe_split(estimator, X_A, y_A, train_A)
    X_A_test, y_A_test = _safe_split(estimator, X_A, y_A, test_A, train_A)

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = _check_fit_params(X_train, fit_params, train_A)

    if y_train is None:
        estimator.fit(X_train, **fit_params)
    else:
        estimator.fit(X_train, y_train, **fit_params)

    if verbose:
        print(estimator)

    scores = _score(estimator, X_A_test, y_A_test, scorer)

    if ret_all:
        probas = np.array(estimator.predict_proba(X_A_test))
        labels = pd.DataFrame({'test_A': test_A})

    if if_compo:
        if len(test_B)!=0:  # Check if y_B_test is not empty
            y_B = y_B % 2
            X_B_test, y_B_test = _safe_split(estimator, X_B, y_B, test_B)
            score_B = _score(estimator, X_B_test, y_B_test, scorer)
            scores = [scores, score_B]

            if ret_all:
                proba_B = np.array(estimator.predict_proba(X_B_test))
                probas = [probas, proba_B]
                labels['test_B'] = pd.Series(test_B)

        else:
            scores = [scores, scores * np.nan]
            if ret_all:
                probas = [probas, probas * np.nan]

        if verbose:
            print('scores', scores, score_B)
            # print('probas', probas.shape, proba_B.shape)

    if ret_all:
        coefs = get_coef(estimator, 'coef_')
        intercept = get_coef(estimator, 'intercept_')
        coefs = np.vstack((intercept[:, np.newaxis], coefs))

    if verbose:
        print('coefs', coefs.shape)

    if ret_all:
        return scores, probas, coefs, labels
    else:
        return scores, 0, 0, 0

def _nofit_and_score_B(
        estimator,
        X_B,
        y_B,
        test_B,
        scorer,
):

    X_B_test, y_B_test = _safe_split(estimator, X_B, y_B, test_B)
    score_B = _score(estimator, X_B_test, y_B_test, scorer)

    return score_B

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
