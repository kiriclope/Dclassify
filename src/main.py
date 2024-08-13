import torch
import numpy as np
from time import perf_counter

def convert_seconds(seconds):
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return h, m, s


def get_classification(model, X, y, X_test=None, y_test=None, RETURN="overlaps", **kwargs):
    start = perf_counter()

    if kwargs["verbose"]:
        print("Features, X,", X.shape, " | labels, y,", y.shape)

    if "scores" in RETURN:
        scores = model.get_cv_scores(
            X, y, kwargs["scoring"], cv=None, X_test=X_test, y_test=y_test
        )
        end = perf_counter()
        print("Elapsed (with compilation) = %dh %dm %ds" % convert_seconds(end - start))
        return scores
    elif "overlaps" in RETURN:
        coefs, bias = model.get_bootstrap_coefs(X_avg, y, n_boots=kwargs["n_boots"])
        print("coefs", coefs.shape)
        overlaps = model.get_bootstrap_overlaps(X)
        end = perf_counter()
        print("Elapsed (with compilation) = %dh %dm %ds" % convert_seconds(end - start))
        return overlaps
    elif "coefs" in RETURN:
        if kwargs["n_boots"] >1:
            print("Bagging Classifier")
            coefs, bias = model.get_bootstrap_coefs(X, y, n_boots=kwargs["n_boots"])
        else:
            model.fit(X, y)
            coefs = model.coefs
            bias = model.coefs

        end = perf_counter()
        print("Elapsed (with compilation) = %dh %dm %ds" % convert_seconds(end - start))
        return coefs, bias
    else:
        return None
