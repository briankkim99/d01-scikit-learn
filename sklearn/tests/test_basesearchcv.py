import pytest

import scipy.sparse as sp

from sklearn.base import clone
from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import assert_almost_equal
from sklearn.utils._testing import ignore_warnings
from sklearn.utils.stats import _weighted_percentile

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn import svm, datasets
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV

iris = datasets.load_iris()

def grid_search(parameters):
  svc = svm.SVC()
  clf = GridSearchCV(svc, parameters,return_train_score=True)
  clf.fit(iris.data, iris.target)
  return clf.best_params_

def test_gridsearchcv_best_single():

  params1 = {'kernel':('linear', 'rbf'), 'C':[1]}
  p1 = grid_search(params1)
  params2 = {'kernel':('linear', 'rbf'), 'C':[1]}
  p2 = grid_search(params2)

  assert p1 == p2

def test_gridsearchcv_best_double():

  params1 = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
  p1 = grid_search(params1)
  params2 = {'kernel':('rbf', 'linear'), 'C':[10, 1]}
  p2 = grid_search(params2)

  assert p1 == p2

def test_gridsearchcv_best_multi():

  params1 = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
  p1 = grid_search(params1)
  params2 = {'kernel':('rbf', 'linear'), 'C':[1, 10]}
  p2 = grid_search(params2)
  params3 = {'kernel':('linear', 'rbf'), 'C':[10, 1]}
  p3 = grid_search(params3)
  params4 = {'kernel':('rbf', 'linear'), 'C':[10, 1]}
  p4 = grid_search(params4)

  assert p1 == p2 == p3 == p4
