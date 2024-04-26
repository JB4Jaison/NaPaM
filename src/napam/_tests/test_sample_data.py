import pytest
from .._sample_data import make_sample_data
from napari import Viewer
from qtpy.QtWidgets import QApplication
import numpy as np
from qtpy.QtCore import Qt


def test_make_sample_data():
    # Example test for make_sample_data
    data = make_sample_data()
    
    assert isinstance(data, list), "Should return a list"
    assert len(data) == 1, "Should return a list with one element"

    assert isinstance(data[0], tuple), "Element should be a tuple"

    assert len(data[0]) == 2, "Tuple should have two elements"

    assert isinstance(data[0][0], np.ndarray), "First element of tuple should be a numpy array"

    assert isinstance(data[0][1], dict), "Second element of tuple should be a dictionary"

    assert len(data[0][1]) == 0, "Second element should be an empty dictionary"