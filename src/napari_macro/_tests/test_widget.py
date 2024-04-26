import pytest
from .._widget import process_image, MacroWidget
from napari import Viewer
from qtpy.QtWidgets import QApplication
import numpy as np
from qtpy.QtCore import Qt

@pytest.fixture
def app(qtbot):
    return QApplication.instance() or QApplication([])

@pytest.fixture
def viewer():
    return Viewer()

@pytest.fixture
def macro_widget(viewer, qtbot):
    widget = MacroWidget(viewer)
    qtbot.addWidget(widget)
    return widget

#TODO: Create mock class for the GUI and test the GUI elements

def test_process_image():
    # Example test for process_image
    image = np.zeros((10, 10))
    code = "result = image.copy()"
    result = process_image(image, code)
    assert np.array_equal(result, image), "The processed image should match the original."

def test_macro_widget_image_change_detection(macro_widget, viewer, qtbot):
    # Testing the widget's ability to detect changes in the image
    initial_image = np.zeros((10, 10))
    viewer.add_image(initial_image)
    changed_image = initial_image.copy()
    changed_image[5, 5] = 1  # Modify the image
    viewer.layers[0].data = changed_image
    # Trigger any function or check that should detect this change
    assert macro_widget.image_has_changed(initial_image, changed_image), "Widget should detect the image change."

def test_image_processing_function():
    # Placeholder for testing a specific image processing function
    image = np.zeros((10, 10))
    # Modify the image or apply a processing function
    processed_image = image  # Replace with actual processing
    assert processed_image is not None, "Processed image should not be None."
    # Add more assertions based on expected behavior
