from napari.layers import Image, Labels, Shapes
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QHBoxLayout, QCheckBox, QComboBox
from PyQt5.Qsci import QsciScintilla, QsciLexerPython
from PyQt5.QtGui import QColor, QFont
import numpy as np
from napari.layers import Labels
from copy import deepcopy
from matplotlib.path import Path
import time
import datetime
import napari
import traceback

# Create a function to process the image based on the user's code
def process_image(image:np.ndarray, code:str):
    '''
    Process the image based on the user's code

    Parameters
    ----------
    image : np.ndarray
        The image to be processed
    code : str
        The code to be executed on the image

    Returns
    -------
    np.ndarray
        The processed image

    '''

    if len(image) == 0:
        return "Error: No image provided"
    
    if not code:
        return "Error: No code provided"
    
    try:
        print("RUNNING USER MACRO ...")
        # Create a namespace for the code to run in
        namespace = {'image': image, 'result': None}
        
        # Time the execution of the code
        start = time.time()
        # Execute the user's code within the namespace
        exec(code, namespace)

        end = time.time()
        print(f"Time Elapsed For Completion - {str(datetime.timedelta(seconds=end - start))}")
        # Retrieve the result from the namespace
        result = namespace.get('result', None)
        # print(result, type(result))
        
        return result
    except Exception as e:
        print("AN ERROR OCCURED ...")
        return f"Error: {traceback.format_exc()}"

class MacroWidget(QWidget):
    def __init__(self, napari_viewer: 'napari.viewer.Viewer'):
        super().__init__()
        self.viewer = napari_viewer

        self.setLayout(QVBoxLayout())

        # Create dropdowns for selecting layers
        self.image_layer_dropdown = QComboBox()
        self.shape_layer_dropdown = QComboBox()
        
        self.layout().addWidget(QLabel("Select Image/Label Layer:"))
        self.layout().addWidget(self.image_layer_dropdown)

        self.layout().addWidget(QLabel("Select Shape Layer:"))
        self.layout().addWidget(self.shape_layer_dropdown)

        self.update_layer_dropdowns()

        # Connect layer events to update the dropdowns
        self.viewer.layers.events.inserted.connect(self.update_layer_dropdowns)
        self.viewer.layers.events.removed.connect(self.update_layer_dropdowns)

        # Create a QTextEdit for the user to input code
        self.code_input = self.add_code_box()

        self.run_button = QPushButton("Run Code")
        self.layout().addWidget(self.run_button)

        self.run_button.clicked.connect(self.run_code)

        self.output_layer_text = self.output_name_text_box()

        # Create a checkbox for selecting ROI
        self.roi_checkbox = QCheckBox("Apply to ROI only")
        self.layout().addWidget(self.roi_checkbox)

    def update_layer_dropdowns(self, event=None):
        print("Updating Layer Dropdowns ...")

        # Retainn the current selection
        current_image_layer = self.image_layer_dropdown.currentText()
        current_shape_layer = self.shape_layer_dropdown.currentText()

        # Clear the dropdown lists
        self.image_layer_dropdown.clear()
        self.shape_layer_dropdown.clear()

        for layer in self.viewer.layers:
            if isinstance(layer, (Image, Labels)):
                self.image_layer_dropdown.addItem(layer.name)
            elif isinstance(layer, Shapes):
                self.shape_layer_dropdown.addItem(layer.name)

        self.image_layer_dropdown.setCurrentText(current_image_layer)
        self.shape_layer_dropdown.setCurrentText(current_shape_layer)
    
    def add_code_box(self):

        code_area = QWidget()
        layout = QVBoxLayout()
        
        layout.setContentsMargins(0, 0, 0, 0)

        new_combo_label = QLabel("Macro Code Area")
        layout.addWidget(new_combo_label) # Adding the first component in the layout - Label

        # Creating a QsciScintilla text editor
        editor = QsciScintilla(self)
        font = QFont('Consolas', 11)
       
        """
        Customization - GENERAL
        """
        # Setting a lexer for syntax highlighting
        lexer = QsciLexerPython()
        editor.setLexer(lexer)
        editor.setFont(font)
        # Set the encoding to UTF-8
        editor.setUtf8(True)
        # Set tab width to 4 spaces
        editor.setTabWidth(4)  
        # Placeholder text
        editor.setText("image #The input variable - your selected image\n'''\nAdd your Processing Steps Here\n'''\nresult= image")

        # Set text wrapping mode
        editor.setWrapMode(QsciScintilla.WrapWord)

        """
        Customization - CARET (the blinking cursor indicator)
        """
        # Set the caret color to red
        caret_fg_color = QColor("#ffff0000")
        editor.setCaretForegroundColor(caret_fg_color)
        # Enable and set the caret line background color to slightly transparent blue
        editor.setCaretLineVisible(True)
        caret_bg_color = QColor("#1f0000ff")
        editor.setCaretLineBackgroundColor(caret_bg_color)
        # Set the caret width of 4 pixels
        editor.setCaretWidth(4)

        """
        Customization - AUTOCOMPLETION
        """
        # Set the autocompletions to case INsensitive
        editor.setAutoCompletionCaseSensitivity(False)
        # Set the autocompletion to not replace the word to the right of the cursor
        editor.setAutoCompletionReplaceWord(False)
        # Set the autocompletion source to be the words in the document
        editor.setAutoCompletionSource(QsciScintilla.AcsDocument)
        # Set the autocompletion dialog to appear as soon as 1 character is typed
        editor.setAutoCompletionThreshold(1)
        

        # Set dark mode colors
        editor.setPaper(QColor('#ff2e2e2e'))  # Background color
         # Set the default style values
        # editor.setColor(QColor(0xff, 0xff, 0xff))
        # editor.setPaper(QColor(0x2e, 0x2e, 0x2e))
        editor.setColor(QColor('#ffffffff'))  # Default text color
        # editor.SendScintilla(QsciScintilla.SCI_STYLESETBACK, QsciScintilla.STYLE_DEFAULT, QColor('#2e2e2e'))
        # editor.SendScintilla(QsciScintilla.SCI_STYLESETFORE, QsciScintilla.STYLE_DEFAULT, QColor('#ffffff'))

        # Set line number margin colors
        editor.setMarginType(0, QsciScintilla.NumberMargin)  # Set margin 0 as line numbers
        editor.setMarginWidth(0, 20)  # Set margin width to fit line numbers
        # editor.setMarginsBackgroundColor(QColor('#2e2e2e'))  # Background color for line number margin
        # editor.setMarginsForegroundColor(QColor('#8c8c8c'))  # Text color for line numbers



        # Configure lexer for dark mode syntax highlighting
        # lexer.setDefaultFont(font)
        # lexer.setPaper(QColor('#2e2e2e'))
        # # lexer.setColor(QColor('#ffffffff'), QsciLexerPython.Default)
        # lexer.setColor(QColor('#00ffffff'), QsciLexerPython.UnclosedString)
        # lexer.setColor(QColor('#ffdcdcaa'), QsciLexerPython.Keyword)
        # lexer.setColor(QColor('#ff9cdcfe'), QsciLexerPython.ClassName)
        # lexer.setColor(QColor('#ff4ec9b0'), QsciLexerPython.FunctionMethodName)
        # lexer.setColor(QColor('#ff509046'), QsciLexerPython.Comment)
        # lexer.setColor(QColor('#ffd16969'), QsciLexerPython.Number)
        # lexer.setColor(QColor('#ffb5cea8'), QsciLexerPython.DoubleQuotedString)
        # lexer.setColor(QColor('#ffb5cea8'), QsciLexerPython.SingleQuotedString)
        # lexer.setColor(QColor("#ffff0000"), QsciLexerPython.TripleSingleQuotedString)

        # for i in range(len(QsciLexerPython.styles)):
        #     lexer.setPaper(QColor(0xff, 0xff, 0xff), i)

        layout.addWidget(editor) # Adding the second component in the layout - QsciScintilla
        code_area.setLayout(layout)
        self.layout().addWidget(code_area)

        return editor
                   
    def output_name_text_box(self):

        output_label_area = QWidget()
        layout = QHBoxLayout()
        
        layout.setContentsMargins(0, 0, 0, 0)

        new_output_label = QLabel("Output Layer Name")
        layout.addWidget(new_output_label) # Adding the first component in the layout - Label

        new_output_text_area = QLineEdit(self)

        new_output_text_area.setText("Processed Image")
        new_output_text_area.setObjectName("OutputNameBox") # Setting the id for the object to reference later
        layout.addWidget(new_output_text_area) # Adding the second component in the layout - Text Area

        output_label_area.setLayout(layout)
        self.layout().addWidget(output_label_area)

        return output_label_area

    def run_code(self):
        code = self.code_input.text()
        selected_shape_layer = None
        image_viewer = self.viewer
    
        # Get the selected image layer
        selected_image_layer_name = self.image_layer_dropdown.currentText()
        selected_layer = image_viewer.layers[selected_image_layer_name]
        # print(f"Data Type of Selected Layer: {data_dtype}")

        data_dtype = selected_layer.data.dtype
        roi_mask = np.ones(selected_layer.data.shape, dtype=data_dtype) # By default you would have the roi being the same size as the image
        image = deepcopy(selected_layer.data)

        # Get the selected shape layer (ROI)
        if self.roi_checkbox.isChecked():
            selected_shape_layer_name = self.shape_layer_dropdown.currentText()
            selected_shape_layer = image_viewer.layers[selected_shape_layer_name]
        
        # Extract the ROI from the selected shape layer
        if selected_shape_layer is not None:

            # Calculate the intersection between the ROI and the image, if checkbox is selected
            # We need to calcuate the intersection to prevent the ROI from selecting areas outside the image space
            if self.roi_checkbox.isChecked():
                print("Calualting the ROI Mask...")
                # Get the shape of the image
                image_shape = image.shape
                
                # print(f"Selected shape layer data type {type(selected_shape_layer.data[0])}")
                # Interesect the given shape with the image selected
                roi_polygon = self.intersect_mask_with_image(selected_shape_layer.data[0], image_shape)

                # print(len(roi_polygon))
                # print(type(roi_polygon))

                # Create the mask with the same shape as the image
                # roi_mask = selected_shape_layer.to_masks(roi_polygon)
                roi_mask = self.roi_to_mask(roi_polygon, image_shape)
                # roi_mask = np.where(roi_mask, False, True)
                roi_mask = roi_mask.astype(data_dtype) # Convert the mask to the same data type as the image
                # print("ROI Mask:", roi_mask)
                # print("No of ones in ROI Mask:", np.sum(roi_mask))
                # print("No of zeros in ROI Mask:", np.sum(roi_mask == 0))
                print("ROI Mask generated...")
                
        elif selected_shape_layer is None and self.roi_checkbox.isChecked(): #CHECK IF THIS IS THE RIGHT WAY TO HANDLE THIS
            print("ROI CHECKBOX CHECKED BUT NO ROI PROVIDED!!! Please provide an ROI using the Shapes layer ...")
            print("Using the entire image as ROI ...")


        if isinstance(selected_layer, (Image)):

            # Get the original image within the ROI space
            original_ROI_image = image * roi_mask
            ROI_copy = deepcopy(original_ROI_image)
            # print("Original ROI Image :", original_ROI_image)
            result_roi = process_image(ROI_copy, code)
            
            # print("Roi Result:", result_roi)

            if isinstance(result_roi,(list, np.ndarray)):
                 
                result_roi = result_roi * roi_mask # Apply the ROI mask to the result to retain only relevant parts
                result_roi = result_roi.astype(data_dtype) # Multiplication and addition between int32 and float64 is ambiguous
                if self.roi_checkbox.isChecked():

                    temp_image = deepcopy(image)
                    # image_viewer.add_labels(image, name=self.output_layer_text.findChild(QLineEdit, "OutputNameBox").text()+" Preview")
                    if len(temp_image.shape) == 3:
                        for stack in range(temp_image.shape[0]):
                            if self.image_has_changed(original_ROI_image[stack], result_roi[stack]):
                                # print(stack, self.image_has_changed(temp_image[stack] * roi_mask, result_roi[stack]))
                                temp_image[stack] *= (1 - roi_mask).astype(data_dtype)  # Set the ROI area to 0
                                temp_image[stack] = temp_image[stack]
                                temp_image[stack] += result_roi[stack]  # Add the modified ROI

                    elif len(temp_image.shape) == 2:
                        if self.image_has_changed(original_ROI_image, result_roi):
                            # print("Image has changed in 2D")
                            temp_image *= (1 - roi_mask).astype(data_dtype)
                            temp_image += result_roi

                    image_viewer.add_image(temp_image, name=self.output_layer_text.findChild(QLineEdit, "OutputNameBox").text(), colormap='gray')
                else:
                    
                    # Create a new image layer to display the result
                    image_viewer.add_image(result_roi, name=self.output_layer_text.findChild(QLineEdit, "OutputNameBox").text(), colormap='gray')

                print("MACRO EXECUTION COMPLETED SUCCESSFULLY ...")
            else:
                print(result_roi) # The error is returned as a string from the function
                

        if isinstance(selected_layer, (Labels)):

            # Get the original image within the ROI space
            original_ROI_image = image * roi_mask
            ROI_copy = deepcopy(original_ROI_image)

            result_roi = process_image(ROI_copy, code) 
            

            if isinstance(result_roi,(list, np.ndarray)):

                result_roi = result_roi * roi_mask 
                result_roi = result_roi.astype(data_dtype)
                if self.roi_checkbox.isChecked():

                    temp_image = deepcopy(image)
                    # image_viewer.add_labels(image, name=self.output_layer_text.findChild(QLineEdit, "OutputNameBox").text()+" Preview")
                    if len(temp_image.shape) == 3:
                        for stack in range(temp_image.shape[0]):
                            if self.image_has_changed(original_ROI_image[stack], result_roi[stack]):
                                # print(stack, self.image_has_changed(temp_image[stack] * roi_mask, result_roi[stack]))
                                temp_image[stack] *= (1 - roi_mask)  # Set the ROI area to 0
                                temp_image[stack] = temp_image[stack].astype(data_dtype)
                                temp_image[stack] += result_roi[stack]  # Add the modified ROI

                    elif len(temp_image.shape) == 2:
                        if self.image_has_changed(original_ROI_image, result_roi):
                            # print("Label has changed in 2D")
                            temp_image *= (1 - roi_mask)
                            temp_image += result_roi
                    
                    image_viewer.add_labels(temp_image, name=self.output_layer_text.findChild(QLineEdit, "OutputNameBox").text())

                else:
                    # Create a new image layer to display the result
                    image_viewer.add_labels(result_roi, name=self.output_layer_text.findChild(QLineEdit, "OutputNameBox").text())
                
                print("MACRO EXECUTION COMPLETED SUCCESSFULLY ...")
            else:
                print(result_roi)

    def intersect_mask_with_image(self, polygon:Shapes, image_shape: tuple) -> Shapes:
        '''
        Finds the intersection of the given shape with the selected image.

        Parameters
        ----------
        polygon : napari.layers.Shapes
            The shape layer polygon that is to be interected with the image
        image_shape : tuple
            The shape of the image that is to be intersected with the mask
    
        Returns
        -------
        new_polygon : napari.layers.Shapes
        The final result of the intersection between the polygon and the image
        '''       
        new_polygon = deepcopy(polygon)

        # print(f"Data Type of Polygon: f{type(polygon)}")
        # print(f"Polygon:{polygon}")
        # print(f"Image Shape: {image_shape}")
        # print(f"Data Type of Image Shape: {type(image_shape)}")

        for vertex in new_polygon:
            # print(f"Vertex: {vertex}")
            if len(vertex) == 3: # This means it is a 3D image
                
                # Z dimension (assuming this is 3D)
                if vertex[0] < 0:
                    vertex[0] = -1 #interesting peculiarity of the napari shapes layer where the first edge start from -1
                elif vertex[0] > image_shape[0]:
                    vertex[0] = image_shape[0]

                # Y dimension
                if vertex[1] < 0:
                    vertex[1] = -1
                elif vertex[1] > image_shape[1]:
                    vertex[1] = image_shape[1]

                # X dimension
                if vertex[2] < 0:
                    vertex[2] = -1
                elif vertex[2] > image_shape[2]:
                    vertex[2] = image_shape[2]

            elif len(vertex) == 2:
                # 2D image

                # Y dimension (assuming this is 2D)
                if vertex[0] < 0:
                    vertex[0] = -1
                elif vertex[0] > image_shape[0]:
                    vertex[0] = image_shape[0]

                # X dimension
                if vertex[1] < 0:
                    vertex[1] = -1
                elif vertex[1] > image_shape[1]:
                    vertex[1] = image_shape[1]
            else:
                Exception("Invalid shape passed - must be a 2D or a 3D shape")


        print("New Polygon:", new_polygon)
        return new_polygon
    
    def roi_to_mask(self, polygon, shape_image_dimensions):
        # Create a grid of coordinates corresponding to the image indices

        modified_polygon = np.array([vertex[-2:] for vertex in polygon]) # Converting 3D coordinates to 2D
        # modified_polygon = np.round(modified_polygon).astype(np.int32)
        
        ny, nx = shape_image_dimensions[-2:]
        y_indices, x_indices = np.mgrid[:ny, :nx]
        coordinates = np.vstack((y_indices.ravel(), x_indices.ravel())).T # We transpose it to get the desired shaped of (N,2)

        # Create a Path object from the polygon vertices
        path = Path(modified_polygon)

        # Include points on the edge by setting a radius
        # The radius should be small, something like 1 pixel width
        # which depends on the resolution and scale of your image.
        edge_radius = 0.001 / np.mean([nx, ny])  # Adjust as necessary

        # Use the Path object to check which coordinates are inside or on the edge of the polygon
        mask_inside = path.contains_points(coordinates)
        # mask_edge = path.contains_points(coordinates, radius=edge_radius)

        # Combine masks for inside and edge
        mask = mask_inside

        # # Use the Path object to check which coordinates are inside the polygon
        # mask = path.contains_points(coordinates)

        # Reshape the mask back into the image shape
        return mask.reshape((ny, nx))
    
    def image_has_changed(self, image, comparison_image):
        '''
        Takes two images and compares them to see if they are different.
        An image is said to be different if the sum of the difference between the two images is not 0

        Parameters
        ----------
        image : np.ndarray
            The original image
        comparison_image : np.ndarray
            The image to compare the original image with

        Returns
        -------
        bool
            True if the images are different, False otherwise
        '''
        
        return np.sum(image - comparison_image) != 0


