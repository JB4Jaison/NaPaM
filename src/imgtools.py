import matplotlib.pyplot as plt
import os
import numpy as np
from tifffile import imread, imwrite
import fnmatch

def plot_mask(img:np.ndarray):
    '''
    Plot a binary mask of the image provided.
    
    Expects the masks to be binary. If the segmentation mask array is not 0 or 1 results may be unexpected
    
     Parameters
    ----------
    img : str
        Image (segmentation mask) to display
    '''
    if len(img.shape) != 2:
        raise ValueError("Dimension mask is more than two. Expects a 2D mask.")
    
    plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    
    return

def read_tiff(filepath:str, encoding:str = 'uint8') -> np.ndarray:
    '''
    Read a tiff file and return it as per encoding.
    
    If the tiff file is empty this returns an empty numpy array. Encodings are expected to be unit or float
    
    Parameters
    ----------
    filepath : str
        String value of the tiff file to read
    encoding : str
        The type of encoding used to convert the file. Default - uint8
 
    Returns
    -------
    tiff_img : np.ndarray
        The contents of the tiff file as an numpy array
    '''
    
    if encoding not in ['uint8', 'uint16', "uint32", "float32", "float64"]:
        raise ValueError("Encoding passed is invalid")
    
    tiff_img = np.array([])
    
    tiff_img = imread(filepath)
    tiff_img = tiff_img.astype(encoding)
    
    return tiff_img

def read_h5(filepath:str, encoding:str = 'uint8') -> np.ndarray:
    '''
    Read a h5 file and return it as per encoding.
    
    If the h5 file is empty this returns an empty numpy array. Encodings are expected to be unit or float
    
    Parameters
    ----------
    filepath : str
        String value of the h5 file to read
    encoding : str
        The type of encoding used to convert the file. Default - uint8
 
    Returns
    -------
    h5_img : np.ndarray
        The contents of the h5 file as an numpy array
    '''
    
    if encoding not in ['uint8', 'uint16', "uint32", "float32", "float64"]:
        raise ValueError("Encoding passed is invalid")
            
    h5_img = np.array([])
    
    with h5py.File(r'C:\Users\jjohn\Documents\temporary\TBB3335_stitched_crop_x3639_y6811_z609_c1_t0_rb20_Probabilities.h5', 'r') as f:
        h5_img = f.get('exported_data')[:,:,:,:]
        print(h5_img[80,:,:,1])
        np.where(h5_img>0)
        plot_mask(h5_img[80,:,:,1])
#         print(h5_img.shape)
    
#     h5_img = imread(filepath)
#     h5_img = tiff_img.astype(encoding)
    
    return h5_img

def get_files(directory:str, extension:str):
    """
    Create an iterator that filters files in a directory by extension.

    The extension needs to be provided without the "." (dot) in the string. The string for the filename has to be raw.
    
    Returns
    -------
    directory: str
        The directory in which the files are to be filtered.
    extension: str
        The file extension to filter by.

    Parameters
    ----------
    filename: filepath
        An iterator over filenames with the specified extension.
    """
    pattern = f'*.{extension}'
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename

def threshold_by_limit(image, threshold):
    return np.where(image> threshold, 1, 0)