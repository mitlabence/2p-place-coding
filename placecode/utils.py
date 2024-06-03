from tkinter import Tk  # use tkinter to open files
from tkinter.filedialog import askopenfilename, askdirectory
import os.path
import scipy
import numpy as np


def raise_above_all(window):
    window.attributes('-topmost', 1)
    window.attributes('-topmost', 0)


def open_file(title: str = "Select file") -> str:
    """Opens a tkinter dialog to select a file. Returns the path of the file.

    Parameters
    ----------
    title : str, optional
        The message to display in the open directory dialog, by default "Select file".
    :return: the absolute path of the directory selected.

    Returns
    -------
    str
        The absolute path of the file selected, or "." if Cancel was pressed.
    """
    root = Tk()
    root.attributes("-topmost", True)
    root.withdraw()  # keep root window from appearing
    return os.path.normpath(askopenfilename(title=title))


def open_dir(title: str = "Select data directory", ending_slash: bool = False) -> str:
    """Opens a tkinter dialog to select a folder. Returns the path of the folder.

    Parameters
    ----------
    title : str, optional
        The message to display in the open directory dialog, by default "Select data directory"
    ending_slash : bool, optional
        _description_, by default False

    Returns
    -------
    str
        _description_
    """
    """

    :param title: 
    :return: the absolute path of the directory selected.
    """
    root = Tk()
    root.attributes("-topmost", True)
    root.withdraw()  # keep root window from appearing
    folder_path = askdirectory(title=title)
    if ending_slash:
        folder_path += "/"
    return os.path.normpath(folder_path)


def read_spatial(A_data, A_indices, A_indptr, A_shape, n_components, resolution, unflatten: bool = False) -> np.array:
    """Given the numpy arrays data, indices, indptr, shape, read the sparse encoded spatial component data and
    reshape it into (n_components, resolution_x, resolution_y)

    Parameters
    ----------
    A_data : np.array
        The data field of the sparse encoding
    A_indices : np.array
        The indices field of the sparse encoding
    A_indptr : np.array
        The indptr field of the sparse encoding
    A_shape : np.array
        The shape field of the sparse encoding
    n_components : int
        the number of components in the CaImAn data
    resolution : tuple(int, int), or [int, int], or np.array(shape=(2,), dtype=dtype("int32"))
        the resolution of the 2p recording. It should be read out from CaImAn dims.
    unflatten : bool
        default: False. If True, the individual spatial components will be converted into 2d arrays. If False,
        left as 1d/flat numpy arrays.
    Returns
    -------
    np.array of shape (n_components, resolution_x * resolution_y ) if unflatten=False, else (n_components, *resolution)
        The dense matrix form of the spatial components.
    """
    spatial = scipy.sparse.csc.csc_matrix(
        (A_data, A_indices, A_indptr), shape=A_shape).todense()  # returns array with dimensions (flat resolution, n_components)
    spatial = np.array(spatial)  # change type to numpy array
    # (262144 -> 512x512, i.e. "unflatten" along imaging resolution)
    spatial = np.swapaxes(spatial, 0, 1)
    if unflatten:
        # TODO: need to test if x and y are in correct order (for asymmetric resolution).
        spatial = np.reshape(spatial, (n_components, *resolution))
    return spatial
