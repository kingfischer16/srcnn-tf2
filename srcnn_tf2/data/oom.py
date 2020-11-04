"""
OOM.PY
======

Tools for handling large sets of training data that
cannot be contained in memory, i.e. 'out-of-memory'.
"""

# Imports
import numpy as np
from tensorflow.keras.utils import Sequence


class SRCNNTrainingGenerator(Sequence):
    """
    A generator for reading a list of files (x- and y-data images),
    and feeding them to a keras.model.fit_generator function. This
    is to be used when the full set of training data is too large.
    """
    def __init__(self, filenames, batch_size):
        """
        Args:
            filenames (list): A list of tuples, (x_filename, y_filename),
             where the filenames are the full path filenames as strings.
            
            batch_size (int): Training batch size.
        """
        self.filenames = filenames
        self.batch_size = batch_size

    def __len__(self) :
        return (np.ceil(len(self.filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx) :
        batch_x = [f[0] for f in self.filenames[idx * self.batch_size : (idx+1) * self.batch_size]]
        batch_y = [f[1] for f in self.filenames[idx * self.batch_size : (idx+1) * self.batch_size]]
        return np.array([np.load(bx) for bx in batch_x]), np.array([np.load(by) for by in batch_y])
