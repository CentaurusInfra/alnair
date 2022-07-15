AlnairJob Library
=================
All datasets that represent a map from keys to data samples should subclass
the AlnairJobDataset class. All subclasses should overwrite: 

* ``__convert__```: supporting pre-processing loaded data. Data are saved as key-value map before calling this method. You are responsible for reshaping the dict to desired array.
* ``__getitem__``: supporting fetching a data sample for a given index.
* ``__len__``: returning the size of the dataset

Installing
============

.. code-block:: bash

    pip3 install pyalnair

Usage
=====

.. code-block:: bash

    >>> from src.example import custom_sklearn
    >>> custom_sklearn.get_sklearn_version()
    '0.24.2'
