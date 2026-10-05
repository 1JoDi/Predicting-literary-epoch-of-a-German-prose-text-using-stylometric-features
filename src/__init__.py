"""Stylometric epoch classification of German prose (Advanced Python for NLP).

The package is split into one module per pipeline step:

* ``config``      -- read the YAML configuration file
* ``catalog``     -- download and filter the Project Gutenberg catalog
* ``epochs``      -- map author death years to literary epochs
* ``download``    -- download the selected texts
* ``cleaning``    -- strip Gutenberg boilerplate, extract a text sample
* ``features``    -- the four stylometric features
* ``dataset``     -- run spaCy over the corpus and build the feature table
* ``centroid``    -- a nearest-centroid classifier written with NumPy
* ``classify``    -- train and evaluate the classifiers
* ``plots``       -- all figures used in the report
"""
