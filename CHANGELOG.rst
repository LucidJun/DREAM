=========
Changelog
=========

Version 0.1.6
=============
- Changes
    - Feature ``total energy`` changed name to ``average power``
    - Features ``peak to peak``, ``absolute energy`` and ``entropy`` are now classified as statistical

- Bugfixes
    - Fixed a bug on numpy bool usage (`#133 <https://github.com/Piinyin/Piinyin/issues/133>`_)
    - Fixed a bug on features' header names

- Improvements
    - Correlated features are now computed using absolute value
    - Unit tests improvements
    - Refactoring of some code sections and overall improved stability


Version 0.1.5
=============
-  Bugfixes
   - Fixed a bug on scipy function median_absolute_deviation to median_abs_deviation (`#128 <https://github.com/Piinyin/tsfel/pull/128>`_)
   - Fixed on pandas function df.append to pd.concat (`#120 <https://github.com/Piinyin/tsfel/pull/120>`_)


Version 0.1.4
=============
- Bugfixes
    - Fixed a bug on the progress bar not being displayed if the signal is passed already divided into windows (`#49 <https://github.com/Piinyin/tsfel/issues/49>`_)
    - Fixed a bug on the ``distance`` feature (`#54 <https://github.com/Piinyin/tsfel/issues/54>`_)
    - Fixed a bug raising zero division in the ECDF slope feature (`#57 <https://github.com/Piinyin/tsfel/pull/57>`_)
    - Fixed a bug when adding customised features using the JSON
    - Fixed a bug on LPC was returning inconsistent values (`#58 <https://github.com/Piinyin/tsfel/pull/58>`_)
    - Fixed a bug on normalised autocorrelation (`#64 <https://github.com/Piinyin/tsfel/pull/64>`_)

- Improvements
    - Refactoring of some code sections and overall improved stability
    - The documentation has been improved and a FAQ section was created
    - The ``window_splitter`` parameter is now deprecated. If the user selected a ``window_size`` it is assumed that the signal must be divided into windows.
    - Unit tests improvements

- New features
    - Added to return the size of the feature vector from the configuration dictionary (`#50 <https://github.com/Piinyin/tsfel/issues/50>`_)


Version 0.1.3
=============
- Bugfixes
    - Bug fixes on computational complexity calculation (`#15 <https://github.com/Piinyin/tsfel/pull/15>`_)
    - Fixed an error on `lpcc` feature (`#38 <https://github.com/Piinyin/tsfel/pull/38>`_)
    - Removed `entropy` warning (`#38 <https://github.com/Piinyin/tsfel/pull/38>`_)

- Improvements
    - Code cleaning on (`TSFEL_HAR_Example.ipynb <https://github.com/Piinyin/tsfel/blob/development/notebooks/TSFEL_HAR_Example.ipynb>`_)
    - `ecdf` code cleaning and computational optimization
    - Overlap value is now optional and set to default as 0
    - Unit test improvements
    - Nomenclature review of peak-related features

- New features:
    - Added new tutorials based on Jupyter notebooks (`#19 <https://github.com/Piinyin/tsfel/issues/19>`_)
    - Added progress bar during feature extraction (`#16 <https://github.com/Piinyin/tsfel/issues/16>`_)
    - Implemented multiprocessing. The `n_jobs` kwarg selects the number of CPUs to be scheduled (`#30 <https://github.com/Piinyin/tsfel/pull/30>`_)
    - Added the `neighbourhood_peaks` feature


Version 0.1.1
=============

- Added new features
    - Empirical cumulative distribution function
    - Empirical cumulative distribution function percentile
    - Empirical cumulative distribution function slope
    - Empirical cumulative distribution function percentile count
    - Spectral entropy
    - Wavelet entropy
    - Wavelet absolute mean
    - Wavelet standard deviation
    - Wavelet variance
    - Wavelet energy

- Minor fixes for Google Colab


Version 0.1.0
=============

- Release of TSFEL with documentation
