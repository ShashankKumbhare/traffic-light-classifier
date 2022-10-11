
# ==================================================================================================================================
# START >> FILE INFO
# ==================================================================================================================================
# File        : traffic_light_classifier/__init__.py
# Author      : Shashank Kumbhare
# Date        : 09/20/2022
# email       : shashankkumbhare8@gmail.com
# Description : This file is a __init__ file for python package 'traffic_light_classifier'.
# ==================================================================================================================================
# END << FILE INFO
# ==================================================================================================================================



# ==================================================================================================================================
# START >> PACKAGE >> traffic_light_classifier
# ==================================================================================================================================
# >>
"""
- Traffic Light Classifier to classify traffic light signals as either red, yellow
  or green.
- This package is a part of a computer vision project 'Traffic Light Classification'.
- A robust probabilistic approach based classifier has been implemented from scratch
  to classify traffic light signal's status using computer vision and machine learning
  techniques.
- Several data cleaning steps, features extraction and a probabilistic metric has
  been implemented.
- All training stages and prediction stages can be throughly visualized & analyzed.
- This package contains a classifier, plotting & feature extraction functionalities,
  and datasets for the project.
- Libraries used: OpenCV-Python, scipy, matplotlib, numpy.
"""

__version__  = '1.0.0'
_name_pkg    = __name__.partition(".")[0]
print("")
print(f"==========================================================================")
print(f"Importing package '{_name_pkg}'...")
print(f"==========================================================================")

# ==================================================================================
# START >> IMPORTS
# ==================================================================================
# >>
# MODULES >>
from .                        import helpers
from .                        import tests
from .                        import plots
from .                        import modify_images
from .                        import extract_feature
from .                        import statistics
# ELEMENTS >>
from .datasets                import datasets
from .model                   import *
# <<
# ==================================================================================
# END << IMPORTS
# ==================================================================================

print(f"==========================================================================")
print(f"Package '{_name_pkg}' imported sucessfully !!")
print(f"==========================================================================")
print(f"version {__version__}")
print("")

# <<
# ==================================================================================================================================
# END << PACKAGE << traffic_light_classifier
# ==================================================================================================================================
