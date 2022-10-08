
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
This package is a part of a computer vision project 'Traffic Light Classification'.
The project was the final project of the online nanodegree program 'Intro to Self
Driving Cars' offered by 'udacity.com'.
This package has utilised the knowledge of computer vision and machine learning
techniques to classify the traffic signal light images as either red, green, or
yellow.
"""

_name_pkg = __name__.partition(".")[0]
print("")
print(f"==========================================================================")
print(f"Importing package '{_name_pkg}'...")
print(f"==========================================================================")

# ==================================================================================
# START >> IMPORTS
# ==================================================================================
# >>
# SUBPACKAGES >>
# MODULES >>
from .                        import helpers
from .                        import plots
from .                        import modify_images
from .                        import extract_feature
from .                        import statistics
from .Model                   import Model
# ELEMENTS >>
from .__data_subpkg__         import datasets
from .__tests_subpkg__        import Tests
# <<
# ==================================================================================
# END << IMPORTS
# ==================================================================================

print("")
print(f"==========================================================================")
print(f"Package '{_name_pkg}' imported sucessfully !!")
print(f"==========================================================================")
print("")

# <<
# ==================================================================================================================================
# END << PACKAGE << traffic_light_classifier
# ==================================================================================================================================
