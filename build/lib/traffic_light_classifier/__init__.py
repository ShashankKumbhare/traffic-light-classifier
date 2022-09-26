
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

_name_pkg_ = __name__.partition(".")[0]
print("")
print(f"==========================================================================")
print(f"Importing package '{_name_pkg_}'...")
print(f"==========================================================================")

# ==================================================================================
# START >> IMPORTS
# ==================================================================================
# >>
# SUBPACKAGES >>
# from .__dependencies_subpkg__ import *
# from .__constants_subpkg__ import *
# from .__auxil_subpkg__ import *
# from .__tests_subpkg__ import *
from .plots_subpkg import *
from .modify_images_subpkg import *
from .extract_feature_subpkg import *
# from .template_subpkg import *
# MODULES >>
# from .template_mod import template_mod_func
# <<
# ==================================================================================
# END >> IMPORTS
# ==================================================================================

print("")
print(f"==========================================================================")
print(f"Package '{_name_pkg_}' imported sucessfully !!")
print(f"==========================================================================")
print("")

# <<
# ==================================================================================================================================
# END << PACKAGE << traffic_light_classifier
# ==================================================================================================================================
