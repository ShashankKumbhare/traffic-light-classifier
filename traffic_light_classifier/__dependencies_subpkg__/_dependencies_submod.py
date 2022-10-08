
# ==================================================================================================================================
# START >> FILE INFO
# ==================================================================================================================================
# File        : traffic_light_classifier/__dependencies_subpkg__/_dependencies_submod.py
# Author      : Shashank Kumbhare
# Date        : 09/20/2022
# email       : shashankkumbhare8@gmail.com
# Description : This file is a python submodule for python subpackage
#               'traffic_light_classifier.__dependencies_subpkg__'.
# ==================================================================================================================================
# END << FILE INFO
# ==================================================================================================================================



# ==================================================================================================================================
# START >> SUBMODULE >> traffic_light_classifier.__dependencies_subpkg__._dependencies_submod
# ==================================================================================================================================
# >>
"""
This submodule imports all the required 3rd party dependency packages/libraries for
the package which are then shared across all the package modules & submodules. All
the 3rd party dependency packages are imported here at one place and any other
dependencies are not to be imported in any module or submodule other than this
submodule.
"""

_name_submod = __name__.partition(".")[-1]
print(f"   + Adding submodule '{_name_submod}'...")

# ==================================================================================
# START >> IMPORTS
# ==================================================================================
# >>
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib.ticker import MultipleLocator
style.use('classic')
style.use('seaborn-white')
import math
import os
import sys
import inspect
import glob                                 # library for loading images from a directory
import matplotlib.image as mpimg
import cv2
import unittest
from scipy.stats import norm, lognorm, truncnorm
from scipy.stats import poisson
from scipy.special import logsumexp
from scipy import optimize
from IPython.display import Markdown, display
from textwrap import wrap
import random
# <<
# ==================================================================================
# END << IMPORTS
# ==================================================================================

print("   - Done!")

# <<
# ==================================================================================================================================
# END << SUBMODULE << traffic_light_classifier.__dependencies_subpkg__._dependencies_submod
# ==================================================================================================================================
