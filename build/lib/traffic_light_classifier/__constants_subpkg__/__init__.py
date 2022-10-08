
# ==================================================================================================================================
# START >> FILE INFO
# ==================================================================================================================================
# File        : traffic_light_classifier/__constants_subpkg__/__init__.py
# Author      : Shashank Kumbhare
# Date        : 09/20/2022
# email       : shashankkumbhare8@gmail.com
# Description : This file is a __init__ file for python subpackage
#               'traffic_light_classifier.__constants_subpkg__'.
# ==================================================================================================================================
# END << FILE INFO
# ==================================================================================================================================



# ==================================================================================================================================
# START >> SUBPACKAGE >> traffic_light_classifier.__constants_subpkg__
# ==================================================================================================================================
# >>
"""
This subpackage is created to store constants required for the package.
These constants will be shared across all the package modules & submodules.
"""

_name_subpkg = __name__.partition(".")[-1]
print("")
print(f" + Adding subpackage '{_name_subpkg}'...")

# ==================================================================================
# START >> IMPORTS
# ==================================================================================
# >>
# SUBMODULES >>
from ._constants_submod import *
# <<
# ==================================================================================
# END << IMPORTS
# ==================================================================================

print(" - Done!")

# <<
# ==================================================================================================================================
# END << SUBPACKAGE << traffic_light_classifier.__constants_subpkg__
# ==================================================================================================================================
