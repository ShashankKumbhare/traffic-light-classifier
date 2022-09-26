
# ==================================================================================================================================
# START >> FILE INFO
# ==================================================================================================================================
# File        : traffic_light_classifier/template_mod.py
# Author      : Shashank Kumbhare
# Date        : 09/20/2022
# email       : shashankkumbhare8@gmail.com
# Description : This file is a python module for python package 'traffic_light_classifier'.
# ==================================================================================================================================
# END << FILE INFO
# ==================================================================================================================================



# ==================================================================================================================================
# START >> MODULE >> traffic_light_classifier.template_mod
# ==================================================================================================================================
# >>
"""
This module is created/used for/to.
MODULE description MODULE description MODULE description MODULE description
MODULE description MODULE description MODULE description MODULE description
MODULE description MODULE description.
"""

_name_mod_ = __name__.partition(".")[-1]
print("")
print(f" + Adding module '{_name_mod_}'...")

# ==================================================================================
# START >> IMPORTS
# ==================================================================================
# >>
from .__auxil_subpkg__ import *
from .template_subpkg import *
# <<
# ==================================================================================
# END >> IMPORTS
# ==================================================================================


# ==================================================================================================================================
# START >> FUNCTION >> template_mod_func
# ==================================================================================================================================
# >>
def template_mod_func   ( p_p_p_p_1 = ""
                        , p_p_p_p_2 = ""
                        ) :
    
    """
    ================================================================================
    START >> DOC >> template_mod_func
    ================================================================================
        
        GENERAL INFO
        ============
            
            t_t_t_t t_t_t t_t_t_t_t t_t t_t_t_t t_t_t t_t_t_t t_t t_t_t_t t_t_t t_t
            t_t_t_t t_t_t t_t_t_t_t t_t t_t_t_t t_t_t t_t_t_t t_t t_t_t_t t_t_t t_t
            t_t_t_t t_t_t t_t_t_t_t t_t t_t_t_t t_t_t t_t_t_t t_t t_t_t_t t_t_t t_t
        
        PARAMETERS
        ==========
            
            p_p_p_p_1 <type>
                
                t_t_t_t t_t_t t_t_t_t_t t_t t_t_t_t t_t_t t_t_t_t_t t_t t_t_t_t t_t
                t_t_t_t t_t_t t_t_t_t_t t_t t_t_t_t t_t_t t_t_t_t_t t_t t_t_t_t t_t
            
            p_p_p_p_2 <type>
                
                t_t_t_t t_t_t t_t_t_t_t t_t t_t_t_t t_t_t t_t_t_t_t t_t t_t_t_t t_t
                t_t_t_t t_t_t t_t_t_t_t t_t t_t_t_t t_t_t t_t_t_t_t t_t t_t_t_t t_t
        
        RETURNS
        =======
            
            r_r_r_r <type>
                
                t_t_t_t t_t_t t_t_t_t_t t_t t_t_t_t t_t_t t_t_t_t_t t_t t_t_t_t t_t
    
    ================================================================================
    END << DOC << template_mod_func
    ================================================================================
    """
    
    _name_func_ = inspect.stack()[0][3]
    print(f"This is a print from '{_name_mod_}.{_name_func_}'{p_p_p_p_1}{p_p_p_p_2}.")
    print("The following line will print from template_subpkg.template_submod")
    template_submod_func()
    
    return None
# <<
# ==================================================================================================================================
# END << FUNCTION << template_mod_func
# ==================================================================================================================================

print(" - Done!")

# <<
# ==================================================================================================================================
# END << MODULE << traffic_light_classifier.template_mod
# ==================================================================================================================================
