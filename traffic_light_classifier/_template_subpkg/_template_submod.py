
# ==================================================================================================================================
# START >> FILE INFO
# ==================================================================================================================================
# File        : traffic_light_classifier/_template_subpkg/_template_submod.py
# Author      : Shashank Kumbhare
# Date        : --/--/----
# email       : shashankkumbhare8@gmail.com
# Description : This file is a python submodule for python subpackage
#               'traffic_light_classifier._template_subpkg'.
# ==================================================================================================================================
# END << FILE INFO
# ==================================================================================================================================



# ==================================================================================================================================
# START >> SUBMODULE >> traffic_light_classifier._template_subpkg._template_submod
# ==================================================================================================================================
# >>
"""
This submodule is created/used for/to
SUBMODULE description SUBMODULE description SUBMODULE
SUBMODULE description SUBMODULE description SUBMODULE description SUBMODULE
SUBMODULE description SUBMODULE description.
"""

_name_subpkg = __name__.partition(".")[-2]
_name_submod = __name__.partition(".")[-1]
print(f"   + Adding submodule '{_name_submod}'...")

# ==================================================================================
# START >> IMPORTS
# ==================================================================================
# >>
from ..__auxil_subpkg__ import *
# <<
# ==================================================================================
# END << IMPORTS
# ==================================================================================


# ==================================================================================================================================
# START >> FUNCTION >> _template_submod_func
# ==================================================================================================================================
# >>
def _template_submod_func   ( p_p_p_p_1 = ""
                            , p_p_p_p_2 = ""
                            ) :
    
    """
    ================================================================================
    START >> DOC >> _template_submod_func
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
    END << DOC << _template_submod_func
    ================================================================================
    """
    
    _name_func = inspect.stack()[0][3]
    print(f"This is a print from '{_name_subpkg}.{_name_submod}.{_name_func}'{p_p_p_p_1}{p_p_p_p_2}.")
    
    return None
# <<
# ==================================================================================================================================
# END << FUNCTION << _template_submod_func
# ==================================================================================================================================

print("   - Done!")

# <<
# ==================================================================================================================================
# END << SUBMODULE << traffic_light_classifier._template_subpkg._template_submod
# ==================================================================================================================================
