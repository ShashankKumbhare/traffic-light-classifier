
# ==================================================================================================================================
# START >> FILE INFO
# ==================================================================================================================================
# File        : traffic_light_classifier/__auxil_subpkg__/_auxil_submod.py
# Author      : Shashank Kumbhare
# Date        : 09/20/2022
# email       : shashankkumbhare8@gmail.com
# Description : This file is a python submodule for python subpackage
#               'traffic_light_classifier.__auxil_subpkg__'.
# ==================================================================================================================================
# END << FILE INFO
# ==================================================================================================================================



# ==================================================================================================================================
# START >> SUBMODULE >> traffic_light_classifier.__auxil_subpkg__._auxil_submod
# ==================================================================================================================================

# >>
"""
This submodule contains some auxiliary functions being used in rest of the modules
and submodules.
"""

_name_subpkg = __name__.partition(".")[-2]
_name_submod = __name__.partition(".")[-1]
print(f"   + Adding submodule '{_name_submod}'...")

# ==================================================================================
# START >> IMPORTS
# ==================================================================================
# >>
from ..__dependencies_subpkg__ import *
from ..__constants_subpkg__    import *
# <<
# ==================================================================================
# END << IMPORTS
# ==================================================================================


# ==================================================================================
# START >> EXPORTS
# ==================================================================================
# >>
__all__ = [ "load_dataset", "one_hot_encode", "one_hot_encode_reverse", "get_title"
          , "update_user_done", "print_title", "get_shape_params_truncnorm" ]
# <<
# ==================================================================================
# END << EXPORTS
# ==================================================================================


# ==================================================================================================================================
# START >> FUNCTION >> load_dataset
# ==================================================================================================================================
# >>
def load_dataset ( image_dir ) :
    
    """
    ================================================================================
    START >> DOC >> load_dataset
    ================================================================================
        
        GENERAL INFO
        ============
            
            This function loads in images and their labels and places them in a list.
            The list contains all images and their associated labels.
            For example, after data is loaded, im_list[0][:] will be the first
            image-label pair in the list.
        
        PARAMETERS
        ==========
            
            image_dir <str>
                
                Location of the directory of images.
        
        RETURNS
        =======
            
            im_list <list>
                
                A list of images.
    
    ================================================================================
    END << DOC << load_dataset
    ================================================================================
    """
    
    # Populate this empty image list
    im_list     = []
    image_types = ["red", "yellow", "green"]
    
    # Iterate through each color folder
    for im_type in image_types:
        
        # Iterate through each image file in each image_type folder
        # glob reads in any image with the extension "image_dir/im_type/*"
        for file in glob.glob(os.path.join(image_dir, im_type, "*")):
            
            # Read in the image
            im = mpimg.imread(file)
            
            # Check if the image exists/if it's been correctly read-in
            if not im is None:
                # Append the image, and it's type (red, green, yellow) to the image list
                im_list.append((im, im_type))
    
    return im_list
# <<
# ==================================================================================================================================
# END << FUNCTION << load_dataset
# ==================================================================================================================================



# ==================================================================================================================================
# START >> FUNCTION >> one_hot_encode
# ==================================================================================================================================
# >>
def one_hot_encode ( label ) :
    
    """
    ================================================================================
    START >> DOC >> one_hot_encode
    ================================================================================
        
        GENERAL INFO
        ============
            
            One hot encode an image label.
        
        PARAMETERS
        ==========
            
            label <str>
                
                Image label.
                Possible arg: "red", "green", "yellow", "r", "g", "y", "R", "G", "Y"
        
        RETURNS
        =======
            
            one_hot_encoded <list>
                
                A list of length 3 with element either 0 or 1.
                Examples:
                one_hot_encode("r") returns: [1, 0, 0]
                one_hot_encode("y") returns: [0, 1, 0]
                one_hot_encode("g") returns: [0, 0, 1]
    
    ================================================================================
    END << DOC << one_hot_encode
    ================================================================================
    """
    
    red    = ["r", "R", "red", "Red", "RED"]
    yellow = ["y", "Y", "yellow", "Yellow", "YELLOW"]
    green  = ["g", "G", "green", "Green", "GREEN"]
    
    if label in red:
        one_hot_encoded = [1, 0, 0]
    elif label in yellow:
        one_hot_encoded = [0, 1, 0]
    elif label in green:
        one_hot_encoded = [0, 0, 1]
    else:
        print("Please input proper label.")

    return one_hot_encoded
# <<
# ==================================================================================================================================
# END << FUNCTION << one_hot_encode
# ==================================================================================================================================



# ==================================================================================================================================
# START >> FUNCTION >> one_hot_encode_reverse
# ==================================================================================================================================
# >>
def one_hot_encode_reverse ( encode ) :
    
    """
    ================================================================================
    START >> DOC >> one_hot_encode_reverse
    ================================================================================
        
        GENERAL INFO
        ============
            
            Reverses one hot encode of an image label giving an image label.
        
        PARAMETERS
        ==========
            
            encode <list>
                
                One hot encode output.
                Possible arg: [1,0,0], [0,1,0], [0,0,1]
        
        RETURNS
        =======
            
            label <str>
                
                Examples:
                one_hot_encode_reverse( [1,0,0] ) returns: "R"
                one_hot_encode_reverse( [0,1,0] ) returns: "Y"
    
    ================================================================================
    END << DOC << one_hot_encode_reverse
    ================================================================================
    """
    
    if encode == [1,0,0]:
        label = "Red"
    elif encode == [0,1,0]:
        label = "Yellow"
    elif encode == [0,0,1]:
        label = "Green"
    else:
        print("Please input proper encode.")
    
    return label
# <<
# ==================================================================================================================================
# END << FUNCTION << one_hot_encode_reverse
# ==================================================================================================================================



# ==================================================================================================================================
# START >> FUNCTION >> get_title
# ==================================================================================================================================
# >>
def get_title( image ) :
    
    """
    ================================================================================
    START >> DOC >> get_title
    ================================================================================
        
        GENERAL INFO
        ============
            
            Generate a title for the input image.
        
        PARAMETERS
        ==========
            
            image <tuple>
                
                A tuple like (rgb, label)   or
                A tuple like (rgb, label_pred, label_true)
        
        RETURNS
        =======
            
            title <str>
                
                A title according to the type of image input.
    
    ================================================================================
    END << DOC << get_title
    ================================================================================
    """
    
    if len(image) == 2:
        if type(image[1]) == list:
            label  = one_hot_encode_reverse(image[1])
        else:
            label  = image[1]
        title      = f"{label.capitalize()}"
    elif len(image) == 3:
        label_pred = image[1]
        label_true = image[2]
        title      = f"True: {label_true}, Pred: {label_pred}"
    else:
        title      = f""
    
    return title
# <<
# ==================================================================================================================================
# END << FUNCTION << get_title
# ==================================================================================================================================



# ==================================================================================================================================
# START >> FUNCTION >> update_user_done
# ==================================================================================================================================
# >>
def update_user_done():
    
    """
    ================================================================================
    START >> DOC >> update_user_done
    ================================================================================
        
        GENERAL INFO
        ============
            
            This function simply prints a message "Done!".
        
        PARAMETERS
        ==========
            
            None
        
        RETURNS
        =======
            
            None
    
    ================================================================================
    END << DOC << update_user_done
    ================================================================================
    """
    
    print(f"Done!")
    print("")
    
    return None
# <<
# ==================================================================================================================================
# END << FUNCTION << update_user_done
# ==================================================================================================================================



# ==================================================================================================================================
# START >> FUNCTION >> print_title
# ==================================================================================================================================
# >>
def print_title(title, sub_title = False):
    
    """
    ================================================================================
    START >> DOC >> print_title
    ================================================================================
        
        GENERAL INFO
        ============
            
            Prints the input title in a decorated form.
        
        PARAMETERS
        ==========
            
            title <str>
                
                Title to be printed
        
        RETURNS
        =======
            
            None
    
    ================================================================================
    END << DOC << print_title
    ================================================================================
    """
    
    lines = wrap(title, 130)
    lines = np.array(lines)
    lens_lines   = [ len(line) for line in lines ]
    len_max_line = max(lens_lines)
    
    if not sub_title: print("="*len_max_line)
    for line in lines:
        print(line)
    print("="*len_max_line)
    
    return None
# <<
# ==================================================================================================================================
# END << FUNCTION << print_title
# ==================================================================================================================================



# ==================================================================================================================================
# START >> FUNCTION >> get_shape_params_truncnorm
# ==================================================================================================================================
# >>
def get_shape_params_truncnorm  ( xa
                                , xb
                                , mu
                                ) :
    
    """
    ================================================================================
    START >> DOC >> get_shape_params_truncnorm
    ================================================================================
        
        GENERAL INFO
        ============
            
            Calculated shape parameters a and b of truncated gaussin distribution.
        
        PARAMETERS
        ==========
            
            xa <float>
                
                Lower bound of truncated gaussin distribution.
            
            xb <float>
                
                Upper bound of truncated gaussin distribution.
            
            mu <float>
                
                Mean of the parent gaussin distribution.
            
            sigma <float>
                
                Standard deviation of the parent gaussin distribution.
        
        RETURNS
        =======
            
            (a, b) <tuble>
                
                Tuple of shape parameters a & b.
    
    ================================================================================
    END << DOC << get_shape_params_truncnorm
    ================================================================================
    """
    
    a = xa - mu
    b = xb - mu
    
    return (a, b)
# <<
# ==================================================================================================================================
# END << FUNCTION << get_shape_params_truncnorm
# ==================================================================================================================================



# ==================================================================================================================================
# START >> FUNCTION >> _template_submod_func
# ==================================================================================================================================
# >>
def template_submod_func    ( p_p_p_p_1 = ""
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
# END << SUBMODULE << traffic_light_classifier.__auxil_subpkg__._auxil_submod
# ==================================================================================================================================