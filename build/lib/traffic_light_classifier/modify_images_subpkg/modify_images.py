
# ==================================================================================================================================
# START >> FILE INFO
# ==================================================================================================================================
# File        : traffic_light_classifier/modify_images_subpkg/modify_images.py
# Author      : Shashank Kumbhare
# Date        : 09/23/2022
# email       : shashankkumbhare8@gmail.com
# Description : This file is a python submodule for python subpackage
#               'traffic_light_classifier.modify_images_subpkg'.
# ==================================================================================================================================
# END << FILE INFO
# ==================================================================================================================================



# ==================================================================================================================================
# START >> SUBMODULE >> traffic_light_classifier.modify_images_subpkg.modify_images
# ==================================================================================================================================
# >>
"""
This submodule contains functionalities to manupulate or modify traffic light
training & test images.
"""

_name_subpkg_ = __name__.partition(".")[-2]
_name_submod_ = __name__.partition(".")[-1]
print(f"   + Adding submodule '{_name_submod_}'...")

# ==================================================================================
# START >> IMPORTS
# ==================================================================================
from ..__auxil_subpkg__ import *
from ..plots_subpkg import *
# ==================================================================================
# END >> IMPORTS
# ==================================================================================


# ==================================================================================================================================
# START >> FUNCTION >> standardize_image
# ==================================================================================================================================
# >>
def standardize_image( image
                     , size = 32
                     ) :
    
    """
    ================================================================================
    START >> DOC >> standardize_image
    ================================================================================
        
        GENERAL INFO
        ============
            
            Standardizes an input RGB image with a desired size.
            It returns a new image of dimension (size, size, 3).
        
        PARAMETERS
        ==========
            
            image <np.array>
                
                Numpy array of rgb image of shape (n_row, n_col, 3).
            
            size <int>
                
                The size of desired standardized image.
        
        RETURNS
        =======
            
            image_std <type>
                
                A standardized version of image of dimension (size, size, 3).
    
    ================================================================================
    END << DOC << standardize_image
    ================================================================================
    """
    
    image_rgb = np.copy(image)
    image_std = cv2.resize(image_rgb, (size, size))
    
    return image_std
# <<
# ==================================================================================================================================
# END << FUNCTION << standardize_image
# ==================================================================================================================================



# ==================================================================================================================================
# START >> FUNCTION >> standardize_images
# ==================================================================================================================================
# >>
def standardize_images( images
                      , size = 32
                      ) :
    
    """
    ================================================================================
    START >> DOC >> standardize_images
    ================================================================================
        
        GENERAL INFO
        ============
            
            Standardizes a list of input RGB images with a desired size.
            It returns a new a list of images of dimension (size, size, 3).
        
        PARAMETERS
        ==========
            
            images <list>
                
                A list of numpy array of rgb image of shape (n_row, n_col, 3).
                
            size <int>
                
                The size of desired standardized image.
        
        RETURNS
        =======
            
            images_std <type>
                
                A list of standardized version of images of dimension (size, size, 3).
    
    ================================================================================
    END << DOC << standardize_images
    ================================================================================
    """
    
    ims_std        = [ standardize_image(image[0], size = size)        for image in images ]
    one_hot_labels = [ helpers.one_hot_encode(image[1])    for image in images ]
    images_std     = [ (im_std, one_hot_label) for im_std, one_hot_label in zip(ims_std, one_hot_labels) ]
    
    return images_std
# <<
# ==================================================================================================================================
# END << FUNCTION << standardize_images
# ==================================================================================================================================



# ==================================================================================================================================
# START >> FUNCTION >> convert_rgb_to_hsv
# ==================================================================================================================================
# >>
def convert_rgb_to_hsv( image_rgb
                      , plot_enabled = False
                      , cmap         = None
                      ) :
    
    """
    ================================================================================
    START >> DOC >> convert_rgb_to_hsv
    ================================================================================
        
        GENERAL INFO
        ============
            
            Converts rgb image to hsv image.
        
        PARAMETERS
        ==========
            
            image_rgb <np.array>
                
                Numpy array of rgb image of shape (n_row, n_col, 3).
            
            plot_enabled <bool>
                
                When enabled plots the image.
        
        RETURNS
        =======
            
            image_hsv <np.array>
                
                Numpy array of hsv image of shape (n_row, n_col, 3).
    
    ================================================================================
    END << DOC << convert_rgb_to_hsv
    ================================================================================
    """
    
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    
    if plot_enabled:
        plots.plot_channels(image_hsv, cmap = cmap, titles = ["HSV Image", "H channel", "S channel", "V channel"])
    
    return image_hsv
# <<
# ==================================================================================================================================
# END << FUNCTION << convert_rgb_to_hsv
# ==================================================================================================================================



# ==================================================================================================================================
# START >> FUNCTION >> convert_hsv_to_rgb
# ==================================================================================================================================
# >>
def convert_hsv_to_rgb( image_hsv
                      , plot_enabled = False
                      , cmap = None
                      ) :
    
    """
    ================================================================================
    START >> DOC >> convert_hsv_to_rgb
    ================================================================================
        
        GENERAL INFO
        ============
            
            Converts hsv image to rgb image.
        
        PARAMETERS
        ==========
            
            image_hsv <np.array>
                
                Numpy array of hsv image of shape (n_row, n_col, 3).
        
        RETURNS
        =======
            
            image_hsv <np.array>
                
                Numpy array of rgb image of shape (n_row, n_col, 3).
    
    ================================================================================
    END << DOC << convert_hsv_to_rgb
    ================================================================================
    """
    
    image_rgb = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
    
    if plot_enabled:
        plots.plot_channels(image_rgb, cmap = cmap, titles = ["RGB Image", "R channel", "G channel", "B channel"])
    
    return image_rgb
# <<
# ==================================================================================================================================
# END << FUNCTION << convert_hsv_to_rgb
# ==================================================================================================================================



# ==================================================================================================================================
# START >> FUNCTION >> mask_image
# ==================================================================================================================================
# >>
def mask_image( image
              , range_mask_x
              , range_mask_y
              , plot_enabled = False
              ) :
    
    """
    ================================================================================
    START >> DOC >> mask_image
    ================================================================================
        
        GENERAL INFO
        ============
            
            Masks the input image for the given range.
        
        PARAMETERS
        ==========
            
            image <np.array>
                
                Numpy array of image of shape (n_row, n_col, 3).
            
            range_crop_x <tuple>
                
                Crop range along x-axis.
            
            range_crop_y <tuple>
                
                Crop range along y-axis.
            
            plot_enabled <bool>
                
                When enabled plots the image.
        
        RETURNS
        =======
            
            image_cropped <np.array>
                
                Numpy array of image of shape (n_row, n_col, 3).
    
    ================================================================================
    END << DOC << mask_image
    ================================================================================
    """
    
    x_left   = range_mask_x[0]
    x_right  = range_mask_x[1]
    y_top    = range_mask_y[0]
    y_bottom = range_mask_y[1]
    
    image_masked = np.copy(image)
    
    width  = len(image[0])
    height = len(image)
    
    image_masked[         :      ,       0:x_left ] = 0
    image_masked[         :      , x_right:width  ] = 0
    image_masked[        0:y_top ,        :       ] = 0
    image_masked[ y_bottom:height,        :       ] = 0
    
    if plot_enabled:
        plots.plot_images( [image, image_masked], enable_grid = False )
    
    return image_masked
# <<
# ==================================================================================================================================
# END << FUNCTION << mask_image
# ==================================================================================================================================



# ==================================================================================================================================
# START >> FUNCTION >> template_submod_func
# ==================================================================================================================================
# >>
def template_submod_func    ( p_p_p_p_1 = ""
                            , p_p_p_p_2 = ""
                            ) :
    
    """
    ================================================================================
    START >> DOC >> template_submod_func
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
    END << DOC << template_submod_func
    ================================================================================
    """
    
    _name_func_ = inspect.stack()[0][3]
    print(f"This is a print from '{_name_subpkg_}.{_name_submod_}.{_name_func_}'{p_p_p_p_1}{p_p_p_p_2}.")
    
    return None
# <<
# ==================================================================================================================================
# END << FUNCTION << template_submod_func
# ==================================================================================================================================

print("   - Done!")

# <<
# ==================================================================================================================================
# END << SUBMODULE << traffic_light_classifier.modify_images_subpkg.modify_images
# ==================================================================================================================================
