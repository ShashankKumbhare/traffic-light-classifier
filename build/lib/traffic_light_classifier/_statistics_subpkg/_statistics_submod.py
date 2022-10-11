
# ==================================================================================================================================
# START >> FILE INFO
# ==================================================================================================================================
# File        : traffic_light_classifier/_statistics_subpkg/_statistics_submod.py
# Author      : Shashank Kumbhare
# Date        : 09/24/2022
# email       : shashankkumbhare8@gmail.com
# Description : This file is a python submodule for python subpackage
#               'traffic_light_classifier._statistics_subpkg'.
# ==================================================================================================================================
# END << FILE INFO
# ==================================================================================================================================



# ==================================================================================================================================
# START >> SUBMODULE >> traffic_light_classifier._statistics_subpkg._statistics_submod
# ==================================================================================================================================
# >>
"""
This submodule contains functionalities to calculate statistical properties.
"""



# ==================================================================================
# START >> IMPORTS
# ==================================================================================
from ..__dependencies_subpkg__ import *
from ..__constants_subpkg__    import *
from ..__auxil_subpkg__        import *
from ..__data_subpkg__         import *
from .._plots_subpkg           import *
from .._modify_images_subpkg   import *
from .._extract_feature_subpkg import *
# ==================================================================================
# END << IMPORTS
# ==================================================================================


# ==================================================================================
# START >> EXPORTS
# ==================================================================================
__all__ = [ "get_distribution_of_channel" ]
# ==================================================================================
# END << EXPORTS
# ==================================================================================


# ==================================================================================================================================
# START >> FUNCTION >> get_distribution_of_channel
# ==================================================================================================================================
# >>
def get_distribution_of_channel ( image_rgb
                                , channels
                                , ch
                                , drop_zeros      = False
                                , remove_outliers = False
                                , plot_enabled    = False
                                ) :
    
    """
    ================================================================================
    START >> DOC >> get_distribution_of_channel
    ================================================================================
        
        GENERAL INFO
        ============
            
            Gets the distribution of channel values in input image in the desired
            range along x and y direction.
        
        PARAMETERS
        ==========
            
            image_rgb <np.array>
                
                Numpy array of rgb image of shape (n_row, n_col, 3).
            
            channels <str>
            
                A string indicating channels type either 'rgb' or 'hsv'.
            
            ch <int>
                
                Channel number (0, 1, or 2).
            
            drop_zeros <bool> (optional)
            
                When enabled drops the zeros from the distribution.
            
            remove_outliers <bool> (optional)
            
                When enabled removes the outliers from the distribution.
            
            plot_enabled <bool> (optional)
                
                When enabled plots the image.
        
        RETURNS
        =======
            
            distribution <tuple>
                
                A tuple of size 3 containing mean, sigma, and the channel values.
    
    ================================================================================
    END << DOC << get_distribution_of_channel
    ================================================================================
    """
    
    # Converting image to hsv if requested >>
    if channels == "hsv":
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    else:
        image = image_rgb
    
    # Separating channel >>
    image_cropped    = image
    ch_image_cropped = image_cropped[:,:,ch]
    
    # Flattening image to 1d array >>
    n_rows = len(ch_image_cropped)
    n_cols = len(ch_image_cropped[0])
    chVals = ch_image_cropped.reshape( (n_rows*n_cols) )
    chVals_float = np.array(chVals, dtype = float)
    
    # Dropping zeros >>
    if drop_zeros:
        chVals_float = chVals_float[chVals_float!=0]
        if len(chVals_float) == 0:
            chVals_float = [0]
    
    # Removing outliers >>
    if remove_outliers:
        Q1  = np.percentile(chVals_float, 10, interpolation = 'midpoint')
        Q3  = np.percentile(chVals_float, 90, interpolation = 'midpoint')
        IQR = Q3 - Q1
        chVals_float = chVals_float[chVals_float<=(Q3+1.5*IQR)]
        chVals_float = chVals_float[chVals_float>=(Q3-1.5*IQR)]
    
    # Getting mean and standard deviation >>
    mu    = np.mean(chVals_float)
    sigma = np.std(chVals_float)
    
    distribution = (mu, sigma, chVals_float)
    
    # Plotting histogram >>
    if plot_enabled:
        _, axes = plt.subplots(1, 1, figsize = (3.33, 3.33))
        axes.hist(chVals_float)
        axes.set_title(f"Histogram of ch {channels[ch]}\n mu = {mu:.3f}, sig = {sigma:.3f}")
    
    return distribution
# <<
# ==================================================================================================================================
# END << FUNCTION << get_distribution_of_channel
# ==================================================================================================================================



# <<
# ==================================================================================================================================
# END << SUBMODULE << traffic_light_classifier._statistics_subpkg._statistics_submod
# ==================================================================================================================================
