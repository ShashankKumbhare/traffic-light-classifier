
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
This submodule contains functionalities to calculate probabilities and likelihood values.
"""

_name_subpkg = __name__.partition(".")[-2]
_name_submod = __name__.partition(".")[-1]
print(f"   + Adding submodule '{_name_submod}'...")

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
__all__ = [ "get_distribution_of_channel", "get_logLikelihood" ]
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
                                , plot_enabled = False
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
            
            rangeX <tuple>
                
                Range along x-axis.
            
            rangeY <tuple>
                
                Range along y-axis.
            
            cmap <str>
                
                Possible value: None or "gray"
        
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



# ==================================================================================================================================
# START >> FUNCTION >> get_logLikelihood
# ==================================================================================================================================
# >>
def get_logLikelihood   ( mu
                        , sigma
                        , xa
                        , xb
                        , data_points
                        ) :
    
    """
    ================================================================================
    START >> DOC >> get_logLikelihood
    ================================================================================
        
        GENERAL INFO
        ============
            
            Computes the log likelihood of the input distribution(mu, sigma) given
            data points.
        
        PARAMETERS
        ==========
            
            mu <float>
                
                Mean of the input distribution.
            
            sigma <float>
                
                Standard deviation of the input distribution.
                
            data_points <list<floats>>
                
                Data points.
        
        RETURNS
        =======
            
            logLikelihood <float>
                
                Log likelihood of the input distribution(mu,sigma) given the data points.
    
    ================================================================================
    END << DOC << get_logLikelihood
    ================================================================================
    """
    
    logLikelihood = 0
    (a, b) = get_shape_params_truncnorm(xa, xb, mu, 0)
    for data_point in data_points:
        if data_point == 0:
            llh = -100
        else:
            llh = truncnorm.logpdf(data_point, a, b, mu, sigma)
        # llh = poisson.logpmf(data_point, mu)
        logLikelihood = logLikelihood + llh
    return logLikelihood
# <<
# ==================================================================================================================================
# END << FUNCTION << get_logLikelihood
# ==================================================================================================================================



# ==================================================================================================================================
# START >> FUNCTION >> _template_submod_func
# ==================================================================================================================================
# >>
def _template_submod_func3    ( p_p_p_p_1 = ""
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



# ==================================================================================================================================
# START >> FUNCTION >> _template_submod_func
# ==================================================================================================================================
# >>
def _template_submod_func4  ( p_p_p_p_1 = ""
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
# END << SUBMODULE << traffic_light_classifier._statistics_subpkg._statistics_submod
# ==================================================================================================================================
