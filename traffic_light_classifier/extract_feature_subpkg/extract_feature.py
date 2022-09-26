
# ==================================================================================================================================
# START >> FILE INFO
# ==================================================================================================================================
# File        : traffic_light_classifier/extract_feature_subpkg/extract_feature.py
# Author      : Shashank Kumbhare
# Date        : 09/23/2022
# email       : shashankkumbhare8@gmail.com
# Description : This file is a python submodule for python subpackage
#               'traffic_light_classifier.extract_feature_subpkg'.
# ==================================================================================================================================
# END << FILE INFO
# ==================================================================================================================================



# ==================================================================================================================================
# START >> SUBMODULE >> traffic_light_classifier.extract_feature_subpkg.extract_feature
# ==================================================================================================================================
# >>
"""
This submodule contains functionalities to extract features from traffic light
image dataset.
"""

_name_subpkg_ = __name__.partition(".")[-2]
_name_submod_ = __name__.partition(".")[-1]
print(f"   + Adding submodule '{_name_submod_}'...")

# ==================================================================================
# START >> IMPORTS
# ==================================================================================
from ..__auxil_subpkg__ import *
from ..plots_subpkg import *
from ..modify_images_subpkg import *
# ==================================================================================
# END >> IMPORTS
# ==================================================================================


# ==================================================================================================================================
# START >> FUNCTION >> get_average_channel
# ==================================================================================================================================
# >>
def get_average_channel( image_rgb
                       , channel
                       ) :
    
    """
    ================================================================================
    START >> DOC >> get_average_channel
    ================================================================================
        
        GENERAL INFO
        ============
            
            Calculates average value of channel requested from the input rgb image.
            Channel can be r/g/b or h/s/v.
        
        PARAMETERS
        ==========
            
            image_rgb <np.array>
                
                Numpy array of rgb image of shape (n_row, n_col, 3).
            
            channel <str>
                
                Channel to extract from rgb image.
                Possible values: "r", "g", "b" or "h", "s", "v".
        
        RETURNS
        =======
            
            avg_channel <float>
                
                Average value of channel requested from the input rgb image.
    
    ================================================================================
    END << DOC << get_average_channel
    ================================================================================
    """
    
    # Setting channel number >>
    if channel in ("h", "s", "v"):
        # Converting image to HSV if channel h/s/v requested >>
        im = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        if channel == "h":
            channel_num = 0
        elif channel == "s":
            channel_num = 1
        elif channel == "v":
            channel_num = 2
    else:
        im = image_rgb
        if channel == "r":
            channel_num = 0
        elif channel == "g":
            channel_num = 1
        elif channel == "b":
            channel_num = 2
    
    # Taking mean >>
    avg_channel = np.mean( im[:,:,channel_num] )
    
    return avg_channel
# <<
# ==================================================================================================================================
# END << FUNCTION << get_average_channel
# ==================================================================================================================================



# ==================================================================================================================================
# START >> FUNCTION >> get_average_channel_along_axis
# ==================================================================================================================================
# >>
def get_average_channel_along_axis( im_rgb
                                  , channel
                                  , axis
                                  ) :
    
    """
    ================================================================================
    START >> DOC >> get_average_channel_along_axis
    ================================================================================
        
        GENERAL INFO
        ============
            
            Calculates average value of channel requested from the input rgb image
            along the requested axis.
            Channel can be r/g/b or h/s/v.
            Axis can be 0 or 1.
        
        PARAMETERS
        ==========
            
            image_rgb <np.array>
                
                Numpy array of rgb image of shape (n_row, n_col, 3).
            
            channel <str>
                
                Channel to extract from rgb image.
                Possible values: "r", "g", "b" or "h", "s", "v".
            
            axis <int>
                
                Axis to take average on. Either 0 or 1.
        
        RETURNS
        =======
            
            avg_im_channel_along_axis <np.array>
                
                1D numpy array.
    
    ================================================================================
    END << DOC << get_average_channel_along_axis
    ================================================================================
    """
    
    # Setting channel number >>
    if channel in ("h", "s", "v"):
        # Converting image to HSV if channel h/s/v requested >>
        im = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2HSV)
        if channel == "h":
            channel_num = 0
        elif channel == "s":
            channel_num = 1
        elif channel == "v":
            channel_num = 2
    else:
        im = im_rgb
        if channel == "r":
            channel_num = 0
        elif channel == "g":
            channel_num = 1
        elif channel == "b":
            channel_num = 2
    
    im_channel                = im[:,:,channel_num]
    sum_im_channel_along_axis = im_channel.sum(axis=axis)
    n_col                     = len(sum_im_channel_along_axis)
    avg_im_channel_along_axis = sum_im_channel_along_axis / n_col
    
    return avg_im_channel_along_axis
# <<
# ==================================================================================================================================
# END << FUNCTION << get_average_channel_along_axis
# ==================================================================================================================================



# ==================================================================================================================================
# START >> FUNCTION >> get_range_of_high_average_channel_along_axis
# ==================================================================================================================================
# >>
def get_range_of_high_average_channel_along_axis( im_rgb
                                                , channel
                                                , axis
                                                , len_range
                                                , plot_enabled = False
                                                ) :
    
    """
    ================================================================================
    START >> DOC >> get_range_of_high_average_channel_along_axis
    ================================================================================
        
        GENERAL INFO
        ============
            
            Extracts the range of the region of high average channel along an axis.
        
        PARAMETERS
        ==========
            
            image_rgb <np.array>
                
                Numpy array of rgb image of shape (n_row, n_col, 3).
            
            channel <str>
                
                Channel to extract from rgb image.
                Possible values: "r", "g", "b" or "h", "s", "v".
            
            axis <int>
                
                Axis to take average on. Either 0 or 1.
            
            len_range <int>
                
                Size of the range to be extracted.
            
            plot_enabled <bool>
                
                If enabled plot a bar chart of the average channel along an axis.
        
        RETURNS
        =======
            
            range_of_high_average_channel_along_axis <tuple>
                
                A tuple of size 2 indicating lower and upper limit.
    
    ================================================================================
    END << DOC << get_range_of_high_average_channel_along_axis
    ================================================================================
    """
    
    avg_channel_along_axis = get_average_channel_along_axis(im_rgb, channel, axis)
    
    sums_along_axis = []
    for i in range( len(avg_channel_along_axis) - len_range ):
        sum_along_axis = np.sum( avg_channel_along_axis[i:i+len_range] )
        sums_along_axis.append(sum_along_axis)
    
    i_sum_max = np.argmax(sums_along_axis)
    
    range_of_high_average_channel_along_axis = (i_sum_max, i_sum_max+len_range)
    
    if plot_enabled:
        plots.plot_bar( avg_channel_along_axis )
    
    return range_of_high_average_channel_along_axis, avg_channel_along_axis
# <<
# ==================================================================================================================================
# END << FUNCTION << get_range_of_high_average_channel_along_axis
# ==================================================================================================================================



# ==================================================================================================================================
# START >> FUNCTION >> get_range_of_high_average_channel
# ==================================================================================================================================
# >>
def get_range_of_high_average_channel( im_rgb
                                     , channel
                                     , len_range
                                     , plot_enabled = False
                                     ) :
    
    """
    ================================================================================
    START >> DOC >> get_range_of_high_average_channel
    ================================================================================
        
        GENERAL INFO
        ============
            
            Extracts the X & Y range of the region of high average channel values
            along both the axis.
        
        PARAMETERS
        ==========
            
            image_rgb <np.array>
                
                Numpy array of rgb image of shape (n_row, n_col, 3).
            
            channel <str>
                
                Channel to extract from rgb image.
                Possible values: "r", "g", "b" or "h", "s", "v".
            
            len_range <int>
                
                Size of the range to be extracted.
            
            plot_enabled <bool>
                
                If enabled plot a bar chart of the average channel along an axis.
        
        RETURNS
        =======
            
            rangeXY_of_high_average_channel <list<tuple>>
                
                A list of tuples of length 2 indicating X & Y lower and upper limits.
            
            sums_channel_along_XY <list>
                
                A list of length 2 indicating sums_channel_along_X & sums_channel_along_Y lower and upper limits.
    
    ================================================================================
    END << DOC << get_range_of_high_average_channel
    ================================================================================
    """
    
    range_X, sums_channel_along_X = get_range_of_high_average_channel_along_axis( im_rgb, channel, 0, len_range, plot_enabled=False)
    range_Y, sums_channel_along_Y = get_range_of_high_average_channel_along_axis( im_rgb, channel, 1, len_range, plot_enabled=False)
    
    rangeXY_of_high_average_channel = [ range_X, range_Y ]
    sums_channel_along_XY           = [sums_channel_along_X, sums_channel_along_Y]
    
    if plot_enabled:
        
        fig, axes = plt.subplots(1, 4, figsize = (4*3.33, 3.33))
        
        axes[0].imshow( im_rgb )
        axes[0].set_title( "rgb image" )
        
        axes[1].imshow( convert_rgb_to_hsv(im_rgb)[:,:,1], cmap = "gray" )
        axes[1].set_title( "S channel" )
        
        x = list(range(len(sums_channel_along_X)))
        axes[2].bar( x, sums_channel_along_XY[0])
        axes[2].set_title( "saturation along X" )
        
        y = list(range(len(sums_channel_along_Y)))
        axes[3].bar( y, sums_channel_along_XY[1])
        axes[3].set_title( "saturation along Y" )
        
        plt.show()
    
    return rangeXY_of_high_average_channel, sums_channel_along_XY
# <<
# ==================================================================================================================================
# END << FUNCTION << get_range_of_high_average_channel
# ==================================================================================================================================



# ==================================================================================================================================
# START >> FUNCTION >> get_average_image
# ==================================================================================================================================
# >>
def get_average_image( images_rgb
                     , channels = "rgb"
                     ) :
    
    """
    ================================================================================
    START >> DOC >> get_average_image
    ================================================================================
        
        GENERAL INFO
        ============
            
            Calculates average channels of all imput images.
        
        PARAMETERS
        ==========
            
            images_rgb <list>
                
                A list of numpy array of rgb image of shape (n_row, n_col, 3).
            
            channels <str>
            
                A string indicating channels type either 'rgb' or 'hsv'.
        
        RETURNS
        =======
            
            image_average <np.array>
                
                Numpy array of of shape (n_row, n_col, 3).
    
    ================================================================================
    END << DOC << get_average_image
    ================================================================================
    """
    
    if channels == "hsv":
        images = np.array( [ cv2.cvtColor(image_rgb[0], cv2.COLOR_RGB2HSV) for image_rgb in images_rgb ] )
    else:
        images = np.array( [ image_rgb[0] for image_rgb in images_rgb ] )
    
    average_channels = np.mean(images, axis = 0)
    average_channels = [ average_channels[:,:,i] for i in range(3) ]
    
    ch0 = average_channels[0]
    ch1 = average_channels[1]
    ch2 = average_channels[2]
    image_average = cv2.merge([ch0, ch1, ch2])
    image_average = np.uint8(image_average)
    
    return image_average
# <<
# ==================================================================================================================================
# END << FUNCTION << get_average_image
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
# END << SUBMODULE << traffic_light_classifier.extract_feature_subpkg.extract_feature
# ==================================================================================================================================
