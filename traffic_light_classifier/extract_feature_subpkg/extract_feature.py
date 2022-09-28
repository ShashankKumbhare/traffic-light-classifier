
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

_name_subpkg = __name__.partition(".")[-2]
_name_submod = __name__.partition(".")[-1]
print(f"   + Adding submodule '{_name_submod}'...")

# ==================================================================================
# START >> IMPORTS
# ==================================================================================
from ..__dependencies_subpkg__ import _dependencies_submod as _dps
from ..__constants_subpkg__ import _constants_submod as _CONSTANTS
# from ..__auxil_subpkg__ import _auxil_submod as _auxil
from ..plots_subpkg import plots as _plots
from ..modify_images_subpkg import modify_images as _modify_images
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
        im = _dps.cv2.cvtColor(image_rgb, _dps.cv2.COLOR_RGB2HSV)
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
    avg_channel = _dps.np.mean( im[:,:,channel_num] )
    
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
        im = _dps.cv2.cvtColor(im_rgb, _dps.cv2.COLOR_RGB2HSV)
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
    
    # avg_channel_along_axis = get_average_channel_along_axis(im_rgb, channel, axis)
    #
    # sums_along_axis = []
    # for i in range( len(avg_channel_along_axis) - len_range ):
    #     sum_along_axis = np.sum( avg_channel_along_axis[i:i+len_range] )
    #     sums_along_axis.append(sum_along_axis)
    #
    # i_sum_max = np.argmax(sums_along_axis)
    # # i_sum_max = np.argpartition(sums_along_axis, len(sums_along_axis) // 2)[len(sums_along_axis) // 2]
    #
    # range_of_high_average_channel_along_axis = (i_sum_max, i_sum_max+len_range)
    #
    # if plot_enabled:
    #     plots.plot_bar( avg_channel_along_axis )
    #
    # return range_of_high_average_channel_along_axis, avg_channel_along_axis
    
    avg_channel_along_axis = get_average_channel_along_axis(im_rgb, channel, axis)
    
    sums_along_axis = []
    for i in range( len(avg_channel_along_axis) - len_range ):
        sum_along_axis = _dps.np.sum( avg_channel_along_axis[i:i+len_range] )
        sums_along_axis.append(sum_along_axis)
    
    i_sum_max = _dps.np.argmax(sums_along_axis)
    
    range_of_high_average_channel_along_axis = (i_sum_max, i_sum_max+len_range)
    
    if plot_enabled:
        _plots.plot_bar( avg_channel_along_axis )
    
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
    
    range_X, sums_channel_along_X = get_range_of_high_average_channel_along_axis( im_rgb, channel, 0, len_range[0], plot_enabled=False)
    range_Y, sums_channel_along_Y = get_range_of_high_average_channel_along_axis( im_rgb, channel, 1, len_range[1], plot_enabled=False)
    
    rangeXY_of_high_average_channel = [ range_X, range_Y ]
    sums_channel_along_XY           = [sums_channel_along_X, sums_channel_along_Y]
    
    if plot_enabled:
        
        fig, axes = _dps.plt.subplots(1, 4, figsize = (4*3.33, 3.33))
        
        axes[0].imshow( im_rgb )
        axes[0].set_title( "rgb image" )
        
        axes[1].imshow( _modify_images.convert_rgb_to_hsv(im_rgb)[:,:,2], cmap = "gray" )
        axes[1].set_title( "S channel" )
        
        x = list(range(len(sums_channel_along_X)))
        axes[2].bar( x, sums_channel_along_XY[0])
        axes[2].set_title( "saturation along X" )
        
        y = list(range(len(sums_channel_along_Y)))
        axes[3].bar( y, sums_channel_along_XY[1])
        axes[3].set_title( "saturation along Y" )
        
        _dps.plt.show()
    
    return rangeXY_of_high_average_channel, sums_channel_along_XY
# <<
# ==================================================================================================================================
# END << FUNCTION << get_range_of_high_average_channel
# ==================================================================================================================================



# ==================================================================================================================================
# START >> FUNCTION >> get_average_image
# ==================================================================================================================================
# >>
def get_average_image   ( images
                        , plot_enabled  = False
                        , type_channels = ""
                        , name_image    = _CONSTANTS.DEFAULT_NAME_IMAGE
                        , is_images_npArrays = False
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
            
            images <list>
                
                A list of numpy array of images of shape (n_row, n_col, 3).
                Default is "" for unknown.
            
            plot_enabled <bool>
                
                If enabled plot a bar chart of the average channel along an axis.
            
            type_channels <str>
                
                A string indicating the type of channels either 'rgb' or 'hsv'.
            
            cmap <str>
                
                Colormap for plot. Possible value: "viridis", "gray", etc.
        
        RETURNS
        =======
            
            image_average <np.array>
                
                Numpy array of shape (n_row, n_col, 3).
    
    ================================================================================
    END << DOC << get_average_image
    ================================================================================
    """
    
    # Making a 4d array to hold all images >>
    if not is_images_npArrays:
        images = _dps.np.array( [ image[0] for image in images ] )
    
    # Taking average of all images (i.e. average along axis 0) >>
    image_average = _dps.np.mean(images, axis = 0)
    
    # Converting dtype from 'float64' to "uint8" >>
    image_average = _dps.np.uint8(image_average)
    
    # Plotting if requested >>
    if plot_enabled:
        _plots.plot_channels( image_average
                            , type_channels = type_channels
                            , name_image    = name_image
                            , cmap          = "gray" )
        
    return image_average
# <<
# ==================================================================================================================================
# END << FUNCTION << get_average_image
# ==================================================================================================================================



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
    
    _name_func = _dps.inspect.stack()[0][3]
    print(f"This is a print from '{_name_subpkg}.{_name_submod}.{_name_func}'{p_p_p_p_1}{p_p_p_p_2}.")
    
    return None
# <<
# ==================================================================================================================================
# END << FUNCTION << _template_submod_func
# ==================================================================================================================================

print("   - Done!")

# <<
# ==================================================================================================================================
# END << SUBMODULE << traffic_light_classifier.extract_feature_subpkg.extract_feature
# ==================================================================================================================================
