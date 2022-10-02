
# ==================================================================================================================================
# START >> FILE INFO
# ==================================================================================================================================
# File        : traffic_light_classifier/_extract_feature_subpkg/_extract_feature_submod.py
# Author      : Shashank Kumbhare
# Date        : 09/23/2022
# email       : shashankkumbhare8@gmail.com
# Description : This file is a python submodule for python subpackage
#               'traffic_light_classifier._extract_feature_subpkg'.
# ==================================================================================================================================
# END << FILE INFO
# ==================================================================================================================================



# ==================================================================================================================================
# START >> SUBMODULE >> traffic_light_classifier._extract_feature_subpkg._extract_feature_submod
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
# >>
from ..__dependencies_subpkg__ import *
from ..__constants_subpkg__    import *
from ..__auxil_subpkg__        import *
from ..__data_subpkg__         import *
from .._plots_subpkg           import *
from .._modify_images_subpkg   import *
# <<
# ==================================================================================
# END << IMPORTS
# ==================================================================================


# ==================================================================================
# START >> IMPORTS
# ==================================================================================
# >>
__all__ =   [ "get_average_channel", "get_average_channel_along_axis"
            , "get_range_of_high_average_channel_along_axis"
            , "get_range_of_high_average_channel", "get_average_image"
            , "get_location_of_light" ]
# <<
# ==================================================================================
# END << IMPORTS
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
    sums_im_channel_along_axis = im_channel.sum(axis=axis)
    n_col                     = len(sums_im_channel_along_axis)
    avgs_im_channel_along_axis = sums_im_channel_along_axis / n_col
    
    return avgs_im_channel_along_axis
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
                                                , extra_channel = None
                                                , plot_enabled  = False
                                                , i = 1
                                                , j = 1
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
    
    avgs_channel_along_axis = get_average_channel_along_axis(im_rgb, channel, axis)
    
    if extra_channel is not None:
        avgs_extra_channel_along_axis = get_average_channel_along_axis(im_rgb, extra_channel, axis)
        avgs_channel_along_axis       = np.array(avgs_channel_along_axis)**i * np.array(avgs_extra_channel_along_axis)**j
    
    sums_along_axis = []
    for i in range( len(avgs_channel_along_axis) - len_range + 1 ):
        sum_along_axis = sum( avgs_channel_along_axis[i:i+len_range] )
        sums_along_axis.append(sum_along_axis)
    
    i_sum_max = np.argmax(sums_along_axis)
    range_of_high_average_channel_along_axis = (i_sum_max, i_sum_max+len_range)
    
    if plot_enabled:
        plot_bar( avgs_channel_along_axis )
    
    return range_of_high_average_channel_along_axis, avgs_channel_along_axis
# <<
# ==================================================================================================================================
# END << FUNCTION << get_range_of_high_average_channel_along_axis
# ==================================================================================================================================



# ==================================================================================================================================
# START >> FUNCTION >> get_range_of_high_average_channel
# ==================================================================================================================================
# >>
def get_range_of_high_average_channel( image_rgb
                                     , channel
                                     , shape_area_search = DEFAULT_SHAPE_AREA_SEARCH
                                     , extra_channel = None
                                     , plot_enabled  = False
                                     , name_image    = DEFAULT_NAME_IMAGE
                                     , i = 1
                                     , j = 1
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
            
            avgs_ch_along_XY <list>
                
                A list of length 2 indicating avgs_ch_along_X & avgs_ch_along_Y lower and upper limits.
    
    ================================================================================
    END << DOC << get_range_of_high_average_channel
    ================================================================================
    """
    
    range_X, avgs_ch_along_X = get_range_of_high_average_channel_along_axis(
                                image_rgb, channel, 0, shape_area_search[0], extra_channel, False, i, j )
    range_Y, avgs_ch_along_Y = get_range_of_high_average_channel_along_axis(
                                image_rgb, channel, 1, shape_area_search[1], extra_channel, False, i, j )
    
    rangeXY_of_high_average_channel = [ range_X, range_Y ]
    avgs_ch_along_XY                = [avgs_ch_along_X, avgs_ch_along_Y]
    
    if plot_enabled:
        
        fig, axes = plt.subplots(1, 7, figsize = (7*3.33, 3.33))
        
        axes[0].imshow( image_rgb )
        axes[0].set_title( name_image )
        
        axes[1].imshow( convert_rgb_to_hsv(image_rgb)[:,:,0], cmap = "gray" )
        axes[1].set_title( "H channel" )
        
        axes[2].imshow( convert_rgb_to_hsv(image_rgb)[:,:,1], cmap = "gray" )
        axes[2].set_title( "S channel" )
        
        axes[3].imshow( convert_rgb_to_hsv(image_rgb)[:,:,2], cmap = "gray" )
        axes[3].set_title( "V channel" )
        
        axes[4].imshow( mask_image( image_rgb, range_X, range_Y ) )
        axes[4].set_title( "masked " + name_image )
        
        x = list(range(len(avgs_ch_along_X)))
        axes[5].bar( x, avgs_ch_along_XY[0])
        axes[5].set_title( "saturation along X" )
        
        y = list(range(len(avgs_ch_along_Y)))
        axes[6].barh(y, avgs_ch_along_XY[1])
        axes[6].invert_yaxis()
        axes[6].set_title( "saturation along Y" )
        
        plt.show()
    
    return rangeXY_of_high_average_channel, avgs_ch_along_XY
# <<
# ==================================================================================================================================
# END << FUNCTION << get_range_of_high_average_channel
# ==================================================================================================================================



# ==================================================================================================================================
# START >> FUNCTION >> get_location_of_light
# ==================================================================================================================================
# >>
def get_location_of_light   ( image_rgb
                            , shape_area_search = DEFAULT_SHAPE_AREA_SEARCH
                            , plot_enabled = False
                            , name_image    = DEFAULT_NAME_IMAGE
                            ) :
    
    """
    ================================================================================
    START >> DOC >> get_location_of_light
    ================================================================================
        
        GENERAL INFO
        ============
            
            Finds the location of the light in the input rgb image by extracting the
            X & Y range of the region of high saturation and brightness channel
            values along both the axis.
        
        PARAMETERS
        ==========
            
            image_rgb <np.array>
                
                Numpy array of rgb image of shape (n_row, n_col, 3).
            
            shape_area_search <tuple>
                
                A tuple of length 2 indicating height & width of the search area.
            
            plot_enabled <bool>
                
                If enabled plot a bar chart of the average channel along an axis.
        
        RETURNS
        =======
            
            corners_location <list<list<tuple>>>
                
                A list of list of tuples where each tuple element is the location of
                the corners of the rectangular area of the light found in the order
                top-left, top-right, bot-left & bot-right.
    
    ================================================================================
    END << DOC << get_location_of_light
    ================================================================================
    """
    
    # Getting range for high saturation region >>
    # range_s_X, avgs_s_along_X = get_range_of_high_average_channel_along_axis(
    #                                             image_rgb, "s", 0, shape_area_search[0], "v", False )
    # range_s_Y, avgs_s_along_Y = get_range_of_high_average_channel_along_axis(
    #                                             image_rgb, "s", 1, shape_area_search[1], "v", False )
    
    range_s_X, avgs_s_along_X = get_range_of_high_average_channel_along_axis(
                                                image_rgb, "s", 0, shape_area_search[0], None, False )
    range_s_Y, avgs_s_along_Y = get_range_of_high_average_channel_along_axis(
                                                image_rgb, "s", 1, shape_area_search[1], None, False )
    
    range_s_XY_of_high_average_s = [range_s_X, range_s_Y]
    avgs_s_along_XY              = [avgs_s_along_X, avgs_s_along_Y]
    
    if plot_enabled:
        
        fig, axes = plt.subplots(1, 5, figsize = (5*3.33, 3.33))
        
        axes[0].imshow( image_rgb )
        axes[0].set_title( name_image )
        
        axes[1].imshow( convert_rgb_to_hsv(image_rgb)[:,:,2], cmap = "gray" )
        axes[1].set_title( "S ch. of " + name_image )
        
        axes[2].imshow( mask_image( image_rgb, range_s_X, range_s_Y ) )
        axes[2].set_title( "masked " + name_image )
        
        x = list(range(len(avgs_s_along_X)))
        axes[3].bar( x, avgs_s_along_XY[0])
        axes[3].set_title( "saturation along X" )
        
        y = list(range(len(avgs_s_along_Y)))
        axes[4].bar( y, avgs_s_along_XY[1])
        axes[4].set_title( "saturation along Y" )
        
        plt.show()
    
    
    return range_s_XY_of_high_average_s
# <<
# ==================================================================================================================================
# END << FUNCTION << get_location_of_light
# ==================================================================================================================================



# ==================================================================================================================================
# START >> FUNCTION >> get_average_image
# ==================================================================================================================================
# >>
def get_average_image   ( images
                        , plot_enabled  = False
                        , type_channels = ""
                        , name_image    = DEFAULT_NAME_IMAGE
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
        images = np.array( [ image[0] for image in images ] )
    
    # Taking average of all images (i.e. average along axis 0) >>
    image_average = np.mean(images, axis = 0)
    
    # Converting dtype from 'float64' to "uint8" >>
    image_average = np.uint8(image_average)
    
    # Plotting if requested >>
    if plot_enabled:
        plot_channels   ( image_average
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
# END << SUBMODULE << traffic_light_classifier._extract_feature_subpkg._extract_feature_submod
# ==================================================================================================================================
