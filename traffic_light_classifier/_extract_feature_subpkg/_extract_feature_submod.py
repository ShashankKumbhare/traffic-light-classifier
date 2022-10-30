
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
# START >> EXPORTS
# ==================================================================================
# >>
__all__ =   [ "get_average_channel", "get_average_channel_along_axis"
            , "get_region_high_avg_channel_along_axis"
            , "get_region_high_avg_channel", "get_average_image"
            , "get_colors_from_image" ]
# <<
# ==================================================================================
# END << EXPORTS
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
# START >> FUNCTION >> get_region_high_avg_channel_along_axis
# ==================================================================================================================================
# >>
def get_region_high_avg_channel_along_axis  ( image_rgb
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
    START >> DOC >> get_region_high_avg_channel_along_axis
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
            
            extra_channel <str> (optional)
                
                Extra channel used to extract from rgb image.
                Possible values: "r", "g", "b" or "h", "s", "v".
            
            plot_enabled <bool> (optional)
                
                If enabled plot a bar chart of the average channel along an axis.
        
        RETURNS
        =======
            
            region_high_average_channel_along_axis <tuple>
                
                A tuple of size 2 indicating lower and upper limit.
            
            avgs_channel_along_axis <np.array>
            
                Numpy array of length n_row if axis = 0 requested or
                Numpy array of length n_col if axis = 1 requested.
    
    ================================================================================
    END << DOC << get_region_high_avg_channel_along_axis
    ================================================================================
    """
    
    avgs_channel_along_axis = get_average_channel_along_axis(image_rgb, channel, axis)
    
    if extra_channel is not None:
        avgs_extra_channel_along_axis = get_average_channel_along_axis(image_rgb, extra_channel, axis)
        avgs_channel_along_axis       = np.array(avgs_channel_along_axis)**i * np.array(avgs_extra_channel_along_axis)**j
    
    sums_along_axis = []
    for i in range( len(avgs_channel_along_axis) - len_range + 1 ):
        sum_along_axis = sum( avgs_channel_along_axis[i:i+len_range] )
        sums_along_axis.append(sum_along_axis)
    
    i_sum_max = np.argmax(sums_along_axis)
    region_high_average_channel_along_axis = (i_sum_max, i_sum_max+len_range)
    
    if plot_enabled:
        plot_bar( avgs_channel_along_axis )
    
    return region_high_average_channel_along_axis, avgs_channel_along_axis
# <<
# ==================================================================================================================================
# END << FUNCTION << get_region_high_avg_channel_along_axis
# ==================================================================================================================================



# ==================================================================================================================================
# START >> FUNCTION >> get_region_high_avg_channel
# ==================================================================================================================================
# >>
def get_region_high_avg_channel ( image_rgb
                                , channel
                                , extra_channel     = None
                                , shape_area_search = DEFAULT_SHAPE_AREA_SEARCH
                                , plot_enabled      = False
                                , name_image        = DEFAULT_NAME_IMAGE
                                , i = 1
                                , j = 1
                                ) :
    
    """
    ================================================================================
    START >> DOC >> get_region_high_avg_channel
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
            
            extra_channel <str> (optional)
                
                Extra channel used to extract from rgb image.
                Possible values: "r", "g", "b" or "h", "s", "v".
            
            shape_area_search <tuple> (optional)
                
                A tuple of sizes of the ranges along X & Y to be extracted.
            
            plot_enabled <bool> (optional)
                
                If enabled plot a bar chart of the average channel along an axis.
            
            name_image <str> (optional)
                
                A string for name of the image.
        
        RETURNS
        =======
            
            region_high_avg_channel <tuple<tuple>>
                
                A tuple of tuples of length 2 indicating X & Y lower and upper limits.
            
            image_masked_high_average_channel <list>
            
                A maksed input image at region of high avg channel of dimension
                (size, size, 3).
    
    ================================================================================
    END << DOC << get_region_high_avg_channel
    ================================================================================
    """
    
    region_X, avgs_ch_along_X = get_region_high_avg_channel_along_axis( image_rgb, channel, 0, shape_area_search[0], extra_channel, False, i, j )
    region_Y, avgs_ch_along_Y = get_region_high_avg_channel_along_axis( image_rgb, channel, 1, shape_area_search[1], extra_channel, False, i, j )
    region_high_avg_channel   = (region_X, region_Y)
    
    image_masked_high_average_channel = mask_image( image_rgb, region_X, region_Y )
    
    if plot_enabled:
        
        n_col = 6 if extra_channel else 5
        fig, axes = plt.subplots(1, n_col, figsize = (n_col*3.33, 3.33))
        
        axes[0].imshow( image_rgb )
        axes[0].set_title( name_image )
        
        axes[1].imshow( convert_rgb_to_hsv(image_rgb)[:,:,1], cmap = "gray" )
        axes[1].set_title( "S channel" )
        
        if extra_channel:
            axes[2].imshow( convert_rgb_to_hsv(image_rgb)[:,:,2], cmap = "gray" )
            axes[2].set_title( "V channel" )
        
        axes[-3].imshow(image_masked_high_average_channel)
        axes[-3].set_title( "masked " + name_image )
        
        x = list(range(len(avgs_ch_along_X)))
        axes[-2].bar( x, avgs_ch_along_X)
        axes[-2].set_title( "light strength along X" )
        
        y = list(range(len(avgs_ch_along_Y)))
        axes[-1].barh(y, avgs_ch_along_Y)
        axes[-1].invert_yaxis()
        axes[-1].set_title( "light strength along Y" )
        
        plt.show()
    
    return region_high_avg_channel, image_masked_high_average_channel
# <<
# ==================================================================================================================================
# END << FUNCTION << get_region_high_avg_channel
# ==================================================================================================================================



# ==================================================================================================================================
# START >> FUNCTION >> get_average_image
# ==================================================================================================================================
# >>
def get_average_image   ( images
                        , plot_enabled  = False
                        , type_channels = ""
                        , name_image    = DEFAULT_NAME_IMAGE
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
            
            plot_enabled <bool> (optional)
                
                If enabled plot a bar chart of the average channel along an axis.
            
            type_channels <str> (optional)
                
                A string indicating the type of channels either 'rgb' or 'hsv'.
            
            name_image <str> (optional)
                
                A string for name of the image.
        
        RETURNS
        =======
            
            image_average <np.array>
                
                Numpy array of shape (n_row, n_col, 3).
    
    ================================================================================
    END << DOC << get_average_image
    ================================================================================
    """
    
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
# START >> FUNCTION >> get_colors_from_image
# ==================================================================================================================================
# >>
def get_colors_from_image   ( image_rgb
                            , range_hue
                            , plot_enabled = False
                            , titles = (DEFAULT_NAME_IMAGE, "colors extracted from \n" + DEFAULT_NAME_IMAGE)
                            ) :
    
    """
    ================================================================================
    START >> DOC >> get_colors_from_image
    ================================================================================
        
        GENERAL INFO
        ============
            
            Masks input image with the input range of hues.
        
        PARAMETERS
        ==========
            
            image_rgb <np.array>
                
                Numpy array of rgb image of shape (n_row, n_col, 3).
            
            range_hue <tuple>
                
                A tuple of length 2 indicating lower and upper bound of hue.
            
            plot_enabled <bool> (optional)
                
                If enabled plot a bar chart of the average channel along an axis.
            
            titles <tuple<str>> (optional)
                
                A tuple of length 2 with names for the title of the plots before and
                after masking.
        
        RETURNS
        =======
            
            image_color_rgb <np.array>
                
                Numpy array of rgb image of shape (n_row, n_col, 3).
    
    ================================================================================
    END << DOC << get_colors_from_image
    ================================================================================
    """
    
    # Convert to HSV >>
    # image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    image_hsv = convert_rgb_to_hsv(image_rgb)
    
    # Defining color selection boundaries in HSV values >>
    lower = np.array([range_hue[0],   0,   0])
    upper = np.array([range_hue[1], 255, 255])
    
    # Extracting colors from the input image >>
    mask_color_hsv                     = cv2.inRange(image_hsv, lower, upper)
    image_color_hsv                    = np.copy(image_hsv)
    image_color_hsv[mask_color_hsv==0] = [0,0,0]
    image_color_rgb                    = convert_hsv_to_rgb(image_color_hsv)
    
    if plot_enabled:
        fig, axes = plt.subplots(1, 4, figsize = (4*DEFAULT_FIGSIZE, DEFAULT_FIGSIZE))
        axes[0].imshow(image_rgb)
        axes[0].set_title(titles[0])
        axes[1].imshow(image_color_rgb)
        axes[1].set_title(titles[1])
        axes[2].imshow(image_color_hsv[:,:,1], cmap = "gray")
        axes[2].set_title("S channel")
        axes[3].imshow(image_color_hsv[:,:,2], cmap = "gray")
        axes[3].set_title("V channel")
        plt.show()
    
    return image_color_rgb
# <<
# ==================================================================================================================================
# END << FUNCTION << get_colors_from_image
# ==================================================================================================================================



# <<
# ==================================================================================================================================
# END << SUBMODULE << traffic_light_classifier._extract_feature_subpkg._extract_feature_submod
# ==================================================================================================================================
