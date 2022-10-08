
# ==================================================================================================================================
# START >> FILE INFO
# ==================================================================================================================================
# File        : traffic_light_classifier/_plots_subpkg/_plots_submod.py
# Author      : Shashank Kumbhare
# Date        : 09/22/2022
# email       : shashankkumbhare8@gmail.com
# Description : This file is a python submodule for python subpackage
#               'traffic_light_classifier._plots_subpkg'.
# ==================================================================================================================================
# END << FILE INFO
# ==================================================================================================================================



# ==================================================================================================================================
# START >> SUBMODULE >> traffic_light_classifier._plots_subpkg._plots_submod
# ==================================================================================================================================
# >>
"""
This submodule is created for the visualization and analyzsis of the image-dataset.
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
# <<
# ==================================================================================
# END << IMPORTS
# ==================================================================================


# ==================================================================================
# START >> EXPORTS
# ==================================================================================
# >>
__all__ = ["plot_images", "plot_channels", "plot_bar", ]
# <<
# ==================================================================================
# END << EXPORTS
# ==================================================================================


# ==================================================================================================================================
# START >> FUNCTION >> plot_images
# ==================================================================================================================================
# >>
def plot_images( images
               , title_enabled = True
               , name_image    = DEFAULT_NAME_IMAGE
               , figsizeScale  = DEFAULT_FIGSIZESCALE
               , enable_grid   = False
               , cmap          = None
               ) :
    
    """
    ================================================================================
    START >> DOC >> plot_images
    ================================================================================
        
        GENERAL INFO
        ============
            
            Plots the input images.
        
        PARAMETERS
        ==========
            
            images <list>
                
                A list of images or a single image to plot.
            
            title_enabled <bool>
                
                When enabled sets title/s to the plot/s.
            
            figsizeScale <float>
                
                A float to scale the size of the output image.
            
            enable_grid <bool>
                
                When enabled draws mesh/grid on plot.
            
            cmap <str>
                
                Colormap for plot. Possible value: "viridis", "gray", etc.
        
        RETURNS
        =======
            
            None
    
    ================================================================================
    END << DOC << plot_images
    ================================================================================
    """
    
    # Plotting images >>
    if type(images) == tuple or type(images) == np.ndarray:
        
        # If input is a single image either as a tuple(rgb,label) or as an array >>
        f, axes = plt.subplots( 1, 1, figsize = (3.33*figsizeScale, 3.33*figsizeScale) )
        
        if type(images) == np.ndarray:
            image   = images
            if title_enabled:
                axes.set_title(name_image)
        elif type(images) == tuple:
            image   = images[0]
            if title_enabled:
                title = get_title(images)
                axes.set_title(title)
        axes.imshow(image, cmap = cmap)
    
    else:
        
        # Deciding the no. of rows and columns of the grid plot >>
        len_images = len(images)
        if len_images < 11:
            n_col = len_images
        else:
            n_col = int( np.ceil(np.sqrt(len_images)) )
        n_row     = int( np.ceil(len_images/n_col) )

        # Creating plot axes >>
        figsize    = (n_col*3.33*figsizeScale, n_row*3.33*figsizeScale)
        f, axes    = plt.subplots(n_row, n_col, figsize = figsize)
        
        # Loop for all images along a grid >>
        i_image = 0
        for i in range(n_row):
            
            if n_col == 1:
                ax = axes
            else:
                ax = axes[i] if n_row != 1 else axes
            
            for j in range(n_col):
                
                if len(images) == 1:
                    ax_curr = ax
                else:
                    ax_curr = ax[j]
                
                if type(images[0]) == tuple:
                    image = images[i_image][0]
                else:
                    image = images[i_image]
                
                if title_enabled:
                    if not type(images[0]) == tuple:
                        title = get_title(images[i_image])
                        ax_curr.set_title(title)
                
                if enable_grid:
                    ax_curr.set_xticks( np.arange(-.5, len(image[0]), 1), minor = True )
                    ax_curr.set_yticks( np.arange(-.5, len(image)   , 1), minor = True )
                    ax_curr.grid( which = 'minor', color = 'k', linestyle = '-', linewidth = 0.5)
                
                ax_curr.imshow(image, cmap = cmap)
                
                i_image = i_image + 1
                if i_image >= len_images:
                    break
    plt.show()
    
    return None
# <<
# ==================================================================================================================================
# END << FUNCTION << plot_images
# ==================================================================================================================================



# ==================================================================================================================================
# START >> FUNCTION >> plot_channels
# ==================================================================================================================================
# >>
def plot_channels   ( image_rgb
                    , type_channels = ""
                    , name_image    = DEFAULT_NAME_IMAGE
                    , cmap          = DEFAULT_CMAP
                    , figsizeScale  = DEFAULT_FIGSIZESCALE
                    ) :
    
    """
    ================================================================================
    START >> DOC >> plot_channels
    ================================================================================
        
        GENERAL INFO
        ============
            
            Plots individual channels of input image.
        
        PARAMETERS
        ==========
            
            image_rgb <np.array>
                
                Numpy array of rgb image of shape (n_row, n_col, 3).
            
            type_channels <str>
                
                A string indicating the type of channels either 'rgb' or 'hsv'.
                Default is "" for unknown.
            
            name_image <str>
                
                A string for name of the image.
            
            cmap <str>
                
                Colormap for plot. Possible value: "viridis", "gray", etc.
            
            figsizeScale <float>
                
                A float to scale the size of the output image.
        
        RETURNS
        =======
            
            None
    
    ================================================================================
    END << DOC << plot_channels
    ================================================================================
    """
    
    if type_channels == "hsv":
        image = np.copy(image_rgb)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    else:
        image = image_rgb
    
    # Getting channels >>
    image_ch0 = image[:,:,0]
    image_ch1 = image[:,:,1]
    image_ch2 = image[:,:,2]
    
    # Creating plot axes for plots >>
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize = (4*3.33*figsizeScale, 4*3.33*figsizeScale))
    
    # Setting titles for plots >>
    if type(type_channels) != str or len(type_channels) == 0 or type_channels is None or type_channels not in ["rgb", "hsv"]:
        type_channels = "012"
    type_channels = type_channels.upper()
    titles = ( f"{name_image}"
             , f"{type_channels[0]} ch. of {name_image}"
             , f"{type_channels[1]} ch. of {name_image}"
             , f"{type_channels[2]} ch. of {name_image}" )
    
    # Plotting the original image and the three channels >>
    if type(cmap) != str or len(cmap) == 0 or cmap is None:
        cmap = DEFAULT_CMAP
    ax1.set_title(titles[0])
    ax1.imshow(image_rgb, cmap = cmap)
    ax2.set_title(titles[1])
    ax2.imshow(image_ch0, cmap = cmap)
    ax3.set_title(titles[2])
    ax3.imshow(image_ch1, cmap = cmap)
    ax4.set_title(titles[3])
    ax4.imshow(image_ch2, cmap = cmap)
    
    plt.show()

    return None
# <<
# ==================================================================================================================================
# END << FUNCTION << plot_channels
# ==================================================================================================================================



# ==================================================================================================================================
# START >> FUNCTION >> plot_bar
# ==================================================================================================================================
# >>
def plot_bar( Y
            , n_row        = 1
            , figsizeScale = 1
            ) :
    
    """
    ================================================================================
    START >> DOC >> plot_bar
    ================================================================================
        
        GENERAL INFO
        ============
            
            Plots bar plot/s for the input data.
        
        PARAMETERS
        ==========
            
            Y <list or np.array>
                
                Data to be ploted.
                Could ne np.array or list of np.array for multiple plot data.
            
            n_row <int>
                
                No. of rows for the plots.
        
        RETURNS
        =======
            
            None
    
    ================================================================================
    END << DOC << plot_bar
    ================================================================================
    """
    
    len_Y = len(Y)
    
    if type(Y) == np.ndarray:
        
        f, axes = plt.subplots(1, 1, figsize = (3.33*figsizeScale, 3.33*figsizeScale))
        x = list(range(len(Y)))
        axes.bar(x,Y)
        
    else:
        
        n_col   = int(len_Y/n_row)
        figsize = (n_col*3.33*figsizeScale, n_row*3.33*figsizeScale)
        f, axes = plt.subplots(n_row, n_col, figsize = figsize)
        
        if n_col == 1:
            
            y = Y[0]
            x = list(range(len(y)))
            axes.bar(x,y)
            
        else:
            
            x   = list(range(len(Y[0])))
            i_Y = 0
            for i in range(n_row):
                axes_x = axes[i] if n_row != 1 else axes
                for j in range(n_col):
                    ax_curr = axes_x[j] if n_col != 1 else axes
                    y = Y[i_Y]
                    ax_curr.bar(x,y)
                    i_Y = i_Y + 1
                    if i_Y >= len_Y:
                        plt.show()
                        break
    plt.show()
        
    return None
# <<
# ==================================================================================================================================
# END << FUNCTION << plot_bar
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
# END << SUBMODULE << traffic_light_classifier._plots_subpkg._plots_submod
# ==================================================================================================================================
