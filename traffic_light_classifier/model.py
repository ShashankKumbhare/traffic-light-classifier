
# ==================================================================================================================================
# START >> FILE INFO
# ==================================================================================================================================
# File        : traffic_light_classifier/model.py
# Author      : Shashank Kumbhare
# Date        : 09/20/2022
# email       : shashankkumbhare8@gmail.com
# Description : This file is a python module for python package 'traffic_light_classifier'.
# ==================================================================================================================================
# END << FILE INFO
# ==================================================================================================================================



# ==================================================================================================================================
# START >> MODULE >> traffic_light_classifier.model
# ==================================================================================================================================
# >>
"""
This module is created/used for/to.
MODULE description MODULE description MODULE description MODULE description
MODULE description MODULE description MODULE description MODULE description
MODULE description MODULE description.
"""

_name_mod = __name__.partition(".")[-1]
print(f"  + Adding module '{_name_mod}'...", )

# ==================================================================================
# START >> IMPORTS
# ==================================================================================
# >>
# from .__dependencies_subpkg__ import *
from .__constants_subpkg__    import *
from .__auxil_subpkg__        import *
from .__data_subpkg__         import *
from ._plots_subpkg           import *
from ._modify_images_subpkg   import *
from ._extract_feature_subpkg import *
from ._statistics_subpkg      import *
# <<
# ==================================================================================
# END << IMPORTS
# ==================================================================================


# ==================================================================================
# START >> EXPORTS
# ==================================================================================
# >>
__all__ = ["Model"]
# <<
# ==================================================================================
# END << EXPORTS
# ==================================================================================



# ==================================================================================================================================
# START >> CLASS >> Model
# ==================================================================================================================================
# >>
class Model:
    
    # ==============================================================================================================================
    # START >> METHOD >> __init__
    # ==============================================================================================================================
    # >>
    def __init__(self) :
        
        self.datasets           = Struct()
        self.datasets.train     = datasets.train
        self.datasets.test      = datasets.test
        self.compilation        = None
        self.prediction         = None
        self.prediction_dataset = None
    # <<
    # ==============================================================================================================================
    # END << METHOD << __init__
    # ==============================================================================================================================
    
    
    # ==============================================================================================================================
    # START >> METHOD >> compile
    # ==============================================================================================================================
    # >>
    def compile(self, show_analysis = False):
        
        print("\nCompilation in progress... Please wait !!")
        plot_enabled = True if show_analysis else False
        
        # Compilation Stage 1: Getting average image for red, yellow and green training images >>
        if show_analysis:
            print_heading(f"Compilation Stage 1: Getting average image for red, yellow and green training images")
        self.compilation                       = Struct()
        self.compilation.stg1_image_avg        = Struct()
        self.compilation.stg1_image_avg.red    = get_average_image(self.datasets.train.images_std.red,    False, "rgb", DEFAULT_NAME_IMAGE_AVG_RED)
        self.compilation.stg1_image_avg.yellow = get_average_image(self.datasets.train.images_std.yellow, False, "rgb", DEFAULT_NAME_IMAGE_AVG_YELLOW)
        self.compilation.stg1_image_avg.green  = get_average_image(self.datasets.train.images_std.green,  False, "rgb", DEFAULT_NAME_IMAGE_AVG_GREEN)
        
        # Plotting hsv channels >>
        if show_analysis:
            print("hsv image of average image")
            plot_channels   ( self.compilation.stg1_image_avg.red,    "hsv", name_image = DEFAULT_NAME_IMAGE_AVG_RED)
            plot_channels   ( self.compilation.stg1_image_avg.yellow, "hsv", name_image = DEFAULT_NAME_IMAGE_AVG_YELLOW)
            plot_channels   ( self.compilation.stg1_image_avg.green,  "hsv", name_image = DEFAULT_NAME_IMAGE_AVG_GREEN)
            update_user_done()
        
        # Compilation Stage 2: Getting region of high saturation in average red, yellow & green images >>
        if show_analysis:
            print_heading(f"Compilation Stage 2: Getting region of high saturation in average red, yellow & green images")
            print("Region of high saturation in average red, yellow & green images")
        self.compilation.stg2a_region_high_s                  = Struct()
        self.compilation.stg2b_image_avg_masked_region_high_s = Struct()
        
        self.compilation.stg2a_region_high_s.red,    self.compilation.stg2b_image_avg_masked_region_high_s.red    = get_region_high_avg_channel(self.compilation.stg1_image_avg.red,    "s", None, DEFAULT_SHAPE_AREA_SEARCH_AVG_IMAGE, plot_enabled, DEFAULT_NAME_IMAGE_AVG_RED )
        self.compilation.stg2a_region_high_s.yellow, self.compilation.stg2b_image_avg_masked_region_high_s.yellow = get_region_high_avg_channel(self.compilation.stg1_image_avg.yellow, "s", None, DEFAULT_SHAPE_AREA_SEARCH_AVG_IMAGE, plot_enabled, DEFAULT_NAME_IMAGE_AVG_YELLOW )
        self.compilation.stg2a_region_high_s.green,  self.compilation.stg2b_image_avg_masked_region_high_s.green  = get_region_high_avg_channel(self.compilation.stg1_image_avg.green,  "s", None, DEFAULT_SHAPE_AREA_SEARCH_AVG_IMAGE, plot_enabled, DEFAULT_NAME_IMAGE_AVG_GREEN )
        
        if show_analysis:
            print(f"Approximate region of high Saturation red    images: X = {self.compilation.stg2a_region_high_s.red[0]}, Y = {self.compilation.stg2a_region_high_s.red[1]}")
            print(f"Approximate region of high Saturation yellow images: X = {self.compilation.stg2a_region_high_s.yellow[0]}, Y = {self.compilation.stg2a_region_high_s.yellow[1]}")
            print(f"Approximate region of high Saturation green  images: X = {self.compilation.stg2a_region_high_s.green[0]}, Y = {self.compilation.stg2a_region_high_s.green[1]}")
            update_user_done()
        
        # Compilation Stage 3: Cropping all training images at their respective color's average image's high saturation region >>
        if show_analysis:
            print_heading(f"Compilation Stage 3: Cropping all training images at their respective color's average image's high saturation region")
        self.compilation.stg3_dataset_images_cropped_high_s_region        = Struct()
        self.compilation.stg3_dataset_images_cropped_high_s_region.red    = [ crop_image( image, self.compilation.stg2a_region_high_s.red[0],    self.compilation.stg2a_region_high_s.red[1] )    for image in self.datasets.train.images_std.red ]
        self.compilation.stg3_dataset_images_cropped_high_s_region.yellow = [ crop_image( image, self.compilation.stg2a_region_high_s.yellow[0], self.compilation.stg2a_region_high_s.yellow[1] ) for image in self.datasets.train.images_std.yellow ]
        self.compilation.stg3_dataset_images_cropped_high_s_region.green  = [ crop_image( image, self.compilation.stg2a_region_high_s.green[0],  self.compilation.stg2a_region_high_s.green[1] )  for image in self.datasets.train.images_std.green ]
        if show_analysis:
            printmd(f"A few cropped lights:", color = "skyblue", is_bold = True)
            images_red    = self.compilation.stg3_dataset_images_cropped_high_s_region.red[0:10]
            images_yellow = self.compilation.stg3_dataset_images_cropped_high_s_region.yellow[0:10]
            images_green  = self.compilation.stg3_dataset_images_cropped_high_s_region.green[0:10]
            plot_images(images_red,    figsizeScale = DEFAULT_FIGSIZESCALE_EXAMPLES)
            plot_images(images_yellow, figsizeScale = DEFAULT_FIGSIZESCALE_EXAMPLES)
            plot_images(images_green,  figsizeScale = DEFAULT_FIGSIZESCALE_EXAMPLES)
            update_user_done()
        
        # Compilation Stage 4: Locating lights in all images by using high saturation and high brightness regions >>
        if show_analysis:
            print_heading("Compilation Stage 4: Locating lights in all images by using high saturation and high brightness regions")
        self.compilation.stg4_locations_light        = Struct()
        self.compilation.stg4_locations_light.red    = [ get_region_high_avg_channel(image, "s", "v", DEFAULT_SHAPE_AREA_SEARCH_LIGHT, i=1, j=7)[0] for image in self.compilation.stg3_dataset_images_cropped_high_s_region.red ]
        self.compilation.stg4_locations_light.yellow = [ get_region_high_avg_channel(image, "s", "v", DEFAULT_SHAPE_AREA_SEARCH_LIGHT, i=1, j=7)[0] for image in self.compilation.stg3_dataset_images_cropped_high_s_region.yellow ]
        self.compilation.stg4_locations_light.green  = [ get_region_high_avg_channel(image, "s", "v", DEFAULT_SHAPE_AREA_SEARCH_LIGHT, i=1, j=7)[0] for image in self.compilation.stg3_dataset_images_cropped_high_s_region.green ]
        if show_analysis:
            locations_of_lights_red    = self.compilation.stg4_locations_light.red[0:5]
            locations_of_lights_yellow = self.compilation.stg4_locations_light.yellow[0:5]
            locations_of_lights_green  = self.compilation.stg4_locations_light.green[0:5]
            print(f"A few locations of lights:")
            print(f"Location of red    lights: {locations_of_lights_red}, etc")
            print(f"Location of yellow lights: {locations_of_lights_yellow}, etc")
            print(f"Location of green  lights: {locations_of_lights_green}, etc")
            update_user_done()
        
        # Compilation Stage 5: Cropping images at their respective light's position >>
        if show_analysis:
            print_heading(f"Compilation Stage 5: Cropping images at their respective light's position")
        self.compilation.stg5_dataset_images_light        = Struct()
        self.compilation.stg5_dataset_images_light.red    = [ crop_image(image, loc[0], loc[1]) for image, loc in zip(self.compilation.stg3_dataset_images_cropped_high_s_region.red,    self.compilation.stg4_locations_light.red) ]
        self.compilation.stg5_dataset_images_light.yellow = [ crop_image(image, loc[0], loc[1]) for image, loc in zip(self.compilation.stg3_dataset_images_cropped_high_s_region.yellow, self.compilation.stg4_locations_light.yellow) ]
        self.compilation.stg5_dataset_images_light.green  = [ crop_image(image, loc[0], loc[1]) for image, loc in zip(self.compilation.stg3_dataset_images_cropped_high_s_region.green,  self.compilation.stg4_locations_light.green) ]
        if show_analysis:
            printmd(f"A few located lights:", color = "skyblue", is_bold = True)
            images_red    = self.compilation.stg5_dataset_images_light.red[0:10]
            images_yellow = self.compilation.stg5_dataset_images_light.yellow[0:10]
            images_green  = self.compilation.stg5_dataset_images_light.green[0:10]
            plot_images(images_red,    figsizeScale = DEFAULT_FIGSIZESCALE_EXAMPLES)
            plot_images(images_yellow, figsizeScale = DEFAULT_FIGSIZESCALE_EXAMPLES)
            plot_images(images_green,  figsizeScale = DEFAULT_FIGSIZESCALE_EXAMPLES)
            update_user_done()
        
        # Compilation Stage 6: Getting images of average red light, average yellow light & average green light... >>
        if show_analysis:
            print_heading("Compilation Stage 6: Getting images of average red light, average yellow light & average green light")
        self.compilation.stg6_image_light_avg        = Struct()
        self.compilation.stg6_image_light_avg.red    = get_average_image(self.compilation.stg5_dataset_images_light.red,    plot_enabled, "hsv", DEFAULT_NAME_IMAGE_AVG_LIGHT_RED)
        self.compilation.stg6_image_light_avg.yellow = get_average_image(self.compilation.stg5_dataset_images_light.yellow, plot_enabled, "hsv", DEFAULT_NAME_IMAGE_AVG_LIGHT_YELLOW)
        self.compilation.stg6_image_light_avg.green  = get_average_image(self.compilation.stg5_dataset_images_light.green,  plot_enabled, "hsv", DEFAULT_NAME_IMAGE_AVG_LIGHT_GREEN)
        if show_analysis:
            update_user_done()
        
        # Compilation Stage 7: Getting hues, saturations and brightnesses of average red light, average yellow light & average green light >>
        if show_analysis:
            print_heading("Compilation Stage 7: Getting hues, saturations and brightnesses of average red light, average yellow light & average green light")
        
        # Compilation Stage 7a: Getting hues of average red light, average yellow light & average green light >>
        if show_analysis:
            print_heading("Compilation Stage 7a: Getting hues of average red light, average yellow light & average green light", heading_level = DEFAULT_SUBHEADING_LEVEL)
        self.compilation.stg7a_hue_avg_light        = Struct()
        self.compilation.stg7a_hue_avg_light.red    = Struct()
        self.compilation.stg7a_hue_avg_light.yellow = Struct()
        self.compilation.stg7a_hue_avg_light.green  = Struct()
        self.compilation.stg7a_hue_avg_light.red.mu,    self.compilation.stg7a_hue_avg_light.red.sigma,    self.compilation.stg7a_hue_avg_light.red.dist    = get_distribution_of_channel ( self.compilation.stg6_image_light_avg.red,    "hsv", ch=0, remove_outliers = True )
        self.compilation.stg7a_hue_avg_light.yellow.mu, self.compilation.stg7a_hue_avg_light.yellow.sigma, self.compilation.stg7a_hue_avg_light.yellow.dist = get_distribution_of_channel ( self.compilation.stg6_image_light_avg.yellow, "hsv", ch=0, remove_outliers = True )
        self.compilation.stg7a_hue_avg_light.green.mu,  self.compilation.stg7a_hue_avg_light.green.sigma,  self.compilation.stg7a_hue_avg_light.green.dist  = get_distribution_of_channel ( self.compilation.stg6_image_light_avg.green,  "hsv", ch=0, remove_outliers = True )
        if show_analysis:
            print(f"Distribution of hue in red   -lights:     mu_hue_red    = {self.compilation.stg7a_hue_avg_light.red.mu:7.3f},     sigma_hue_red    = {self.compilation.stg7a_hue_avg_light.red.sigma:4.3f}")
            print(f"Distribution of hue in yellow-lights:     mu_hue_yellow = {self.compilation.stg7a_hue_avg_light.yellow.mu:7.3f},     sigma_hue_yellow = {self.compilation.stg7a_hue_avg_light.yellow.sigma:4.3f}")
            print(f"Distribution of hue in green -lights:     mu_hue_green  = {self.compilation.stg7a_hue_avg_light.green.mu:7.3f},     sigma_hue_green  = {self.compilation.stg7a_hue_avg_light.green.sigma:4.3f}")
            _, axes = plt.subplots( 1, 1, figsize = (20,5) )
            axes.set_title("Distribution of hues in red, yellow and green lights")
            x = np.linspace(0,179,180)
            (a_red,    b_red)    = get_shape_params_truncnorm( 0, 179, self.compilation.stg7a_hue_avg_light.red.mu )
            (a_yellow, b_yellow) = get_shape_params_truncnorm( 0, 179, self.compilation.stg7a_hue_avg_light.yellow.mu )
            (a_green,  b_green)  = get_shape_params_truncnorm( 0, 179, self.compilation.stg7a_hue_avg_light.green.mu )
            y_red    = truncnorm.pdf(x, a_red,    b_red,    self.compilation.stg7a_hue_avg_light.red.mu,    self.compilation.stg7a_hue_avg_light.red.sigma)
            y_yellow = truncnorm.pdf(x, a_yellow, b_yellow, self.compilation.stg7a_hue_avg_light.yellow.mu, self.compilation.stg7a_hue_avg_light.yellow.sigma)
            y_green  = truncnorm.pdf(x, a_green,  b_green,  self.compilation.stg7a_hue_avg_light.green.mu,  self.compilation.stg7a_hue_avg_light.green.sigma)
            axes.bar(x, y_red,    alpha = DEFAULT_ALPHA, color = DEFAULT_COLOR_RED,    edgecolor = DEFAULT_COLOR_RED,    label = DEFAULT_LABEL_LIGHTS_RED)
            axes.bar(x, y_yellow, alpha = DEFAULT_ALPHA, color = DEFAULT_COLOR_YELLOW, edgecolor = DEFAULT_COLOR_YELLOW, label = DEFAULT_LABEL_LIGHTS_YELLOW)
            axes.bar(x, y_green,  alpha = DEFAULT_ALPHA, color = DEFAULT_COLOR_GREEN,  edgecolor = DEFAULT_COLOR_GREEN,  label = DEFAULT_LABEL_LIGHTS_GREEN)
            axes.legend()
            axes.set_xlim([-10, 190])
            axes.xaxis.set_major_locator(MultipleLocator(20))
            axes.xaxis.set_tick_params(labelbottom = True)
            plt.show()
            update_user_done()
        
        # Compilation Stage 7b: Getting saturations of average red light, average yellow light & average green light >>
        if show_analysis:
            print_heading("Compilation Stage 7b: Getting saturations of average red light, average yellow light & average green light", heading_level = DEFAULT_SUBHEADING_LEVEL)
        self.compilation.stg7b_sat_avg_light        = Struct()
        self.compilation.stg7b_sat_avg_light.red    = Struct()
        self.compilation.stg7b_sat_avg_light.yellow = Struct()
        self.compilation.stg7b_sat_avg_light.green  = Struct()
        self.compilation.stg7b_sat_avg_light.red.mu,    self.compilation.stg7b_sat_avg_light.red.sigma,    self.compilation.stg7b_sat_avg_light.red.dist    = get_distribution_of_channel ( self.compilation.stg6_image_light_avg.red,    "hsv", ch=1, remove_outliers = True )
        self.compilation.stg7b_sat_avg_light.yellow.mu, self.compilation.stg7b_sat_avg_light.yellow.sigma, self.compilation.stg7b_sat_avg_light.yellow.dist = get_distribution_of_channel ( self.compilation.stg6_image_light_avg.yellow, "hsv", ch=1, remove_outliers = True )
        self.compilation.stg7b_sat_avg_light.green.mu,  self.compilation.stg7b_sat_avg_light.green.sigma,  self.compilation.stg7b_sat_avg_light.green.dist  = get_distribution_of_channel ( self.compilation.stg6_image_light_avg.green,  "hsv", ch=1, remove_outliers = True )
        if show_analysis:
            print(f"Distribution of saturation in red   -lights:     mu_sat_red    = {self.compilation.stg7b_sat_avg_light.red.mu:7.3f},     sigma_sat_red    = {self.compilation.stg7b_sat_avg_light.red.sigma:4.3f}")
            print(f"Distribution of saturation in yellow-lights:     mu_sat_yellow = {self.compilation.stg7b_sat_avg_light.yellow.mu:7.3f},     sigma_sat_yellow = {self.compilation.stg7b_sat_avg_light.yellow.sigma:4.3f}")
            print(f"Distribution of saturation in green -lights:     mu_sat_green  = {self.compilation.stg7b_sat_avg_light.green.mu:7.3f},     sigma_sat_green  = {self.compilation.stg7b_sat_avg_light.green.sigma:4.3f}")
            _, axes = plt.subplots( 1, 1, figsize = (20,5) )
            axes.set_title("Distribution of saturations in red, yellow and green lights")
            x = np.linspace(0,254,255)
            (a_red,    b_red)    = get_shape_params_truncnorm( 0, 255, self.compilation.stg7b_sat_avg_light.red.mu )
            (a_yellow, b_yellow) = get_shape_params_truncnorm( 0, 255, self.compilation.stg7b_sat_avg_light.yellow.mu )
            (a_green,  b_green)  = get_shape_params_truncnorm( 0, 255, self.compilation.stg7b_sat_avg_light.green.mu )
            y_red    = truncnorm.pdf(x, a_red,    b_red,    self.compilation.stg7b_sat_avg_light.red.mu,    self.compilation.stg7b_sat_avg_light.red.sigma)
            y_yellow = truncnorm.pdf(x, a_yellow, b_yellow, self.compilation.stg7b_sat_avg_light.yellow.mu, self.compilation.stg7b_sat_avg_light.yellow.sigma)
            y_green  = truncnorm.pdf(x, a_green,  b_green,  self.compilation.stg7b_sat_avg_light.green.mu,  self.compilation.stg7b_sat_avg_light.green.sigma)
            axes.bar(x, y_red,    alpha = DEFAULT_ALPHA, color = DEFAULT_COLOR_RED,    edgecolor = DEFAULT_COLOR_RED,    label = DEFAULT_LABEL_LIGHTS_RED)
            axes.bar(x, y_yellow, alpha = DEFAULT_ALPHA, color = DEFAULT_COLOR_YELLOW, edgecolor = DEFAULT_COLOR_YELLOW, label = DEFAULT_LABEL_LIGHTS_YELLOW)
            axes.bar(x, y_green,  alpha = DEFAULT_ALPHA, color = DEFAULT_COLOR_GREEN,  edgecolor = DEFAULT_COLOR_GREEN,  label = DEFAULT_LABEL_LIGHTS_GREEN)
            axes.legend()
            axes.set_xlim([-10, 265])
            axes.xaxis.set_major_locator(MultipleLocator(20))
            axes.xaxis.set_tick_params(labelbottom = True)
            plt.show()
            update_user_done()
        
        # Compilation Stage 7c: Getting brightnesses of average red light, average yellow light & average green light >>
        if show_analysis:
            print_heading("Compilation Stage 7c: Getting brightnesses of average red light, average yellow light & average green light", heading_level = DEFAULT_SUBHEADING_LEVEL)
        self.compilation.stg7c_brt_avg_light        = Struct()
        self.compilation.stg7c_brt_avg_light.red    = Struct()
        self.compilation.stg7c_brt_avg_light.yellow = Struct()
        self.compilation.stg7c_brt_avg_light.green  = Struct()
        self.compilation.stg7c_brt_avg_light.red.mu,    self.compilation.stg7c_brt_avg_light.red.sigma,    self.compilation.stg7c_brt_avg_light.red.dist    = get_distribution_of_channel ( self.compilation.stg6_image_light_avg.red,    "hsv", ch=2, remove_outliers = True )
        self.compilation.stg7c_brt_avg_light.yellow.mu, self.compilation.stg7c_brt_avg_light.yellow.sigma, self.compilation.stg7c_brt_avg_light.yellow.dist = get_distribution_of_channel ( self.compilation.stg6_image_light_avg.yellow, "hsv", ch=2, remove_outliers = True )
        self.compilation.stg7c_brt_avg_light.green.mu,  self.compilation.stg7c_brt_avg_light.green.sigma,  self.compilation.stg7c_brt_avg_light.green.dist  = get_distribution_of_channel ( self.compilation.stg6_image_light_avg.green,  "hsv", ch=2, remove_outliers = True )
        if show_analysis:
            print(f"Distribution of brightness in red   -lights:     mu_brt_red    = {self.compilation.stg7c_brt_avg_light.red.mu:7.3f},     sigma_brt_red    = {self.compilation.stg7c_brt_avg_light.red.sigma:4.3f}")
            print(f"Distribution of brightness in yellow-lights:     mu_brt_yellow = {self.compilation.stg7c_brt_avg_light.yellow.mu:7.3f},     sigma_brt_yellow = {self.compilation.stg7c_brt_avg_light.yellow.sigma:4.3f}")
            print(f"Distribution of brightness in green -lights:     mu_brt_green  = {self.compilation.stg7c_brt_avg_light.green.mu:7.3f},     sigma_brt_green  = {self.compilation.stg7c_brt_avg_light.green.sigma:4.3f}")
            _, axes = plt.subplots( 1, 1, figsize = (20,5) )
            axes.set_title("Distribution of brightness in red, yellow and green lights")
            x = np.linspace(0,254,255)
            (a_red,    b_red)    = get_shape_params_truncnorm( 0, 255, self.compilation.stg7c_brt_avg_light.red.mu )
            (a_yellow, b_yellow) = get_shape_params_truncnorm( 0, 255, self.compilation.stg7c_brt_avg_light.yellow.mu )
            (a_green,  b_green)  = get_shape_params_truncnorm( 0, 255, self.compilation.stg7c_brt_avg_light.green.mu )
            y_red    = truncnorm.pdf(x, a_red,    b_red,    self.compilation.stg7c_brt_avg_light.red.mu,    self.compilation.stg7c_brt_avg_light.red.sigma)
            y_yellow = truncnorm.pdf(x, a_yellow, b_yellow, self.compilation.stg7c_brt_avg_light.yellow.mu, self.compilation.stg7c_brt_avg_light.yellow.sigma)
            y_green  = truncnorm.pdf(x, a_green,  b_green,  self.compilation.stg7c_brt_avg_light.green.mu,  self.compilation.stg7c_brt_avg_light.green.sigma)
            axes.bar(x, y_red,    alpha = DEFAULT_ALPHA, color = DEFAULT_COLOR_RED,    edgecolor = DEFAULT_COLOR_RED,    label = DEFAULT_LABEL_LIGHTS_RED)
            axes.bar(x, y_yellow, alpha = DEFAULT_ALPHA, color = DEFAULT_COLOR_YELLOW, edgecolor = DEFAULT_COLOR_YELLOW, label = DEFAULT_LABEL_LIGHTS_YELLOW)
            axes.bar(x, y_green,  alpha = DEFAULT_ALPHA, color = DEFAULT_COLOR_GREEN,  edgecolor = DEFAULT_COLOR_GREEN,  label = DEFAULT_LABEL_LIGHTS_GREEN)
            axes.legend()
            axes.set_xlim([-10, 265])
            axes.xaxis.set_major_locator(MultipleLocator(20))
            axes.xaxis.set_tick_params(labelbottom = True)
            plt.show()
            update_user_done()
        
        # Compilation Stage 8: Optimizing classifier's metric's parameters for red, yellow and green lights >>
        if show_analysis:
            print_heading("Compilation Stage 8: Optimizing classifier's metric's parameters for red, yellow and green lights")
            printmd(DEFAULT_MODEL_PROBABILISTIC)
            printmd("The current compilation stage is optimizing the parameters $a$ & $b$ for maximum accuracy.", is_bold = True)
            printmd("Please wait...")
        
        params_initial      = [1, 3]
        result_optimization = optimize.minimize(self._get_neg_accuracy_dataset_train, x0 = params_initial, args = (True, self.datasets.train.images_std.all, self.datasets.train.labels_std), method = 'COBYLA')
        self.compilation.stg8_params_optimised = result_optimization.x
        
        if show_analysis:
            printmd(f"Optimization complete !!\n", is_bold = True)
            printmd(f"Optimized parameters: a = {self.compilation.stg8_params_optimised[0]:7.6f}, b = {self.compilation.stg8_params_optimised[1]:7.6f}", color = "green")
            update_user_done()
        
        # Compilation Stage 9: Prediction analysis & accuracy of training dataset for classifier's optimized parameters >>
        if show_analysis:
            print_heading("Compilation Stage 9: Prediction analysis & accuracy of training dataset for classifier's optimized parameters")
        
        self.compilation.stg9a_dataset_train_analysis = Struct()
        self.compilation.stg9b_misclassified          = Struct()
        self.compilation.stg9c_accuracy_train         = Struct()
        
        self.compilation.stg9c_accuracy_train.red,    self.compilation.stg9a_dataset_train_analysis.red,    self.compilation.stg9b_misclassified.red    = self._get_accuracy_dataset_train(self.datasets.train.images_std.red,     [1,0,0], show_analysis, "red")
        self.compilation.stg9c_accuracy_train.yellow, self.compilation.stg9a_dataset_train_analysis.yellow, self.compilation.stg9b_misclassified.yellow = self._get_accuracy_dataset_train(self.datasets.train.images_std.yellow,  [0,1,0], show_analysis, "yellow")
        self.compilation.stg9c_accuracy_train.green,  self.compilation.stg9a_dataset_train_analysis.green,  self.compilation.stg9b_misclassified.green  = self._get_accuracy_dataset_train(self.datasets.train.images_std.green,   [0,0,1], show_analysis, "green")
        self.compilation.stg9c_accuracy_train.all     = -self._get_neg_accuracy_dataset_train(params = None, is_optimization = False, images_std = self.datasets.train.images_std.all, labels_true_std = self.datasets.train.labels_std, show_analysis = show_analysis)
        
        if show_analysis:
            update_user_done()
        print_heading(f"Compilation complete !!", heading_level = DEFAULT_HEADING_LEVEL, color = "green")
        
        return None
    # <<
    # ==============================================================================================================================
    # END << METHOD << compile
    # ==============================================================================================================================
    
    
    # ==============================================================================================================================
    # START >> METHOD >> _predict
    # ==============================================================================================================================
    # >>
    def _predict    ( self
                    , image_rgb
                    , label_true         = None
                    , params             = None
                    , is_optimization    = False
                    , show_analysis      = False
                    , show_probabilities = False
                    ) :
        
        self.prediction = Struct()
        self.prediction.image_input = image_rgb
        if label_true is not None: self.prediction.label_true  = label_true
        
        plot_enabled = True if show_analysis else False
        
        if is_optimization:
            a = params[0]
            b = params[1]
        else:
            a = self.compilation.stg8_params_optimised[0]
            b = self.compilation.stg8_params_optimised[1]
        
        # Prediction Stage 1: Cropping image at model's optimal high saturation region for red, yellow, green light's position >>
        if show_analysis:
            print_heading(f"Prediction Stage 1: Cropping image at model's optimal high saturation region for red, yellow, green light's position")
            print("This procedure uses saturation features.")
        self.prediction.stg1_image_high_s_region        = Struct()
        self.prediction.stg1_image_high_s_region.red    = crop_image( self.prediction.image_input, self.compilation.stg2a_region_high_s.red[0],    self.compilation.stg2a_region_high_s.red[1],    plot_enabled, titles = (DEFAULT_NAME_IMAGE_INPUT, "high S region of \n model's red images") )
        self.prediction.stg1_image_high_s_region.yellow = crop_image( self.prediction.image_input, self.compilation.stg2a_region_high_s.yellow[0], self.compilation.stg2a_region_high_s.yellow[1], plot_enabled, titles = (DEFAULT_NAME_IMAGE_INPUT, "high S region of \n model's yellow images") )
        self.prediction.stg1_image_high_s_region.green  = crop_image( self.prediction.image_input, self.compilation.stg2a_region_high_s.green[0],  self.compilation.stg2a_region_high_s.green[1],  plot_enabled, titles = (DEFAULT_NAME_IMAGE_INPUT, "high S region of \n model's green images") )
        if show_analysis:
            update_user_done()
        
        # Prediction Stage 2: Locating light in model's optimal region of red, yellow, green lights by using high saturation and high brightness regions >>
        if show_analysis:
            print_heading("Prediction Stage 2: Locating light in model's optimal regions of red, yellow, green lights")
            print("This procedure uses saturation and brightness features.")
        self.prediction.stg2_locations_light           = Struct()
        self.prediction.stg2_locations_light.red,    _ = get_region_high_avg_channel(self.prediction.stg1_image_high_s_region.red,    "s", "v", DEFAULT_SHAPE_AREA_SEARCH_LIGHT, plot_enabled, DEFAULT_NAME_IMAGE_INPUT + " in red region",    a, b)
        self.prediction.stg2_locations_light.yellow, _ = get_region_high_avg_channel(self.prediction.stg1_image_high_s_region.yellow, "s", "v", DEFAULT_SHAPE_AREA_SEARCH_LIGHT, plot_enabled, DEFAULT_NAME_IMAGE_INPUT + " in yellow region", a, b)
        self.prediction.stg2_locations_light.green,  _ = get_region_high_avg_channel(self.prediction.stg1_image_high_s_region.green,  "s", "v", DEFAULT_SHAPE_AREA_SEARCH_LIGHT, plot_enabled, DEFAULT_NAME_IMAGE_INPUT + " in green region",  a, b)
        if show_analysis:
            update_user_done()
        
        # Prediction Stage 3: Cropping image at model's optimal region of red, yellow, green lights >>
        if show_analysis:
            print_heading(f"Prediction Stage 3: Cropping image at model's optimal region of red, yellow, green lights")
        self.prediction.stg3_image_light        = Struct()
        self.prediction.stg3_image_light.red    = crop_image(self.prediction.stg1_image_high_s_region.red,    self.prediction.stg2_locations_light.red[0],    self.prediction.stg2_locations_light.red[1] )
        self.prediction.stg3_image_light.yellow = crop_image(self.prediction.stg1_image_high_s_region.yellow, self.prediction.stg2_locations_light.yellow[0], self.prediction.stg2_locations_light.yellow[1] )
        self.prediction.stg3_image_light.green  = crop_image(self.prediction.stg1_image_high_s_region.green,  self.prediction.stg2_locations_light.green[0],  self.prediction.stg2_locations_light.green[1] )
        if show_analysis:
            update_user_done()
        
        # Prediction Stage 4: Extracting model's red, yellow, green light's colors from the respective cropped parts of the image >>
        if show_analysis:
            print_heading(f"Prediction Stage 4: Extracting model's red, yellow, green light's colors from the respective cropped parts of the input image")
        
        self.prediction.stg4_image_colors_extracted = Struct()
        sigma = 15
        hue_lower_red    = self.compilation.stg7a_hue_avg_light.red.mu    - 0.5*sigma
        hue_upper_red    = self.compilation.stg7a_hue_avg_light.red.mu    + 0.5*sigma
        hue_lower_yellow = self.compilation.stg7a_hue_avg_light.yellow.mu - 0.4*sigma
        hue_upper_yellow = self.compilation.stg7a_hue_avg_light.yellow.mu + 0.5*sigma
        hue_lower_green  = self.compilation.stg7a_hue_avg_light.green.mu  -   1*sigma
        hue_upper_green  = self.compilation.stg7a_hue_avg_light.green.mu  + 0.6*sigma
        
        self.prediction.stg4_image_colors_extracted.red    =  get_colors_from_image( self.prediction.stg3_image_light.red,    (hue_lower_red,    hue_upper_red),    plot_enabled = plot_enabled, titles = (DEFAULT_NAME_LOCATED_LIGHT + " in red region",    "model's red colors extracted \n from "    + DEFAULT_NAME_LOCATED_LIGHT) )
        self.prediction.stg4_image_colors_extracted.yellow =  get_colors_from_image( self.prediction.stg3_image_light.yellow, (hue_lower_yellow, hue_upper_yellow), plot_enabled = plot_enabled, titles = (DEFAULT_NAME_LOCATED_LIGHT + " in yellow region", "model's yellow colors extracted \n from " + DEFAULT_NAME_LOCATED_LIGHT) )
        self.prediction.stg4_image_colors_extracted.green  =  get_colors_from_image( self.prediction.stg3_image_light.green,  (hue_lower_green,  hue_upper_green),  plot_enabled = plot_enabled, titles = (DEFAULT_NAME_LOCATED_LIGHT + " in green region",  "model's green colors extracted \n from "  + DEFAULT_NAME_LOCATED_LIGHT) )
        
        if show_analysis:
            update_user_done()
        
        # Prediction Stage 5: Getting hues, saturations & brightnesses from the extracted colors at model's optimal region of red, yellow, green lights >>
        if show_analysis:
            print_heading("Prediction Stage 5a: Getting hues from the extracted colors at model's optimal region of red, yellow, green lights")
        
        # Prediction Stage 5a: Getting hues from the extracted colors at model's optimal region of red, yellow, green lights >>
        if show_analysis:
            print_heading("Prediction Stage 5a: Getting hues from the extracted colors at model's optimal region of red, yellow, green lights", heading_level = DEFAULT_SUBHEADING_LEVEL)
        self.prediction.stg5a_hue_input_light        = Struct()
        self.prediction.stg5a_hue_input_light.red    = Struct()
        self.prediction.stg5a_hue_input_light.yellow = Struct()
        self.prediction.stg5a_hue_input_light.green  = Struct()
        self.prediction.stg5a_hue_input_light.red.mu,    self.prediction.stg5a_hue_input_light.red.sigma,    self.prediction.stg5a_hue_input_light.red.dist    = get_distribution_of_channel( self.prediction.stg4_image_colors_extracted.red,    "hsv", ch=0, drop_zeros = True )
        self.prediction.stg5a_hue_input_light.yellow.mu, self.prediction.stg5a_hue_input_light.yellow.sigma, self.prediction.stg5a_hue_input_light.yellow.dist = get_distribution_of_channel( self.prediction.stg4_image_colors_extracted.yellow, "hsv", ch=0, drop_zeros = True )
        self.prediction.stg5a_hue_input_light.green.mu,  self.prediction.stg5a_hue_input_light.green.sigma,  self.prediction.stg5a_hue_input_light.green.dist  = get_distribution_of_channel( self.prediction.stg4_image_colors_extracted.green,  "hsv", ch=0, drop_zeros = True )
        if show_analysis:
            print(f"Distribution of hues at model's red    light region:     mu_hue_red    = {self.prediction.stg5a_hue_input_light.red.mu:7.3f},     sigma_hue_red    = {self.prediction.stg5a_hue_input_light.red.sigma:4.3f}")
            print(f"Distribution of hues at model's yellow light region:     mu_hue_yellow = {self.prediction.stg5a_hue_input_light.yellow.mu:7.3f},     sigma_hue_yellow = {self.prediction.stg5a_hue_input_light.yellow.sigma:4.3f}")
            print(f"Distribution of hues at model's green  light region:     mu_hue_green  = {self.prediction.stg5a_hue_input_light.green.mu:7.3f},     sigma_hue_green  = {self.prediction.stg5a_hue_input_light.green.sigma:4.3f}")
            update_user_done()
        
        # Prediction Stage 5b: Getting saturations from the extracted colors at model's optimal region of red, yellow, green lights >>
        if show_analysis:
            print_heading("Prediction Stage 5b: Getting saturations from the extracted colors at model's optimal region of red, yellow, green lights", heading_level = DEFAULT_SUBHEADING_LEVEL)
        self.prediction.stg5b_sat_input_light        = Struct()
        self.prediction.stg5b_sat_input_light.red    = Struct()
        self.prediction.stg5b_sat_input_light.yellow = Struct()
        self.prediction.stg5b_sat_input_light.green  = Struct()
        self.prediction.stg5b_sat_input_light.red.mu,    self.prediction.stg5b_sat_input_light.red.sigma,    self.prediction.stg5b_sat_input_light.red.dist    = get_distribution_of_channel( self.prediction.stg4_image_colors_extracted.red,    "hsv", ch=1 )
        self.prediction.stg5b_sat_input_light.yellow.mu, self.prediction.stg5b_sat_input_light.yellow.sigma, self.prediction.stg5b_sat_input_light.yellow.dist = get_distribution_of_channel( self.prediction.stg4_image_colors_extracted.yellow, "hsv", ch=1 )
        self.prediction.stg5b_sat_input_light.green.mu,  self.prediction.stg5b_sat_input_light.green.sigma,  self.prediction.stg5b_sat_input_light.green.dist  = get_distribution_of_channel( self.prediction.stg4_image_colors_extracted.green,  "hsv", ch=1 )
        if show_analysis:
            print(f"Distribution of saturations at model's red    light region:     mu_sat_red    = {self.prediction.stg5b_sat_input_light.red.mu:7.3f},     sigma_sat_red    = {self.prediction.stg5b_sat_input_light.red.sigma:4.3f}")
            print(f"Distribution of saturations at model's yellow light region:     mu_sat_yellow = {self.prediction.stg5b_sat_input_light.yellow.mu:7.3f},     sigma_sat_yellow = {self.prediction.stg5b_sat_input_light.yellow.sigma:4.3f}")
            print(f"Distribution of saturations at model's green  light region:     mu_sat_green  = {self.prediction.stg5b_sat_input_light.green.mu:7.3f},     sigma_sat_green  = {self.prediction.stg5b_sat_input_light.green.sigma:4.3f}")
            update_user_done()
            
        # Prediction Stage 5c: Getting brightnesses from the extracted colors at model's optimal region of red, yellow, green lights >>
        if show_analysis:
            print_heading("Prediction Stage 5c: Getting brightnesses from the extracted colors at model's optimal region of red, yellow, green lights", heading_level = DEFAULT_SUBHEADING_LEVEL)
        self.prediction.stg5c_brt_input_light        = Struct()
        self.prediction.stg5c_brt_input_light.red    = Struct()
        self.prediction.stg5c_brt_input_light.yellow = Struct()
        self.prediction.stg5c_brt_input_light.green  = Struct()
        self.prediction.stg5c_brt_input_light.red.mu,    self.prediction.stg5c_brt_input_light.red.sigma,    self.prediction.stg5c_brt_input_light.red.dist    = get_distribution_of_channel( self.prediction.stg4_image_colors_extracted.red,    "hsv", ch=2 )
        self.prediction.stg5c_brt_input_light.yellow.mu, self.prediction.stg5c_brt_input_light.yellow.sigma, self.prediction.stg5c_brt_input_light.yellow.dist = get_distribution_of_channel( self.prediction.stg4_image_colors_extracted.yellow, "hsv", ch=2 )
        self.prediction.stg5c_brt_input_light.green.mu,  self.prediction.stg5c_brt_input_light.green.sigma,  self.prediction.stg5c_brt_input_light.green.dist  = get_distribution_of_channel( self.prediction.stg4_image_colors_extracted.green,  "hsv", ch=2 )
        if show_analysis:
            print(f"Distribution of brightnesses at model's red    light region:     mu_brt_red    = {self.prediction.stg5c_brt_input_light.red.mu:7.3f},     sigma_brt_red    = {self.prediction.stg5c_brt_input_light.red.sigma:4.3f}")
            print(f"Distribution of brightnesses at model's yellow light region:     mu_brt_yellow = {self.prediction.stg5c_brt_input_light.yellow.mu:7.3f},     sigma_brt_yellow = {self.prediction.stg5c_brt_input_light.yellow.sigma:4.3f}")
            print(f"Distribution of brightnesses at model's green  light region:     mu_brt_green  = {self.prediction.stg5c_brt_input_light.green.mu:7.3f},     sigma_brt_green  = {self.prediction.stg5c_brt_input_light.green.sigma:4.3f}")
            update_user_done()
        
        # Prediction Stage 6: Calculating probabilities of image being red, yellow and green >>
        if show_analysis:
            print_heading("Prediction Stage 6: Calculating probabilities of image being red, yellow and green")
            printmd(DEFAULT_MODEL_PROBABILISTIC)
        
        self.prediction.stg6_probabilities = Struct()
        
        if self.prediction.stg5b_sat_input_light.red.mu    == 0: self.prediction.stg5b_sat_input_light.red.mu    = 1
        if self.prediction.stg5b_sat_input_light.yellow.mu == 0: self.prediction.stg5b_sat_input_light.yellow.mu = 1
        if self.prediction.stg5b_sat_input_light.green.mu  == 0: self.prediction.stg5b_sat_input_light.green.mu  = 1
        if self.prediction.stg5c_brt_input_light.red.mu    == 0: self.prediction.stg5c_brt_input_light.red.mu    = 1
        if self.prediction.stg5c_brt_input_light.yellow.mu == 0: self.prediction.stg5c_brt_input_light.yellow.mu = 1
        if self.prediction.stg5c_brt_input_light.green.mu  == 0: self.prediction.stg5c_brt_input_light.green.mu  = 1
        
        strength_red    = self.prediction.stg5b_sat_input_light.red.mu**a    * self.prediction.stg5c_brt_input_light.red.mu**b
        strength_yellow = self.prediction.stg5b_sat_input_light.yellow.mu**a * self.prediction.stg5c_brt_input_light.yellow.mu**b
        strength_green  = self.prediction.stg5b_sat_input_light.green.mu**a  * self.prediction.stg5c_brt_input_light.green.mu**b
        deno            = strength_red + strength_yellow + strength_green
        self.prediction.stg6_probabilities.image_being_red    = strength_red    / deno
        self.prediction.stg6_probabilities.image_being_yellow = strength_yellow / deno
        self.prediction.stg6_probabilities.image_being_green  = strength_green  / deno
        
        if show_analysis or show_probabilities:
            print(f"Probability of image being red    = {self.prediction.stg6_probabilities.image_being_red:4.3f}")
            print(f"Probability of image being yellow = {self.prediction.stg6_probabilities.image_being_yellow:4.3f}")
            print(f"Probability of image being green  = {self.prediction.stg6_probabilities.image_being_green:4.3f}")
            if show_analysis: update_user_done()
        
        # Prediction Stage 7: Predicting image >>
        if show_analysis:
            print_heading("Prediction Stage 7: Predicting image")
        probabilities = [self.prediction.stg6_probabilities.image_being_red, self.prediction.stg6_probabilities.image_being_yellow, self.prediction.stg6_probabilities.image_being_green]
        i_max_prob = np.argmax(probabilities)
        if i_max_prob == 0:
            label = "red"
        elif i_max_prob == 1:
            label = "yellow"
        elif i_max_prob == 2:
            label = "green"
        
        if show_analysis or show_probabilities:
            print(f"This image is '{label.upper()}'")
            _, axes = plt.subplots( 1, 2, figsize = (10,3) )
            axes[0].set_title("input image")
            axes[0].imshow(image_rgb)
            x      = ["red", "yellow", "green"]
            colors = ['red', 'orange', 'mediumspringgreen']
            axes[1].set_title("Probabilities")
            axes[1].bar(x, probabilities, color = colors, edgecolor = colors)
            plt.show()
            if show_analysis: update_user_done()
        
        label_pred = one_hot_encode(label)
        self.prediction.stg7_label_predicted     = label_pred
        self.prediction.stg7_label_predicted_str = label
        
        return label_pred, self.prediction
    # <<
    # ==============================================================================================================================
    # END << METHOD << _predict
    # ==============================================================================================================================
    
    
    # ==============================================================================================================================
    # START >> METHOD >> predict
    # ==============================================================================================================================
    # >>
    def predict(self, image_rgb, show_analysis = False, show_probabilities = False):
        
        label_pred, _ = self._predict(image_rgb, show_analysis = show_analysis, show_probabilities = show_probabilities)
        
        return label_pred
    # <<
    # ==============================================================================================================================
    # END << METHOD << predict
    # ==============================================================================================================================
    
    
    # ==============================================================================================================================
    # START >> METHOD >> predict_dataset
    # ==============================================================================================================================
    # >>
    def predict_dataset(self, images_std, labels_true_std = None):
        
        #  Creating a struct object to store results >>
        self.prediction_dataset = Struct()
        if labels_true_std is not None: self.prediction_dataset.labels_true = labels_true_std
        
        # Predicting and storing results >>
        results_pred                             = [ self._predict(image_std, label_true) for image_std, label_true in zip(images_std, labels_true_std) ]
        self.prediction_dataset.labels_pred      = [ result_pred[0] for result_pred in results_pred ]
        self.prediction_dataset.dataset_analysis = [ result_pred[1] for result_pred in results_pred ]
        
        # If true labels are probided then store misclassified >>
        if labels_true_std is not None:
            self.prediction_dataset.misclassified = [ result_pred[1] for result_pred, labels_true in zip(results_pred, labels_true_std) if result_pred[1].stg7_label_predicted != labels_true ]
            n_total_all      = len(self.prediction_dataset.labels_pred)
            n_pred_correct   = sum(a == b for a, b in zip(self.prediction_dataset.labels_pred, labels_true_std))
            accuracy_overall = n_pred_correct / n_total_all
            self.prediction_dataset.accuracy = accuracy_overall
        
        return accuracy_overall
    # <<
    # ==============================================================================================================================
    # END << METHOD << predict_dataset
    # ==============================================================================================================================
    
    
    # ==============================================================================================================================
    # START >> METHOD >> _get_neg_accuracy_dataset_train
    # ==============================================================================================================================
    # >>
    def _get_neg_accuracy_dataset_train(self, params, is_optimization, images_std, labels_true_std, show_analysis = False):
        
        labels_pred      = [ self._predict(image_std, params = params, is_optimization = is_optimization)[0] for image_std in images_std ]
        n_total_all      = len(labels_pred)
        n_pred_correct   = sum(a == b for a, b in zip(labels_pred, labels_true_std))
        accuracy_overall = n_pred_correct / n_total_all
        if show_analysis:
            printmd(f"Overall accuracy", color = DEFAULT_COLOR_SUBHEADING)
            print(f"Total images     = {n_total_all}")
            print(f"Pred correctly   = {n_pred_correct}")
            printmd(f"Accuracy overall = {accuracy_overall*100:.2f}%", color = "green", is_bold = True)
        
        return -accuracy_overall
    # <<
    # ==============================================================================================================================
    # END << METHOD << _get_neg_accuracy_dataset_train
    # ==============================================================================================================================
    
    
    # ==============================================================================================================================
    # START >> METHOD >> _get_accuracy_dataset_train
    # ==============================================================================================================================
    # >>
    def _get_accuracy_dataset_train(self, images_std, label_true, show_analysis = False, name_images = ""):
        
        results_pred     = [ self._predict(image_std, label_true) for image_std in images_std ]
        labels_pred      = [ result_pred[0] for result_pred in results_pred ]
        analyses_pred    = [ result_pred[1] for result_pred in results_pred ]
        misclassified    = [ result_pred[1] for result_pred in results_pred if result_pred[1].stg7_label_predicted != label_true ]
        n_pred_all       = len(labels_pred)
        n_pred_red       = labels_pred.count([1,0,0])
        n_pred_yellow    = labels_pred.count([0,1,0])
        n_pred_green     = labels_pred.count([0,0,1])
        n_pred_requested = labels_pred.count(list(label_true))
        accuracy         = n_pred_requested / n_pred_all
        
        if show_analysis:
            
            images_extracted_hue_red    = [ analysis_pred.stg4_image_colors_extracted.red                 for analysis_pred in analyses_pred[0:10] ]
            probabilities_being_red     = [ round(analysis_pred.stg6_probabilities.image_being_red, 5)    for analysis_pred in analyses_pred[0:10] ]
            
            images_extracted_hue_yellow = [ analysis_pred.stg4_image_colors_extracted.yellow              for analysis_pred in analyses_pred[0:10] ]
            probabilities_being_yellow  = [ round(analysis_pred.stg6_probabilities.image_being_yellow, 5) for analysis_pred in analyses_pred[0:10] ]
            
            images_extracted_hue_green  = [ analysis_pred.stg4_image_colors_extracted.green               for analysis_pred in analyses_pred[0:10] ]
            probabilities_being_green   = [ round(analysis_pred.stg6_probabilities.image_being_green, 5)  for analysis_pred in analyses_pred[0:10] ]
            
            labels_str_pred             = [ analysis_pred.stg7_label_predicted_str for analysis_pred in analyses_pred[0:10] ]
            
            printmd(f"{name_images.capitalize()} light training dataset (a few examples)", color = "skyblue", is_bold = True)
            
            printmd(f"Model's red hues extracted from the red lights located in model's red light region:", is_bold = True)
            plot_images(images_extracted_hue_red, figsizeScale = DEFAULT_FIGSIZESCALE_EXAMPLES)
            print(f"Probability of being red: {probabilities_being_red}")
            
            printmd(f"Model's yellow hues extracted from the yellow lights located in model's yellow light region:", is_bold = True)
            plot_images(images_extracted_hue_yellow, figsizeScale = DEFAULT_FIGSIZESCALE_EXAMPLES)
            print(f"Probability of being yellow: {probabilities_being_yellow}")
            
            printmd(f"Model's green hues extracted from the yellow lights located in model's yellow light region:", is_bold = True)
            plot_images(images_extracted_hue_green, figsizeScale = DEFAULT_FIGSIZESCALE_EXAMPLES)
            print(f"Probability of being green: {probabilities_being_green}")
            
            print(f"Predicted labels: {labels_str_pred}")

            print(f"Total {name_images} images: {len(images_std)}")
            print(f"Predicted Red    = {n_pred_red}")
            print(f"Predicted Yellow = {n_pred_yellow}")
            print(f"Predicted Green  = {n_pred_green}")
            printmd(f"Accuracy         = {accuracy*100:.2f}%", color = "green", is_bold = True)
        
        return accuracy, analyses_pred, misclassified
    # <<
    # ==============================================================================================================================
    # END << METHOD << _get_accuracy_dataset_train
    # ==============================================================================================================================
    
    
    # ==============================================================================================================================
    # START >> METHOD >> _template_method
    # ==============================================================================================================================
    # >>
    # def _template_method(self):
    #
    #     return None
    # <<
    # ==============================================================================================================================
    # END << METHOD << _template_method
    # ==============================================================================================================================
    
# <<
# ==================================================================================================================================
# END << CLASS << Model
# ==================================================================================================================================



# ==================================================================================================================================
# START >> FUNCTION >> _template_mod_func
# ==================================================================================================================================
# >>
def _template_mod_func  ( p_p_p_p_1 = ""
                        , p_p_p_p_2 = ""
                        ) :
    
    """
    ================================================================================
    START >> DOC >> _template_mod_func
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
    END << DOC << _template_mod_func
    ================================================================================
    """
    
    _name_func = inspect.stack()[0][3]
    print(f"This is a print from '{_name_mod}.{_name_func}'{p_p_p_p_1}{p_p_p_p_2}.")
    
    return None
# <<
# ==================================================================================================================================
# END << FUNCTION << _template_mod_func
# ==================================================================================================================================

print("  - Done!")

# <<
# ==================================================================================================================================
# END << MODULE << traffic_light_classifier.model
# ==================================================================================================================================
