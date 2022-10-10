
# ==================================================================================================================================
# START >> FILE INFO
# ==================================================================================================================================
# File        : traffic_light_classifier/_template_subpkg/_template_submod.py
# Author      : Shashank Kumbhare
# Date        : --/--/----
# email       : shashankkumbhare8@gmail.com
# Description : This file is a python submodule for python subpackage
#               'traffic_light_classifier._template_subpkg'.
# ==================================================================================================================================
# END << FILE INFO
# ==================================================================================================================================



# ==================================================================================================================================
# START >> SUBMODULE >> traffic_light_classifier._template_subpkg._template_submod
# ==================================================================================================================================
# >>
"""
This submodule is created to manage all the datasets related variables.
"""



# ==================================================================================
# START >> IMPORTS
# ==================================================================================
# >>
from ..__dependencies_subpkg__ import *
from ..__constants_subpkg__    import *
from ..__auxil_subpkg__        import *
from .._modify_images_subpkg   import *
# <<
# ==================================================================================
# END << IMPORTS
# ==================================================================================


# ==================================================================================
# START >> EXPORTS
# ==================================================================================
# >>
__all__ = ["datasets"]
# <<
# ==================================================================================
# END << EXPORTS
# ==================================================================================


# ==================================================================================================================================
# START >> DATASET
# ==================================================================================================================================
# >>
datasets       = Struct()
datasets.train = Struct()
datasets.test  = Struct()

# Assigning dataset directories >>
datasets.train._dir = "traffic_light_classifier/__data_subpkg__/dataset_train"
datasets.test._dir  = "traffic_light_classifier/__data_subpkg__/dataset_test"

# Loading datasets >>
datasets.train.images_and_labels = load_dataset(datasets.train._dir)
datasets.test.images_and_labels  = load_dataset(datasets.test._dir)

# Separating labels and images for training dataset >>
datasets.train.labels        = [ image_and_label[1] for image_and_label in datasets.train.images_and_labels ]
datasets.train.images        = Struct()
datasets.train.images.all    = [ image_and_label[0] for image_and_label in datasets.train.images_and_labels ]
datasets.train.images.red    = [ image for image, label in zip(datasets.train.images.all, datasets.train.labels) if label == "red" ]
datasets.train.images.yellow = [ image for image, label in zip(datasets.train.images.all, datasets.train.labels) if label == "yellow" ]
datasets.train.images.green  = [ image for image, label in zip(datasets.train.images.all, datasets.train.labels) if label == "green" ]

# Separating labels and images for test dataset >>
datasets.test.labels        = [ image_and_label[1] for image_and_label in datasets.test.images_and_labels ]
datasets.test.images        = Struct()
datasets.test.images.all    = [ image_and_label[0] for image_and_label in datasets.test.images_and_labels ]
datasets.test.images.red    = [ image for image, label in zip(datasets.test.images.all, datasets.test.labels) if label == "red" ]
datasets.test.images.yellow = [ image for image, label in zip(datasets.test.images.all, datasets.test.labels) if label == "yellow" ]
datasets.test.images.green  = [ image for image, label in zip(datasets.test.images.all, datasets.test.labels) if label == "green" ]

# Standardizing training dataset >>
datasets.train.images_and_labels_std = standardize_images(datasets.train.images_and_labels)
datasets.train.labels_std            = [ image_and_label_std[1] for image_and_label_std in datasets.train.images_and_labels_std ]
datasets.train.images_std            = Struct()
datasets.train.images_std.all        = [ image_and_label_std[0] for image_and_label_std in datasets.train.images_and_labels_std ]
datasets.train.images_std.red        = [ image for image, label in zip(datasets.train.images_std.all, datasets.train.labels_std) if label == [1,0,0] ]
datasets.train.images_std.yellow     = [ image for image, label in zip(datasets.train.images_std.all, datasets.train.labels_std) if label == [0,1,0] ]
datasets.train.images_std.green      = [ image for image, label in zip(datasets.train.images_std.all, datasets.train.labels_std) if label == [0,0,1] ]

# Standardizing training dataset >>
datasets.test.images_and_labels_std = standardize_images(datasets.test.images_and_labels)
datasets.test.labels_std            = [ image_and_label_std[1] for image_and_label_std  in datasets.test.images_and_labels_std ]
datasets.test.images_std            = Struct()
datasets.test.images_std.all        = [ image_and_label_std[0] for image_and_label_std in datasets.test.images_and_labels_std ]
datasets.test.images_std.red        = [ image for image, label in zip(datasets.test.images_std.all, datasets.test.labels_std) if label == [1,0,0] ]
datasets.test.images_std.yellow     = [ image for image, label in zip(datasets.test.images_std.all, datasets.test.labels_std) if label == [0,1,0] ]
datasets.test.images_std.green      = [ image for image, label in zip(datasets.test.images_std.all, datasets.test.labels_std) if label == [0,0,1] ]
# <<
# ==================================================================================================================================
# END << DATASET
# ==================================================================================================================================



# <<
# ==================================================================================================================================
# END << SUBMODULE << traffic_light_classifier._template_subpkg._template_submod
# ==================================================================================================================================
