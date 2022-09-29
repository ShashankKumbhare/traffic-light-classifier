
from setuptools import setup

# ==================================================================================
# START >> NOTE
# ==================================================================================
# >>
# If you have setup.py in your project and you use find_packages() within it, it is
# necessary to have an __init__.py file in every directory for packages to be
# automatically found.
# Packages are only recognized if they include an __init__.py file
# <<
# ==================================================================================
# END << NOTE
# ==================================================================================

setup(
    name         = 'traffic_light_classifier',
    version      = '0.0.1',
    description  = 'A computer vision & probabilistic approach based traffic light classifier.',
    license      = 'MIT',
    package_dir  = {
                    'traffic_light_classifier'                        : 'traffic_light_classifier',
                    'traffic_light_classifier.__dependencies_subpkg__': 'traffic_light_classifier/__dependencies_subpkg__',
                    'traffic_light_classifier.__constants_subpkg__'   : 'traffic_light_classifier/__constants_subpkg__',
                    'traffic_light_classifier.__auxil_subpkg__'       : 'traffic_light_classifier/__auxil_subpkg__',
                    'traffic_light_classifier.__tests_subpkg__'       : 'traffic_light_classifier/__tests_subpkg__',
                    'traffic_light_classifier.plots_subpkg'           : 'traffic_light_classifier/plots_subpkg',
                    'traffic_light_classifier.modify_images_subpkg'   : 'traffic_light_classifier/modify_images_subpkg',
                    'traffic_light_classifier.extract_feature_subpkg' : 'traffic_light_classifier/extract_feature_subpkg'
                   },
    packages     = [
                    'traffic_light_classifier',
                    'traffic_light_classifier.__dependencies_subpkg__',
                    'traffic_light_classifier.__constants_subpkg__',
                    'traffic_light_classifier.__auxil_subpkg__',
                    'traffic_light_classifier.__tests_subpkg__',
                    'traffic_light_classifier.plots_subpkg',
                    'traffic_light_classifier.modify_images_subpkg',
                    'traffic_light_classifier.extract_feature_subpkg'
                   ],
    author       = 'Shashank Kumbhare',
    author_email = 'shashankkumbhare8@gmail.com',
    keywords     = ['python', 'opencv', 'computer vision', 'ML', 'machine learning', 'traffic-light-classifier', 'AI', 'artificial intelligence'],
    url          = 'https://github.com/ShashankKumbhare/traffic-light-classifier'
)
