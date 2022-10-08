
# from setuptools import setup
# from setuptools import find_packages

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

# Loading the README.md file >>
with open(file = "README.md", mode = "r") as readme_handle:
    long_description = readme_handle.read()

setup(
    name             = 'traffic_light_classifier',
    version          = '1.0.0',
    author           = 'Shashank Kumbhare',
    author_email     = 'shashankkumbhare8@gmail.com',
    url              = 'https://github.com/ShashankKumbhare/traffic-light-classifier',
    description      = 'A computer vision & probabilistic approach based traffic light classifier',
    long_description = long_description,
    long_description_content_type = "text/markdown",
    license          = 'MIT',
    keywords         = ['python', 'opencv', 'computer vision', 'ML', 'machine learning', 'traffic-light-classifier',
                        'self driving cars', 'AI', 'artificial intelligence'],
    install_requires = [
                        'numpy==1.21.5',
                        'matplotlib==3.5.1',
                        'scipy==1.7.3',
                        'IPython==8.2.0',
                       ],
    packages         = find_packages(),
    package_data     = {
                        'traffic_light_classifier' : ['__data_subpkg__/dataset_train/red/*',
                                                      '__data_subpkg__/dataset_train/yellow/*',
                                                      '__data_subpkg__/dataset_train/green/*',
                                                      '__data_subpkg__/dataset_test/red/*',
                                                      '__data_subpkg__/dataset_test/yellow/*',
                                                      '__data_subpkg__/dataset_test/green/*']
                       },
    include_package_data = True,
    classifiers      = ['License :: OSI Approved :: MIT',
                        'Natural Language :: English',
                        'Operating Syayerm :: OS Independent'
                        'Programming Labguage :: Python :: 3'
                       ]
    # package_dir      = {
    #                     'traffic_light_classifier'                        : 'traffic_light_classifier',
    #                     'traffic_light_classifier.__dependencies_subpkg__': 'traffic_light_classifier/__dependencies_subpkg__',
    #                     'traffic_light_classifier.__constants_subpkg__'   : 'traffic_light_classifier/__constants_subpkg__',
    #                     'traffic_light_classifier.__auxil_subpkg__'       : 'traffic_light_classifier/__auxil_subpkg__',
    #                     'traffic_light_classifier.__tests_subpkg__'       : 'traffic_light_classifier/__tests_subpkg__',
    #                     'traffic_light_classifier.plots_subpkg'           : 'traffic_light_classifier/plots_subpkg',
    #                     'traffic_light_classifier.modify_images_subpkg'   : 'traffic_light_classifier/modify_images_subpkg',
    #                     'traffic_light_classifier.extract_feature_subpkg' : 'traffic_light_classifier/extract_feature_subpkg'
    #                    }
    # packages         = [
    #                     'traffic_light_classifier',
    #                     'traffic_light_classifier.__dependencies_subpkg__',
    #                     'traffic_light_classifier.__constants_subpkg__',
    #                     'traffic_light_classifier.__auxil_subpkg__',
    #                     'traffic_light_classifier.__tests_subpkg__',
    #                     'traffic_light_classifier.plots_subpkg',
    #                     'traffic_light_classifier.modify_images_subpkg',
    #                     'traffic_light_classifier.extract_feature_subpkg'
    #                    ]
)
