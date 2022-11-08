
from setuptools import setup
from setuptools import find_packages

with open(file = "README.md", mode = "r") as readme_handle:
    long_description = readme_handle.read()

setup(
    name             = 'traffic_light_classifier',
    version          = '1.0.1',
    author           = 'Shashank Kumbhare',
    author_email     = 'shashankkumbhare8@gmail.com',
    url              = 'https://github.com/ShashankKumbhare/traffic-light-classifier',
    description      = 'A computer vision & probabilistic approach based traffic light classifier.',
    long_description = long_description,
    long_description_content_type = "text/markdown",
    license          = 'MIT',
    keywords         = ['python', 'opencv', 'computer vision', 'ML', 'machine learning', 'traffic-light-classifier',
                        'self driving cars', 'AI', 'artificial intelligence'],
    install_requires = [
                        'opencv-python',
                        'numpy',
                        'matplotlib',
                        'scipy',
                        'IPython'
                       ],
    packages         = find_packages(),
    package_data     = {
                        'traffic_light_classifier' : ['__data_subpkg__/dataset_train/red/*',
                                                      '__data_subpkg__/dataset_train/yellow/*',
                                                      '__data_subpkg__/dataset_train/green/*',
                                                      '__data_subpkg__/dataset_test/red/*',
                                                      '__data_subpkg__/dataset_test/yellow/*',
                                                      '__data_subpkg__/dataset_test/green/*',
                                                      '__auxil_subpkg__/images/*',
                                                      '__constants_subpkg__/*',]
                       },
    include_package_data = True,
    classifiers      = ['License :: OSI Approved :: MIT License',
                        'Natural Language :: English',
                        'Operating System :: OS Independent',
                        'Programming Language :: Python :: 3'
                       ]
)
