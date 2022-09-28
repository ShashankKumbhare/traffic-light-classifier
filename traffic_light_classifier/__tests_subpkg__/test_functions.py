
# ==================================================================================================================================
# START >> FILE INFO
# ==================================================================================================================================
# File        : traffic_light_classifier/__tests_subpkg__/test_functions.py
# Author      : Shashank Kumbhare
# Date        : 09/20/2022
# email       : shashankkumbhare8@gmail.com
# Description : This file is a python submodule for python subpackage
#               'traffic_light_classifier.__tests_subpkg__'.
# ==================================================================================================================================
# END << FILE INFO
# ==================================================================================================================================



# ==================================================================================================================================
# START >> SUBMODULE >> traffic_light_classifier.__tests_subpkg__.test_functions
# ==================================================================================================================================
# >>
"""
This submodule contains tools to perform unittests on the functionalities of the
package.
"""

_name_subpkg = __name__.partition(".")[-2]
_name_submod = __name__.partition(".")[-1]
print(f"   + Adding submodule '{_name_submod}'...")

# ==================================================================================
# START >> IMPORTS
# ==================================================================================
# >>
from ..__dependencies_subpkg__ import _dependencies_submod as _dps
# <<
# ==================================================================================
# END >> IMPORTS
# ==================================================================================


# ==================================================================================================================================
# START >> FUNCTIONS
# ==================================================================================================================================
# >>
# Helper functions for printing markdown text (text in color/bold/etc)
def _printmd(string):
    _dps.display(_dps.Markdown(string))

# Print a test failed message, given an error
def _print_fail():
    _printmd('**<span style="color: red;">TEST FAILED</span>**')
    
# Print a test passed message
def _print_pass():
    _printmd('**<span style="color: green;">TEST PASSED</span>**')
# <<
# ==================================================================================================================================
# END << FUNCTIONS
# ==================================================================================================================================



# ==================================================================================================================================
# START >> CLASS >> Tests
# ==================================================================================================================================
# >>
class Tests(_dps.unittest.TestCase):
    
    """
    A class holding all tests.
    """
    
    # Tests the `one_hot_encode` function, which is passed in as an argument
    def test_one_hot(self, one_hot_function):
        
        # Test that the generated one-hot labels match the expected one-hot label
        # For all three cases (red, yellow, green)
        try:
            self.assertEqual([1,0,0], one_hot_function('red'))
            self.assertEqual([0,1,0], one_hot_function('yellow'))
            self.assertEqual([0,0,1], one_hot_function('green'))
        
        # If the function does *not* pass all 3 tests above, it enters this exception
        except self.failureException as e:
            # Print out an error message
            _print_fail()
            print("Your function did not return the expected one-hot label.")
            print('\n'+str(e))
            return
        
        # Print out a "test passed" message
        _print_pass()
    
    # Tests if any misclassified images are red but mistakenly classified as green
    def test_red_as_green(self, misclassified_images):
        # Loop through each misclassified image and the labels
        for im, predicted_label, true_label in misclassified_images:
            
            # Check if the image is one of a red light
            if true_label == [1,0,0]:
                
                try:
                    # Check that it is NOT labeled as a green light
                    self.assertNotEqual(predicted_label, [0, 0, 1])
                except self.failureException as e:
                    # Print out an error message
                    _print_fail()
                    print("Warning: A red light is classified as green.")
                    print('\n'+str(e))
                    return
        
        # No red lights are classified as green; test passed
        _print_pass()
# <<
# ==================================================================================================================================
# END << CLASS << Tests
# ==================================================================================================================================

print("   - Done!")

# <<
# ==================================================================================================================================
# END << SUBMODULE << traffic_light_classifier.__tests_subpkg__.test_functions
# ==================================================================================================================================
