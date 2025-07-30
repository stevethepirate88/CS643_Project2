# This is there I am going to write the code to train the model

import argparse

# The program needs to accept the path to the test file as a command line parameter.
parser = argparse.ArgumentParser(description='Getting the filepath for training')
parser.add_argument("path", help="The path to the files that will be used for training and validation. Don't include the file name.")
path = parser.parse_args()

