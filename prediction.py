# This is where I am going to write the prediction code
import sys
import argparse

# Need to take the pathname to the test file as a command line parameter. This pathname contains the directory location and the filename of the test file.

parser = argparse.ArgumentParser(description='Getting the filepath for training')
parser.add_argument("path", help="The path to the files that will be used for training and validation. Don't include the file name.")
path = parser.parse_args()


# May use the ValidationDataset.csv as the test

# The output needs to be a measure of prediction performance, specifically the F1 score available in MLlib