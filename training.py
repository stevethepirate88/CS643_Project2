# This is there I am going to write the code to train the model

import argparse

# The program needs to accept TrainingDataset.csv and ValidationDataset.csv
parser = argparse.ArgumentParser(description='Getting the filepath for training')
parser.add_argument("path", help="The path to the files that will be used for training and validation. Don't include the file name.")
path = parser.parse_args()

# Need to run training in parallel on multiple EC2 instances

# Need to validate the model seperately

