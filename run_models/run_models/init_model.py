#!/usr/bin/env python
from run_models import utils
import argparse
import sys
import os

import time
import logging


################################ ARGUMENTS ################################
parser = argparse.ArgumentParser()
parser.add_argument("-r", "--root_dir", help="root directory",type=str,	default='')
parser.add_argument("-m", "--model", help="Model python file",type=str,	default="model.py")
parser.add_argument("-tr", "--train", help="Training file names",type=str,	default=("data/X_train.txt", "data/y_train.txt"))
parser.add_argument("-te", "--test", help="Testing file names",type=str,	default=("data/X_test.txt", "data/y_test.txt"))
parser.add_argument("-mp", "--model_pre", help="directory of output files",type=str, default="trained_model")
parser.add_argument("-ps", "--print", help="print summary",type=str, default=True)
args = parser.parse_args()

############################### CHECK INPUTS ###############################
input_string = "MODEL ROOT DIRECTORY: "
train_string = "Training data: "
test_string = "Testing data: "
mi_string = "Model script: "
mo_string = "Trained model prefix: "
maxlen = max([len(i) for i in [input_string,train_string, test_string, mi_string, mo_string]]) + 10
print()
print("{}{:.>{l}}".format(input_string,'',l=maxlen-len(input_string)),"{} (-i)".format(str(args.root_dir)))
print("{}{:.>{l}}".format(mi_string,'',l=maxlen-len(mi_string)),"{} (-m)".format(str(args.model)))
print("{}{:.>{l}}".format(train_string,'',l=maxlen-len(train_string)),"{} (-tr)".format(str(args.train)))
print("{}{:.>{l}}".format(test_string,'',l=maxlen-len(test_string)),"{} (-te)".format(str(args.test)))
print("{}{:.>{l}}".format(mo_string,'',l=maxlen-len(mo_string)),"{} (-mp)".format(str(args.model_pre)))
print()

#args.
def main():
	#utils.load_data
	utils.sanitised_input()
	print("\nLoading data and model...\n")
	inputs = "{}*{}*{}*{}*{}".format(args.root_dir, *args.train, *args.test)
	model = os.path.join(args.root_dir, args.model)
	os.system('python {} {}'.format(model, inputs))