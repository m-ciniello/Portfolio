import time
import sys
import os
import numpy as np

def sanitised_input():
	ans_yes = ["Y","y","Yes","yes"]
	ans_no = ["N","n","No","no"]
	ans_sd = ["S","s"]
	ask1 ="Are these parameters correct? (Y:Continue, N:Exit, S:Self_Destruct): "
	ask2 = "Please enter either 'Y' for continue, 'N' to exit, or 'S' to self destruct: "
	ui = input(ask1)
	while True:
		if ui in ans_yes:
			break
		elif ui in ans_no:
			print("Exiting script... Goodbye!")
			sys.exit()
		elif ui in ans_sd:
			cntd = 10
			for i in range(1,cntd):
				if i < cntd-1:
					print("Initiating self destruct sequence in: " + str(cntd-i), end="\r")
					time.sleep(1)
				else: 
					print("Initiating self destruct sequence in: " + str(cntd-i))
			time.sleep(2)
			#sys.stdout.flush
			print("JK! Exiting script... Goodbye!")
			sys.exit()
		else:
			ui = input(ask2)

def get_data(arg_list, print_=True):
	root_dir, X_train, y_train, X_test, y_test = arg_list.split("*")
	X_train = np.loadtxt(os.path.join(root_dir,X_train))
	X_test = np.loadtxt(os.path.join(root_dir,X_test)) 
	y_train = np.loadtxt(os.path.join(root_dir,y_train))
	y_test =  np.loadtxt(os.path.join(root_dir,y_test))
	if print_:
		print("\n{:#<7} INPUT DATA {:#>7}".format('',''))
		print("X_train.shape: ", X_train.shape)
		print("y_train.shape: ", y_train.shape)
		print("X_test.shape: ", X_test.shape)
		print("y_test.shape: ", y_test.shape)
	return X_train, X_test, y_train, y_test


def inputPosInt(arg_name, message):
	while True:
		# Confirm int
		try:
			userInput = int(input(message))
			assert userInput > 0 
		except (ValueError, AssertionError):
			print("Not an integer (greater than 0)! Try again.")
			continue
		else:
			return userInput 
			break

def inputFloat(arg_name, message):
	while True:
		# Confirm int
		try:
			userInput = float(input(message))
		except ValueError:
			print("Not a float value! Try again.")
			continue
		else:
			return userInput 
			break

def get_arg(arg_name, arg_type='int'):
	message = "\nPlease enter {} value of type {}: ".format(arg_name,arg_type)
	if arg_type == 'int':
		return inputPosInt(arg_name, message)
	elif arg_type == 'float':
		return inputFloat(arg_name, message)
	else:
		return input(message)

def check_epochs():
	epoch_int = int(input("Input number of epochs: "))
	if epoch_int <= 0:
		raise ValueError("This needs to be a positive integer dumdum!!!")
	return epoch_int
