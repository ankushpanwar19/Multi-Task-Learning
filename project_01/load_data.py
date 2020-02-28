# Deep Learning for Autonomous Driving
# Material for Project 1
# For further questions contact Ozan Unal, ouenal@ee.ethz.ch

import pickle

def load_data(data_path):
	''' 
    Load data dictionary from data_path.
    '''
	with open(data_path, 'rb') as fp:
	    data = pickle.load(fp)
	return data