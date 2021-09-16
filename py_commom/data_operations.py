import logging
from copy import deepcopy
import numpy as np
import h5py
import os
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit


def flattenList(data, dimension=2, target_dimension=1):
	if not data:
		logging.error("data must not be empty")
		return False
	if dimension < target_dimension:
		logging.error('dimension < target_dimension')
		return False

	for i in range(dimension - target_dimension):
		data_flatten = []
		for d in data:
			data_flatten += d
			data = deepcopy(data_flatten)

	return data

def round_mean(li, round_num=2):
	return round(np.mean(li), round_num)

def round_var(li, round_num=2):
	return round(np.var(li), round_num)

def round_std(li, round_num=2):
	return round(np.std(li), round_num)

def readHDF5(file, mode='r'):
    logging.info('readHDF5: '+ file)
    if mode == 'r':
    	data = h5py.File(file, 'r')
    elif mode == 'a':
    	data = h5py.File(file, 'a')
    return data

def writeHDF5(file, data):
	hf = h5py.File(file, 'w')
	for key,value in data.items():
		#logging.debug(key)
		hf.create_dataset(key, data=value)
	hf.close()
	logging.info('writeHDF5: '+ file)

def getH5Files(path, file_num=None):
    logging.info('getH5Files: '+ path)
    h5_files = [path+p for p in os.listdir(path) if '.h5' in p]
    if file_num:
        try:
            h5_files = h5_files[:file_num]
        except:
            logging.warning('getH5Files: not so much h5 files, already select all h5 files')
    return h5_files

def array2Image(a, file_name='', cmap='Reds', title='', xlabel='', ylabel='', colorbar=True):
	plt.clf()

	if title:
		plt.title(title)
	if xlabel:
		plt.xlabel(xlabel)
	if ylabel:
		plt.ylabel(ylabel)


	try:
		plt.imshow(Image.fromarray(a), cmap=cmap, origin='lower')
	except Exception as e:
		logging.error(e)
		plt.imshow(Image.fromarray(a), cmap='Reds', origin='lower')

	if colorbar:
		plt.colorbar()

	if filename:
		plt.savefig(filename)
	else:
		plt.show()



	return

def drawHist(data_dict, bins=100, file_name='', title='', xlabel='', ylabel='', x_range=None):
	plt.clf()

	data_min = min([min(v) for v in data_dict.values()])
	data_max = max([max(v) for v in data_dict.values()])

	if x_range: 
		data_min, data_max = x_range[0], x_range[1]

	for k,v in data_dict.items():
		plt.hist(v, bins=bins, range=[data_min, data_max], histtype='step', label=k)
	
	if title:
		plt.title(title)
	if xlabel:
		plt.xlabel(xlabel)
	if ylabel:
		plt.ylabel(ylabel)

	plt.legend()
	plt.savefig(file_name)
	plt.clf()
	return

def drawScatter(d1, d2, file_name='', title='', xlabel='', ylabel='', fit=False, xlim='', ylim='', line=''):
	plt.clf()
	plt.scatter(d1, d2, s=20, c='red', alpha=0.5)

	plt.xlim(min(d1), max(d1))
	plt.ylim(min(d2), max(d2))

	if xlabel:
		plt.xlabel(xlabel)
	if ylabel:
		plt.ylabel(ylabel)

	if xlim:
		plt.xlim(xlim)
	if ylim:
		plt.ylim(ylim)

	if fit:
		popt, pcov = curve_fit(multiplyFunc, d1, d2)
		x_fit = np.linspace(min(d1), max(d1), 100)
		y_fit = multiplyFunc(x_fit, popt[0])
		plt.plot(x_fit, y_fit, linestyle='--')
		plt.legend()

	if line:
		x = np.linspace(min(d1), max(d1), 100)
		y = line[0] * x + line[1]
		plt.plot(x, y, label='y = {}*x + {}'.format(line[0], line[1]), linestyle='--')
		plt.legend()

	plt.title(title)
	plt.savefig(file_name)
	plt.clf()
	return



