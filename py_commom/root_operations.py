import ROOT as rt
import time
import os
import numpy as np
import logging
import copy

#from data_operations import *

def readTree(path, tree='DataCollSvc'):
	logging.info("readTree: reading " + path)
	try:
		myfile = rt.TFile(path)
	except:
		logging.error("can not open file: " + path)
		return False
	mytree = myfile.Get(tree)
	entries = mytree.GetEntriesFast()
	#entries = Tree.GetEntries()
	return myfile, mytree, entries


def readChain(files=[], tree_name=''):
	if not files:
		logging.error('files must not be empty')
		return False
	if not tree_name:
		logging.error('please specify tree_name')
		return False
	chain = rt.TChain(tree_name)
	for f in files:	
		if '.root' not in f:
			if f[-1] != '/':
				f += '/'
			f += '*.root'
		chain.Add(f)
		logging.info('add '+f+' to chain')
	entries = chain.GetEntries()
	logging.info('total '+str(entries)+' entries')
	return chain, entries

def getBranchData(tree, entry, branch_name='', flatten=False):
	#if branch_name not in tree
	if branch_name not in [b.GetName() for b in list(tree.GetListOfBranches())]:
		logging.warning("can not find a branch named " + branch_name)
		return False

	data = []
	tree.GetEntry(entry)
	data = (getattr(tree, branch_name))
	if flatten:
		return flattenList(data)
	else:
		return data

def list2TVector(li):
	v = rt.TVector(0, len(li)-1)
	for i in range(len(li)):
		v[i] = li[i]
	return v

def getAxisRange(li, axis='y', factor=0.1):
	data = li
	if axis=='y':
		data = flattenList(li)
	l_max = max(data)
	l_min = min(data)
	l_diff = l_max - l_min
	return [l_min-factor*l_diff, l_max+factor*l_diff]

def createTGraph(x, y, title="", line_color=1, line_width=1, line_style=1, x_axis_range=None, 
	y_axis_range=None, x_axis_title="", y_axis_title="", marker_size=0.5, 
	marker_color=1, marker_style=None):

	if len(x) != len(y):
		logging.error('len(x) != len(y)')
		return False

	logging.info("creating TGraph " + title)

	x_vector = list2TVector(x)
	y_vector = list2TVector(y)

	g = rt.TGraph(x_vector, y_vector)
	g.SetLineColor(line_color)
	g.SetLineWidth(line_width)
	g.SetLineStyle(line_style)

	if marker_style:
		g.SetMarkerStyle(marker_style)

	g.SetMarkerSize(marker_size)
	g.SetMarkerColor(marker_color)

	x_axis = g.GetXaxis()
	y_axis = g.GetYaxis()

	x_axis.SetTitle(x_axis_title)
	y_axis.SetTitle(y_axis_title)

	if x_axis_range:		
		x_axis.SetRangeUser(x_axis_range[0], x_axis_range[1])
	if y_axis_range:		
		y_axis.SetRangeUser(y_axis_range[0], y_axis_range[1])

	g.SetTitle(title)

	return g

def createLegend(d, position=[0.7,0.8,0.9,0.9]):
	logging.info("creating legend...")
	logging.info('d')
	logging.info(d)

	legend = rt.TLegend(position[0], position[1], position[2], position[3])

	logging.info('sorted(d.keys())')
	logging.info(sorted(d.keys()))

	for key in sorted(d.keys()):
		legend.AddEntry(d[key], key )

	return legend

def getBestLegendSize(y_keys, fig_size):
	logging.debug('fig_size')
	logging.debug(fig_size)
	max_length = max([len(key) for key in y_keys])
	legend_size = ((0.15+0.015*max_length)/(1.0*fig_size[0]/fig_size[1]), 0.05*len(y_keys))
	logging.info('legend_size')
	logging.info(legend_size)
	return legend_size

def getBestLegendPosition(x, y_values, x_axis_range, y_axis_range, legend_size, stride=0.1):
	def normalize(data, axis_range):
		return [0.1 + 0.8*(d - axis_range[0] ) / axis_range[1] for d in data]

	x = normalize(x, x_axis_range)
	points_dict = {}
	for i in range(len(x)):
		points_dict[x[i]] = normalize([y[i] for y in y_values], y_axis_range)

	logging.debug('points_dict')
	logging.debug(points_dict)

	def countPoints(points_dict, x_range, y_range):
		xs = [x for x in points_dict.keys() if x>x_range[0] and x< x_range[1]]
		return sum([len([y for y in points_dict[x] if y>y_range[0] and y<y_range[1]]) for x in xs])

	best_position = [1, 1]
	min_num = len(x) * len(y_values)

	positions = []
	x_range = list(np.arange(0.9, 0.1+legend_size[0], stride*(-1)))
	positions += zip(x_range, [0.9 for _ in x_range])
	y_range = list(np.arange(0.9, 0.1+legend_size[1], stride*(-1)))
	positions += zip([0.9 for _ in y_range], y_range)
	logging.debug('positions')
	logging.debug(positions)

	for pos in positions:
		num = countPoints(points_dict, (pos[0] - legend_size[0], pos[0]), (pos[1] - legend_size[1], pos[1]))
		logging.debug('pos')
		logging.debug(pos)
		logging.debug('num')
		logging.debug(num)
		if num < min_num:
			best_position = pos
			min_num = num
		if min_num == 0:
			return (best_position[0] - legend_size[0], best_position[1] - legend_size[1], best_position[0], best_position[1])

	logging.info('best legend position')
	logging.info(best_position)
	logging.info('min cover point number')
	logging.info(min_num)		
	return (best_position[0] - legend_size[0], best_position[1] - legend_size[1], best_position[0], best_position[1])


def drawTGraph(x, y_keys, y_values, title='', 
	x_axis_title="", y_axis_title="",
	line=True, line_width=1, line_style=1,  	 
	marker=True, marker_size=1, marker_style='auto',
	fig_size=(1200,800), save_file='', legend='auto',
	legend_mean=False, mean_line=False):
	
	logging.debug('x')
	logging.debug(x)

	fig_num = len(y_keys) 
	
	if fig_num < 1:
		logging,error('y_keys is empty')
		return False

	def rectLength(li, target_num, complete_value=1):
		if len(li) <= target_num:
			li += [complete_value for i in range(target_num-len(li))]
		else:
			li = li[:target_num]
		return li

	#marker
	if isinstance(marker, list):
		marker = [True if i else False for i in marker]
		marker = rectLength(marker, fig_num, complete_value=True)
	else:
		marker = [True if marker else False for i in range(fig_num)]

	

	#marker_style
	if isinstance(marker_style, str):
		if marker_style == 'auto':
			marker_style = range(20, fig_num+20)
	elif isinstance(marker_style, int):
		marker_style = [marker_style for i in range(fig_num)]
	elif isinstance(marker_style, list):
		marker_style = rectLength(marker_style, fig_num, complete_value=20)
	else:
		marker_style = range(20, fig_num)

	#marker_size
	if not isinstance(marker_size, list):
		marker_size = [marker_size for i in range(fig_num)]
	elif isinstance(marker_size, list):
		marker_size = rectLength(marker_size, fig_num, complete_value=1)
	else:
		marker_size += [1 for i in range(fig_num)]

	#line
	if isinstance(line, list):
		line = [True if i else False for i in line]
		line = rectLength(line, fig_num, complete_value=True)
	else:
		line = [True if line else False for i in range(fig_num)]

	#line_width
	if isinstance(line_width, int):
		line_width = [line_width for i in range(fig_num)]
	elif isinstance(line_width, list):
		line_width = rectLength(line_width, fig_num, complete_value=1)
	else:
		line_width += [1 for i in range(fig_num)]

	#line_style
	if isinstance(line_style, int):
		line_style = [line_style for i in range(fig_num)]
	elif isinstance(line_style, list):
		line_style = rectLength(line_style, fig_num, complete_value=1)
	else:
		line_style = range(1, fig_num)

	#color
	color_list = [2,4,6,8,9,13,26,30,36,42,46,49]
	color = color_list[:fig_num]
	logging.debug('color')
	logging.debug(color)

	#mean_line
	if isinstance(mean_line, list):
		mean_line = rectLength(line_style, fig_num, complete_value=False)
	else:
		mean_line = [True if mean_line else False for i in range(fig_num)]


	#y_axis_range
	x_axis_range = getAxisRange(x, axis='x')
	y_axis_range = getAxisRange(y_values)
	logging.debug('y_axis_range')
	logging.debug(y_axis_range)

	#style
	style=['' for i in range(fig_num)]
	style[0] = 'A'
	for i in range(fig_num):
		if line[i]:
			style[i] += 'L'
		if marker[i]:
			style[i] += 'P'
		if i != 0:
			style[i] += ' SAME'

	logging.debug('style')
	logging.debug(style)

	logging.debug('fig_size')
	logging.debug(fig_size)

	#create Canvas
	c = rt.TCanvas("c1","lustre_eos",200,10,fig_size[0], fig_size[1])
	#create TGraphs
	graphs=[]
	mean_graphs = []
	legend_dict={}

	for i in range(fig_num):
		g = createTGraph(x, y_values[i], title=title, 
				line_color=color[i], line_width=line_width[i], 
				line_style=line_style[i], 
				marker_color=color[i], marker_size=marker_size[i], 
				marker_style=marker_style[i],  
				x_axis_title=x_axis_title, y_axis_title=y_axis_title, 
				x_axis_range=x_axis_range, y_axis_range=y_axis_range)
		graphs.append(g)
		if mean_line[i]:
			g_mean = createTGraph(x, [np.mean(y_values[i])]*len(x), 
				line_color=color[i], line_style=2)
			mean_graphs.append(g_mean)

		#mean and var
		if legend_mean:
			y_keys[i] += ', mean='+str(round_mean(y_values[i]))
			y_keys[i] += ', var='+str(round_var(y_values[i]))
		legend_dict[y_keys[i]] = g

	#create legend
	if len(legend) != 4:
		logging.warning('len(legend) != 4, set legend = auto')
		legend = 'auto'
	if legend == 'auto':
		legend_size = getBestLegendSize(y_keys, fig_size)
		legend = getBestLegendPosition(x, y_values, x_axis_range, y_axis_range, legend_size)
	logging.debug('legend')
	logging.debug(legend)
	legend = createLegend(legend_dict, legend)


	#draw
	
	for i in range(fig_num):
		graphs[i].Draw(style[i])	
	for mg in mean_graphs:
		mg.Draw('L SAME')
	
	#graphs[1].Draw(style[0])

	legend.Draw("SAME")

	if save_file:
		try:
			c.SaveAs(save_file)
			logging.info('save '+save_file)
		except:
			logging.error('save '+save_file+' failed')

	return True

def createTH1F(bins, x_range, title = '', \
	fig_size=(1200,800), legend=[], legend_position=[0.7,0.8,0.9,0.9], \
	x_axis_title='', y_axis_title=''):
	rt.gStyle.SetOptStat(0)
	c = rt.TCanvas('c1', 'test', \
		200, 10, fig_size[0], fig_size[1])

	d = {}
	h_list = []

	for i in range(len(legend)):
		h = rt.TH1F('h', title, bins, x_range[0], x_range[1])

		x_axis = h.GetXaxis()
		y_axis = h.GetYaxis()

		x_axis.SetTitle(x_axis_title)
		y_axis.SetTitle(y_axis_title)

		

		h.SetLineColor(i+1)
		h_list.append(h)

		d[legend[i]] = h
	legend = createLegend(d, position=legend_position)

	return c, h_list, legend

def normalizeTH1F(h_list, norm=1):
	for h in h_list:
		scale = norm / h.Integral()
		h.Scale(scale)

	return h_list

def saveTH1F(c, h_list, legend, filename='th1f.pdf', normalize=False, y_log=False, \
	y_range=None):
	if not h_list:
		logging.error('not h_list')
		return False

	if normalize:
		h_list = normalizeTH1F(h_list)

	y_axis = h_list[0].GetYaxis()
	if y_range:
		y_axis.SetRangeUser(y_range[0], y_range[1])

	h_list[0].Draw('HIST')
	for h in h_list[1:]:
		h.Draw('HIST SAME')

	if legend:
		legend.Draw("SAME")

	c.SaveAs(filename)

