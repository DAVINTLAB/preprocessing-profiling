# -*- coding: utf-8 -*-
"""Plot distribution of datasets"""

import random
import base64
from distutils.version import LooseVersion
import preprocessing_profiling.base as base
from pkg_resources import resource_filename
import matplotlib
from matplotlib import gridspec
from matplotlib.patches import Patch
from matplotlib import cm
import numpy as np
from matplotlib.colors import ListedColormap
import missingno as msno
from scipy.cluster import hierarchy
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import tree
import pandas as pd
import warnings

BACKEND = matplotlib.get_backend()
if matplotlib.get_backend().lower() != BACKEND.lower():
	# If backend is not set properly a call to describe will hang
	matplotlib.use(BACKEND)
from matplotlib import pyplot as plt
try:
	from StringIO import BytesIO
except ImportError:
	from io import BytesIO
try:
	from urllib import quote
except ImportError:
	from urllib.parse import quote

def matrix(df,
		   filter=None, n=0, p=0, sort=None,
		   figsize=(25, 10), width_ratios=(15, 1), color=(0.25, 0.25, 0.25),
		   fontsize=16, labels=None, sparkline=True, inline=False,
		   freq=None, ax=None, predictions=False):
	"""
	A matrix visualization of the nullity of the given DataFrame.
	
	:param df: The `DataFrame` being mapped.
	:param filter: The filter to apply to the heatmap. Should be one of "top", "bottom", or None (default).
	:param n: The max number of columns to include in the filtered DataFrame.
	:param p: The max percentage fill of the columns in the filtered DataFrame.
	:param sort: The row sort order to apply. Can be "ascending", "descending", or None.
	:param figsize: The size of the figure to display.
	:param fontsize: The figure's font size. Default to 16.
	:param labels: Whether or not to display the column names. Defaults to the underlying data labels when there are
	50 columns or less, and no labels when there are more than 50 columns.
	:param sparkline: Whether or not to display the sparkline. Defaults to True.
	:param width_ratios: The ratio of the width of the matrix to the width of the sparkline. Defaults to `(15, 1)`.
	Does nothing if `sparkline=False`.
	:param color: The color of the filled columns. Default is `(0.25, 0.25, 0.25)`.
	:return: If `inline` is False, the underlying `matplotlib.figure` object. Else, nothing.
	"""
	df = base.nullity_filter(df, filter=filter, n=n, p=p)
	df = base.nullity_sort(df, sort=sort, axis='columns')
	
	if(predictions):
		predicted = df[df.columns[-1]]
		actual = df[df.columns[-2]]
		df = df.drop(columns = df.columns[-1])
		classes = actual.unique()
		classes.sort()
		combinations = []
		colorNumber = 0
		for classA in classes:
			for classB in classes:
				combinations.append((classA, classB))
				colorNumber += 1
		colors = cm.get_cmap("rainbow")(np.linspace(0, 1, colorNumber))
	height = df.shape[0]
	width = df.shape[1]
	
	# z is the color-mask array, g is a NxNx3 matrix. Apply the z color-mask to set the RGB of each pixel.
	z0 = df.notnull().values
	g = np.zeros((height, width, 3))
	
	g[z0 < 0.5] = [1, 1, 1]
	if(predictions):
		legend_elements = []
		j = 0
		for combination in combinations:
			j += 1
			if combination[0] == combination[1]:
				color1 = color
			else:
				color1 = colors[j][0:3]
			z1 = z0.copy()
			for i in range(0, height):
				if not(actual.iloc[i] == combination[0] and predicted.iloc[i] == combination[1]):
					z1[i] = np.zeros(width, dtype=bool)
			g[z1 > 0.5] = color1
			if np.any(z1) and combination[0] != combination[1]:
				legend_elements.append(Patch(facecolor = color1, edgecolor = 'black', label = "Class " + str(combination[0]) + " → " + "Class " + str(combination[1])))
	else:
		g[z0 > 0.5] = color
	
	# Set up the matplotlib grid layout. A unary subplot if no sparkline, a left-right splot if yes sparkline.
	if ax is None:
		plt.figure(figsize=figsize)
		if sparkline:
			gs = gridspec.GridSpec(1, 2, width_ratios=width_ratios)
			gs.update(wspace=0.08)
			ax1 = plt.subplot(gs[1])
		else:
			gs = gridspec.GridSpec(1, 1)
		ax0 = plt.subplot(gs[0])
	else:
		if sparkline is not False:
			warnings.warn(
				"Plotting a sparkline on an existing axis is not currently supported. "
				"To remove this warning, set sparkline=False."
			)
			sparkline = False
		ax0 = ax
	
	# Create the nullity plot.
	ax0.imshow(g, interpolation='none')
	
	# Remove extraneous default visual elements.
	ax0.set_aspect('auto')
	ax0.grid(b=False)
	ax0.xaxis.tick_top()
	ax0.xaxis.set_ticks_position('none')
	ax0.yaxis.set_ticks_position('none')
	ax0.spines['top'].set_visible(False)
	ax0.spines['right'].set_visible(False)
	ax0.spines['bottom'].set_visible(False)
	ax0.spines['left'].set_visible(False)
	
	# Set up and rotate the column ticks. The labels argument is set to None by default. If the user specifies it in
	# the argument, respect that specification. Otherwise display for <= 50 columns and do not display for > 50.
	if labels or (labels is None and len(df.columns) <= 50):
		ha = 'left'
		ax0.set_xticks(list(range(0, width)))
		ax0.set_xticklabels(list(df.columns), rotation=45, ha=ha, fontsize=fontsize)
	else:
		ax0.set_xticks([])
	
	# Adds Timestamps ticks if freq is not None, else set up the two top-bottom row ticks.
	if freq:
		ts_list = []
		
		if type(df.index) == pd.PeriodIndex:
			ts_array = pd.date_range(df.index.to_timestamp().date[0],
									 df.index.to_timestamp().date[-1],
									 freq=freq).values
			
			ts_ticks = pd.date_range(df.index.to_timestamp().date[0],
									 df.index.to_timestamp().date[-1],
									 freq=freq).map(lambda t:
													t.strftime('%Y-%m-%d'))
			
		elif type(df.index) == pd.DatetimeIndex:
			ts_array = pd.date_range(df.index.date[0], df.index.date[-1],
									 freq=freq).values
			
			ts_ticks = pd.date_range(df.index.date[0], df.index.date[-1],
									 freq=freq).map(lambda t:
													t.strftime('%Y-%m-%d'))
		else:
			raise KeyError('Dataframe index must be PeriodIndex or DatetimeIndex.')
		try:
			for value in ts_array:
				ts_list.append(df.index.get_loc(value))
		except KeyError:
			raise KeyError('Could not divide time index into desired frequency.')
	
		ax0.set_yticks(ts_list)
		ax0.set_yticklabels(ts_ticks, fontsize=int(fontsize / 16 * 20), rotation=0)
	else:
		ax0.set_yticks([0, df.shape[0] - 1])
		ax0.set_yticklabels([1, df.shape[0]], fontsize=int(fontsize / 16 * 20), rotation=0)
	
	# Create the inter-column vertical grid.
	in_between_point = [x + 0.5 for x in range(0, width - 1)]
	for in_between_point in in_between_point:
		ax0.axvline(in_between_point, linestyle='-', color='white')
	
	if sparkline:
		# Calculate row-wise completeness for the sparkline.
		completeness_srs = df.notnull().astype(bool).sum(axis=1)
		x_domain = list(range(0, height))
		y_range = list(reversed(completeness_srs.values))
		min_completeness = min(y_range)
		max_completeness = max(y_range)
		min_completeness_index = y_range.index(min_completeness)
		max_completeness_index = y_range.index(max_completeness)
		
		# Set up the sparkline, remove the border element.
		ax1.grid(b=False)
		ax1.set_aspect('auto')
		# GH 25
		if int(matplotlib.__version__[0]) <= 1:
			ax1.set_axis_bgcolor((1, 1, 1))
		else:
			ax1.set_facecolor((1, 1, 1))
		ax1.spines['top'].set_visible(False)
		ax1.spines['right'].set_visible(False)
		ax1.spines['bottom'].set_visible(False)
		ax1.spines['left'].set_visible(False)
		ax1.set_ymargin(0)
		
		# Plot sparkline---plot is sideways so the x and y axis are reversed.
		ax1.plot(y_range, x_domain, color=color)
		
		if labels:
			# Figure out what case to display the label in: mixed, upper, lower.
			label = 'Data Completeness'
			if str(df.columns[0]).islower():
				label = label.lower()
			if str(df.columns[0]).isupper():
				label = label.upper()
			
			# Set up and rotate the sparkline label.
			ha = 'left'
			ax1.set_xticks([min_completeness + (max_completeness - min_completeness) / 2])
			ax1.set_xticklabels([label], rotation=45, ha=ha, fontsize=fontsize)
			ax1.xaxis.tick_top()
			ax1.set_yticks([])
		else:
			ax1.set_xticks([])
			ax1.set_yticks([])
		
		# Add maximum and minimum labels, circles.
		ax1.annotate(max_completeness,
					 xy=(max_completeness, max_completeness_index),
					 xytext=(max_completeness + 2, max_completeness_index),
					 fontsize=int(fontsize / 16 * 14),
					 va='center',
					 ha='left')
		ax1.annotate(min_completeness,
					 xy=(min_completeness, min_completeness_index),
					 xytext=(min_completeness - 2, min_completeness_index),
					 fontsize=int(fontsize / 16 * 14),
					 va='center',
					 ha='right')
		
		ax1.set_xlim([min_completeness - 2, max_completeness + 2])  # Otherwise the circles are cut off.
		ax1.plot([min_completeness], [min_completeness_index], '.', color=color, markersize=10.0)
		ax1.plot([max_completeness], [max_completeness_index], '.', color=color, markersize=10.0)
	
		# Remove tick mark (only works after plotting).
		ax1.xaxis.set_ticks_position('none')
	
	if(predictions):
		box = ax0.get_position()
		ax0.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
		box = ax1.get_position()
		ax1.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
		legend_elements.append(Patch(facecolor = color, edgecolor = 'black', label = "Valid"))
		legend_elements.append(Patch(facecolor = [1, 1, 1], edgecolor = 'black', label = "Missing"))
		ax0.legend(handles = legend_elements, loc = 'upper center', bbox_to_anchor = (0.5, -0.05), ncol = len(legend_elements), fontsize=25)
	
	if inline:
		warnings.warn(
			"The 'inline' argument has been deprecated, and will be removed in a future version "
			"of missingno."
		)
		plt.show()
	else:
		return ax0

def _plot_histogram(series, bins=10, figsize=(6, 4), facecolor='#337ab7'):
	"""Plot an histogram from the data and return the AxesSubplot object.
	
	Parameters
	----------
	series : Series
		The data to plot
	figsize : tuple
		The size of the figure (width, height) in inches, default (6,4)
	facecolor : str
		The color code.
	
	Returns
	-------
	matplotlib.AxesSubplot
		The plot.
	"""
	if base.get_vartype(series) == base.TYPE_DATE:
		# TODO: These calls should be merged
		fig = plt.figure(figsize=figsize)
		plot = fig.add_subplot(111)
		plot.set_ylabel('Frequency')
		try:
			plot.hist(series.dropna().values, facecolor=facecolor, bins=bins)
		except TypeError: # matplotlib 1.4 can't plot dates so will show empty plot instead
			pass
	else:
		plot = series.plot(kind='hist', figsize=figsize,
						   facecolor=facecolor,
						   bins=bins)  # TODO when running on server, send this off to a different thread
	return plot

def histogram(series, **kwargs):
	"""Plot an histogram of the data.
	
	Parameters
	----------
	series: Series
		The data to plot.
	
	Returns
	-------
	str
		The resulting image encoded as a string.
	"""
	imgdata = BytesIO()
	plot = _plot_histogram(series, **kwargs)
	plot.figure.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.1, wspace=0, hspace=0)
	plot.figure.savefig(imgdata)
	imgdata.seek(0)
	result_string = 'data:image/png;base64,' + quote(base64.b64encode(imgdata.getvalue()))
	# TODO Think about writing this to disk instead of caching them in strings
	plt.close(plot.figure)
	return result_string

def mini_histogram(series, **kwargs):
	"""Plot a small (mini) histogram of the data.
	
	Parameters
	----------
	series: Series
		The data to plot.
	
	Returns
	-------
	str
		The resulting image encoded as a string.
	"""
	imgdata = BytesIO()
	#plot = _plot_histogram(series, figsize=(2, 0.75), **kwargs)
	plot = _plot_histogram(series, figsize=(4, 2), **kwargs)
	#plot.axes.get_yaxis().set_visible(False)
	
	if LooseVersion(matplotlib.__version__) <= '1.5.9':
		plot.set_axis_bgcolor("w")
	else:
		plot.set_facecolor("w")
	
	xticks = plot.xaxis.get_major_ticks()
	#for tick in xticks[1:-1]:
	#	tick.set_visible(False)
	#	tick.label.set_visible(False)
	for tick in (xticks[0], xticks[-1]):
		tick.label.set_fontsize(8)
	every_nth = 2
	for n, label in enumerate(plot.xaxis.get_ticklabels()):
		if n % every_nth == 0:
			label.set_visible(False)
	#plot.figure.subplots_adjust(left=0.15, right=0.85, top=1, bottom=0.35, wspace=0, hspace=0)
	plot.figure.subplots_adjust(left=0.2, right=0.95, top=0.95 , wspace=0, hspace=0)
	plot.figure.savefig(imgdata)
	imgdata.seek(0)
	result_string = 'data:image/png;base64,' + quote(base64.b64encode(imgdata.getvalue()))
	plt.close(plot.figure)
	return result_string

def correlation_matrix(corrdf, title, **kwargs):
	"""Plot image of a matrix correlation.
	Parameters
	----------
	corrdf: DataFrame
		The matrix correlation to plot.
	title: str
		The matrix title
	Returns
	-------
	str, The resulting image encoded as a string.
	"""
	imgdata = BytesIO()
	fig_cor, axes_cor = plt.subplots(1, 1)
	labels = corrdf.columns
	N = 256
	blues = np.ones((N, 4))
	blues[:, 0] = np.linspace(1, 66/256, N)
	blues[:, 1] = np.linspace(1, 136/256, N)
	blues[:, 2] = np.linspace(1, 181/256, N)
	reds = np.ones((N, 4))
	reds[:, 0] = np.linspace(209/256, 1, N)
	reds[:, 1] = np.linspace(60/256, 1, N)
	reds[:, 2] = np.linspace(75/256, 1, N)
	newcmp = ListedColormap(np.concatenate((reds, blues)))
	matrix_image = axes_cor.imshow(corrdf, vmin=-1, vmax=1, interpolation="nearest", cmap=newcmp)
	plt.title(title, size=18)
	plt.colorbar(matrix_image)
	axes_cor.set_xticks(np.arange(0, corrdf.shape[0], corrdf.shape[0] * 1.0 / len(labels)))
	axes_cor.set_yticks(np.arange(0, corrdf.shape[1], corrdf.shape[1] * 1.0 / len(labels)))
	axes_cor.set_xticklabels(labels, rotation=90)
	axes_cor.set_yticklabels(labels)
	
	matrix_image.figure.savefig(imgdata, bbox_inches='tight')
	imgdata.seek(0)
	result_string = 'data:image/png;base64,' + quote(base64.b64encode(imgdata.getvalue()))
	plt.close(matrix_image.figure)
	return result_string

def missing_matrix(df, predictions = False):
	"""Plot a missingno matrix
	
	Parameters
	----------
	df: DataFrame
		The dataframe.
	
	Returns
	-------
	str
		The resulting image encoded as a string.
	"""
	imgdata = BytesIO()
	if(predictions):
		plot = matrix(df, predictions = True)
	else:
		plot = matrix(df)
	plot.figure.savefig(imgdata)
	imgdata.seek(0)
	result_string = 'data:image/png;base64,' + quote(base64.b64encode(imgdata.getvalue()))
	plt.close(plot.figure)
	return result_string

def missing_bar(df):
	"""Plot a missingno bar chart
	
	Parameters
	----------
	df: DataFrame
		The dataframe.
	
	Returns
	-------
	str
		The resulting image encoded as a string.
	"""
	imgdata = BytesIO()
	plot = msno.bar(df)
	plot.figure.savefig(imgdata)
	imgdata.seek(0)
	result_string = 'data:image/png;base64,' + quote(base64.b64encode(imgdata.getvalue()))
	plt.close(plot.figure)
	return result_string

def missing_heat(df):
	"""Plot a missingno heat map
	
	Parameters
	----------
	df: DataFrame
		The dataframe.
	
	Returns
	-------
	str
		The resulting image encoded as a string.
	"""
	imgdata = BytesIO()
	plot = msno.heatmap(df)
	plot.figure.savefig(imgdata)
	imgdata.seek(0)
	result_string = 'data:image/png;base64,' + quote(base64.b64encode(imgdata.getvalue()))
	plt.close(plot.figure)
	return result_string

def missing_dendrogram(df):
	"""Plot a missingno dendrogram
	
	Parameters
	----------
	df: DataFrame
		The dataframe.
	
	Returns
	-------
	str
		The resulting image encoded as a string.
	"""
	imgdata = BytesIO()
	plot = msno.dendrogram(df)
	plot.figure.savefig(imgdata)
	imgdata.seek(0)
	result_string = 'data:image/png;base64,' + quote(base64.b64encode(imgdata.getvalue()))
	plt.close(plot.figure)
	return result_string

def generate_report_visualizations(report):
	# Receives a report and returns it with all the matplotlib based visualizations
	
	try:
		# reset matplotlib style before use
		# Fails in matplotlib 1.4.x so plot might look bad
		matplotlib.style.use("default")
	except:
		pass
	matplotlib.style.use(resource_filename(__name__, "preprocessing_profiling.mplstyle"))
	
	report['missing_matrix'] = missing_matrix(report['dataframe']['modified'])
	
	# Generate the missing matrixes with the color coded prediction errors. For each strategy, a matrixes will be generated for the different ways to order the rows.
	for strategy in report['strategy_classifications']:
		df = report['dataframe']['test'].copy()
		df['pred'] = report['strategy_classifications'][strategy]['result']['pred']
		report['strategy_classifications'][strategy]['prediction_matrixes'] = [{"name": "Original Dataset", "image": missing_matrix(df, predictions = True), "quantity": np.count_nonzero(df.iloc[:, -1] != df.iloc[:, -2]), "total": df.shape[0]}]
		combinations = np.array(df[df.iloc[:, -1] != df.iloc[:, -2]].iloc[:, -2:].drop_duplicates()) # Select every distinct combination in the last two columns where they are not the same(the result is a list with every distinct actual-predicted pair that represents a wrong prediction)
		for pair in combinations:
			matrix = {"name": str(pair[0])+"→"+str(pair[1])}
			matrix['image'] = missing_matrix(df[(df.iloc[:, -2] == pair[0]) & (df.iloc[:, -1] == pair[-1])].append(df[~((df.iloc[:, -2] == pair[0]) & (df.iloc[:, -1] == pair[-1]))]), predictions = True) # Order the list with the prediction error in question on the top and generate the matrix
			matrix['quantity'] = df[(df.iloc[:, -2] == pair[0]) & (df.iloc[:, -1] == pair[-1])].shape[0]
			matrix['total'] = df.shape[0]
			report['strategy_classifications'][strategy]['prediction_matrixes'].append(matrix)
	
	return report
