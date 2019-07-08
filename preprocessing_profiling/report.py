# -*- coding: utf-8 -*-
import sys
import six
import pandas as pd
import preprocessing_profiling.formatters as formatters
import preprocessing_profiling.templates as templates
import datetime as dt
from preprocessing_profiling.plot import missing_matrix


def to_html(sample, stats_object):

	info_html = templates.template('info').render(sample_table_html=sample.to_html(classes="sample"), missingMatrix = missing_matrix(stats_object['dataframe']))
	
	overview_html = templates.template('overview').render(plots = stats_object['classifications'])
	
	# Baseline
	classifications_html = templates.template('classification').render(stratCode = "0", strategy = "Baseline", plots = stats_object['classifications']['baseline'])
	
	# Constant Imputation
	classifications_html += templates.template('classification').render(stratCode = "1", strategy = "Constant Imputation", plots = stats_object['classifications']['constant'])
	
	# Mean Imputation
	classifications_html += templates.template('classification').render(stratCode = "2", strategy = "Mean Imputation", plots = stats_object['classifications']['mean'])
	
	# Median Imputation
	classifications_html += templates.template('classification').render(stratCode = "3", strategy = "Median Imputation", plots = stats_object['classifications']['median'])
	
	# Most Frequent Imputation
	classifications_html += templates.template('classification').render(stratCode = "4", strategy = "Most Frequent Imputation", plots = stats_object['classifications']['mostFrequent'])
	
	return templates.template('base').render(info_html = info_html, overview_html = overview_html, classifications_html = classifications_html)
