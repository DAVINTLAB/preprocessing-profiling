# -*- coding: utf-8 -*-
import json
import preprocessing_profiling.templates as templates
from preprocessing_profiling.plot import missing_matrix
from preprocessing_profiling.base import default


def to_html(sample, stats_object):
	
	info_html = templates.template('info').render(sample_table_html=sample.to_html(classes="sample"), missingMatrix = missing_matrix(stats_object['dataframe']))
	
	overview_html = templates.template('overview').render(values = stats_object['classifications'])
	
	# Baseline
	stats_object['classifications']['baseline']['errorDistributionDict'] = json.dumps(stats_object['classifications']['baseline']['errorDistributionDict'], default = default)
	print(stats_object['classifications']['baseline']['errorDistributionDict'])
	classifications_html = templates.template('classification').render(stratCode = "0", strategy = "Baseline", plots = stats_object['classifications']['baseline'])
	
	# Constant Imputation
	stats_object['classifications']['constant']['errorDistributionDict'] = json.dumps(stats_object['classifications']['constant']['errorDistributionDict'], default = default)
	classifications_html += templates.template('classification').render(stratCode = "1", strategy = "Constant Imputation (= zero)", plots = stats_object['classifications']['constant'])
	
	# Mean Imputation
	stats_object['classifications']['mean']['errorDistributionDict'] = json.dumps(stats_object['classifications']['mean']['errorDistributionDict'], default = default)
	classifications_html += templates.template('classification').render(stratCode = "2", strategy = "Mean Imputation", plots = stats_object['classifications']['mean'])
	
	classifications_html += templates.template('more_classifications_header').render()
	
	# Median Imputation
	stats_object['classifications']['median']['errorDistributionDict'] = json.dumps(stats_object['classifications']['median']['errorDistributionDict'], default = default)
	classifications_html += templates.template('classification').render(stratCode = "3", strategy = "Median Imputation", plots = stats_object['classifications']['median'])
	
	# Most Frequent Imputation
	stats_object['classifications']['mostFrequent']['errorDistributionDict'] = json.dumps(stats_object['classifications']['mostFrequent']['errorDistributionDict'], default = default)
	classifications_html += templates.template('classification').render(stratCode = "4", strategy = "Most Frequent Imputation", plots = stats_object['classifications']['mostFrequent'])
	
	classifications_html += templates.template('more_classifications_footer').render()
	
	return templates.template('base').render(info_html = info_html, overview_html = overview_html, classifications_html = classifications_html)
