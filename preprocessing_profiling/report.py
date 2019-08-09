# -*- coding: utf-8 -*-
import json
import preprocessing_profiling.templates as templates
from preprocessing_profiling.plot import missing_matrix
from preprocessing_profiling.base import default


def to_html(sample, stats_object):
	info_html = templates.template('info').render(sample_table_html=sample.to_html(classes="sample"), missingMatrix = missing_matrix(stats_object['dataframe']), generated_missing_values = stats_object['generated_missing_values'])
	
	overview_rows_html = ""
	for strategy in stats_object['classifications']:
		overview_rows_html += templates.template('overview_row').render(content = stats_object['classifications'][strategy])
	overview_html = templates.template('overview').render(tableContent = overview_rows_html)
	
	classifications_html = ""
	count = 0
	for strategy in stats_object['classifications']:
		count += 1
		if(count == 4):
			classifications_html += templates.template('more_classifications_header').render()
		classification = stats_object['classifications'][strategy].copy()
		classification['errorDistributionDict'] = json.dumps(stats_object['classifications'][strategy]['errorDistributionDict'], default = default)
		classifications_html += templates.template('classification').render(classification = classification)
	if(count >=4):
		classifications_html += templates.template('more_classifications_footer').render()
	
	error_distribution_dicts = []
	for strategy in stats_object['classifications']:
		error_distribution_dicts.append(stats_object['classifications'][strategy]['errorDistributionDict'])
		error_distribution_dicts[-1]['strategy'] = stats_object['classifications'][strategy]['stratName']
	
	missingMatrixes = []
	try:
		for k in stats_object['test_results']:
			missingMatrixes.append({"name": k, "matrix": missing_matrix(stats_object['test_results'][k], predictions=True)})
	except:
		missingMatrixes = [{"name": "default", "matrix": missing_matrix(stats_object['test_results']['default'])}]
	
	diving_html = templates.template('diving').render(error_distribution_dicts = error_distribution_dicts, missingMatrixes = json.dumps(missingMatrixes))
	
	return templates.template('base').render(info_html = info_html, overview_html = overview_html, classifications_html = classifications_html, diving_html = diving_html)
