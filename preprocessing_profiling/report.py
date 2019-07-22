# -*- coding: utf-8 -*-
import json
import preprocessing_profiling.templates as templates
from preprocessing_profiling.plot import missing_matrix
from preprocessing_profiling.base import default


def to_html(sample, stats_object):
	
	info_html = templates.template('info').render(sankey_data = json.dumps(stats_object['classifications']['baseline']['errorDistributionDict'], default = default), sample_table_html=sample.to_html(classes="sample"), missingMatrix = missing_matrix(stats_object['dataframe']))
	
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
		stats_object['classifications'][strategy]['errorDistributionDict'] = json.dumps(stats_object['classifications'][strategy]['errorDistributionDict'], default = default)
		classifications_html += templates.template('classification').render(classification = stats_object['classifications'][strategy])
	if(count >=4):
		classifications_html += templates.template('more_classifications_footer').render()
	
	return templates.template('base').render(info_html = info_html, overview_html = overview_html, classifications_html = classifications_html)
