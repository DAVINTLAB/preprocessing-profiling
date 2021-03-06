# -*- coding: utf-8 -*-
"""Uses the test results and the templates to create an HTML page with the information"""
from preprocessing_profiling.templates import template

def report(report, messages):
	
	messages_html = ""
	if(len(messages) > 0):
		for message in messages:
			messages_html += template('message').render(message = message)
		messages_html = template('messages').render(messages_html = messages_html)
	
	info_html = template('info').render(sample_table_html = report['dataframe']['modified'].head().to_html(classes="sample"), missing_matrix = report['missing_matrix'], generated_missing_values = report['generated_missing_values'])
	
	overview_rows_html = template('overview_row').render(classification = report['baseline'])
	for strategy in report['strategy_classifications']:
		overview_rows_html += template('overview_row').render(classification = report['strategy_classifications'][strategy])
	overview_html = template('overview').render(tableContent = overview_rows_html, model_name = report['model_name'])
	
	classifications_html = template('classification').render(classification = report['baseline'])
	count = 1
	for strategy in report['strategy_classifications']:
		count += 1
		if(count == 4):
			classifications_html += template('more_classifications_header').render()
		classifications_html += template('classification').render(classification = report['strategy_classifications'][strategy])
	if(count >= 4):
		classifications_html += template('more_classifications_footer').render()
	
	missing_matrixes = []
	for strategy in report['strategy_classifications']:
		missing_matrixes.append({"name": report['strategy_classifications'][strategy]['strat_name'], "missingMatrixes": report['strategy_classifications'][strategy]['prediction_matrixes']})
	
	report['baseline']['error_distribution_dict']['strategy'] = report['baseline']['strat_name']
	error_distribution_dicts = [report['baseline']['error_distribution_dict']]
	for strategy in report['strategy_classifications']:
		error_distribution_dicts.append(report['strategy_classifications'][strategy]['error_distribution_dict'])
		error_distribution_dicts[-1]['strategy'] = report['strategy_classifications'][strategy]['strat_name']
	
	diving_html = template('diving').render(missing_matrixes = missing_matrixes, error_distribution_dicts = error_distribution_dicts)
	
	return template('base').render(messages_html = messages_html, info_html = info_html, overview_html = overview_html, classifications_html = classifications_html, diving_html = diving_html)

def importer():
	return template('importer').render()

def wrap(content):
	return template('wrapper').render(content = content)

def downloadable():
	return template('downloadable').render()