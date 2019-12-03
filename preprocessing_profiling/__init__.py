# -*- coding: utf-8 -*-
import warnings
import codecs
import pandas as pd
from .classification import strategy_comparison
import preprocessing_profiling.html as html
from .base import infer_missing_entries
from .plot import generate_report_visualizations
from IPython.display import display

DEFAULT_OUTPUTFILE = "report.html"

class ProfileReport(object):
	html = ''
	file = None
	
	def __init__(self, df, format_missing_values = True, model="DecisionTreeClassifier"):
		if not isinstance(df, pd.DataFrame):
			raise TypeError("df must be of type pandas.DataFrame")
		if df.empty:
			raise ValueError("df can not be empty")
		
		if (format_missing_values):
			df = infer_missing_entries(df)
		
		with warnings.catch_warnings(record=True) as w:
			report = strategy_comparison(df, model)
		messages = set(map(lambda warning: warning.message.__str__(), w))
		
		report = generate_report_visualizations(report)
		
		self.html = html.report(report, messages)
		
		self.description_set = report
		
	
	def get_description(self):
		"""Return the description (a raw statistical summary) of the dataset.
	
		Returns
		-------
		dict
			Containing the following keys:
				* table: general statistics on the dataset
				* variables: summary statistics for each variable
				* freq: frequency table
		"""
		return self.description_set
	
	def get_rejected_variables(self, threshold=0.9):
		"""Return a list of variable names being rejected for high
		correlation with one of remaining variables.
	
		Parameters:
		----------
		threshold : float
			Correlation value which is above the threshold are rejected
		
		Returns
		-------
		list
			The list of rejected variables or an empty list if the correlation has not been computed.
		"""
		variable_profile = self.description_set['variables']
		result = []
		if hasattr(variable_profile, 'correlation'):
			result = variable_profile.index[variable_profile.correlation > threshold].tolist()
		return  result
	
	def to_file(self, outputfile=DEFAULT_OUTPUTFILE):
		"""Write the report to a file.
		
		By default a name is generated.
		
		Parameters:
		----------
		outputfile : str
			The name or the path of the file to generale including the extension (.html).
		"""
		file = open(outputfile, "w", encoding="utf8")
		file.write(self.to_html())
		file.close()
	
	def to_html(self):
		"""Generate and return complete template as lengthy string
			for using with frameworks.
		
		Returns
		-------
		str
			The HTML output.
		"""
		return html.wrap(self.html)
	
	def _repr_html_(self):
		"""Used to output the HTML representation to a Jupyter notebook
		
		Returns
		-------
		str
			The HTML internal representation.
		"""
		
		class Importer:
			def __init__(self):
				self.html = html.importer()
			def _repr_html_(self):
				return self.html
		display(Importer())
		
		file = open("report.html", "w", encoding="utf8")
		file.write(self.to_html())
		file.close()
		
		return self.html + html.downloadable()
	
	def __str__(self):
		"""Overwrite of the str method.
		
		Returns
		-------
		str
			A string representation of the object.
		"""
		return "Output written to file " + str(self.file.name)
