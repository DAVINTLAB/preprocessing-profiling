# -*- coding: utf-8 -*-
import codecs
from io import StringIO
import pandas as pd
import numpy as np
import preprocessing_profiling.templates as templates
from .describe import describe
from .report import to_html

NO_OUTPUTFILE = "preprocessing_profiling.no_outputfile"
DEFAULT_OUTPUTFILE = "preprocessing_profiling.default_outputfile"

class ProfileReport(object):
	html = ''
	file = None

	def __init__(self, df, **kwargs):
		missingCount = {'?' : 0, 'na' : 0, 'n/a' : 0, 'empty' : 0, 'null' : 0}
		if kwargs.get("format_missing_values", True):
			is_datetime = []
			for column in df:
				if(type(df[column][0]) == str):
					df[column] = df[column].str.strip()
			for column in df:
				for j in range(len(df[column])):
					if(type(df[column][j]) == str):
						if(df[column][j] == "?"):
							missingCount['?'] += 1
							df.loc[j, column] = np.nan
						elif(df[column][j].lower() == "na"):
							missingCount['na'] += 1
							df.loc[j, column] = np.nan
						elif(df[column][j].lower() == "n/a"):
							missingCount['n/a'] += 1
							df.loc[j, column] = np.nan
						elif(df[column][j].lower() == "empty"):
							missingCount['empty'] += 1
							df.loc[j, column] = np.nan
						elif(df[column][j].lower() == "null"):
							missingCount['null'] += 1
							df.loc[j, column] = np.nan
			df = pd.read_csv(StringIO(df.to_csv(index=False, date_format='%Y-%m-%d %H:%M:%S')), parse_dates=list(df.select_dtypes(include=[np.datetime64]).columns))
		if "format_missing_values" in kwargs:
			kwargs.pop("format_missing_values")
		
		sample = kwargs.get('sample', df.head())

		description_set = describe(df, **kwargs)

		self.html = to_html(sample,
							description_set)

		self.description_set = description_set
		

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
	
		if outputfile != NO_OUTPUTFILE:
			if outputfile == DEFAULT_OUTPUTFILE:
				outputfile = 'profile_' + str(hash(self)) + ".html"
			# TODO: should be done in the template
			with codecs.open(outputfile, 'w+b', encoding='utf8') as self.file:
				self.file.write(templates.template('wrapper').render(content=self.html))

	def to_html(self):
		"""Generate and return complete template as lengthy string
			for using with frameworks.
		
		Returns
		-------
		str
			The HTML output.
		"""
		return templates.template('wrapper').render(content=self.html)

	def _repr_html_(self):
		"""Used to output the HTML representation to a Jupyter notebook
		
		Returns
		-------
		str
			The HTML internal representation.
		"""
		return self.html

	def __str__(self):
		"""Overwrite of the str method.
		
		Returns
		-------
		str
			A string representation of the object.
		"""
		return "Output written to file " + str(self.file.name)
