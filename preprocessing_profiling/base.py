# -*- coding: utf-8 -*-
"""Common parts to all other modules, mainly utility functions.
"""
import pandas as pd
import numpy as np
import random
from io import StringIO

TYPE_CAT = 'CAT'
"""String: A categorical variable"""

TYPE_BOOL = 'BOOL'
"""String: A boolean variable"""

TYPE_NUM = 'NUM'
"""String: A numerical variable"""

TYPE_DATE = 'DATE'
"""String: A numeric variable"""

S_TYPE_CONST = 'CONST'
"""String: A constant variable"""

S_TYPE_UNIQUE = 'UNIQUE'
"""String: A unique variable"""

S_TYPE_UNSUPPORTED = 'UNSUPPORTED'
"""String: An unsupported variable"""

_VALUE_COUNTS_MEMO = {}

def nullity_sort(df, sort=None, axis='columns'):
    """
    Sorts a DataFrame according to its nullity, in either ascending or descending order.

    :param df: The DataFrame object being sorted.
    :param sort: The sorting method: either "ascending", "descending", or None (default).
    :return: The nullity-sorted DataFrame.
    """
    if sort is None:
        return df
    elif sort not in ['ascending', 'descending']:
        raise ValueError('The "sort" parameter must be set to "ascending" or "descending".')

    if axis not in ['rows', 'columns']:
        raise ValueError('The "axis" parameter must be set to "rows" or "columns".')

    if axis == 'columns':
        if sort == 'ascending':
            return df.iloc[np.argsort(df.count(axis='columns').values), :]
        elif sort == 'descending':
            return df.iloc[np.flipud(np.argsort(df.count(axis='columns').values)), :]
    elif axis == 'rows':
        if sort == 'ascending':
            return df.iloc[:, np.argsort(df.count(axis='rows').values)]
        elif sort == 'descending':
            return df.iloc[:, np.flipud(np.argsort(df.count(axis='rows').values))]


def nullity_filter(df, filter=None, p=0, n=0):
    """
    Filters a DataFrame according to its nullity, using some combination of 'top' and 'bottom' numerical and
    percentage values. Percentages and numerical thresholds can be specified simultaneously: for example,
    to get a DataFrame with columns of at least 75% completeness but with no more than 5 columns, use
    `nullity_filter(df, filter='top', p=.75, n=5)`.

    :param df: The DataFrame whose columns are being filtered.
    :param filter: The orientation of the filter being applied to the DataFrame. One of, "top", "bottom",
    or None (default). The filter will simply return the DataFrame if you leave the filter argument unspecified or
    as None.
    :param p: A completeness ratio cut-off. If non-zero the filter will limit the DataFrame to columns with at least p
    completeness. Input should be in the range [0, 1].
    :param n: A numerical cut-off. If non-zero no more than this number of columns will be returned.
    :return: The nullity-filtered `DataFrame`.
    """
    if filter == 'top':
        if p:
            df = df.iloc[:, [c >= p for c in df.count(axis='rows').values / len(df)]]
        if n:
            df = df.iloc[:, np.sort(np.argsort(df.count(axis='rows').values)[-n:])]
    elif filter == 'bottom':
        if p:
            df = df.iloc[:, [c <= p for c in df.count(axis='rows').values / len(df)]]
        if n:
            df = df.iloc[:, np.sort(np.argsort(df.count(axis='rows').values)[:n])]
    return df

def get_groupby_statistic(data):
	"""Calculate value counts and distinct count of a variable (technically a Series).

	The result is cached by column name in a global variable to avoid recomputing.

	Parameters
	----------
	data : Series
		The data type of the Series.

	Returns
	-------
	list
		value count and distinct count
	"""
	if data.name is not None and data.name in _VALUE_COUNTS_MEMO:
		return _VALUE_COUNTS_MEMO[data.name]

	value_counts_with_nan = data.value_counts(dropna=False)
	value_counts_without_nan = value_counts_with_nan.loc[value_counts_with_nan.index.dropna()]
	distinct_count_with_nan = value_counts_with_nan.count()

	# When the inferred type of the index is just "mixed" probably the types within the series are tuple, dict, list and so on...
	if value_counts_without_nan.index.inferred_type == "mixed":
		raise TypeError('Not supported mixed type')

	result = [value_counts_without_nan, distinct_count_with_nan]

	if data.name is not None:
		_VALUE_COUNTS_MEMO[data.name] = result

	return result

_MEMO = {}
def get_vartype(data):
	"""Infer the type of a variable (technically a Series).

	The types supported are split in standard types and special types.

	Standard types:
		* Categorical (`TYPE_CAT`): the default type if no other one can be determined
		* Numerical (`TYPE_NUM`): if it contains numbers
		* Boolean (`TYPE_BOOL`): at this time only detected if it contains boolean values, see todo
		* Date (`TYPE_DATE`): if it contains datetime

	Special types:
		* Constant (`S_TYPE_CONST`): if all values in the variable are equal
		* Unique (`S_TYPE_UNIQUE`): if all values in the variable are different
		* Unsupported (`S_TYPE_UNSUPPORTED`): if the variable is unsupported

	 The result is cached by column name in a global variable to avoid recomputing.

	Parameters
	----------
	data : Series
		The data type of the Series.

	Returns
	-------
	str
		The data type of the Series.

	Notes
	----
		* #72: Numeric with low Distinct count should be treated as "Categorical"
	"""
	if data.name is not None and data.name in _MEMO:
		return _MEMO[data.name]

	vartype = None
	try:
		distinct_count = get_groupby_statistic(data)[1]
		leng = len(data)

		if distinct_count <= 1:
			vartype = S_TYPE_CONST
		elif pd.api.types.is_bool_dtype(data) or ((distinct_count == 2 or (distinct_count == 3 and data.hasnans)) and pd.api.types.is_numeric_dtype(data)):
			vartype = TYPE_BOOL
		elif pd.api.types.is_numeric_dtype(data):
			vartype = TYPE_NUM
		elif pd.api.types.is_datetime64_dtype(data):
			vartype = TYPE_DATE
		elif distinct_count == leng:
			vartype = S_TYPE_UNIQUE
		else:
			vartype = TYPE_CAT
	except:
		vartype = S_TYPE_UNSUPPORTED

	if data.name is not None:
		_MEMO[data.name] = vartype

	return vartype

def has_bool(df):
	for column in df.columns[0:-1]:
		if(get_vartype(df[column]) == TYPE_BOOL):
			return True
	return False

def clear_cache():
	"""Clear the cache stored as global variables"""
	global _MEMO, _VALUE_COUNTS_MEMO
	_MEMO = {}
	_VALUE_COUNTS_MEMO = {}

def infer_missing_entries(df):
	# Finds strings that represent a missing entry and replace them with missing values
	# TO DO: Rewrite this method without converting the data into a csv and parsing it again
	missingCount = {'?' : 0, 'na' : 0, 'n/a' : 0, 'empty' : 0, 'null' : 0}
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
	return pd.read_csv(StringIO(df.to_csv(index=False, date_format='%Y-%m-%d %H:%M:%S')), parse_dates=list(df.select_dtypes(include=[np.datetime64]).columns))

def generate_missing_values(df, p):
	# Generates missing values on p(a percentage) of the rows in the received dataframe(maximum of one per row)
	
	df = df.copy()
	
	to_generate = int(df.shape[0] * p)
	not_chosen = list(range(0, df.shape[0] - 1))
	columns = df.columns[:-1] # The y column shouldn't have missing values
	while(to_generate > 0):
		chosen = not_chosen.pop(random.randint(0, len(not_chosen) - 1))
		df.at[chosen, columns[random.randint(0, len(columns) - 1)]] = np.nan
		to_generate-=1
	return df
