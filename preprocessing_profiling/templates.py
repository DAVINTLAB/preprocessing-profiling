# coding=UTF-8
"""Contains all templates used for generating the HTML profile report"""

from jinja2 import Environment, PackageLoader

# Initializing Jinja
pl = PackageLoader('preprocessing_profiling', 'templates')
jinja2_env = Environment(lstrip_blocks=True, trim_blocks=True, loader=pl)

# Mapping between template name and file
templates = {'info': 'info.html',
			 'base': 'base.html',
			 'classification': 'classification.html',
			 'overview': 'overview.html'
			 }

# Mapping between row type and var type
var_type = {'NUM': 'Numeric',
			'DATE': 'Date',
			'CAT': 'Categorical',
			'UNIQUE': 'Categorical, Unique',
			'BOOL': 'Boolean',
			'CONST': 'Constant',
			'CORR': 'Highly correlated',
			'RECODED': 'Recoded',
			'UNSUPPORTED': 'Unsupported'
			}


def template(template_name):
	"""Return a jinja template ready for rendering. If needed, global variables are initialized.

	Parameters
	----------
	template_name: str, the name of the template as defined in the templates mapping

	Returns
	-------
	The Jinja template ready for rendering
	"""
	globals = None
	if template_name.startswith('row_'):
		# This is a row template setting global variable
		globals = dict()
		globals['vartype'] = var_type[template_name.split('_')[1].upper()]
	return jinja2_env.get_template(templates[template_name], globals=globals)

# The number of column to use in the display of the frequency table according to the category
mini_freq_table_nb_col = {'CAT': 6, 'BOOL': 3}
