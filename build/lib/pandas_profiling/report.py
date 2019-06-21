# -*- coding: utf-8 -*-
"""Generate reports"""
import sys
import six
import pandas as pd
import pandas_profiling.formatters as formatters
import pandas_profiling.templates as templates
import pandas_profiling.plot as plot
import datetime as dt


def to_html(sample, stats_object, missing_count):
    """Generate a HTML report from summary statistics and a given sample.

    Parameters
    ----------
    sample : DataFrame
        the sample you want to print
    stats_object : dict
        Summary statistics. Should be generated with an appropriate describe() function

    Returns
    -------
    str
        containing profile report in HTML format

    Notes
    -----
        * This function as to be refactored since it's huge and it contains inner functions
    """

    n_obs = stats_object['table']['n']

    value_formatters = formatters.value_formatters
    row_formatters = formatters.row_formatters

    if not isinstance(sample, pd.DataFrame):
        raise TypeError("sample must be of type pandas.DataFrame")

    if not isinstance(stats_object, dict):
        raise TypeError("stats_object must be of type dict. Did you generate this using the pandas_profiling.describe() function?")

    if not set({'table', 'variables', 'freq', 'correlations'}).issubset(set(stats_object.keys())):
        raise TypeError(
            "stats_object badly formatted. Did you generate this using the pandas_profiling.describe() function?")

    def fmt(value, name):
        if pd.isnull(value):
            return ""
        if name in value_formatters:
            return value_formatters[name](value)
        elif isinstance(value, float):
            return value_formatters[formatters.DEFAULT_FLOAT_FORMATTER](value)
        else:
            try:
                return unicode(value)  # Python 2
            except NameError:
                return str(value)      # Python 3
                

    def _format_row(freq, label, max_freq, row_template, n, extra_class=''):
            if max_freq != 0:
                width = int(freq / max_freq * 99) + 1
            else:
                width = 1

            if width > 20:
                label_in_bar = freq
                label_after_bar = ""
            else:
                label_in_bar = "&nbsp;"
                label_after_bar = freq

            return row_template.render(label=label,
                                       width=width,
                                       count=freq,
                                       percentage='{:2.1f}'.format(freq / n * 100),
                                       extra_class=extra_class,
                                       label_in_bar=label_in_bar,
                                       label_after_bar=label_after_bar)

    def freq_table(freqtable, n, table_template, row_template, max_number_to_print, nb_col=6):

        freq_rows_html = u''

        if max_number_to_print > n:
                max_number_to_print=n

        if max_number_to_print < len(freqtable):
            freq_other = sum(freqtable.iloc[max_number_to_print:])
            min_freq = freqtable.values[max_number_to_print]
        else:
            freq_other = 0
            min_freq = 0

        freq_missing = n - sum(freqtable)
        max_freq = max(freqtable.values[0], freq_other, freq_missing)

        # TODO: Correctly sort missing and other

        for label, freq in six.iteritems(freqtable.iloc[0:max_number_to_print]):
            freq_rows_html += _format_row(freq, label, max_freq, row_template, n)

        if freq_other > min_freq:
            freq_rows_html += _format_row(freq_other,
                                         "Other values (%s)" % (freqtable.count() - max_number_to_print), max_freq, row_template, n,
                                         extra_class='other')

        if freq_missing > min_freq:
            freq_rows_html += _format_row(freq_missing, "(Missing)", max_freq, row_template, n, extra_class='missing')

        return table_template.render(rows=freq_rows_html, varid=hash(idx), nb_col=nb_col)

    def extreme_obs_table(freqtable, table_template, row_template, number_to_print, n, ascending = True):

        # If it's mixed between base types (str, int) convert to str. Pure "mixed" types are filtered during type discovery
        if "mixed" in freqtable.index.inferred_type:
            freqtable.index = freqtable.index.astype(str)

        sorted_freqTable = freqtable.sort_index()

        if ascending:
            obs_to_print = sorted_freqTable.iloc[:number_to_print]
        else:
            obs_to_print = sorted_freqTable.iloc[-number_to_print:]

        freq_rows_html = ''
        max_freq = max(obs_to_print.values)

        for label, freq in six.iteritems(obs_to_print):
            freq_rows_html += _format_row(freq, label, max_freq, row_template, n)

        return table_template.render(rows=freq_rows_html)

    # Variables
    rows_html = u""
    messages = []
    variable = 0

    for idx, row in stats_object['variables'].iterrows():

        df = stats_object['dataframe']
        formatted_values = {'varname': idx, 'varid': hash(idx), 'column_values': df[idx].dropna().tolist()}
        row_classes = {}

        for col, value in six.iteritems(row):
            formatted_values[col] = fmt(value, col)

        for col in set(row.index) & six.viewkeys(row_formatters):
            row_classes[col] = row_formatters[col](row[col])
            if row_classes[col] == "alert" and col in templates.messages:
                messages.append(templates.messages[col].format(formatted_values, varname = idx))

        if row['type'] in {'CAT', 'BOOL'}:
            formatted_values['minifreqtable'] = freq_table(stats_object['freq'][idx], n_obs,
                                                           templates.template('mini_freq_table'), 
                                                           templates.template('mini_freq_table_row'), 
                                                           3, 
                                                           templates.mini_freq_table_nb_col[row['type']])

            if row['distinct_count'] > 50:
                messages.append(templates.messages['HIGH_CARDINALITY'].format(formatted_values, varname = idx))
                row_classes['distinct_count'] = "alert"
            else:
                row_classes['distinct_count'] = ""

        if row['type'] == 'UNIQUE':
            obs = stats_object['freq'][idx].index

            formatted_values['firstn'] = pd.DataFrame(obs[0:3], columns=["First 3 values"]).to_html(classes="example_values", index=False)
            formatted_values['lastn'] = pd.DataFrame(obs[-3:], columns=["Last 3 values"]).to_html(classes="example_values", index=False)
        
        aux = []
        if row['type'] == 'DATE':
            for date in formatted_values['column_values']:
                aux.append((date - dt.datetime(1970,1,1)).total_seconds())
            formatted_values['column_values'] = aux
        
        if row['type'] == 'UNSUPPORTED':
            formatted_values['varname'] = idx
            messages.append(templates.messages[row['type']].format(formatted_values))
        elif row['type'] in {'CORR', 'CONST', 'RECODED'}:
            formatted_values['varname'] = idx
            messages.append(templates.messages[row['type']].format(formatted_values))
        else:
            formatted_values['freqtable'] = freq_table(stats_object['freq'][idx], n_obs,
                                                       templates.template('freq_table'), templates.template('freq_table_row'), 10)
            formatted_values['firstn_expanded'] = extreme_obs_table(stats_object['freq'][idx], templates.template('freq_table'), templates.template('freq_table_row'), 5, n_obs, ascending = True)
            formatted_values['lastn_expanded'] = extreme_obs_table(stats_object['freq'][idx], templates.template('freq_table'), templates.template('freq_table_row'), 5, n_obs, ascending = False)

        if variable == 3:
            rows_html += templates.template('more_variables_header').render()
        rows_html += templates.row_templates_dict[row['type']].render(values=formatted_values, row_classes=row_classes)
        variable += 1
    if variable > 3:
        rows_html += templates.template('more_variables_footer').render()
    # Overview
    formatted_values = {k: fmt(v, k) for k, v in six.iteritems(stats_object['table'])}
    formatted_values['missing_count'] = missing_count
    row_classes={}
    for col in six.viewkeys(stats_object['table']) & six.viewkeys(row_formatters):
        row_classes[col] = row_formatters[col](stats_object['table'][col])
        if row_classes[col] == "alert" and col in templates.messages:
            messages.append(templates.messages[col].format(formatted_values, varname = idx))

    messages_html = u''
    for msg in messages:
        messages_html += templates.message_row.format(message=msg)

    overview_html = templates.template('overview').render(values=formatted_values, row_classes = row_classes, messages=messages_html)

    # Add plot of matrix correlation
    pearson_matrix = plot.correlation_matrix(stats_object['correlations']['pearson'], 'Pearson')
    spearman_matrix = plot.correlation_matrix(stats_object['correlations']['spearman'], 'Spearman')
    correlations_html = templates.template('correlations').render(
        values={'pearson_matrix': pearson_matrix, 'pearson_numeric': df.corr().to_html(), 'spearman_matrix': spearman_matrix})

    # Add sample
    sample_html = templates.template('sample').render(sample_table_html=sample.to_html(classes="sample"))
    # TODO: should be done in the template
    
    missing = df.isnull().values.any()
    if(missing):
        values = {}
        values['matrix'] = plot.missing_matrix(df)
        values['bar'] = plot.missing_bar(df)
        values['heat'] = plot.missing_heat(df)
        values['dendrogram'] = plot.missing_dendrogram(df)
        missing_html = templates.template('missing').render(values=values)
    
    sections = {
        'overview_html': overview_html,
        'rows_html': rows_html,
        'sample_html': sample_html,
        'correlation_html': correlations_html
    }
    if(missing):
        sections['missing_html'] = missing_html
    
    return templates.template('base').render(sections, values = {'missing': missing})
