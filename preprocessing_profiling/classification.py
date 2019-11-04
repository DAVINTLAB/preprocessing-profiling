# -*- coding: utf-8 -*-
"""Run a machine learning algorithm for the baseline and the imputation strategies and generate a report for each set of results"""
import numpy as np
import pandas as pd
import random
from sklearn.impute import SimpleImputer
from sklearn import tree
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from preprocessing_profiling.base import generate_missing_values, clear_cache, has_bool

def decision_tree_report(df):
	# Run predictions using a decision tree and generate a report of the results
	
	report = {"dataframe": df}
	
	# Split the dataframe into x and y. Then split x and y into train and test.
	x = np.array(df.iloc[:, :-1])
	y = np.array(df.iloc[:, -1])
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = seed)
	
	model = tree.tree.DecisionTreeClassifier()
	model.fit(x_train, y_train)
	y_pred = model.predict(x_test)
	
	# Create a dataframe with the test rows and an additional column that contains the predictions made by the model
	test = pd.DataFrame(x_test, columns = df.columns[:-1])
	test[df.columns[-1]] = y_test
	test['pred'] = y_pred
	report['result'] = test
	
	# Create a dictionary that stores the number of occurrences for each Actual -> Predicted pair.
	nodes = []
	for classification in np.sort(test[df.columns[-1]].unique()):
		nodes.append({"name": classification})
	nodes += nodes
	links = []
	for i in range(0, len(nodes) // 2):
		for j in range(len(nodes) // 2, len(nodes)):
			occurences = test[(test.iloc[:, -2] == nodes[i]['name']) & (test.iloc[:, -1] == nodes[j]['name'])].iloc[:, :-2]
			if(len(occurences) > 0):
				links.append({"source":i, "target": j, "value": len(occurences), "occurences": occurences.values.tolist()})
	report['error_distribution_dict'] = {"nodes": nodes, "links": links, "variables": df.columns.tolist()[:-1]}
	
	report['classification_report'] = classification_report(y_test, y_pred, output_dict = True)
	report['accuracy'] = round(accuracy_score(y_test, y_pred) * 100, 1)
	
	report['split'] = {"x_train": x_train, "x_test": x_test, "y_train": y_train, "y_test": y_test}
	
	return report

def strategy_comparison(df):

	def run_baseline(df):
		# Select only the entries with no missing values and generate report
		
		df = df.copy()
		
		df = df.dropna()
		
		report = decision_tree_report(df)
		
		report['strat_code'] = 0
		report['strat_name'] = "Baseline (without missing values)"
		
		return report
	
	def run_strategy(strategy, df):
		# Impute missing values and generate report
		
		df = df.copy()
		
		df.iloc[:,:-1] = SimpleImputer(strategy=strategy).fit_transform(df.iloc[:, :-1].values) # Imputes all the missing entries using the requested strategy
		
		report = decision_tree_report(df)
		
		if(strategy == "mean"):
			report['strat_code'] = 1
			report['strat_name'] = "Mean Imputation"
		elif(strategy == "median"):
			report['strat_code'] = 2
			report['strat_name'] = "Median Imputation"
		elif(strategy == "most_frequent"):
			report['strat_code'] = 3
			report['strat_name'] = "Most Frequent Imputation"
		elif(strategy == "constant"):
			report['strat_code'] = 4
			report['strat_name'] = "Constant Imputation (= zero)"
		
		return report
	
	# Clearing the cache before computing stats
	clear_cache()
	
	global seed
	seed = random.randint(1, 1000000)
	
	report = {"dataframe":{"original": df}}
	
	report['baseline'] = run_baseline(df)
	
	# If the dataframe doesn't have missing values, generate missing values. Otherwise, use the existing ones.
	report['generated_missing_values'] = not df.isnull().values.any()
	if(report['generated_missing_values']):
		df = generate_missing_values(df, 0.75)
	report['dataframe']['modified'] = df
	x = np.array(df.iloc[:, :-1])
	y = np.array(df.iloc[:, -1])
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = seed)
	report['dataframe']['test'] = pd.DataFrame(x_test, columns = df.columns[:-1])
	report['dataframe']['test'][df.columns[-1]] = y_test
	
	report['strategy_classifications'] = {}
	report['strategy_classifications']['mean'] = run_strategy("mean", df)
	if(not has_bool(df)):
		report['strategy_classifications']['median'] = run_strategy("median", df)
		report['strategy_classifications']['mode'] = run_strategy("most_frequent", df)
	report['strategy_classifications']['zero'] = run_strategy("constant", df)
	
	return report
