# -*- coding: utf-8 -*-
"""Compute statistical description of datasets"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn import tree
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ConfusionMatrix
from yellowbrick.classifier import ClassPredictionError
from yellowbrick.classifier import PrecisionRecallCurve
from pkg_resources import resource_filename
import preprocessing_profiling.base as base
from preprocessing_profiling.plot import yellowbrick_to_img

def get_classificationReportDecisionTree(x_train, x_test, y_train, y_test, classes):
	'''
	This function is trainning the model with the received parameters.
	Then it evaluate the results of prediction and print/plot the Reports: 
		ClassificationReport, ConfusionMatrix, ClassPredictionError, and PrecisionRecallCurve.
	
	It is using the yellowbrick.classifier library.
	
	The split of the dataset is happening out of this function to make sure 
	all the reports will be using the same data, for comparison purposes.
	
	@param x_train: contains the training features
	@param x_test: contains the testing features
	@param y_train: contains the training label
	@param y_test: contains the testing labels   
	@param classes: contains the list of classes/labels   
	
	@return: none (it is using print + yellowbrick plot)
	'''
	
	#Creating the model
	model=tree.tree.DecisionTreeClassifier()
	model.fit(x_train, y_train)

	#Getting the predictions from the trained model
	y_pred = model.predict(x_test)

	#Showing the Classification Report (Text)
	result ={}
	result['classificationReportText'] = classification_report(y_test, y_pred)
	result['accuracy'] = round(accuracy_score(y_test, y_pred) * 100, 1)
	
	'''
		The yellowbrick_to_img and plt.close methods 
		shouldn't be called from here. Unfortunately, 
		a bug in yellowbrick and the way it is structured
		make it impossible to pass the objects forward and
		call these methods later in report.py.
	'''
	
	#Showing the Classification Report (Chart)
	visualizer = ClassificationReport(model, classes=classes, cmap='RdBu', support=True)
	visualizer.score(x_test, y_test)  # Evaluate the model on the test data
	result['classificationReport'] = yellowbrick_to_img(visualizer)
	plt.close()
	
	# The ConfusionMatrix visualizer taxes a model
	#cm = ConfusionMatrix(model, classes=classes, label_encoder={0: 'setosa', 1: 'versicolor', 2: 'virginica'})
	cm = ConfusionMatrix(model, classes=classes, cmap='RdBu')
	cm.score(x_test, y_test)  # Evaluate the model on the test data
	result['confusionMatrix'] = yellowbrick_to_img(cm, rearrange_x_labels = True)
	plt.close()
	
	# The Class Prediction Error Distribution
	cp = ClassPredictionError(model, classes=classes)
	cp.score(x_test, y_test)  # Evaluate the model on the test data
	result['errorDistribution'] = yellowbrick_to_img(cp)
	plt.close()

	# Precision-Recall Curves
	pc = PrecisionRecallCurve(model, classes=classes)
	pc.fit(x_train, y_train)  # Fit the visualizer and the model
	pc.score(x_test, y_test)  # Evaluate the model on the test data
	result['precisionRecall'] = yellowbrick_to_img(pc)
	plt.close()

	# Precision-Recall Curves Multi-Label Classification  
	if (classes.size > 2):
		pc2 = PrecisionRecallCurve(model, per_class=True, iso_f1_curves=True, fill_area=False, micro=False)
		pc2.fit(x_train, y_train)  # Fit the visualizer and the model
		pc2.score(x_test, y_test)  # Evaluate the model on the test data
		result['precisionRecallInd'] = yellowbrick_to_img(pc2)
		plt.close()
	
	# Dictionary used to generate the sankey diagram
	nodes = []
	for e in cp._ClassificationScoreVisualizer__classes:
		nodes.append({"name": e})
	nodes += nodes
	links = []
	for i in range(0, len(cp.predictions_)):
		for j in range(0, len(cp.predictions_[i])):
			if cp.predictions_[i][j] != 0:
				links.append({"source": i, "target": len(cp.predictions_) + j, "value": cp.predictions_[i][j], "occurences": x_test[(y_test == nodes[i]['name']) & (y_pred == nodes[len(cp.predictions_) + j]['name'])].values.tolist()})
	result['errorDistributionDict'] = {"nodes": nodes, "links": links}
	
	return result

def get_ComparisonDecisionTree(dataset):
	
	'''
	This function is receiving the dataset (Pandas daframe) and the main actions are:
		1) validate if the dataset has missing values, in case not then generate missing values
		2) run the baseline = classification without any missing values
		3) run the model/classification for each preprocessing strategy
		4) plot all the details/results
			
	@param dataset: contains the dataset to be evaluated
	
	@return: none (it is using print + the output of other functions)
	'''	   
	
	#---------------------------------------------------------------------------
	# Pending implementation:
	# *** confirm if the y label is string, in case yes, then change to numbers
	# *** review the code for classes variable  
	#---------------------------------------------------------------------------
	
	
	#---------------------------------------------------------------------------
	# variables
	#---------------------------------------------------------------------------
	is_missing = dataset.isnull().values.any()
	X_full = np.array(dataset.iloc[:,:-1])
	y_full = np.array(dataset.iloc[:,-1])
	classifications = {}
	
	# *** review after the latest code changes!
	y = pd.DataFrame(y_full)
	classes = y[0].unique()
	
	
	#---------------------------------------------------------------------------
	# Baseline
	# The Baseline are running with a different split of train and test data!!!!
	#---------------------------------------------------------------------------
	
	# Check if the original dataset has missing values
	if (is_missing==False):

		# In case of the original dataset without missing (e.g., iris) then use it as is to run the baseline
		x_trainB, x_testB, y_trainB, y_testB = train_test_split(X_full, y_full, test_size=0.30)
		x_train0 = pd.DataFrame(x_trainB)
		x_test0 = pd.DataFrame(x_testB)
		classifications['baseline'] = get_classificationReportDecisionTree(x_train0, x_test0, y_trainB, y_testB, classes)
		classifications['baseline']['stratName'] = "Baseline (without missing values)"
		classifications['baseline']['stratCode'] = "0"
		#---------------------------------------------------------------------------
		# Add missing values in 75% of the lines randomly
		#---------------------------------------------------------------------------
		n_samples = X_full.shape[0]
		n_features = X_full.shape[1]
		rng = np.random.RandomState(0)
		missing_rate = 0.75
		
		n_missing_samples = int(np.floor(n_samples * missing_rate))
		missing_samples = np.hstack((np.zeros(n_samples - n_missing_samples, dtype=np.bool), np.ones(n_missing_samples,dtype=np.bool)))
		rng.shuffle(missing_samples)
		missing_features = rng.randint(0, n_features, n_missing_samples)
		X_missing = X_full.copy()
		X_missing[np.where(missing_samples)[0], missing_features] = np.nan
	
	else:

		# If originall missing data then clean all this lines and run baseline
		df_clean = dataset.dropna()
		X_baseline = np.array(df_clean.iloc[:,:-1])
		y_baseline = np.array(df_clean.iloc[:,-1])
		x_trainB, x_testB, y_trainB, y_testB = train_test_split(X_baseline, y_baseline, test_size=0.30)
		#*** review after the latest code changes! acho que nao precisa mais disso - pode passar direto!
		x_train0 = pd.DataFrame(x_trainB)
		x_test0 = pd.DataFrame(x_testB)
		classifications['baseline'] = get_classificationReportDecisionTree(x_train0, x_test0, y_trainB, y_testB, classes)
		classifications['baseline']['stratName'] = "Baseline (without missing values)"
		classifications['baseline']['stratCode'] = "0"
		
		#All the other tests will continue with the same variable name
		X_missing = X_full.copy()
		
	#No Changes to the class/label column
	y_missing = y_full.copy()
	
	
	#---------------------------------------------------------------------------
	# Spliting the data of training (70%) and test (30%)
	# Same set of data for all preprocessing comparisons! =)
	#---------------------------------------------------------------------------
	x_train, x_test, y_train, y_test = train_test_split(X_missing, y_missing, test_size=0.30)

	#---------------------------------------------------------------------------
	# replacing missing values by 0
	# If “constant”, then replace missing values with fill_value. Can be used with strings or numeric data.
	#---------------------------------------------------------------------------
	imp1 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
	x_train1 = pd.DataFrame(imp1.fit_transform(x_train)) # Rever!! pq transforma pandas?
	x_test1 = pd.DataFrame(imp1.fit_transform(x_test))
	classifications['constant'] = get_classificationReportDecisionTree(x_train1, x_test1, y_train, y_test, classes)
	classifications['constant']['stratName'] = "Constant Imputation (= zero)"
	classifications['constant']['stratCode'] = "1"

	if(not base.has_bool(dataset)):
		#---------------------------------------------------------------------------
		# Imputation (mean strategy) of the missing values
		# If “mean”, then replace missing values using the mean along each column. Can only be used with numeric data.
		#---------------------------------------------------------------------------
		imp2 = SimpleImputer(missing_values=np.nan, strategy="mean")
		x_train2 = pd.DataFrame(imp2.fit_transform(x_train)) # Rever!! pq transforma pandas?
		x_test2 = pd.DataFrame(imp2.fit_transform(x_test))
		classifications['mean'] = get_classificationReportDecisionTree(x_train2, x_test2, y_train, y_test, classes)
		classifications['mean']['stratName'] = "Mean Imputation"
		classifications['mean']['stratCode'] = "2"


		#---------------------------------------------------------------------------
		# Estimate the score after imputation (median strategy) of the missing values
		# If “median”, then replace missing values using the median along each column. Can only be used with numeric data.
		#---------------------------------------------------------------------------
		imp3 = SimpleImputer(missing_values=np.nan, strategy="median")
		x_train3 = pd.DataFrame(imp3.fit_transform(x_train)) # Rever!! pq transforma pandas?
		x_test3 = pd.DataFrame(imp3.fit_transform(x_test))
		classifications['median'] = get_classificationReportDecisionTree(x_train3, x_test3, y_train, y_test, classes)
		classifications['median']['stratName'] = "Median Imputation"
		classifications['median']['stratCode'] = "3"

	#---------------------------------------------------------------------------
	# Estimate the score after imputation (most_frequent strategy) of the missing values
	# If “most_frequent”, then replace missing using the most frequent value along each column. Can be used with strings or numeric data.
	#---------------------------------------------------------------------------
	imp4 = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
	x_train4 = pd.DataFrame(imp4.fit_transform(x_train)) # Rever!! pq transforma pandas?
	x_test4 = pd.DataFrame(imp4.fit_transform(x_test))
	classifications['mostFrequent'] = get_classificationReportDecisionTree(x_train4, x_test4, y_train, y_test, classes)
	classifications['mostFrequent']['stratName'] = "Most Frequent Imputation"
	classifications['mostFrequent']['stratCode'] = "4"
	
	df = pd.DataFrame(X_missing)
	df['y'] = y_missing
	
	generated_missing_values = not is_missing
	
	model=tree.tree.DecisionTreeClassifier()
	model.fit(x_train1, y_train)
	y_pred = model.predict(x_test1)
	test_results = {"default": pd.DataFrame(x_test)}
	test_results['default']['y'] = y_test
	test_results['default']['yPred'] = y_pred
	for classA in classes:
		for classB in classes:
			if classA != classB and np.any((test_results['default']['y'] == classA) & (test_results['default']['yPred'] == classB)):
				test_results[str(classA)+"->"+str(classB)] = test_results['default'][((test_results['default']['y'] == classA) & (test_results['default']['yPred'] == classB))].append(test_results['default'][((test_results['default']['y'] != classA) | (test_results['default']['yPred'] != classB))], ignore_index=True)
	return classifications, df, generated_missing_values, test_results

def describe(df):

	if not isinstance(df, pd.DataFrame):
		raise TypeError("df must be of type pandas.DataFrame")
	if df.empty:
		raise ValueError("df can not be empty")

	try:
		# reset matplotlib style before use
		# Fails in matplotlib 1.4.x so plot might look bad
		matplotlib.style.use("default")
	except:
		pass

	matplotlib.style.use(resource_filename(__name__, "preprocessing_profiling.mplstyle"))

	# Clearing the cache before computing stats
	base.clear_cache()
	
	classifications, df_missing, generated_missing_values, test_results = get_ComparisonDecisionTree(df)

	return {
		'dataframe': df_missing,
		'classifications': classifications,
		'generated_missing_values': generated_missing_values,
		'test_results': test_results
	}
