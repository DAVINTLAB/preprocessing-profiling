# Preprocessing Profiling [![](https://img.shields.io/badge/python-3-blue.svg)](https://www.python.org/)

Preprocessing Profiling is a tool for evaluating the results of different preprocessing techniques on Tabular Datasets. When a dataset with missing values is received, a machine learning algorithm will be tested for a series of possible imputation techniques. When the received dataset has no missing values, the missing values will be created at random (just for exercise purpose). Next, the results of the testing are displayed in an organized report with various visualizations, e.g., Nullity Matrix, Classification Report, Confusion Matrix, and other options explained further in this document.

- [Installation](#installation)
- [Usage](#usage)
  - [Getting started](#getting-started)
  - [Visualizations](#visualizations)
    - [Nullity matrix](#nullity-matrix)
    - [Classification report](#classification-report)
    - [Confusion matrix](#confusion-matrix)
    - [Error distribution](#error-distribution)
    - [Flow of classes](#flow-of-classes)
    - [Multiple strategy flow of classes](#multiple-strategy-flow-of-classes)
    - [Matrix of nullity + class prediction error](#matrix-of-nullity--class-prediction-error)
  - [Other functionalities](#other-functionalities)
- [Performance](#performance)
- [Documentation](#documentation)
- [Dependencies](#dependencies)
- [Backlog](#backlog)
- [Citation](#citation)
- [About the Authors](#about-the-authors)

## Installation

Preprocessing Profiling can be installed by running ```pip install https://github.com/DAVINTLAB/preprocessing-profiling/archive/master.zip```.

## Usage

Preprocessing Profiling will return its report in the form of a page written in HTML like [this one](https://github.com/DAVINTLAB/preprocessing-profiling/blob/master/example_report.html).

### Getting started

The use of Jupyter Notebook is recommended as it can make the experience more interactive. The first step is to import the necessary libraries.
```python
import pandas as pd
from preprocessing_profiling import ProfileReport
```
A [pandas](https://pandas.pydata.org/) dataframe will serve as the dataset that will be used to generate the report. In this example, we are using the Iris dataset.
```python
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", encoding='UTF-8')
```
In Jupyter Notebook, simply calling the report will display it.
```python
ProfileReport(df)
```
Otherwise, a file can be written.
```python
ProfileReport(df).to_file(outputfile = "./path/to/file.html")
```

### Visualizations

#### Nullity matrix

This matrix from [missingno](https://github.com/ResidentMario/missingno) is a way of visualizing the distribution of missing values. It is particularly useful in identifying patterns of the missing values in the data. Missing values are displayed in white and regular entries are displayed in black.

![alt text][missingno matrix]

[missingno matrix]: https://imgur.com/kzd21sT.png

#### Classification report

The classification report will display the main machine learning classification metrics. The precision, recall, f1-score and support of each individual class can be seen.

![alt text][classification report]

[classification report]: https://imgur.com/wAwmnbZ.png

#### Confusion matrix

The confusion matrix shows the predictions in a matrix. In this matrix, the rows represent the actual classes and the columns represent the predictions that were made. The main diagonal contains the correct predictions and is shown in blue. Every other prediction is shown in red. In the example below, we can see two instances where the class was virginica but the algorithm classified them as versicolor.

![alt text][confusion matrix]

[confusion matrix]: https://imgur.com/WLvz1Ft.png

#### Error distribution

In this stacked bar chart, the distribution of the classes for each prediction can be seen. The actual classes are color coded and each stack represents one of the possible classes for prediction.

![alt text][error distribution]

[error distribution]: https://imgur.com/HcSvK4u.png

#### Flow of classes

This sankey diagram will show the flow between the actual classes (left) and the predicted classes (right). Correct predictions are displayed in blue and the incorrect ones are displayed in yellow.

![alt text][flow of classes]

[flow of classes]: https://imgur.com/TDtfq9N.png

#### Multiple strategy flow of classes

A variation of the flow of classes where, instead of a single strategy being covered, all the strategies are displayed side by side. This format favors comparisons between different strategies.

![alt text][big flow of classes]

[big flow of classes]: https://imgur.com/SWQX75P.png

#### Matrix of nullity + class prediction error

A variation of the nullity matrix where the prediction errors are color-coded. This visualiztion facilitates the process of identification of correlations between the missing values and the prediction errors.

![alt text][prediction matrix]

[prediction matrix]: https://imgur.com/H25rGIc.png

### Other functionalities

It is possible to choose the model that will be tested.
```python
ProfileReport(df, model = "MLPClassifier")
```
A [scikit-learn](https://scikit-learn.org/stable/) classifier will be accepted too.
```python
from sklearn.svm import SVC
svc = SVC(gamma = 'auto')
ProfileReport(df, model = svc)
```

## Performance

### Overview of the Results

| ![alt text][cervical_generation] | ![alt text][cervical_loading] |
|----------------------------------|-------------------------------|
| ![alt text][iris_generation]     | ![alt text][iris_loading]     |

[cervical_generation]: https://imgur.com/CgnkMRb.png

[iris_generation]: https://imgur.com/pHlrqWx.png

[cervical_loading]: https://imgur.com/HzAYQNi.png

[iris_loading]: https://imgur.com/2JNZKrP.png

Please find more details about the tests perfomed on [here](https://docs.google.com/spreadsheets/d/1DT6cxFVpfRAD2uzwX8e0A_pXuf1M-1XP2zUKO0os3SY/edit?usp=sharing).

## Documentation

The documentation can be found [here](https://github.com/DAVINTLAB/preprocessing-profiling/blob/master/Documentation.md).

## Dependencies

[Python 3](https://www.python.org/) is required in order to run Preprocessing Profiling. Also, the following Python libraries are used:

| Library                                               | Version |
|-------------------------------------------------------|---------|
| [pandas](https://pandas.pydata.org/)                  | 0.23.4  |
| [numpy](https://numpy.org/)                           | 1.15.4  |
| [matplotlib](https://matplotlib.org/2.1.2/index.html) | 3.0.2   |
| [jinja](http://jinja.pocoo.org/docs/2.10/)            | 2.10.0  |
| [sklearn](https://scikit-learn.org/stable/)           | 0.21.2  |

Internet access is necessary to load the JavaScript libraries. The following JavaScript libraries are used:

| Library                                      | Version |
|----------------------------------------------|---------|
| [d3](https://d3js.org/)                      | 5.9.7   |
| [d3 array](https://github.com/d3/d3-array)   | 1.2.4   |
| [d3 path](https://github.com/d3/d3-path)     | 1.0.7   |
| [d3 shape](https://github.com/d3/d3-shape)   | 1.3.5   |
| [d3 sankey](https://github.com/d3/d3-sankey) | 0.12.1  |
| [jquery](https://jquery.com/)                | 3.4.1   |
| [bootstrap](https://getbootstrap.com/)       | 3.3.6   |

## Backlog

It is the first version of this tool. However, we have already identified different items to be considered as part of the backlog for future releases. These items are summarized in the list below.

- A new feature to allow the user to upload their dataset and then show all the fields with correspondent types and the information if there are missing values or not for each column. Next, the tool should allow the user to select the imputation strategy is desired for each field. 
- Usability enhancements are also planned to increase the interactivity of the visualizations, particularly for "Flow of classes" and "Matrix of nullity + class prediction error" visualizations.
- Issues with downloading/saving the report in Google Chrome and the Jupyter Notebook versions.
- A new horizontal menu fixed on the top of the report.


## Citation

Please refer to this work by citing the dissertation indicate below.

Milani, A. M. P.. Preprocessing profiling model for visual analytics. http://tede2.pucrs.br/tede2/handle/tede/9007. 2019.


## About the Authors

We are members of the Data Visualization and Interaction Lab (DaVInt) at PUCRS:
- Isabel H. Manssour -- Professor Coordinator of DaVInt -- 2017-current.
- Alessandra M. P. Milani -- Master Student in Computer Science -- 2017-2019. 
- Lucas A. Loges -- Undergraduate Student in Computer Science -- 2019-current.

More information can be found [here](https://www.inf.pucrs.br/davint/).

