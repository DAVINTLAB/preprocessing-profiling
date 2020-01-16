## preprocessing\_profiling.ProfileReport<sub><sup>(df, format\_missing\_values=True, model=“DecisionTreeClassifier”)</sup></sub>

### Parameters

**df**: pandas dataframe

> The dataset itself.

**format\_missing\_values**: boolean, optional

> When **format\_missing\_values** is True, strings that (probably) represent missing values (such as “?”, “na”, “n/a”...) will be replaced with numpy.NaN.

**model**: string or scikit-learn classifier, optional

> The model which will be trained and used to predict the classes of the entries in the test set. Can be either a scikit-learn model or a string with the name of a scikit-learn model.
> 
> **Currently accepted strings**
> - “MLPClassifier”
> - “KNeighborsClassifier”
> - “SVC”
> - “GaussianProcessClassifier”
> - “DecisionTreeClassifier”
> - “RandomForestClassifier”
> - “AdaBoostClassifier”
> - “GaussianNB”
> - “QuadraticDiscriminantAnalysis”
> - “DummyClassifier”

### Returns

A **ProfileReport** object which contains methods to display the report in various ways.

### Methods

**to\_file**(outputfile=”report.html”)
> Writes the report to a file. The optional parameter **outputfile** defines the path to the file.

**\_repr\_html\_**()
> When the object is returned to IPython, this method will be called and it will return the report in a format that can be displayed inside, for instance, a Jupyter Notebook cell. The report will have a download button, allowing the user to save the report.*
> 
> *The download button may not work in Google Chrome because of a compatibility issue between Jupyter Notebook (version 5 or greater) and Chrome. To be able to save the report, use the **to\_file** method instead.

**to\_html()**
> Returns a string with the html of the report.
