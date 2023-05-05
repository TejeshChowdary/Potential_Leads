# Potential_Leads
This repository contains code for analyzing and selecting potential leads for ABC Education based on factors that contribute to a high conversion rate. Lead scores are assigned to prioritize potential leads with a higher probability of conversion.

We are using necessary libraries and specific classes and functions from popular Python libraries for data analysis and machine learning. 

- `numpy` is a library for numerical computing and provides support for arrays, matrices, and mathematical functions.
- `pandas` is a library for data manipulation and analysis, offering data structures and tools for handling tabular data.
- `matplotlib.pyplot` is a library for plotting data in Python and provides a variety of options for creating static visualizations.
- `seaborn` is a library for advanced data visualization built on top of matplotlib, offering more complex and aesthetically pleasing visualizations.
- `plotly` is a library for interactive data visualization, enabling the creation of interactive plots and dashboards.

For machine learning, the following specific classes and functions are imported from the scikit-learn library:

- `LogisticRegression` for performing logistic regression, a commonly used classification algorithm.
- `RandomForestClassifier` for performing random forest classification, another popular classification algorithm.
- `DecisionTreeRegressor` for performing decision tree regression, a type of regression algorithm.

Finally, functions for evaluating machine learning models are imported, including:

- `classification_report` for generating a report of classification metrics such as precision, recall, and F1 score.
- `roc_curve` and `auc` for computing the Receiver Operating Characteristic (ROC) curve and its area under the curve (AUC) respectively, which are used for evaluating classification performance.
- `confusion_matrix` and `ConfusionMatrixDisplay` for generating a confusion matrix and visualizing it.
- `cross_val_score` for performing cross-validation, which is a method for assessing how well a machine learning model generalizes to new data.
- `train_test_split` for splitting data into training and testing sets, which is a common step in machine learning.

# The Data for the Analysis is avaiable in Kaggle
Link to Kaggle Datset : https://www.kaggle.com/datasets/ashydv/leads-dataset
Download and Extract Leads.csv file, and change the path in the Potential_Leads.ipynb, to the system local path before running the consecutive cells.
