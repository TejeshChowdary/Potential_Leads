#!/usr/bin/env python
# coding: utf-8

# # Identifying and Selecting Potential Leads
# 
# ABC Education is a company that provides online courses, and they are seeking assistance in identifying and selecting potential leads that are most likely to convert into paying customers. The company aims to improve their customer conversion rate, and in order to achieve this, they want to analyze and determine the various factors that contribute to a higher likelihood of conversion.
# 
# To achieve this goal, the project team plans to create a lead scoring system. This system will allow the team to assign scores to each potential lead based on various factors such as their level of engagement with the company's website, their past purchasing behavior, and any other relevant information that may indicate a higher probability of conversion.
# 
# The lead scoring system will enable the team to prioritize potential leads with higher scores, as these are more likely to convert into paying customers. By focusing their efforts on these high-priority leads, the team hopes to improve their overall customer conversion rate.
# 
# Overall, the goal of this project is to help ABC Education identify and select potential leads more effectively, and ultimately improve their customer conversion rate by targeting those leads that are most likely to convert.

# # Importing Libraries

# In[1]:

# Importing necessary libraries for data analysis and machine learning
import numpy as np              # For numerical computing
import pandas as pd             # For data manipulation and analysis
import matplotlib.pyplot as plt # For plotting data
import seaborn as sns           # For more advanced plotting options
import plotly                   # For interactive data visualization

# Importing specific classes and functions from scikit-learn
from sklearn.linear_model import LogisticRegression      # For logistic regression
from sklearn.ensemble import RandomForestClassifier      # For random forest classification
from sklearn.tree import DecisionTreeRegressor           # For decision tree regression

# Importing functions for evaluating machine learning models
from sklearn.metrics import classification_report, roc_curve, auc  # For classification metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # For confusion matrix
from sklearn.model_selection import cross_val_score     # For cross-validation
from sklearn.model_selection import train_test_split    # For Splitting Data to Training and Testing

# Supress warnings
import warnings
warnings.filterwarnings('ignore')


# # Exploratory Data Analysis

# Exploratory Data Analysis (EDA) is an important step in data analysis that involves examining and understanding the data before applying any statistical or machine learning techniques. The goal of EDA is to gain insights into the data, identify patterns and relationships, and detect potential issues such as missing values or outliers.

# Here are some common steps involved in EDA:
# 
# Data collection: Collecting the data from various sources, such as databases, spreadsheets, or APIs.
# 
# Data cleaning: Cleaning the data to remove any missing values, duplicates, or outliers that can affect the analysis. This step may also involve converting data types, encoding categorical variables, or imputing missing values.
# 
# Descriptive statistics: Calculating summary statistics such as mean, median, standard deviation, and percentiles to get an overview of the data.
# 
# Data visualization: Creating various plots such as histograms, boxplots, scatterplots, and heatmaps to visualize the distribution of the data and identify any patterns or relationships.
# 
# Feature engineering: Creating new features or transforming existing ones to better represent the underlying data and improve the performance of machine learning models.

# # Download Data from Kaggle
# 
# Sure, here's an explanation of how to download the dataset from Kaggle and load it into a Jupyter notebook:
# 
# 1. First, go to the Kaggle dataset page at https://www.kaggle.com/datasets/ashydv/leads-dataset.
# 
# 2. Click the "Download" button to download the dataset in ZIP format.
# 
# 3. Extract the ZIP file to a local directory on your computer, such as "/Users/<foldername/>/Downloads/".
# 
# 4. In your Jupyter notebook, add the following line of code to load the dataset into a pandas DataFrame:
# 
#    ```
#    df = pd.read_csv("/Users/....../Downloads/Leads.csv")
#    
#    or
#    
#    df = pf.read_csv("C:/Users/...../Downloads/Leads.csv")
#    ```
#    
#    Note that the path specified in this example may need to be changed to reflect the actual location where you extracted the dataset.
# 
# 5. Before running the code, make sure to update the path to match the location where you saved the dataset on your own computer.
# 
# By following these steps, you should be able to download the dataset from Kaggle and load it into a Jupyter notebook for further analysis.

# In[2]:


#Load the dataset
df = pd.read_csv("/Users/mac/Downloads/Leads.csv")


# The above line of code is used to load a dataset into a Jupyter notebook using the pandas library. Specifically, it uses the `read_csv()` function of pandas to read a CSV file and create a DataFrame object that represents the data.
# 
# The path specified in this example is "/Users/mac/Downloads/Leads.csv", which assumes that the CSV file is located in the "Downloads" folder of the user "mac" on a Mac operating system.
# 
# To load a dataset into your own Jupyter notebook, you will need to modify the path to reflect the location where the CSV file is saved on your own computer. For example, if you have saved the CSV file in a folder called "Data" on your desktop, you might use the following line of code:
# 
# ```
# df = pd.read_csv("~/Desktop/Data/Leads.csv")
# ```
# 
# Here, the "~" symbol represents the home directory of the user, and the path "~/Desktop/Data/Leads.csv" assumes that the CSV file is located in the "Data" folder on the user's desktop.
# 
# By running this line of code, you should be able to load the dataset into your Jupyter notebook and begin exploring and analyzing the data using pandas and other Python libraries.

# In[3]:


#First 5 rows of the dataset
df.head(5)


# The above code snippet is used to display the first 5 rows of the loaded dataset in a Jupyter notebook. Specifically, it uses the head() function of pandas to display the first 5 rows of the DataFrame object that represents the loaded dataset.
# 
# By running this code in a Jupyter notebook after loading the dataset using the read_csv() function, you should be able to see the first 5 rows of the dataset. This can be useful for getting a quick overview of the data and understanding what columns and values are included in the dataset.

# In[4]:


# Check data types, null and other basic details column-wise
df.info()


# The above code snippet is used to display basic information about the loaded dataset in a Jupyter notebook. Specifically, it uses the `info()` function of pandas to display the data types of each column, the number of non-null values in each column, and the total number of rows in the DataFrame object that represents the loaded dataset.
# 
# By running this code in a Jupyter notebook, you should be able to see a summary of the basic information about the dataset, including the number of columns, the data types of each column, and whether there are any missing values in the dataset.
# 
# This information can be useful for understanding the structure of the dataset and identifying any potential issues that may need to be addressed before performing data analysis or modeling. For example, if there are missing values in the dataset, you may need to decide how to handle these missing values before proceeding with your analysis.
# 
# Overall, the `info()` function is a useful tool for getting a quick overview of the basic details of a loaded dataset in a Jupyter notebook.

# In[5]:


# Get the details of all the numeric columns
df.describe()


# The above code snippet is used to display the statistical summary of the numeric columns in the loaded dataset in a Jupyter notebook. Specifically, it uses the `describe()` function of pandas to compute various summary statistics of the numeric columns in the DataFrame object that represents the loaded dataset, including count, mean, standard deviation, minimum and maximum values, and various percentiles.
# 
# By running this code in a Jupyter notebook, you should be able to see the statistical summary of the numeric columns in the dataset. This can be useful for getting a sense of the distribution and range of values in each column, as well as identifying potential outliers or anomalies in the data.
# 
# It's worth noting that the `describe()` function only works on numeric columns, so any non-numeric columns in the dataset will be excluded from the summary statistics.
# 
# Overall, the `describe()` function is a useful tool for quickly getting an overview of the distribution and summary statistics of the numeric columns in a loaded dataset.

# In[6]:


# Get the details of all columns including categorical variables
df.describe(include='all').T


# The above code snippet is used to display the statistical summary of all columns in the loaded dataset in a Jupyter notebook, including both numeric and categorical variables. Specifically, it uses the `describe()` function of pandas with the `include` parameter set to `'all'` to compute various summary statistics of all columns in the DataFrame object that represents the loaded dataset, including count, unique values, top value, and frequency of the top value for categorical columns.
# 
# By running this code in a Jupyter notebook, you should be able to see the statistical summary of all columns in the dataset, including both numeric and categorical variables. This can be useful for getting a more complete understanding of the structure and content of the dataset, and for identifying potential issues or patterns that may need to be addressed in subsequent data analysis or modeling.
# 
# It's worth noting that the `describe()` function with `include='all'` can be computationally expensive for large datasets, and may not be necessary or practical for every analysis. However, for smaller datasets or exploratory analysis, it can be a useful tool for gaining a comprehensive view of the data.
# 
# Overall, the `describe()` function with `include='all'` is a useful tool for quickly getting a comprehensive statistical summary of all columns in a loaded dataset, including both numeric and categorical variables.

# In[7]:


# Replacing the 'Select' with NaN
df = df.replace('Select', np.NAN)
df.head()


# The above code snippet is used to replace the string value `'Select'` in the loaded dataset with NaN values. Specifically, it uses the `replace()` function of pandas to replace all occurrences of `'Select'` with `np.NAN`, which is the NumPy representation of NaN (not a number).
# 
# By running this code in a Jupyter notebook, you should be able to see the updated DataFrame object that represents the loaded dataset, with all occurrences of `'Select'` replaced with NaN values. This can be useful for cleaning up the data and preparing it for subsequent analysis or modeling.
# 
# The value `'Select'` is often used as a placeholder or default value in online forms or surveys, and may not provide useful information for data analysis or modeling. By replacing it with NaN values, we can treat it as missing data and handle it accordingly in subsequent analysis.
# 
# Overall, the `replace()` function with `'Select'` and `np.NAN` is a useful tool for cleaning up data and handling missing or unknown values in a loaded dataset.

# In[8]:


# Removing columns of no significance
df = df.drop(['Prospect ID', 'Lead Number'], axis=1)
df.shape


# The above code snippet removes two columns from the DataFrame object `df`, namely `'Prospect ID'` and `'Lead Number'`. This is done using the `drop()` function of pandas, which allows you to drop specified labels from rows or columns of a DataFrame.
# 
# The `drop()` function takes two arguments, `labels` and `axis`. In this case, we pass `['Prospect ID', 'Lead Number']` as the `labels` argument, and `axis=1` to indicate that we want to drop columns rather than rows. The resulting DataFrame object has two fewer columns than before.
# 
# By running this code in a Jupyter notebook, you should be able to see the updated DataFrame object that represents the loaded dataset, with the two specified columns removed. This can be useful for simplifying the dataset and focusing on the columns that are most relevant for analysis or modeling.
# 
# Overall, the `drop()` function is a useful tool for removing columns or rows of a DataFrame that are not needed or are redundant for subsequent analysis.

# In[9]:


# Lets check the percentage of missing values
round(df.isnull().sum() * 100 / len(df), 2)


# The above code snippet computes the percentage of missing values in each column of the DataFrame object `df`. This is done using the `isnull()` function of pandas, which returns a DataFrame of the same shape as `df`, but with Boolean values indicating which elements are missing (i.e., NaN values).
# 
# The resulting Boolean DataFrame is then passed to the `sum()` function, which returns a Series object containing the sum of missing values for each column. We then multiply this by 100 and divide by the length of `df` to obtain the percentage of missing values in each column, which is rounded to two decimal places using the `round()` function.
# 
# By running this code in a Jupyter notebook, you should be able to see the resulting Series object that represents the percentage of missing values in each column of the loaded dataset. This can be useful for identifying columns with a high proportion of missing values and deciding how to handle them in subsequent analysis or modeling.
# 
# Overall, computing the percentage of missing values is an important step in data cleaning and preparation, as it can provide insights into the quality and completeness of the data, and guide decisions about how to handle missing or unknown values in subsequent analysis.

# In[10]:


# Function to remove the columns having more than threshold values
def rmissingvaluecol(dff, threshold):
    col = []
    col = list(dff.drop(dff.loc[:,list((100*(dff.isnull().sum()/len(dff.index)) >= threshold))].columns, 1).columns.values)
    print("Columns having more than %s percent missing values: "%threshold, (dff.shape[1] - len(col)))
    removed_cols = list(set(list((dff.columns.values))) - set(col))
    [print(i) for i in removed_cols]   
    return col

# Removing columns having 40% missing values
col = rmissingvaluecol(df, 40)
modified_df = df[col]
modified_df.head()


# The above code snippet defines a function called `rmissingvaluecol()` that takes two arguments: `dff`, which is a DataFrame object, and `threshold`, which is a float representing the maximum percentage of missing values allowed in a column. The function returns a list of columns from the input DataFrame that have fewer missing values than the specified threshold.
# 
# The function works by first computing the percentage of missing values for each column in the input DataFrame `dff` using the `isnull()` and `sum()` functions, as described in the previous explanation. It then identifies the columns that have more missing values than the specified threshold by comparing the computed percentage to the `threshold` argument using a Boolean mask.
# 
# The resulting Boolean mask is then used to select the columns that have fewer missing values than the specified threshold using the `loc[]` function, and the resulting DataFrame is transformed into a list of column names using the `columns.values` attribute. This list is then printed to the console along with the number of removed columns.
# 
# Finally, the function returns the list of columns that have fewer missing values than the specified threshold, which is assigned to the variable `col`. The original DataFrame `df` is then filtered to include only the selected columns, and the resulting DataFrame is assigned to the variable `modified_df`, which is printed to the console using the `head()` function to display the first few rows of the modified DataFrame.
# 
# By running this code in a Jupyter notebook, you can see which columns are removed based on the specified threshold of 40% as per our requirement, and the resulting modified DataFrame that includes only the columns with fewer missing values than the threshold. This can be useful for reducing the dimensionality of the dataset and removing noisy or irrelevant features that are unlikely to contribute to the analysis or modeling.

# #### Let's check if there is any imbalance in the distrubution between Converted/Not Converted classes

# In[11]:


import plotly.express as px
dfg=df.groupby('Converted').count().reset_index()
dfg['Count'] = dfg['Lead Origin']
dfg['Converted'] = dfg['Converted'].astype(str)
fig = px.bar(dfg,x='Converted',y='Count',color='Converted',text='Count',title='Checking Output Class Balance',color_discrete_map={'0': 'red','1': 'blue'})
fig.show()


# #### Since the class distrubution is not highly skewed we do not need to apply any over sampling or under sampling techniques

# This code is using the Plotly Express library to create a bar chart showing the count of leads in each of the two output classes ('Converted' being the target variable) in the dataset. The bar chart is colored based on the output class and the count of each class is displayed above the respective bars. 
# 
# The code groups the dataframe by the 'Converted' column and then counts the number of occurrences in each class using the `count()` function. It then creates a new column 'Count' to hold the counts for each class. Finally, it uses the Plotly Express `bar()` function to create a bar chart with 'Converted' as the x-axis and 'Count' as the y-axis. The chart is titled 'Checking Output Class Balance' and uses red and blue as colors for the two output classes, respectively, using the `color_discrete_map` parameter.

# ### Analyzing Numerical Variables

# In[12]:


#Correlation
fig = px.imshow(df.corr(), text_auto=True,aspect="auto",color_continuous_scale='blues')
fig.show()


# This code will create a heatmap using Plotly Express library to visualize the correlation between all the numeric columns of the dataframe `df`. The heatmap will show the correlation coefficient values ranging from -1 to 1, with -1 indicating a strong negative correlation and 1 indicating a strong positive correlation. The `text_auto=True` argument will display the correlation coefficient values on the heatmap. The `aspect="auto"` argument will adjust the aspect ratio of the heatmap to fit the available space. The `color_continuous_scale='blues'` argument will set the color scheme for the heatmap to shades of blue. The resulting heatmap will help us identify which features are highly correlated and potentially redundant.

# In[13]:


fig = px.scatter(modified_df,  x="Total Time Spent on Website", color="Converted",title="Scatter plot between Leads and Total Time Spent")
fig.show()


# Converted has the highest correlation with Total Time Spend on Website which tells us that the more time someone spends on this website the higher the chances of being converted as a paying customer
# 
# Also, there is more than 50% correlation between "Page Views Per Visit" and "TotalVisit"

# #### Checking for Outliers using Boxplots

# In[14]:


# Outliers in "TotalVisits"
fig = px.box(modified_df, y="TotalVisits")
fig.show()


# This code is using the `box` function from the Plotly Express library to create a box plot of the "TotalVisits" column in the dataset. A box plot is a graphical representation of the distribution of the data that shows the median, quartiles, and outliers of a set of values. In this case, the box plot is used to detect any outliers in the "TotalVisits" column. The resulting plot will show a box that covers the first and third quartiles of the data, a line that represents the median value, and dots that represent any outliers outside the whiskers of the box.

# #### We can see presence of outliers here

# In[15]:


# Removing top and bottom 1 percentile 
upperlimit = modified_df.TotalVisits.quantile(0.99)
modified_df = modified_df[(modified_df.TotalVisits <= upperlimit)]
lowerlimit = modified_df.TotalVisits.quantile(0.01)
modified_df = modified_df[(modified_df.TotalVisits >= lowerlimit)]
# Fill NaN with the mode
modified_df['TotalVisits'] = modified_df['TotalVisits'].fillna(modified_df['TotalVisits'].mode())
fig = px.box(modified_df, y="TotalVisits",title="After removing the outliers")
fig.show()


# In[16]:


# Outliers in "Total Time Spent on Website"
fig = px.box(modified_df, y="Total Time Spent on Website")
fig.show()


# #### No outliers present in "Total Time Spent on Website"

# In[17]:


# Outliers in "Page Views Per Visit"
fig = px.box(modified_df, y="Page Views Per Visit")
fig.show()


# #### We can see presence of outliers here as well

# In[18]:


# Removing top and bottom 1% 
upperlimit = modified_df['Page Views Per Visit'].quantile(0.99)
modified_df = modified_df[(modified_df['Page Views Per Visit'] <= upperlimit)]
lowerlimit = modified_df['Page Views Per Visit'].quantile(0.01)
modified_df = modified_df[(modified_df['Page Views Per Visit'] >= lowerlimit)]
fig = px.box(modified_df, y="Page Views Per Visit",title="After removing the outliers")
fig.show()


# In[19]:


modified_df.shape


# #### Let us analyze the categorical variables

# In[20]:


# City
px.histogram(modified_df,x='City',y='Converted', color='Converted',title='Count of Converted & Non Converted for all values of - City',
                       barmode='group', 
                       histfunc='count',text_auto=True, color_discrete_map={0: 'red',1: 'blue'})


# In[21]:


#Imputing nulls in city column by its most repeated value (mode)
modified_df['City'] = modified_df['City'].replace(np.nan,'Not Specified')


# In[22]:


# Specialization
px.histogram(modified_df,x='Specialization',y='Converted', color='Converted',title='Count of Converted & Non Converted for all values of - Specialization',
                       barmode='group', 
                       histfunc='count',text_auto=True,color_discrete_map={0: 'red',1: 'blue'})


# The above plot shows the distribution of leads and conversions across different values of the "Specialization" variable. We can see that the specialization values containing the term "Management" have both higher number of leads as well as higher number of leads that get converted. Therefore, this variable is significant in determining whether a lead will convert or not and should not be dropped from the analysis.

# In[23]:


#combining Management Specializations because they show similar trends
modified_df['Specialization'] = modified_df['Specialization'].replace(['Finance Management','Human Resource Management',
                                                           'Marketing Management','Operations Management',
                                                           'IT Projects Management','Supply Chain Management',
                                                    'Healthcare Management','Hospitality Management',
                                                           'Retail Management'] ,'Management_Specializations')  

# Lead may not have mentioned specialization because it was not in the list or maybe they are a students 
# and don't have a specialization yet. So we will replace NaN values here with 'Not Specified'
modified_df['Specialization'] = modified_df['Specialization'].replace(np.nan, 'Not Specified')


# In[24]:


#What is your current occupation
px.histogram(modified_df,x='What is your current occupation',y='Converted', color='Converted',title='Count of Converted & Non Converted for all values of - What is your current occupation',
                       barmode='group', 
                       histfunc='count',text_auto=True,color_discrete_map={0: 'red',1: 'blue'})


# The histogram plot for the variable "What is your current occupation" shows that leads who have indicated their occupation as working professionals have the highest chance of being converted. On the other hand, there are the highest number of leads who have indicated their occupation as unemployed, but it is not clear whether they are interested in the course or not.

# In[25]:


#imputing NaN values with mode "Unemployed"
modified_df['What is your current occupation'] = modified_df['What is your current occupation'].replace(np.nan, 'Not Specified')


# In[26]:


# Lead Source
px.histogram(modified_df,x='Lead Source',y='Converted', color='Converted',title='Count of Converted & Non Converted for all values of - Lead Source',
                       barmode='group', 
                       histfunc='count',text_auto=True,color_discrete_map={0: 'red',1: 'blue'})


# - The majority of leads are generated from Direct Traffic and Google Search sources.
# - Reference and Welingak Website are the top sources with the highest conversion rate, with more than 90 percent of leads getting converted.
# - To increase the overall lead conversion rate, there should be a focus on improving the lead conversion of Olark Chat, Organic Search, Direct Traffic, and Google leads. Additionally, generating more leads from Reference and Welingak Website can also help to improve the conversion rate.

# In[27]:


# replacing Nan Values and combining low frequency values
modified_df['Lead Source'] = modified_df['Lead Source'].replace(np.nan,'Others')

# Replacing "google" with "Google" as they appear as two different entities
modified_df['Lead Source'] = modified_df['Lead Source'].replace('google','Google')
modified_df['Lead Source'] = modified_df['Lead Source'].replace('Facebook','Social Media')

# Replacing low frequency values with others
modified_df['Lead Source'] = modified_df['Lead Source'].replace(['bing','Click2call','Press_Release',
                                                     'youtubechannel','welearnblog_Home',
                                                     'WeLearn','blog','Pay per Click Ads',
                                                    'testone','NC_EDM'] ,'Others')            


# In[28]:


# Tags
px.histogram(modified_df,x='Tags',y='Converted', color='Converted',title='Count of Converted & Non Converted for all values of - Tags',
                       barmode='group', 
                       histfunc='count',text_auto=True,color_discrete_map={0: 'red',1: 'blue'})


# The "Tags" column in the dataset provides valuable information about the customer's interests and requirements, but this data is only available after the sales team has contacted the customer. Therefore, it cannot be used to model new leads because this information is not yet available.

# In[29]:


# Imputing NaN in Tags column with Not Specified
modified_df['Tags'] = modified_df['Tags'].replace(np.nan,'Not Specified')

#replacing tags with low frequency with "Other Tags"
modified_df['Tags'] = modified_df['Tags'].replace(['In confusion whether part time or DLP', 'in touch with EINS','Diploma holder (Not Eligible)',
                                     'Approached upfront','Graduation in progress','number not provided', 'opp hangup','Still Thinking',
                                    'Lost to Others','Shall take in the next coming month','Lateral student','Interested in Next batch',
                                    'Recognition issue (DEC approval)','Want to take admission but has financial problems',
                                    'University not recognized'], 'Other_Tags')

modified_df['Tags'] = modified_df['Tags'].replace(['switched off',
                                      'Already a student',
                                       'Not doing further education',
                                       'invalid number',
                                       'wrong number given',
                                       'Interested  in full time MBA'] , 'Other_Tags')


# In[30]:


# Last Activity
px.histogram(modified_df,x='Last Activity',y='Converted', color='Converted',title='Count of Converted & Non Converted for all values of - Last Activity',
                       barmode='group', 
                       histfunc='count',text_auto=True,color_discrete_map={0: 'red',1: 'blue'})


# In[31]:


#replacing Nan Values and combining low frequency values
modified_df['Last Activity'] = modified_df['Last Activity'].replace(np.nan,'Others')
modified_df['Last Activity'] = modified_df['Last Activity'].replace(['Unreachable','Unsubscribed',
                                                        'Had a Phone Conversation', 
                                                        'Approached upfront',
                                                        'View in browser link Clicked',       
                                                        'Email Marked Spam',                  
                                                        'Email Received','Resubscribed to emails',
                                                         'Visited Booth in Tradeshow'],'Others')


# In[32]:


#replacing Nan values with Mode "Better Career Prospects"
modified_df['What matters most to you in choosing a course'] = modified_df['What matters most to you in choosing a course'].replace(np.nan,'Better Career Prospects')


# In[33]:


#Let us drop the highly skewed columns as they would only add bias to the model
cols_to_drop = ['Country','Do Not Call','Search','Magazine','Newspaper Article','X Education Forums','Newspaper',
                 'Digital Advertisement','Through Recommendations','Receive More Updates About Our Courses',
                 'Update me on Supply Chain Content',
                 'Get updates on DM Content','I agree to pay the amount through cheque','Tags']


# In[34]:


# Plot histograms for highly skewed columns
for i in cols_to_drop:
    fig = px.histogram(modified_df,x=i,y='Converted', color='Converted',title='Count of Converted & Non Converted for all values of - {}'.format(i),
                           barmode='group', 
                           histfunc='count',text_auto=True,color_discrete_map={0: 'red',1: 'green'})
fig.show()


# In[35]:


modified_df = modified_df.drop(cols_to_drop,1)
modified_df.info()


# In[36]:


modified_df.shape


# In[37]:


# Check if there are still any null values
round(100*(modified_df.isnull().sum()/len(modified_df.index)), 2)


# There are no more NaNs in our selected columns

# In[38]:


fig = px.histogram(modified_df, x= "Lead Source", y="Total Time Spent on Website",title='Lead Source to Time Spent')
fig.show()


# Most of the customers are visiting through Organic Search, Direct Traffic, Google

# ### Feature Engineering

# #### To be processed by the model, all columns in the dataset should contain numerical values. Therefore, categorical variables must be transformed into numerical ones. One commonly used method for this task is one hot encoding.

# In[39]:


#getting a list of categorical columns
cat_cols= modified_df.select_dtypes(include=['object']).columns
cat_cols


# In[40]:


df_with_dummies = pd.get_dummies(modified_df,drop_first=True)


# In[41]:


df_with_dummies.shape


# In[42]:


df_with_dummies


# #### Train Test Splitting
# #### 80% for model training and 20% for testing

# In[43]:


# train-test split
X = df_with_dummies.loc[:, df_with_dummies.columns != 'Converted']
y = df_with_dummies['Converted']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# This code is splitting the dataset into training and testing sets. The input features are stored in X, and the output target variable is stored in y. Here, the dataset is split into 80% for training and 20% for testing. The split is performed randomly using a seed of 42 to ensure reproducibility of results.

# In[44]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# These print statements show the dimensions of the train and test data after splitting. X_train has 7162 rows and 64 columns, y_train has 7162 elements, X_test has 1791 rows and 64 columns, and y_test has 1791 elements. This indicates that 80% of the data is being used for training and 20% for testing.

# In[45]:


# Helper function to evaluate a model by printing its accuracy on train set, test set, confusion matrix, ROC curve
# This function can be resued for all the models
def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Make predictions on the training and test sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate accuracy on the training and test sets
    train_accuracy = np.mean(y_train_pred == y_train)
    test_accuracy = np.mean(y_test_pred == y_test)

    print('***************************************************************')

    # Print the accuracy scores
    print("Train accuracy:", train_accuracy)
    print()
    print("Test accuracy:", test_accuracy)

    print('***************************************************************')
    # Print the classification report
    print("Classification report:")
    print()
    print(classification_report(y_test, y_test_pred))
    
    print('***************************************************************')

    # Plot the Receiver Operating Characteristic (ROC) curve
    fpr, tpr, _ = roc_curve(y_test, y_test_pred)
    roc_auc = auc(fpr, tpr)
    

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(4, 2))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.xticks([0,1])
    plt.yticks([0,1])
    # Print the values inside the plot
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.xlabel('Predicted label')
    plt.ylabel('Actual label')
    plt.show()
    plt.show()


# This is a helper function that can be used to evaluate the performance of a binary classification model. It takes in a trained model, training and test sets, and calculates the accuracy on the training and test sets. It also prints out a classification report, which gives more details about the performance of the model. Additionally, it plots a Receiver Operating Characteristic (ROC) curve, which shows the relationship between the true positive rate and the false positive rate of the model, and a confusion matrix, which shows the number of true positives, false positives, true negatives, and false negatives of the model's predictions.

# ### Model Building

# #### 1. Logistic Regression

# In[46]:


lr_model = LogisticRegression(random_state=0)
lr_model.fit(X_train, y_train)
evaluate_model(lr_model, X_train, y_train, X_test, y_test)


# Cross validation for Logistic Regression ( 5 Folds)

# In[47]:


all_models = {}
scores = cross_val_score(lr_model, X, y, cv=5)
all_models['Logistic Regression'] = round(scores.mean()*100,2)
print("%0.2f percent accuracy with a standard deviation of %0.2f" % (scores.mean()*100, scores.std()))


# #### 2. K-Nearest Neighbours (KNN)

# In[48]:


from sklearn.neighbors import KNeighborsClassifier
knn_model  = KNeighborsClassifier(n_neighbors=10)
knn_model.fit(X_train, y_train)
evaluate_model(knn_model, X_train, y_train, X_test, y_test)


# Cross validation for KNN ( 5 Folds)

# In[49]:


scores = cross_val_score(knn_model, X, y, cv=5)
all_models['KNN'] = round(scores.mean()*100,2)
print("%0.2f percent accuracy with a standard deviation of %0.2f" % (scores.mean()*100, scores.std()))


# #### 3. Support Vector Machine (SVM)

# In[50]:


from sklearn import svm
svm_model = svm.SVC(C=10, kernel='rbf')
svm_model.fit(X_train, y_train)
evaluate_model(svm_model, X_train, y_train, X_test, y_test)


# Cross validation for SVM ( 5 Folds)

# In[51]:


scores = cross_val_score(svm_model, X, y, cv=5)
all_models['Support Vector Machine'] = round(scores.mean()*100,2)
print("%0.2f percent accuracy with a standard deviation of %0.2f" % (scores.mean()*100, scores.std()))


# #### 4. Decision Tree

# In[52]:


from sklearn.tree import DecisionTreeClassifier

decision_tree_model = DecisionTreeClassifier(max_depth=50)
dt_model = decision_tree_model.fit(X_train, y_train)
evaluate_model(dt_model, X_train, y_train, X_test, y_test)


# Cross validation for Decision Tree ( 5 Folds)

# In[53]:


scores = cross_val_score(dt_model, X, y, cv=5)
all_models['Decision Tree'] = round(scores.mean()*100,2)
print("%0.2f percent accuracy with a standard deviation of %0.2f" % (scores.mean()*100, scores.std()))


# #### 5. Random Forest Classifier

# In[54]:


rf_model = RandomForestClassifier(random_state=42,max_depth=None, n_estimators=100,criterion='gini')
rf_model.fit(X_train, y_train)
evaluate_model(rf_model, X_train, y_train, X_test, y_test)


# In[55]:


scores = cross_val_score(rf_model, X, y, cv=5)
all_models['Random Forest'] = round(scores.mean()*100,2)
print("%0.2f percent accuracy with a standard deviation of %0.2f" % (scores.mean()*100, scores.std()))


# #### 6. XGBoost

# In[56]:


get_ipython().system('pip install xgboost')


# Comment the above command if xgboost is already installed

# In[57]:


from xgboost import XGBClassifier
xgboost_model = XGBClassifier(n_estimators=100,max_depth=None, random_state=42)
xgboost_model.fit(X_train, y_train)
evaluate_model(xgboost_model, X_train, y_train, X_test, y_test)


# In[58]:


scores = cross_val_score(xgboost_model, X, y, cv=5)
all_models['XGBoost'] = round(scores.mean()*100,2)
print("%0.2f percent accuracy with a standard deviation of %0.2f" % (scores.mean()*100, scores.std()))


# #### Hyper Parameter Tuning

# The results of the XGBoost model are encouraging, and there may be room for further improvement through the use of GridCV from sklearn for hyperparameter tuning.

# In[59]:


from sklearn.model_selection import GridSearchCV
params = { 'max_depth': [10,50],
           'learning_rate': [0.01, 0.05],
           'n_estimators': [50, 100,200]
          }

xgboost_model_ = XGBClassifier()

clf = GridSearchCV(estimator=xgboost_model_, 
                   param_grid=params,
                   verbose=1)
clf.fit(X, y)
print("Best parameters:", clf.best_params_)


# In[60]:


all_models['XGBoost_FineTuned'] = round(clf.best_score_*100,2)
clf.best_score_


# In[61]:


from xgboost import plot_importance
plot_importance(xgboost_model, max_num_features=10) # top 10 most important features
plt.show()


# This code uses the XGBoost plot_importance() function to plot the importance of each feature in the XGBoost model. The function takes the trained XGBoost model and an optional argument max_num_features which specifies the maximum number of features to show in the plot. The code specifies max_num_features=10 to show the top 10 most important features. Finally, the matplotlib plt.show() function is used to display the plot.

# In[62]:


#all_models


# In[63]:


fig = px.bar(x=all_models.keys(),y=all_models.values(),title='All Models Comparison',text=all_models.values(),labels=dict(x="Model", y="Accuracy"),color=all_models.keys())
fig.show()


# After training at all models, we evaluate those models and check for accuracy, precision, recall, F1 score and AUC-ROC score on both train and test data. And based on the metrics we decided to go with XGBoost as the metrics were comparatively better and closer for both train and test datasets. 
# 
# 
# Additionally, the implementation of this model is expected to help ABC Education Save significant amounts of Money and Resources by targeting potential leads that are more likely to convert into paying customers, rather than pursuing all leads without discrimination.
