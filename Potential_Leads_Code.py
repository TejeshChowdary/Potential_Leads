

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

#Load the dataset
df = pd.read_csv("/Users/mac/Downloads/Leads.csv")


#First 5 rows of the dataset
df.head(5)

# Check data types, null and other basic details column-wise
df.info()

# Get the details of all the numeric columns
df.describe()

# Get the details of all columns including categorical variables
df.describe(include='all').T

# Replacing the 'Select' with NaN
df = df.replace('Select', np.NAN)
df.head()

# Removing columns of no significance
df = df.drop(['Prospect ID', 'Lead Number'], axis=1)
df.shape

# Lets check the percentage of missing values
round(df.isnull().sum() * 100 / len(df), 2)

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

import plotly.express as px
dfg=df.groupby('Converted').count().reset_index()
dfg['Count'] = dfg['Lead Origin']
dfg['Converted'] = dfg['Converted'].astype(str)
fig = px.bar(dfg,x='Converted',y='Count',color='Converted',text='Count',title='Checking Output Class Balance',color_discrete_map={'0': 'red','1': 'blue'})
fig.show()

#Correlation
fig = px.imshow(df.corr(), text_auto=True,aspect="auto",color_continuous_scale='blues')
fig.show()

fig = px.scatter(modified_df,  x="Total Time Spent on Website", color="Converted",title="Scatter plot between Leads and Total Time Spent")
fig.show()

# Outliers in "TotalVisits"
fig = px.box(modified_df, y="TotalVisits")
fig.show()

# Removing top and bottom 1 percentile 
upperlimit = modified_df.TotalVisits.quantile(0.99)
modified_df = modified_df[(modified_df.TotalVisits <= upperlimit)]
lowerlimit = modified_df.TotalVisits.quantile(0.01)
modified_df = modified_df[(modified_df.TotalVisits >= lowerlimit)]
# Fill NaN with the mode
modified_df['TotalVisits'] = modified_df['TotalVisits'].fillna(modified_df['TotalVisits'].mode())
fig = px.box(modified_df, y="TotalVisits",title="After removing the outliers")
fig.show()

# Outliers in "Total Time Spent on Website"
fig = px.box(modified_df, y="Total Time Spent on Website")
fig.show()

# Outliers in "Page Views Per Visit"
fig = px.box(modified_df, y="Page Views Per Visit")
fig.show()

# Removing top and bottom 1% 
upperlimit = modified_df['Page Views Per Visit'].quantile(0.99)
modified_df = modified_df[(modified_df['Page Views Per Visit'] <= upperlimit)]
lowerlimit = modified_df['Page Views Per Visit'].quantile(0.01)
modified_df = modified_df[(modified_df['Page Views Per Visit'] >= lowerlimit)]
fig = px.box(modified_df, y="Page Views Per Visit",title="After removing the outliers")
fig.show()


modified_df.shape

# City
px.histogram(modified_df,x='City',y='Converted', color='Converted',title='Count of Converted & Non Converted for all values of - City',
                       barmode='group', 
                       histfunc='count',text_auto=True, color_discrete_map={0: 'red',1: 'blue'})


#Imputing nulls in city column by its most repeated value (mode)
modified_df['City'] = modified_df['City'].replace(np.nan,'Not Specified')

# Specialization
px.histogram(modified_df,x='Specialization',y='Converted', color='Converted',title='Count of Converted & Non Converted for all values of - Specialization',
                       barmode='group', 
                       histfunc='count',text_auto=True,color_discrete_map={0: 'red',1: 'blue'})


#combining Management Specializations because they show similar trends
modified_df['Specialization'] = modified_df['Specialization'].replace(['Finance Management','Human Resource Management',
                                                           'Marketing Management','Operations Management',
                                                           'IT Projects Management','Supply Chain Management',
                                                    'Healthcare Management','Hospitality Management',
                                                           'Retail Management'] ,'Management_Specializations')  

# Lead may not have mentioned specialization because it was not in the list or maybe they are a students 
# and don't have a specialization yet. So we will replace NaN values here with 'Not Specified'
modified_df['Specialization'] = modified_df['Specialization'].replace(np.nan, 'Not Specified')

#What is your current occupation
px.histogram(modified_df,x='What is your current occupation',y='Converted', color='Converted',title='Count of Converted & Non Converted for all values of - What is your current occupation',
                       barmode='group', 
                       histfunc='count',text_auto=True,color_discrete_map={0: 'red',1: 'blue'})


#imputing NaN values with mode "Unemployed"
modified_df['What is your current occupation'] = modified_df['What is your current occupation'].replace(np.nan, 'Not Specified')


# In[26]:


# Lead Source
px.histogram(modified_df,x='Lead Source',y='Converted', color='Converted',title='Count of Converted & Non Converted for all values of - Lead Source',
                       barmode='group', 
                       histfunc='count',text_auto=True,color_discrete_map={0: 'red',1: 'blue'})

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

# Tags
px.histogram(modified_df,x='Tags',y='Converted', color='Converted',title='Count of Converted & Non Converted for all values of - Tags',
                       barmode='group', 
                       histfunc='count',text_auto=True,color_discrete_map={0: 'red',1: 'blue'})

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



#getting a list of categorical columns
cat_cols= modified_df.select_dtypes(include=['object']).columns
cat_cols


# In[40]:


df_with_dummies = pd.get_dummies(modified_df,drop_first=True)


# In[41]:


df_with_dummies.shape


# In[42]:


df_with_dummies

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



# In[63]:


fig = px.bar(x=all_models.keys(),y=all_models.values(),title='All Models Comparison',text=all_models.values(),labels=dict(x="Model", y="Accuracy"),color=all_models.keys())
fig.show()
