#!/usr/bin/env python
# coding: utf-8

# ## ENVIRONMENT SET-UP

# In[146]:


# importing of necessary libraries and packages

import numpy as np
import pandas as pd

# Visualisation
import seaborn as sns
import matplotlib.pyplot as plt

# Feature Engineering, Data Splitting, Machine Learning, Metrics
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif # feature selection

# ROC and AUC evaluation and visualisation
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression

# For Filtering the warnings
import warnings
warnings.filterwarnings('ignore')


# ## Exploratory Data Analysis and Data Pre-processing

# EDA (Exploratory Data Analysis) and Data Pre-processing are crucial steps in the data science process, but they serve different purposes and are applied at different stages.
# 
# EDA involves exploring and investigating the data to understand its characteristics, identify patterns, discover anomalies, and formulate hypotheses. This initial step is about getting to know our data intimately before we start any serious modeling or analysis. 
# 
# Data Pre-processing is the stage where we prepare the data for modeling or analysis. It typically happens after EDA and before feature engineering and model building. Typical steps of Data Pre-processing includes:
# 
# * Cleaning: Handling missing values (imputation or removal), removing duplicates, correcting inconsistencies, and dealing with outliers.
# * Transformation: Scaling numerical features (normalization or standardization), encoding categorical variables (one-hot encoding or label encoding), and converting data types.
# * Reduction: Reducing the dimensionality of the data (feature selection or dimensionality reduction techniques like PCA).
# 
# Essentially, EDA and Data Pre-processing are primarily concerned with understanding and preparing data for analysis or modeling and this is what we will start with when initially loading the dataset.

# In[202]:


# loading the data and the variable description using the link of the dataset on github
url_data = (r"https://raw.githubusercontent.com/chrisbw/Hollard-Data-Science/refs/heads/main/data.csv")
url_desc = (r"https://raw.githubusercontent.com/chrisbw/Hollard-Data-Science/refs/heads/main/VariableDescription.csv")

# reading of data using pandas read_csv function 
df = pd.read_csv(url_data)
data_desc = pd.read_csv(url_desc)

# quick glance of the dataset shape
num_rows, num_columns = df.shape
print(f"The focus is an insurance dataframe that contains {num_rows} rows and {num_columns} columns.")


# ### The dataframe variable description is as follows:
# 
# 
# **Customer Id:** Identification number for the Policy holder
# 
# **YearOfObservation:** Year of observation for the insured policy
# 
# **Insured_Period:** Duration of insurance policy in Olusola Insurance. (Ex: Full year insurance, Policy Duration = 1; 6 months = 0.5)
# 
# **Residential:** is the building a residential building or not
# 
# **Building_Painted:** Is the building painted or not (N-Painted, V-Not Painted)
# 
# **Building_Fenced:** Is the building fence or not (N-Fenced, V-Not Fenced)
# 
# **Garden:** building has garden or not (V-has garden; O-no garden)
# 
# **Settlement:** Area where the building is located (R-rural area; U-urban area)
# 
# **Building Dimension:** Size of the insured building in meters squared
# 
# **Building_Type:** The type of building (Type 1, 2, 3, 4)
# 
# **Date_of_Occupancy:** Date building was first occupied
# 
# **NumberOfWindows:** Number of windows in the building
# 
# **Geo Code:** Geographical Code of the Insured building
# 
# **Claim:** Target variable. (0: no claim, 1: at least one claim over insured period)

# In[203]:


# view of the first 10 rows of our dataset
print(f"The first 10 rows of our dataset can provide a quick visual overview of the type of variables each attribute contains \nas indicated in the variable description above. These could be numerical, categorical, ordinal, etc")
print()
df.head(10).transpose()


# In[204]:


# the type of variables can be confirmed by checking the column types. 
# this also helps to determine if the column types rightfully align with what has been outlined in the variable description
df.info()


# Carefully considering the different types of columns in the dataset. The first thing to understand is the type of information under each feature and any unique values present. This would show a somewhat detail of the composition of the data points under each variable. This would eventually be visually shown for easier understanding and readability.
# 
# The dataset information showed that a number of the columns are recognised as objects, floats, and integers. 
# 
# However, according to the variable description, the dataset contains dates (YearOfObservation and Date_of_Occupancy), categorical(Residential, Building_Painted, Building_Fenced, Garden, Settlement, Building_Type, NumberOfWindows, Geo_Code, Claim), and numerical (Insured_Period, Building Dimension) attributes.
# 
# The dates columns could be used either as categorical or date columns depending on the use case of our analysis and modeling. 
# 
# To convert the object type columns to columns, we would need to understand the reasoning behind doing so. 
# 
# * The 'object' data type is a general-purpose type for text or mixed data.
# 
# * The 'categorical' data type is optimized for columns with a limited set of unique values, offering better memory efficiency and performance for certain operations.
# 
# Thus, several of the object type columns are to be converted to categorical type variables. This however will be confirmed after viewing the unique values present under each object data type.

# In[205]:


# function to extract the categorical and numerical attributes to group them for further analysis
def extract_cat_num(data):
    categorical_col = [col for col in df.columns if df[col].dtype == 'object']
    numerical_col = [col for col in df.columns if df[col].dtype == 'float64']
    return categorical_col,numerical_col

categorical_col, numerical_col = extract_cat_num(df)


# In[206]:


# categorical attributes extracted from the dataframe
categorical_col


# In[207]:


# to determine the unique values under each categorical attribute, which can help determine if there are any anomalies
for col in categorical_col:
    print('\nUnique Values in {} has\n{} '.format(col,df[col].value_counts()))
    print('\n')


# In[208]:


numerical_col


# Observations:
# 
# * The Customer Id column is a unique attribute.
# * The unique values in NumberOfWindows show a category of 1 to greater and equal to 10 for each building - however, it also includes 3551 rows that have a period symbol representing the number of windows. This brings to question if this is considered a missing value or  a building with no (0) windows. Requires further investigation.
# * The YearOfObservation, Residential, Building_Type, Date_of_Occupancy and Claim columns have been wrongly recognised as numerical columns and require conversion.
# * The Claim attribute is our target. Which is a categorical column. 
# * The dates columns (YearOfObservation and Date_of_Occupancy) could be used either as categorical or date columns depending on the use case of our analysis and modeling. At this point they will be considered categorical.
# 
# To convert the object type columns to columns, we would need to understand the reasoning behind doing so. 
# 
# 1. The 'object' data type is a general-purpose type for text or mixed data.
# 
# 2. The 'categorical' data type is optimized for columns with a limited set of unique values, offering better memory efficiency and performance for certain operations.
# 
# Thus, several of the object type columns are to be converted to categorical type variables.

# In[209]:


# changing of column types
# first, the Date_of_Occupancy column has float values
# which we first convert to integer to get rid of the decimal before converting to categorical
df['Date_of_Occupancy'] = pd.to_numeric(df['Date_of_Occupancy'], errors='coerce').astype('Int64')
df.head(10).transpose()


# In[210]:


df.info()


# In[211]:


# list of columns to convert to categorical
columns_to_convert = [
    'YearOfObservation', 'Residential', 'Building_Painted', 'Building_Fenced', 'Garden', 
    'Settlement', 'Building_Type','NumberOfWindows', 'Date_of_Occupancy', 'Geo_Code', 'Claim'
]

# convert the specified columns to categorical
df[columns_to_convert] = df[columns_to_convert].astype('category')
df.info()


# In[212]:


# determining the unique values contained within our target attribute Claim
print('\nUnique Values in {} has\n{} '.format('Claim',df['Claim'].value_counts()))
print('\n')


# Our target variable 'Claim' has been observed to be imbalanced, with 5,526 observations indicating that the building has at least a claim over the insured period and 1,634 of the observations indicating that the building doesn’t have a claim over the insured period.

# After conversion, it is also clear from the count that there exists missing values based on the total expected rows and the count of non-null rows in the information displayed above. The missing values can be further analysed to understand the extent at which they occur. 
# 
# The potentially missing values in NumberOfWindows has to also be explored, to understand what the symbol may be trying to represent. To do this, we begin to visualise the relationships between the variables.

# In[158]:


# visualising the distribution of buildings in residential (1) and non-residential (0) areas and the number of windows each building has
plt.figure(figsize=(4, 4), dpi=100)
sns.countplot(x='NumberOfWindows',hue="Residential", data=df)
plt.show()


# According to the visualisation, a large number of buildings with an unknown number of windows are found in non-residential areas. The second largest number are buildings found in residential areas. Non-residential areas are zones designated for purposes other than living or dwelling and contain structures that utilise space in a manner that does not require or have the need for the same infrastructure found in residential builds. Non-residential areas include:
# 
# - Commercial:Office buildings, shopping malls, retail stores, restaurants, and hotels.
# - Industrial: Factories, warehouses, manufacturing plants, and industrial parks.
# - Institutional: Schools, universities, hospitals, government buildings, and religious institutions.
# - Recreational: Parks, sports complexes, theaters, and entertainment venues.
# - Agricultural: Farms, barns, greenhouses, and storage facilities for agricultural products.
# - Transportation Hubs: Airports, train stations, bus terminals, and parking garages.
# - Utilities and Infrastructure: Power plants, water treatment facilities, and communication towers.
# 
# Windowless infrastructure that can also be found in residential areas can also include storage facilities and parking garages. These various structures can be insured to protect against a wide range of potential risks such as property damage. The presence of windows in non-residential buildings depends on the purpose and design of the building.

# In[135]:


# visualising the distribution of buildings that are fenced (N) and not fenced (V) areas and the number of windows each building has
plt.figure(figsize=(4, 4), dpi=100)
sns.countplot(x='NumberOfWindows',hue="Building_Fenced", data=df)
plt.show()


# The above visualisation indicates that all the buildings with the symbol '.' indicating number of windows are not fenced (V). This could be another indicator that these buildings are those that typically do not have windows such as those found in non-residential areas or commercial buildings and parking garages that can be found in residential areas.

# In[136]:


# visualising the distribution of buildings that are fenced (N) and not fenced (V) areas and the number of windows each building has
plt.figure(figsize=(4, 4), dpi=100)
sns.countplot(x='NumberOfWindows',hue="Garden", data=df)
plt.show()


# The above visualisation shows that the buildings with the symbol '.' for the number of windows that a building has do not have a garden and could also indicate industrial infrastructure such as factories, warehouses or parking garages - ruling out that they are not residential. This can still be investigated further.

# In[137]:


# visualising the distribution of buildings that are fenced (N) and not fenced (V) areas and the number of windows each building has
plt.figure(figsize=(4, 4), dpi=100)
sns.countplot(x='NumberOfWindows',hue="Settlement", data=df)
plt.show()


# The above visualisation shows that the settlement at which these buildings with the '.' symbol for the number of windows that a building are located in urban areas. Urban and rural areas differ significantly in terms of population density, land use, infrastructure, and types of buildings. Urban areas have a large population density and are primarily used for residential, commercial, and industrial purposes. This does not confirm nor deny if the buildings have a typical requirement for windows or not, however, they state their specific location of buildings in those locations with an unknown number of windows.
# 
# The analysis has brought me to the conclusion that these buildings could be considered windowless and thus the symbol '.' representing the Number of windows will be converted to a zero (0) going forward.

# In[213]:


# now to edit the categorical data of the NumberOfWindows attribute
df['NumberOfWindows'].replace(to_replace = {'   .':'0'},inplace=True)
# to determine the unique values under each categorical attribute, which can help determine if there are any anomalies
print('\nUnique Values in {} has\n{} '.format('NumberOfWindows',df['NumberOfWindows'].value_counts()))
print('\n')


# Now we can look at any missing values that could be found in the other attributes within our dataset.

# In[214]:


# Visualizing the missing data using a matrix display image
# Percentage of missing data per column
(df.isnull().sum() / df.shape[0] * 100.00).round(2)


# In[215]:


df.isnull().sum()


# At most, less than 10% of the values within the dataset - specifically within the Date_of_Occupancy attribute are missing. Depending on the extent at which there are missing values, they can be dealt with in the following ways:
# 
# * All the rows with missing data can be eliminated
# * All the columns with a lot of missing data can be eliminated
# * Missing data can be imputated
# 
# Imputation of missing data can be performed by replacing replacing the values with the mean, median or mode, or using k-nearest neighbour to replace the data values with similar computed values based on similarity and nearness.
# 
# Regarding our current dataset, the missing values are imputated using the mean and mode.

# In[216]:


# fill missing values with mode
df['Garden'].fillna(df['Garden'].mode()[0],inplace=True)
df['Date_of_Occupancy'].fillna(df['Date_of_Occupancy'].mode()[0],inplace=True)
df['Geo_Code'].fillna(df['Geo_Code'].mode()[0],inplace=True)

# fill missing values with mean
df['Building Dimension'].fillna(df['Building Dimension'].mean(),inplace=True)

# verify that there are no missing values
(df.isnull().sum() / df.shape[0] * 100.00).round(2)


# # FEATURE ENGINEERING

# Feature engineering occurs after cleaning the dataset and before splitting the data. Feature engineering transforms raw data into meaningful features that improve model performance.

# Now that we have dealt with missing features and converted our data types, we shall now consider if our dataset requires the scaling of numerical features, encoding categorical variables and dimensionality reduction.
# 
# Our dataset contains a column called Insured_Period. According to the variable description, these values are to represent the duration of insurance policy in Olusola Insurance. (Ex: Full year insurance, Policy Duration = 1; 6 months = 0.5). Further exploration is required to understand this numerical column.

# In[218]:


print('\nUnique Values in {} has\n{} '.format('Insured_Period',df['Insured_Period'].value_counts()))
print('\n')


# The values above show that the column Insured_Period has values ranging from 0.0 to 1.0, which slightly opposes the initial understanding of the description outlined of our attribute in terms of expected values to indicate that an insured period of 1 year is represented by the value 1 and 6 months is represented by the value 0.5. However, this does align with the intention of the attribute. It is thus understood that the value of 0.0 indicates that the insured period is 0 months, 0.5 would indicate that the insured period is 6 months (half a year or 0.5 year) and 1.0 indicates that the insured period is 12 months (1 full year of insurance or coverage). The values in between 0.0 and 1.0 indicate varying lengths of insured period within a 12-month (full year) period. This attribute shall remain a numerical value.
# 
# The Building Dimension attribute is a column that indicates the area of the building in meters squared. This value is numerical, and shall remain so.
# 
# The difference in scale between these values require normalization/standarization. The reason for applying scaling/normalization is because:
# 
# 1. Many machine learning algorithms are sensitive to the scale of the input features. Features with larger values can dominate the model, leading to biased results. Scaling ensures that all features contribute equally to the model.
# 2. Scaling can improve the convergence speed and stability of some algorithms, especially gradient-based methods like linear regression, logistic regression, and neural networks.  These algorithms often converge faster when the features are on a similar scale.
# 3. If one feature has a much larger range of values than others, it can overshadow the importance of other features. Scaling prevents this dominance.
# 

# Our feature engineering shall include the following steps:
# 
# 1. First we drop the Customer Id column, as it contains unique values.
# 2. We then encode the categorical variables using one-hot encoding.
# 3. Scale the numerical features:to fit within the same range of 0-1.
# 4. We split the data into test and train sets
# 5. Then we perform feature selection using a filter method.

# In[221]:


df.info()


# # MODELING

# In[222]:


# Drop Customer Id
df2 = df.drop(['Geo_Code','Customer Id'], axis=1)
df2.info()


# In[223]:


df2.info()


# In[224]:


# separate features (X) and target (y)
X = df2.drop('Claim', axis=1)
y = df2['Claim']

# re-identify categorical and numerical columns
categorical_cols = [col for col in X.columns if X[col].dtype == 'category']
numerical_cols = [col for col in X.columns if X[col].dtype == 'float64']

# one-hot encode the categorical features
X_categorical_df = pd.get_dummies(X[categorical_cols], drop_first=True)

# scale the numerical features
scaler = StandardScaler()
X_numerical_scaled = scaler.fit_transform(X[numerical_cols])
X_numerical_df = pd.DataFrame(X_numerical_scaled, columns=numerical_cols)

# combine the preprocessed features
X_processed = pd.concat([X_categorical_df, X_numerical_df], axis=1)

# then split into training and testing sets (applied before feature selection)
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=42)


# In[235]:


# training of our model
model = LogisticRegression()
model.fit(X_train, y_train)

# to get the probabilities for the training set
y_train_proba = model.predict_proba(X_train)[:, 1]  # Probabilities for class 1

# to also get the probabilities for the test set
y_test_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for class 1

# to calculate ROC AUC for the training set
roc_auc_train = roc_auc_score(y_train, y_train_proba)
print("Train ROC AUC:", roc_auc_train)

# to also calculate ROC AUC for the test set
roc_auc_test = roc_auc_score(y_test, y_test_proba)
print("Test ROC AUC:", roc_auc_test)

# the plotting of the ROC curve for the training set
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
roc_auc_curve_train = auc(fpr_train, tpr_train)

plt.figure()
plt.plot(fpr_train, tpr_train, color='darkorange', lw=2, label=f'Train ROC curve (area = {roc_auc_curve_train:.2f})')

# the plot ROC curve for the test set
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)
roc_auc_curve_test = auc(fpr_test, tpr_test)

plt.plot(fpr_test, tpr_test, color='navy', lw=2, label=f'Test ROC curve (area = {roc_auc_curve_test:.2f})')

plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')  # Random classifier line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# An important recall:
# 
# The target variable is imbalanced, and for an imbalanced dataset, the model created is best evaluated using the ROC (Receiver Operating Characteristic) and AUC (Area Under the Curve). These metrics handle class imbalance better than evaluating the performance of a model using accuracy. These metrics focus on the model's ability to correctly classify both classes, regardless of their imbalance. AUC also particularly measures how well the model ranks positive instances higher than negative instances, which is particularly useful in this instance where we need to evaluate if a building will have an insurance claim during a certain period (1) or not (0).
# 
# Our model finally results in:
# Train ROC AUC: 0.73
# Test ROC AUC: 0.70

# In[ ]:




