#!/usr/bin/env python
# coding: utf-8

# ## Exercise 
# 

# For this exercise, you will be working with the [House Price Dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/code).
# 
# Please grab the train.csv file from Kaggle and explore this dataset. You need to perform explroatory data analysis and see if there is any correlation between the variables and analyze the distribution of the dataset. The question is open-ended and basically you're asked to perform ED

# 1- Write a summary of your findings in one page (e.g., summary statistics, plots) and submit the pdf file. Therefore, for part 3 of your assignment, you need to submit at least one jupyter notebook file and one pdf file.
# 
# 
# 2- Push your code and project to github and provide the link to your code here. Ensure that your github project is organized to at least couple of main folders, ensure that you have the README file as well:
# 
# - Src
# - Data
# - Docs
# - Results
# 
# Read this link for further info:  https://gist.github.com/ericmjl/27e50331f24db3e8f957d1fe7bbbe510

# https://github.com/Karlacristheljp/Housing--Exploratory-Data-Analysis/blob/main/Src/House-price%20project.ipynb

# **Personnal notes**
# 
# #### Purpose of the project
# - **Training Set (`train.csv`)**: Contains the features of the houses (size, rooms, year built) along with their true target value (`SalePrice`). We use this as our "answer key" so the machine learning model can learn mathematical patterns (e.g., how an extra square foot or better quality affects the price).
# - **Test Set (`test.csv`)**: Contains features for new houses, but the `SalePrice` column is completely missing.
# - **The Goal**: To Perform an EDA of the given dataset.
# - 
# #### Purpose of EDA (Exploratory Data Analysis)
# EDA is the preliminary diagnostic phase before building any model. Its main purposes are:
# - **Understand the Data**: Detect errors, missing values (e.g., houses without a garage), or extreme outliers that could ruin predictions.
# - **Find Relationships**: Discover which features matter the most (e.g., how size or overall quality drives the price up or down).
# - **Make Decisions**: Help us decide which columns to clean, which ones to drop, and what transformations the model needs to perform better.
# 

# In[77]:


# Import galleries
import os
import pandas as pd #library to manipulate data
import matplotlib.pyplot as plt #library for graphics
import seaborn as sns

# 2. Configurar el estilo visual de los gráficos
sns.set_theme(style="whitegrid")

# Read the dataset
# Como nuestro notebook está en 'Src', subimos un nivel ('..') y entramos a 'Data'
ruta_train = os.path.join("..", "Data", "train.csv") 
#os.path.join("..", "Data", "train.csv") we use this to create a path, I said to python jump out of "Scr" and go in to data and search
#read the document .cvs
df_train = pd.read_csv(ruta_train)



# ## 1. Structure and dimension ##

# #### 1.1 Dataframe dimension and preview

# In[78]:


df_train.shape


# In[3]:


df_train.head(5)


# #### 1.2 General information and data types

# In[6]:


print("General information and null values")
df_train.info()


# #### 1.3 Staistical summary and Numerical variables

# **Statistical summary**

# In[7]:


print("Statistical summary")
df_train.describe().T[["count", "mean", "std", "min", "50%", "max"]].head(10|)


# **Numerical variables**

# In[55]:


numeric_cols = df_train.select_dtypes(include=['float64', 'int64'])
numeric_cols


# ## 2. Missing Values Analysis

# **2.1 Quantification of Null Data**

# In[14]:


#Check for missing values and sort them descending
missing_data = df_train.isnull().sum()

#Python, from my  missing_data, create a variable " missing_data"
# That comes from " missing_data" and the values in there I want the >0 ones
missing_data = missing_data[ missing_data > 0].sort_values(ascending = False)
print ("Columns with missing values:")
print (missing_data.head(10))


# **2.2 Percentage of Null Data**

# In[58]:


# Calculate percentage of missing values
#Calculates the percentage of missing values in each column
missing_data = (df_train.isnull().sum() / len(df_train)) * 100
missing_data = missing_data[missing_data > 0].sort_values(ascending=False).head(10)
missing_data


# **2.3 Bar plot: Top Features with Missing values**

# In[38]:


plt.figure(figsize=(8, 4))
sns.barplot(x=missing_data.index, y=missing_data.values, palette='Purples_r', hue=missing_data.index, legend=False)

plt.title('Top Features with Missing Values (%)', fontsize=12)
plt.xlabel('Features', fontsize=10)
plt.ylabel('Missing Percentage (%)', fontsize=10)
plt.xticks(rotation=45, ha='right')

# Save the plot in the Results folder for the PDF report
plt.savefig('../Results/missing_values_top.png', bbox_inches='tight')
plt.show()


# ## 3. Univariate Analysis: Numerical Variables ##

# ### 3.1 Histogram of KDE Curve for Sale Price and Living Area

# In[26]:


# Two subplots to analyze distributions of SalePrice and GrLivArea

#Subplot ( 1 column, 2 rows)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot histogram and KDE for SalePrice
#From df_train take column ['SalePrice'] #kde= provides the density of probability axes[0][1]= left and right
sns.histplot(df_train['SalePrice'], kde=True, color='royalblue', ax=axes[0], bins=30)
axes[0].set_title('Distribution of House Sale Prices', fontsize=12)
axes[0].set_xlabel('Sale Price ($)', fontsize=10)
axes[0].set_ylabel('Frequency', fontsize=10)


# Plot histogram and KDE for GrLivArea (Above Grade Living Area)
#From df_train take column ['GrLivArea'] #kde= provides the density of probability axes[0][1]= left and right
sns.histplot(df_train['GrLivArea'], kde=True, color='darkorange', ax=axes[1], bins=30)
axes[1].set_title('Distribution of Above Grade Living Area', fontsize=12)
axes[1].set_xlabel('Living Area (sq ft)', fontsize=10)
axes[1].set_ylabel('Frequency', fontsize=10)


plt.tight_layout()
plt.savefig('../Results/distributions_univariate.png', bbox_inches='tight')
plt.show()


# In[66]:


df_train[['SalePrice',"GrLivArea"] ].describe().T[["count", "mean", "std", "min", "50%", "max", "75%", "25%"]].head(10)


# ### 3.2 Frequency Distribution of Property Qualities

# In[80]:


df_train["OverallQual"].describe().T[["count", "mean", "std", "min", "50%", "max", "75%", "25%"]].head(10)


# In[28]:


# Count plot for OverallQual to see the frequency distribution of property qualities
#Plot count for continuous variables, counts how many housee belongs to each categorical discrete variable
plt.figure(figsize=(8, 4))
sns.countplot(x='OverallQual', data=df_train, palette='viridis', hue='OverallQual', legend=False)
plt.title('Distribution of Overall Material and Finish Quality', fontsize=12)
plt.xlabel('Overall Quality (1-10)', fontsize=10)
plt.ylabel('Number of Homes (Frequency)', fontsize=10)

# Save the plot in the Results folder for the PDF report
plt.savefig('../Results/overall_quality_count.png', bbox_inches='tight')
plt.show()


# ## 4. Bivariate Analysis: Correlation and Numerical Relationships ##

# ### 4.1 Filtering numerical variables and Correlation Matrix

# In[60]:


# To find which variables influence on the prices

#Select only numerical variables to calculate correlations
#Filter numerical variables by columns
numeric_cols = df_train.select_dtypes(include=['float64', 'int64'])

#Create a correlation matrix with the previous numerical variables .corr()
corr_matrix = numeric_cols.corr()
corr_matrix.head(10)


# #### 4.2 Correlation Heatmap with Sale Price

# In[11]:


#Plot the top variables most correlated with SalePrice > 0.5
plt.figure(figsize=(10, 8))

# Takes the values corr_matrix["SalePrice"] (corr_matrix["SalePrice"])in the corr_matrix 
# agains the target variable "SalePrice"
# absolute value greater than 5 abs(...) > 0.5
# corr_matrix.index[...] extracts and save the names of the columns that fit in the condition
top_corr_features = corr_matrix.index[abs(corr_matrix["SalePrice"]) > 0.5]


#(df_train[top_corr_features].corr from dataframe, take the values related with "Sales price"
# The ones obtained in the last code
#annot prints the values inside each square
#fmt rounds  to 2 decimals 
sns.heatmap(df_train[top_corr_features].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Top Correlations with SalePrice', fontsize=1)

# Save the plot in the Results folder for the PDF report
plt.savefig('../Results/correlation_matrix.png', bbox_inches='tight')

plt.show()


# #### 4.3 Relationship between Living Area and Price (Outlier Detection)

# **Scatterplot between GrLivArea and SalePrice**

# In[82]:


df_train["GrLivArea"].describe().T[["count", "mean", "std", "min", "50%", "max", "75%", "25%"]]


# In[23]:


plt.figure(figsize=(9, 6))
sns.scatterplot(x='GrLivArea', y='SalePrice', data=df_train, color='royalblue', alpha=0.7)
plt.title('GrLivArea vs SalePrice (Outlier Detection)', fontsize=14)
plt.xlabel('Above Grade Living Area (sq ft)', fontsize=12)
plt.ylabel('Sale Price ($)', fontsize=12)

# Save the plot in the Results folder for the PDF report
plt.savefig('../Results/grlivarea_vs_saleprice.png', bbox_inches='tight')

plt.show()


# #### 4.4 Impact of Overall Quality on Price

# **Box Plot of the relationship between Overall Quality and SalePrice**

# In[50]:


# Analyze the bivariate relationship between Overall Quality and SalePrice 
#Shows the sorted quality levels from 1 to 10
plt.figure(figsize=(9, 5))
sns.boxplot(x='OverallQual', y='SalePrice', data=df_train, palette='Set2', hue='OverallQual', legend=False)
plt.title('Sale Price Distribution by Overall Material and Finish Quality', fontsize=12)
plt.xlabel('Overall Quality (1-10)', fontsize=10)
plt.ylabel('Sale Price ($)', fontsize=10)

# Save the plot in the Results folder for the PDF report
plt.savefig('../Results/overall_qual_vs_saleprice.png', bbox_inches='tight')
plt.show()


# In[ ]:





# **Scatterplot between Overall Quality and SalePrice**

# In[75]:


plt.figure(figsize=(9, 6))
sns.scatterplot(x='OverallQual', y='SalePrice', data=df_train, color='orange', alpha=0.7)
plt.title('Overall Quality vs SalePrice (Outlier Detection)', fontsize=14)
plt.xlabel('Overall Quality', fontsize=12)
plt.ylabel('Sale Price ($)', fontsize=12)

# Save the plot in the Results folder for the PDF report
plt.savefig('../Results/grlivarea_vs_saleprice.png', bbox_inches='tight')

plt.show()


# #### 4.4 Relationship between Garage Capacity and Price

# **Box plot of Garage Capacity (Cars) and SalePrice**

# In[79]:


# Analyze the bivariate relationship between Garage Capacity (Cars) and SalePrice using a boxplot
plt.figure(figsize=(8, 4))
sns.boxplot(x='GarageCars', y='SalePrice', data=df_train, palette='Greens_r', hue='GarageCars', legend=False)
plt.title('Sale Price Distribution by Garage Capacity (Cars)', fontsize=12)
plt.xlabel('Garage Capacity (Number of Cars)', fontsize=10)
plt.ylabel('Sale Price ($)', fontsize=10)

# Save the plot in the Results folder for the PDF report
plt.savefig('../Results/garage_cars_vs_saleprice.png', bbox_inches='tight')
plt.show()


# #### 4.5 Relationship Between Total Rooms and Price

# **Box Plot of the relationship between Total Rooms Above Grade and SalePrice**

# In[53]:


# Analyze the bivariate relationship between Total Rooms Above Grade and SalePrice using a boxplot
plt.figure(figsize=(10, 5))
sns.boxplot(x='TotRmsAbvGrd', y='SalePrice', data=df_train, palette='Purples', hue='TotRmsAbvGrd', legend=False)
plt.title('Sale Price Distribution by Total Rooms Above Ground', fontsize=12)
plt.xlabel('Total Rooms Above Grade (excluding bathrooms)', fontsize=10)
plt.ylabel('Sale Price ($)', fontsize=10)

# Save the plot in the Results folder for the PDF report
plt.savefig('../Results/tot_rms_vs_saleprice.png', bbox_inches='tight')
plt.show()


# ## 5. Categorical variables Impact ##

# #### 5.1 Distribution of Sale Price by Neighborhood

# **Boxplot of Sale Price by Neighborhood**

# In[46]:


plt.figure(figsize=(14, 6))
sns.boxplot(x='Neighborhood', y='SalePrice', data=df_train, palette='Set3', hue='Neighborhood', legend=False)
plt.xticks(rotation=90) #Rotate the Neighborhood names
plt.title('Sale Price Distribution by Neighborhood', fontsize=14)
plt.xlabel('Neighborhood', fontsize=12)
plt.ylabel('Sale Price ($)', fontsize=12)
plt.savefig('../Results/saleprice_by_neighborhood.png', bbox_inches='tight')

plt.show()


# In[ ]:





# In[ ]:




