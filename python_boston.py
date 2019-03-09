#To find our datasets
#import sklearn.datasets
#[func for fuc in dir(sklearn.datasets) if fuc.startswith("load")]

from sklearn.datasets import load_boston
boston_data = load_boston()

#To find iris_data functions:
#print(dir(boston_data))

#print("DESC:", iris_data['DESCR']) -->> Description
#print("FILENAME:", iris_data['filename']) -->> File path on device
#print("TARGET:" , iris_data['target']) -->> 1xrow of predicted value (calsses / values)
#print("TARGET NAMES:" , iris_data['target_names']) -->> Name of predicted calsses
#print("FEATURES:" , iris_data['feature_names']) -->> Column names
#print("DATA:" , iris_data['data']) -->> Numpy array of data values

#1. Visualize the data (pandas)

import pandas as pd
import numpy as np

df = pd.DataFrame(data= boston_data['data'], columns= boston_data['feature_names'])
df["MEDV"] = boston_data.target
print(df.head())


#2. Apply KNN Classifier / Regressor
#2.1 Split the data into training and testing

from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, test_size = 0.5, random_state = 5)

print(df_train.shape)
print(df_test.shape)

#2.2 Regression
from sklearn.linear_model import LinearRegression

regr = LinearRegression()
print("Train rows: ", df_train.iloc[:,:-1].shape)
print("Train Y: ", df_train.iloc[:,-1].shape)

#Plotting:

import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

#Create Directories
if not (os.path.exists('./Cols-Histograms') or os.path.exists('./Cols-Scatters')):
    os.makedirs('./Cols-Histograms')
    os.makedirs('./Cols-Scatters')

print("Please wait it'll take a minute!")

#Save a pairplot for all data (overview)
sns.pairplot(df)
plt.savefig("pairplot.png")
plt.close()

#Histogram for each col.

for col in df:
    idx = df.columns.get_loc(col)

    sns.distplot(df[col].values,rug=False,bins=50).set_title("Histogram of {0}".format(col))
    plt.savefig("Cols-Histograms/{0}_{1}.png".format(idx,col), dpi=100)
    plt.close()

#Scatterplot and a regression line for each 2 columns    
    for nxt_col in df.iloc[:,idx+1:]:

        sns.regplot(df[col], df[nxt_col], color='r')
        plt.xlabel('Value of {0}'.format(col))
        plt.ylabel('Value of {0}'.format(nxt_col))
        plt.title('Scatter plot of {0} and {1}'.format(col,nxt_col))

        plt.savefig("Cols-Scatters/{0}_{1}_{2}".format(idx,col,nxt_col), dpi=200)
        plt.close()

print("-------------- FINISHED! --------------")

#4. Print the output "score" or "performance" of the classifier/regressor

