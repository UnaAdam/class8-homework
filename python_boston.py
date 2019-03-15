#!usr/bin/env python

#0. Load data
from sklearn.datasets import load_boston
boston_data = load_boston()

#1. Visualize the data (pandas)
import pandas as pd
import numpy as np

df = pd.DataFrame(data= boston_data['data'], columns= boston_data['feature_names'])
df["MEDV"] = boston_data['target']

#2. Plotting:

import os
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(color_codes=True)

#2.1 Plot & calculate some overviews
print("Creating overview!")

#2.1.1 Pairplot
sns.pairplot(df)
plt.savefig("Pairplot.png")
plt.close()

#2.1.2 Correlation matrix
correlation_matrix = df.corr().round(2)
plt.figure(figsize=(20, 15))
sns.heatmap(data=correlation_matrix, annot=True)
plt.savefig("Correlation_Matrix.png")
plt.close()

#2.1.3 Max & Min Corr. to MEDV
medv_corr = correlation_matrix.iloc[13, :-1]
maxcor_col = medv_corr.idxmax()
mincor_col = medv_corr.idxmin()

print("Max Correlation with MEDV: {0}, Corr. value = {1}".format(
    maxcor_col, max(medv_corr)))
print("Min Correlation with MEDV: {0}, Corr. value = {1}".format(
    mincor_col, min(medv_corr)))


#2.2 Plot Features
#2.2.1 Create Directories to save figs
if not (os.path.exists('./Cols-Histograms')):
    os.makedirs('./Cols-Histograms')

if not (os.path.exists('./Cols-Scatters')):
    os.makedirs('./Cols-Scatters')

if not (os.path.exists('./multiple_features_plotly')):
    os.makedirs('./multiple_features_plotly')

#2.2.2 Histogram for each col.

print("Creating histograms and scatter plots!")

for col in df:
    idx = df.columns.get_loc(col)

    sns.distplot(df[col].values,rug=False,bins=50).set_title("Histogram of {0}".format(col))
    plt.savefig("Cols-Histograms/{0}_{1}.png".format(idx,col), dpi=100)
    plt.close()

#2.2.3 Scatterplot and a regression line for each 2 columns
    
    for nxt_col in df.iloc[:,idx+1:]:

        sns.regplot(df[col], df[nxt_col], color='r')
        plt.xlabel('Value of {0}'.format(col))
        plt.ylabel('Value of {0}'.format(nxt_col))
        plt.title('Scatter plot of {0} and {1}'.format(col,nxt_col))

        plt.savefig("Cols-Scatters/{0}_{1}_{2}".format(idx,col,nxt_col), dpi=200)
        plt.close()

#2.2.4 Scatterplot for +3 features
print("Creating plots for 5 features!")

import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


sorted_df = df.sort_values("MEDV")
sorted_df = sorted_df.reset_index(drop=True)

for col in sorted_df:

    if(col == maxcor_col or col == mincor_col or col == 'MEDV'):
        continue

    idx = df.columns.get_loc(col)

    trace0 = go.Scatter(
        x=sorted_df['MEDV'],
        y=sorted_df[maxcor_col],
        mode='lines',
        name=maxcor_col
    )

    trace1 = go.Scatter(
        x=sorted_df['MEDV'],
        y=sorted_df[mincor_col],
        mode='lines',
        name=mincor_col
    )

    trace2 = go.Scatter(
        x=sorted_df['MEDV'],
        y=sorted_df[col],
        mode='lines',
        opacity=0.8,
        name=col
    )

    for col2 in sorted_df.iloc[:, idx+1:]:
        if(col2 == maxcor_col or col2 == mincor_col or col2 == "MEDV"):
            continue

        trace3 = go.Scatter(
            x=sorted_df['MEDV'],
            y=sorted_df[col2],
            mode='lines',
            opacity=0.8,
            name=col2
        )

        data = [trace0, trace1, trace2, trace3]
        layout = go.Layout(
            title='MEDV vs {0}, {1}, {2}, {3}'.format(
                maxcor_col, mincor_col, col, col2),
            yaxis=dict(title='MEDV'),
            xaxis=dict(title='{0}, {1}, {2}, {3}'.format(
                maxcor_col, mincor_col, col, col2)),
            plot_bgcolor="#f3f3f3"
        )

        fig = go.Figure(data=data, layout=layout)
        plot(fig, filename="./multiple_features_plotly/{0}_{1}_{2}.html".format(
            idx, col, col2), auto_open=False)


#3. Apply KNN Classifier / Regressor
#3.1 Split the data into training and testing
from sklearn.model_selection import train_test_split
print("Creating and fitting Regression Model!")

df_train, df_test, medv_train, medv_test = train_test_split(boston_data["data"], boston_data["target"])
#3.2 Regression
from sklearn.linear_model import LinearRegression

regr = LinearRegression()
regr.fit(df_train, medv_train)

predicted_medv = regr.predict(df_test)
expected_medv = medv_test

#3.3 Regression performance

sns.regplot(expected_medv, predicted_medv, color='g')

plt.xlabel('Expected Value')
plt.ylabel('Predicted Value')
plt.title('Expected vs Predicted Value')

plt.savefig("Regression.png")
plt.close()

print("-------------- FINISHED! --------------")
