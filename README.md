# Mini-Project--Application-of-NN


(Expected the following details in the report )
## Project Title:
Rainfall Prediction.
## Project Description 
Rainfall Prediction is the application of science and technology to predict the amount of rainfall over a region. It is important to exactly determine the rainfall for effective use of water resources, crop productivity and pre-planning of water structures.
## Algorithm:
1.Import necessary libraries.

2.Apply the rainfall dataset to algoritm.

3.Read the dataset.

4.Plot the graph and correlation matrix.

5.Study the final output.
## Program:
```
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] 
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()
    
    def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number])
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] 
    columnNames = list(df)
    if len(columnNames) > 10: 
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()
    
    nRowsRead = 1000
df2 = pd.read_csv('rainfall.csv', delimiter=',', nrows = nRowsRead)
df2.dataframeName = 'rainfall in india 1901-2015.csv'
nRow, nCol = df2.shape
print(f'There are {nRow} rows and {nCol} columns')

df2.head(5)
plotPerColumnDistribution(df2, 10, 5)
plotCorrelationMatrix(df2, 8)
plotScatterMatrix(df2, 20, 10)
```
## Output:
![image](https://user-images.githubusercontent.com/94165103/205506293-b69393d6-261b-4019-a984-3cda79a2c0d5.png)
![image](https://user-images.githubusercontent.com/94165103/205506328-22380e8b-8fdd-4e77-8355-853471198459.png)
![image](https://user-images.githubusercontent.com/94165103/205506361-3351b8b1-6b52-4110-b66c-bcf6b5ebce8e.png)
![image](https://user-images.githubusercontent.com/94165103/205506382-7fb949e8-f24d-49ac-b371-9c396d3819cc.png)
![image](https://user-images.githubusercontent.com/94165103/205506404-735cdcad-9019-45ec-9ca4-e0bf8ed6da60.png)
## Advantage :
Rainfall prediction is important as heavy rainfall can lead to many disasters. The prediction helps people to take preventive measures and moreover the prediction should be accurate. There are two types of prediction short term rainfall prediction and long term rainfall.
## Result:
Thus Implementation of Rainfall Prediction was executed successfully.
