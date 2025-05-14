#Data Exploration

import pandas as pd

sfrs = pd.read_csv("sfr_test.csv")

sfrs.sample(n=5, random_state=42)

'''
Columns that contain the dollar $ sign are: ipa_funding, ma_premium, ma_risk_score, mbr_with_rx_rebates, partd_premium, pcp_cap, pcp_ffs, plan_premium, prof, reinsurance, risk_score_partd, rx, rx_rebates, rx_with_rebates, rx_without_rebates and spec_cap. Exactly 16 out of the 28 available. For easier manipulation, we will put them in a variable.
'''

financial_columns = [
    "ipa_funding",
    "ma_premium",
    "ma_risk_score",
    "mbr_with_rx_rebates",
    "partd_premium",
    "pcp_cap",
    "pcp_ffs",
    "plan_premium",
    "prof",
    "reinsurance",
    "risk_score_partd",
    "rx",
    "rx_rebates",
    "rx_with_rebates",
    "rx_without_rebates",
    "spec_cap"
]

# remove the dollar sign from the financial columns
sfrs[financial_columns] = sfrs[financial_columns].replace('[\$,]', '', regex=True).astype(float)

'''
We will consider two approaches to the outlier detection problem we are having: uni- and multi-variate.

'''

#Uni-Variate Outlier Detection

sfrs[financial_columns].describe().T

'''
Here are some bullet points about the data that might indicate some outliers:

the difference between the 75th and the max value of mbr_with_rx_rebates is very big compared to other quantiles
the 75th percentile of pcp_ffs is still zero, while its max is 6056.44
the 75th percentile of prof is still zero, while its max is 69516.96
all values of risk_score_partd and rx_rebates are zeros - these can be removed from the data
rx_with_rebates and rx_without_rebates have almost identical values
'''

financial_columns.remove("risk_score_partd")
financial_columns.remove("rx_rebates")

import seaborn as sns
from matplotlib import pyplot as plt

for i, financial_column in enumerate(financial_columns):
    plt.figure(figsize=(6, 4))
    plt.title(financial_column)
    sns.boxplot(data=sfrs[financial_column])

#Let's try to plot their distribution of values with a histogram and see if we can notice something interesting.

_ = sfrs[financial_columns].hist(figsize=(18, 12))

'''
As expected, we can see that the columns which had a lot of outliers have the most skewed distribution of values, e.g., rx, rx_with_rebates, prof, etc. However, looking at these plots we can't conclude row-wise outliers; they only give us some sense of the range of values of each column and what value could potentially be treated as an outlier.

We will try one last approach toward the uni-variate outlier detection part of this assignment. We will standardize the value of each column, i.e., calculate its z-score. We'll then try to threshold the values further than 3 standard deviations from the mean so that they are considered outliers.
'''

from scipy import stats
import numpy as np


z = np.abs(stats.zscore(sfrs[financial_columns]))

print(np.where(z > 3))

sfrs_outliers = sfrs[financial_columns][(z > 3).any(axis=1)]
sfrs.iloc[sfrs_outliers.index]
'''
Moreover, we can print the rows which have more than one outlier. Since we have 14 columns, we will illustrate the example where the rows have more than 7 (half) values as outliers.

'''

outliers_row_indices = np.where(z > 3)[0]
unique_values, counts = np.unique(outliers_row_indices, return_counts=True)
outlier_counts = np.array((unique_values, counts)).T

# try changing the threshold (number of columns that are considered outliers), and see the results
threshold = 7

outliers = [x for x in outlier_counts if x[1] > threshold]

indices = [i[0] for i in outliers]
sfrs.iloc[indices]

'''
Multi-Variate Outlier Detection
In this section, we will apply machine learning techniques to detect outliers. We will train some models that take all financial columns as input and output some kind of classification or categorization of whether the input is an outlier or not.

The sklearn library has implemented some of the most widely known and used machine learning algorithms for outlier detection: isolation forest and local outlier factor.

'''

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

isolation_forest = IsolationForest()

isolation_forest = isolation_forest.fit(sfrs[financial_columns])

inference = isolation_forest.predict(sfrs[financial_columns])


indices_if = np.where(inference == -1)[0]  # np.where() returns a tuple of one element

print("There are", len(indices_if), "members marked as outliers")

sfrs.iloc[indices_if]

isolation_forest = IsolationForest()

isolation_forest = isolation_forest.fit(sfrs[financial_columns])

inference = isolation_forest.predict(sfrs[financial_columns])


indices_if = np.where(inference == -1)[0]  # np.where() returns a tuple of one element

print("There are", len(indices_if), "members marked as outliers")

sfrs.iloc[indices_if]

common_indices = list(set(indices_if).intersection(set(indices_lof)))
sfrs.iloc[common_indices]

'''
111 members are identified as outliers by both algorithms. As a final presentation, we can plot the distribution of the values in the financial columns only for the detected outliers and try to compare them with the previous one over the entire data set. Try if you can notice any differences and explain why these members were identified as outliers.


'''
_ = sfrs.iloc[common_indices][financial_columns].hist(figsize=(19, 13))


