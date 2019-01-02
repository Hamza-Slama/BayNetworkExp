# import necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np
import pandas as pd
from urllib.request import urlopen 
import os
from matplotlib import pyplot as plt

names=['age', 'year_of_treatment', 'positive_lymph_nodes', 'survival_status_after_5_years']
c1,c2,c3,c4= np.loadtxt('../input/data.csv',unpack=True,delimiter = ',')
cancer_df = pd.read_csv('../input/data.csv', header=None, names=names)
print(cancer_df.head())

cancer_df.head()
cancer_df.tail()

cancer_df['survival_status_after_5_years'] = cancer_df['survival_status_after_5_years'].map({1:"yes", 2:"no"})
cancer_df['survival_status_after_5_years'] = cancer_df['survival_status_after_5_years'].astype('category')

for idx, feature in enumerate(list(cancer_df.columns)[:-1]):
    fg = sns.FacetGrid(cancer_df, hue='survival_status_after_5_years', size=5)
    fg.map(sns.distplot, feature).add_legend()
    plt.show()
    
print("The age of patients are :",c1[0:5])
print("The year of operation are:",c2[0:5])
print("Number of positive axillary nodes detected are:",c3[0:5])
print("The labes (survived in 5 years[1-yes,2-no])",c4[0:5])
x=np.column_stack((c1,c3))
y=c4

plt.scatter(c1,c3,c=c4)
plt.colorbar(ticks=[ 1, 2])
plt.xlabel("Age of the patient")
plt.ylabel("No of positive axillary nodes")

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(x,y)
predictions=clf.predict(x)

from sklearn.metrics import accuracy_score
accuracy_score(y,predictions)

count = 0
for i in range(0,len(predictions)):
    if predictions[i]==y[i]:
        count+=1
    else:
        pass
accuracy = count/len(predictions)
print(accuracy)
