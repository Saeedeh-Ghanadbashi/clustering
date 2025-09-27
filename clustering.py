import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('./income.csv')

df.head()
plt.scatter(df['Age'],df['Income($)'])
plt.xlabel('Age')
plt.ylabel('Income($)')

plt.show()

from sklearn.cluster import KMeans

km_model = KMeans(n_clusters=3)

df = df.drop('Name', axis = 'columns')
km_model.fit(df[['Age','Income($)']])
km_model.fit(df)
y_pred = km_model.predict(df)

df['cluster'] = y_pred

df.head()

km_model.score(df)

km_model.cluster_centers_

df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
plt.scatter(df1.Age,df1['Income($)'],color='green')
plt.scatter(df2.Age,df2['Income($)'],color='red')
plt.scatter(df3.Age,df3['Income($)'],color='black')
plt.scatter(km_model.cluster_centers_[:,0],km_model.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.xlabel('Age')
plt.ylabel('Income ($)')
plt.legend()
plt.show()

from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
df['Age'] = mms.fit_transform(df[['Age']])

df['Income($)'] = mms.fit_transform(df[['Income($)']])
df.head()

cluster_pred = km_model.fit_predict(df[['Age', 'Income($)']])

df['new cluster'] = cluster_pred

df.head()

df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
plt.scatter(df1.Age,df1['Income($)'],color='green')
plt.scatter(df2.Age,df2['Income($)'],color='red')
plt.scatter(df3.Age,df3['Income($)'],color='black')
plt.scatter(km_model.cluster_centers_[:,0],km_model.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.xlabel('Age')
plt.ylabel('Income ($)')
plt.legend()
plt.show()

sse = []
k_rng = range(1,10)
for i in k_rng:
    km = KMeans(n_clusters=i)
    km.fit(df[['Age', 'Income($)']])
    sse.append(km.inertia_)

plt.xlabel('K')
plt.ylabel('Sum of Squared Error')
plt.plot(k_rng, sse)
plt.show()

#K means clustering finished


