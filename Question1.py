from scipy import stats
from sklearn.cluster import KMeans
import json
import pandas as pd


with open('/home/user/Téléchargements/DataAnalysisChallenge/city_search.json') as f:
    data = json.load(f)

#
# Getting data to a dataframe
n = 3
lists = [[] for _ in range(n)]
for i in range(len(data)):
    lists[0].append(data[i]['user'][0][0]['country'])
    lists[1].append(data[i]['user'][0][0]['user_id'])
    lists[2].append(data[i]['unix_timestamp'][0])

# df is a dataframe version of the json file
df = pd.DataFrame()
df['country'] = lists[0]
df['user_id'] = lists[1]
df['unix_timestamp'] = lists[2]

#Get the number of country
num_country = len(set(lists[0])) - 1
print(num_country, set(lists[0]))

#
# We use the get_dummies fct to convert categorical variable into dummy/indicator variables
# In our case the countries are the categorical variables
df = pd.get_dummies(df, columns=['country'])

# Since our variables are of incomparable units, we should standardize variables
cols = ['unix_timestamp', 'user_id','country_', 'country_DE',
         'country_FR','country_IT','country_US','country_UK','country_ES']

df_std = stats.zscore(df[cols])

# We want to know if our missing data belong to a country from our database, so we try 6 clusters,
kmeans = KMeans(n_clusters=num_country, random_state=0).fit(df_std)
labels = kmeans.labels_
df['clusters'] = labels
cols.extend(['clusters'])

# We take a look to the differences between the clusters
print(df[cols].groupby(['clusters']).mean())