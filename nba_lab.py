# %%
# Importing packages
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import plotly
# %%
# Importing the datasets
salary_data = pd.read_csv('2025_salaries.csv', header = 1)
stats = pd.read_csv('nba_2025.txt', sep= ",")
# %%
# Merging the data together, dropping any rows that are duplicates of earlier rows
merged_data = pd.merge(salary_data, stats, on="Player")
duplicates = merged_data[merged_data.duplicated(subset="Player", keep=False)]
merged_data = merged_data.drop_duplicates(subset="Player")
merged_data.head()
# %%
# Making the salary column a string and creating a new column that measures salary in thousands of dollars
merged_data['2025-26'] = (merged_data['2025-26'].str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.strip().astype(float))
merged_data['2025-26/thou'] = merged_data['2025-26'] / 1000
# %%
# Adding new columns that calculate a player's per game statistics
merged_data['RPG'] = merged_data['TRB'] / merged_data ['G']
merged_data['APG'] = merged_data['AST'] / merged_data ['G']
merged_data['SPG'] = merged_data['STL'] / merged_data ['G']
merged_data['BPG'] = merged_data['BLK'] / merged_data ['G']
merged_data['PPG'] = merged_data['PTS'] / merged_data ['G']
merged_data['MPG'] = merged_data['MP'] / merged_data['G']
merged_data.tail()

# %%
# Creating a column that classifies a player's salary into one of four bins: Low, Below Average, Above Average, and High
merged_data['salary_value'] = pd.cut(
    merged_data['2025-26/thou'],
    bins=[-float('inf'), 6000, 16000, 26000, float('inf')],
    labels=['Low Salary',
            'Below Average Salary',
            'Above Average Salary',
            'High Salary']
)
merged_data.tail()
# %%
# Checking for rows that contain missing values and creating a new dataset that removes them
merged_data[['2025-26', 'PPG', 'RPG', 'APG', 'SPG', 'BPG', 'MPG']].isna().sum()
clean_data = merged_data[['Player', 'Age','Rk','Tm','2025-26/thou', 'salary_value','PPG', 'RPG', 'APG', 'SPG', 'BPG', 'MPG']].dropna()

# %%
# Creating new metrics that calculate a player's ability on either side of the court (offense and defense)
clean_data['offensive_rating'] = clean_data['PPG'] + clean_data['APG'] + clean_data['RPG']
clean_data['stocks'] = clean_data['BPG'] + clean_data['SPG']
clean_data.tail()
# %%
# Creating a numeric section of the dataset to be scaled and clustered
X = clean_data[['2025-26/thou', 'Age', 'Rk', 'PPG', 'RPG', 'APG', 'SPG', 'BPG', 'MPG']]
X.head()
# %%
# Scaling the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# %%
# Performing kMeans clustering on the scaled data (with k = 2 clusters)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=26)
kmeans.fit(X_scaled)
clean_data.loc[X.index, 'Cluster_2'] = kmeans.labels_
# %%
# Seeing where each cluster is located for each statistic
cluster2_summary = (
    clean_data
    .groupby('Cluster_2')
    [['2025-26/thou','Rk', 'Age', 'PPG','APG','RPG','SPG','BPG','offensive_rating','stocks', 'MPG']]
    .mean()
)

cluster2_summary
# %%
# Plotting a player's offensive rating based on the minutes per game they play
import plotly.express as px

fig = px.scatter(
    clean_data,
    x='offensive_rating',
    y='MPG',
    color='salary_value',
    symbol='Cluster_2',
    hover_name='Player',
    title='Offensive Rating vs Minutes Per Game by Salary Tier and Cluster'
)

fig.update_layout(
    xaxis_title='Offensive Rating Per Game',
    yaxis_title='Minutes Per Game'
)

fig.show()

# %%
# Plotting a player's combined steals and blocks per game based on the amount of time they play per game
fig2 = px.scatter(
    clean_data,
    x='stocks',
    y='MPG',
    color='salary_value',
    symbol='Cluster_2',
    hover_name='Player',
    title='Stocks per Game vs Minutes Per Game by Salary Tier and Cluster'
)

fig2.update_layout(
    xaxis_title='Stocks Per Game',
    yaxis_title='Minutes Per Game'
)

fig2.show()
# %%
# Looking a player's statistical ranking based on their salary for the current season
fig3 = px.scatter(
    clean_data,
    x='2025-26/thou',
    y='Rk',
    color='salary_value',
    symbol='Cluster_2',
    hover_name='Player',
    title='Rank vs Salary by Salary Tier and Cluster'
)

fig3.update_layout(
    xaxis_title='Salary (thousands)',
    yaxis_title='Rank'
)

fig3.show()
# %%
# Calculating the variance explained of the clustering model with 2 clusters
total_variance = np.var(X_scaled, axis=0).sum() * X_scaled.shape[0]
within_variance = kmeans.inertia_
variance_explained = (total_variance - within_variance) / total_variance
print("Variance Explained:", variance_explained)
# %%
# Calculating the silhouette score of the clustering model with 2 clusters
from sklearn.metrics import silhouette_score
sil_score = silhouette_score(X_scaled, kmeans.labels_)
print("Silhouette Score:", sil_score)
# %%
# Finding the inertia levels and silhouette scores for each cluster level from k = 2 to k = 11
inertia = []
silhouette_scores = []
k_values = range(2, 11)

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    
    inertia.append(km.inertia_)
    silhouette_scores.append(
        silhouette_score(X_scaled, km.labels_)
    )
# %%
# Performing the elbow method for inertia level
plt.figure()
plt.plot(k_values, inertia)
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

# %%
# Analyzing the silhouette scores per k level
plt.figure()
plt.plot(k_values, silhouette_scores)
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Scores by k")
plt.show()
# %%
# Performing kMeans with k = 4 clusters for a closer analysis
kmeans = KMeans(n_clusters=4, random_state=26)
kmeans.fit(X_scaled)
clean_data.loc[X.index, 'Cluster_4'] = kmeans.labels_

# %%
# Seeing where the four clusters will be for each statistic
cluster4_summary = (
    clean_data
    .groupby('Cluster_4')
    [['2025-26/thou','Rk', 'Age', 'PPG','APG','RPG','SPG','BPG','offensive_rating','stocks', 'MPG']]
    .mean()
)
cluster4_summary
# %%
# Plotting a player's offensive rating based on the minutes per game they play
fig = px.scatter(
    clean_data,
    x='offensive_rating',
    y='MPG',
    color='salary_value',
    symbol='Cluster_4',
    hover_name='Player',
    title='Offensive Rating vs Minutes Per Game by Salary Tier and Cluster'
)

fig.update_layout(
    xaxis_title='Offensive Rating Per Game',
    yaxis_title='Minutes Per Game'
)

fig.show()

# %%
# Plotting a player's combined steals and blocks per game based on the amount of time they play per game
fig2 = px.scatter(
    clean_data,
    x='stocks',
    y='MPG',
    color='salary_value',
    symbol='Cluster_4',
    hover_name='Player',
    title='Stocks per Game vs Minutes Per Game by Salary Tier and Cluster'
)

fig2.update_layout(
    xaxis_title='Stocks Per Game',
    yaxis_title='Minutes Per Game'
)

fig2.show()
# %%
# Looking a player's statistical ranking based on their salary for the current season
fig3 = px.scatter(
    clean_data,
    x='2025-26/thou',
    y='Rk',
    color='salary_value',
    symbol='Cluster_4',
    hover_name='Player',
    title='Rank vs Salary by Salary Tier and Cluster'
)

fig3.update_layout(
    xaxis_title='Salary (thousands)',
    yaxis_title='Rank'
)

fig3.show()

# %%
# Calculating the variance explained for the 4-cluster model
total_variance = np.var(X_scaled, axis=0).sum() * X_scaled.shape[0]
within_variance = kmeans.inertia_
variance_explained = (total_variance - within_variance) / total_variance
print("Variance Explained:", variance_explained)
# %%
# Calculating the silhouette score for the 4-cluster model
sil_score = silhouette_score(X_scaled, kmeans.labels_)
print("Silhouette Score:", sil_score)

# %%
merged_data['Ty Jerome']