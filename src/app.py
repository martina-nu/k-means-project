import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


url = 'https://raw.githubusercontent.com/4GeeksAcademy/k-means-project-tutorial/main/housing.csv'
df_raw = pd.read_csv(url)

df = df_raw.loc[:, ["MedInc", "Latitude", "Longitude"]]

scaler = StandardScaler() 
scaled_df = scaler.fit_transform(df) 

scaled_df = pd.DataFrame(scaled_df,columns=df.columns) # use this df when k=6

scaled_df2 = scaled_df.copy() #find optimal k (2)

# 6 clusters

kmeans = KMeans(n_clusters=6)
scaled_df["Cluster"] = kmeans.fit_predict(scaled_df)
scaled_df["Cluster"] = scaled_df["Cluster"].astype("category")

#Optimal number of clusters: 2

kmeans = KMeans(n_clusters=2)
scaled_df2["Cluster"] = kmeans.fit_predict(scaled_df2)

scaled_df2["Cluster"] = scaled_df2["Cluster"].astype("category")


