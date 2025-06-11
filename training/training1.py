import pandas as pd
df=pd.read_csv("training\dataset\synthetic_logs.csv")

#print(df.head())

df.source.unique()
df.target_label.unique()

#write code to cluster logmessages using dbscan algorithm use sentence transformer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
# Load the pre-trained SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')
# Encode the log messages
log_embeddings = model.encode(df['log_message'].tolist(), show_progress_bar=True)
# Standardize the embeddings
log_embeddings = StandardScaler().fit_transform(log_embeddings)
# Perform DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5, metric='cosine')
clusters = dbscan.fit_predict(log_embeddings)
df['cluster'] = clusters
df.head()
print(df[df.cluster==1])