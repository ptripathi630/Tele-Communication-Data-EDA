#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Title:Telecommunication User Behavior Analysis and Usage Prediction System
#Objective: To analyze large-scale mobile network usage data collected from
#telecom subscribers in order to extract meaningful insights about user behavior,
#app usage trends, and network demand. The project also aims to identify high data users and
#develop predictive models that can assist in decision-making for customer segmentation,
#network optimization, and personalized marketing.


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from IPython.display import display
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
import pickle


# ## Load telecom data 

# In[3]:


df=pd.read_csv(r"C:\Users\Priya Tripathi\Downloads\telcom_data (2).xlsx - Sheet1.csv")


# In[4]:


df.head()


# In[5]:


df.info()


# ## Basic information

# In[6]:


df.shape


# In[7]:


df.shape


# In[8]:


df.describe()


# In[9]:


df.duplicated().sum()


# In[10]:


df.dtypes[df.dtypes=='object']


# In[11]:


df.dtypes[df.dtypes!='object']


# In[12]:


df.isnull().sum()


# ## IMPUTATION ITERATE OVER THE COLUMNS

# In[13]:


df = pd.DataFrame(df)
for column in df.columns:
    if df[column].dtype == 'object':  # For object (categorical) columns
        df[column].fillna(df[column].mode()[0], inplace=True)
    elif pd.api.types.is_numeric_dtype(df[column]):  # For numeric columns
        df[column].fillna(df[column].mean(), inplace=True)


# In[14]:


df.isnull().sum()


# ## Multivarient and univarient visualisation

# In[15]:


df["Handset Type"].unique()
df["Handset Type"].value_counts()


# 
# ## correlation matrix
# 
# 

# In[16]:


numeric=df.select_dtypes(include=["float","int"])
correlation_matrix=numeric.corr()


# In[17]:


plt.figure(figsize=(12,8))
correlation_matrix = correlation_matrix[correlation_matrix.abs() > 0.8]
sn.heatmap(correlation_matrix,fmt='.2g',cmap="coolwarm",annot=True)
plt.title('Correlation Matrix Heatmap')
plt.show()


# In[18]:


pd.set_option('display.max_columns', None)
df


# ## Removed highely correlated data
# 

# In[19]:


df= df.drop(columns=["Nb of sec with 6250B < Vol DL < 31250B", "Dur. (ms).1", "Avg RTT UL (ms)","Nb of sec with 1250B < Vol UL < 6250B"],axis=1,
        )


# In[20]:


df


# ## Multivarient analysis
# 

# In[21]:


fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

# Box plot for Dataset 1
axes[0].boxplot(x="Dur. (ms)", patch_artist=True, data=df, boxprops=dict(facecolor='lightblue'))
axes[0].set_title("Dataset 1")
axes[0].set_ylabel("Values")

# Box plot for Dataset 2
axes[1].boxplot(x="IMSI", patch_artist=True,data=df, boxprops=dict(facecolor='red'))
axes[1].set_title("Dataset 2")

# Box plot for Dataset 3
axes[2].boxplot(x="MSISDN/Number", patch_artist=True, data=df,boxprops=dict(facecolor='lightcoral'))
axes[2].set_title("Dataset 3")

# Adjust spacing
plt.tight_layout()

# Show the plots
plt.show()


# ## FEATURE ENGENEERING
# 

# In[22]:


df["Handset Manufacturer"].unique()
df["Handset Manufacturer"].value_counts()


# Adding two indentical columns to simplify the data
# 

# In[23]:


df["Total youtube traffic (bytes)"]=df["Youtube DL (Bytes)"] +df["Youtube UL (Bytes)"]
df["Total Social media (bytes)"]=df["Social Media DL (Bytes)"]+df["Social Media UL (Bytes)"]
df["Total Gaming (bytes)"]=df["Gaming DL (Bytes)"]+df["Gaming UL (Bytes)"]
df["Total Netflix (bytes)"]=df["Netflix DL (Bytes)"]+df["Netflix UL (Bytes)"]
df["Total Gmail (bytes)"]=df["Google DL (Bytes)"]+df["Google UL (Bytes)"]
df["Total email (bytes)"]=df["Email DL (Bytes)"]+df["Email UL (Bytes)"]
df["Total Other (bytes)"]=df["Other DL (Bytes)"]+df["Other UL (Bytes)"]


# In[24]:


df


# In[25]:


y=df["Total youtube traffic (bytes)"].sum()
ga=df["Total Gaming (bytes)"].sum()
NET=df["Total Netflix (bytes)"].sum()
GM=df["Total Gmail (bytes)"].sum()
EM=df["Total email (bytes)"].sum()
sm=df["Total Social media (bytes)"].sum()


# ## Gmail consume the most data
# 

# In[26]:


print(y,ga,NET,GM,EM,sm)


# In[27]:


avg_price_bldg_type = df.groupby('Handset Type')['Total Gaming (bytes)'].mean().sort_values(ascending=False)

# Plot the data
plt.figure(figsize=(10, 6))
sn.barplot(x=avg_price_bldg_type.index, y=avg_price_bldg_type.values, palette='Reds')
plt.title('handset type and total gaming ', fontsize=16)
plt.xlabel('handset', fontsize=12)
plt.ylabel('Total gaoming in bytes', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Output the average price for reference
print("total gaming in bytes consumed by handset stype:")



# In[28]:


plt.figure(figsize=(10, 6))
sn.scatterplot(x='Handset Manufacturer', y='IMEI', data=df, color='red')
plt.title('unique IMEI belongs to handset ')
plt.xlabel('handset')
plt.ylabel('IMEI')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[29]:


df


# In[30]:


df["Handset Manufacturer"].unique()
df["Handset Manufacturer"].value_counts()


# ## APPLE is the largest  handset manufacture 

# In[31]:


df["Handset Manufacturer"].unique()
df["Handset Manufacturer"].value_counts()


# In[32]:


plt.figure(figsize=(10, 6))
sn.scatterplot(
    data=df,
    x='Handset Manufacturer',
    y='Total Gmail (bytes)',
    hue='Total youtube traffic (bytes)',
    palette='viridis',
    alpha=0.7
)

# Add labels and title
plt.title('handset manufacture consume gamil bytes', fontsize=14)
plt.xlabel('handset type', fontsize=12)
plt.ylabel('gamil(bytes)', fontsize=12)
plt.grid(True)

# Show the plot
plt.show()


# In[33]:


total_social_bytes_consumed_by_handset = df.groupby('Handset Type')['Total Social media (bytes)'].mean()


plt.figure(figsize=(10, 6))
total_social_bytes_consumed_by_handset.sort_values().plot(kind='bar', color='purple', edgecolor='blue')


plt.title('total_social_bytes_consumed_by_handset', fontsize=14)
plt.xlabel('handset', fontsize=12)
plt.ylabel('Total social media (bytes)', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)


plt.show()


# ## REMOVING UNNECESSORY COLUMNS FROM THE DATA
# 

# In[34]:


df = df.drop(columns=['Start','End'],axis=1,)


# In[35]:


df


# ## KFOLD CLUSTER  ML model for unsupervised data 

# In[36]:


from sklearn.model_selection import KFold
df = pd.DataFrame(df)

# Initialize KFold
k = 5  # Number of splits
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Perform K-fold split
fold = 1
for train_index, test_index in kf.split(df):
    train_data = df.iloc[train_index]  # Select training data
    test_data = df.iloc[test_index]    # Select testing data

    print(f"Fold {fold}:")
    print("Train Data:")
    print(train_data)
    print("\nTest Data:")
    print(test_data)
    print("-" * 50)
    fold += 1



# In[37]:


df = pd.DataFrame(df)

# Initialize KFold
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Visualization
plt.figure(figsize=(10, 6))
fold = 1
for train_index, test_index in kf.split(df):
    plt.scatter(train_index, [fold] * len(train_index), c='blue', label='Train' if fold == 1 else "")
    plt.scatter(test_index, [fold] * len(test_index), c='orange', label='Test' if fold == 1 else "")
    fold += 1

plt.xlabel("Indices in DataFrame")
plt.ylabel("Fold")
plt.title("Graphical Representation of K-Fold Splits")
plt.legend(loc="upper right")
plt.show()


# ## label endcoder to convert object data type to binery 

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder

for column in df.select_dtypes(include=['object']).columns:
    label_encoder = LabelEncoder()
    df[column] = label_encoder.fit_transform(df[column])

# Now, proceed with your K-fold split and KMeans clustering:
df = pd.DataFrame(df)
X = df.values

kf = KFold(n_splits=5, shuffle=True, random_state=42)
silhouette_scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]

    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X_train)

    test_labels = kmeans.predict(X_test)

    score = silhouette_score(X_test, test_labels)
    silhouette_scores.append(score)

print("Silhouette Scores for each fold:", silhouette_scores)
print("Average Silhouette Score:", np.mean(silhouette_scores))


# ## Graphical representation of clustering and centriod 

# In[ ]:


df = pd.DataFrame(df)


X = df.values


kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_


plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X', label='Centroids')
plt.title('K-Means Clustering Results', fontsize=14)
plt.xlabel('Feature1', fontsize=12)
plt.ylabel('Feature2', fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# ## MODEL saved in pickel file 

# In[ ]:


with open('kmeans_model.pkl', 'wb') as model_file:
    pickle.dump(kmeans, model_file)

with open('data.pkl', 'wb') as data_file:
    pickle.dump(df, data_file)

print("Model and data saved to pickle files!")


with open('kmeans_model.pkl', 'rb') as model_file:
    loaded_kmeans = pickle.load(model_file)

with open('data.pkl', 'rb') as data_file:
    loaded_data = pickle.load(data_file)

print("Model and data loaded successfully!")
print("Cluster Centers from Loaded Model:")
print(loaded_kmeans.cluster_centers_)


# ## Prepare the Data for SQL

# In[ ]:


# Rename columns to avoid SQL issues (e.g., spaces, special characters)
df.columns = df.columns.str.replace(' ', '_')

# Display the cleaned dataset
print(df)


# ## project executed

# In[ ]:





# In[ ]:





# In[ ]:




