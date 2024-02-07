import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt  # corrected import statement
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer  # corrected import statement
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

data = pd.read_csv("online_retail_II.csv")
data.dropna(subset="Customer ID", axis=0, inplace=True)
data = data[~data.Invoice.str.contains('C', na=False)]
data = data.drop_duplicates(keep="first")

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return up_limit, low_limit

def replace_with_threshold(dataframe, variable):
    up_limit, low_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
    replace_with_threshold(data, "Quantity")
    replace_with_threshold(data, "Price")

data["Revenue"] = data["Quantity"] * data["Price"]
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

Latest_Date = dt.datetime(2011, 12, 10)

RFM = data.groupby('Customer ID').agg({'InvoiceDate': lambda x: (Latest_Date - x.max()).days, 'Invoice': lambda x: x.nunique(), "Revenue": lambda x: x.sum()})
RFM['InvoiceDate'] = RFM['InvoiceDate'].astype(int)
RFM.rename(columns={'InvoiceDate': 'Recency',
                    'Invoice': 'Frequency',
                    'Revenue': 'Monetary'}, inplace=True)
RFM.reset_index()
RFM = RFM[(RFM["Frequency"] > 1)]
Shopping_Cycle = data.groupby('Customer ID').agg({'InvoiceDate': lambda x: ((x.max() - x.min()).days)})
RFM["Shopping_Cycle"] = Shopping_Cycle["InvoiceDate"]
RFM["Interpurchase_Time"] = RFM["Shopping_Cycle"] // RFM["Frequency"]
RFMT = RFM[["Recency", "Frequency", "Monetary", "Interpurchase_Time"]]
st.header("isi dataset :")
st.write(data)

st.header("setelah menggunakan metode rfm :")
st.write(RFM)

st.header("penyesuaian dataset untuk clustering : ")
st.write(RFMT)

scaler = MinMaxScaler()
RFMT_normalized = scaler.fit_transform(RFMT)
clusters = []
for i in range(2, 15):
    km = KMeans(n_clusters=i).fit(RFMT_normalized)
    clusters.append(km.inertia_)

fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(x=list(range(2, 15)), y=clusters, ax=ax)
ax.set_title('Mencari Elbow')
ax.set_xlabel('Number of Clusters')
ax.set_ylabel('Inertia')

st.set_option('deprecation.showPyplotGlobalUse', False)
elbow_plot = st.pyplot()

st.sidebar.subheader("Nilai K")
clust = st.sidebar.slider('Jumlah K yang diinginkan:', 2, 14, 3, 1)

def k_means(n_clust):
    kmean = KMeans(n_clusters=n_clust).fit(RFMT_normalized)
    return kmean.labels_

if st.button('Run K-means'):
    clusters = k_means(clust)
    RFMT_with_clusters = pd.DataFrame(RFMT_normalized, columns=RFMT.columns)
    RFMT_with_clusters['Cluster'] = clusters
    st.write('Clusters:')
    st.write(RFMT_with_clusters)

    sil_coef = silhouette_score(RFMT_normalized, clusters)
    st.write('hasil pengujian dengan Silhouette Coefficient:', sil_coef)
