import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# Web App Title
st.markdown('''
# **The Insighter**


---
''')

# Upload CSV data
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("""
[Example CSV input file](https://www.kaggle.com/datasets/devansodariya/student-performance-data)""")

# Pandas Profiling Report
if uploaded_file is not None:
    @st.cache_data(ttl=1)
    def load_csv():
        csv = pd.read_csv(uploaded_file)
        return csv
    df = load_csv()
    
    pr = ProfileReport(df, explorative=True)
    st.header('**Input DataFrame**')
    st.write(df)
    st.write('---')
    st.write('Checking for missing values..')
    st.write(df.isnull().values.any())
    if df.isnull().values.any():
        print("The dataset contains missing values. What would you like to do?")
        print("1. Delete rows with missing values")
        print("2. Fill missing values with zeroes")
        missing_choice = int(input("Enter your choice: "))

        if missing_choice == 1:
            df.dropna(inplace=True)
            print("Rows with missing values have been deleted.")
        elif missing_choice == 2:
            df.fillna(0, inplace=True)
            print("Missing values have been filled with zeroes.")
        else:
            print("Invalid choice. Please try again!")

    st.write(df.nunique())
    st.header('**Pandas Profiling Report**')
    st_profile_report(pr)

    # K-Means Clustering
    st.header('**K-Means Clustering**')
    n_clusters = st.slider('Select the number of clusters', 2, 10, 3)
    numerical_cols = df.select_dtypes(include=[np.number])
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(numerical_cols)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

# Display clustering results
    st.write('**Cluster Labels**')
    st.write(labels)
    st.write('**Cluster Centroids**')
    st.write(centroids)

# Visualize clustering results using a scatter plot
    import matplotlib.pyplot as plt
    plt.scatter(numerical_cols.iloc[:, 0], numerical_cols.iloc[:, 1], c=labels)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('K-Means Clustering')
    st.pyplot(plt)

else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        # Example data
        # Example data
        @st.cache_data(ttl=1)
        def load_data():
            df=pd.read_csv("student_data.csv")
            #data = np.random.rand(100, 5)
            #df = pd.DataFrame(data, columns=['a', 'b', 'c', 'd', 'e'])
            return df
            
        df = load_data()
        pr = ProfileReport(df, explorative=True)
        st.header('**Input DataFrame**')
        st.write(df)
        st.write('---')
        st.header('**Pandas Profiling Report**')
        st_profile_report(pr)

        # K-Means Clustering
        st.header('**K-Means Clustering**')
        n_clusters = st.slider('Select the number of clusters', 2, 10, 3)
        numerical_cols = df.select_dtypes(include=[np.number]).values  # Convert to NumPy array
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(numerical_cols)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        # Display clustering results
        st.write('**Cluster Labels**')
        st.write(labels)
        st.write('**Cluster Centroids**')
        st.write(centroids)

        # Visualize clustering results using a scatter plot
        import matplotlib.pyplot as plt
        plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=labels)
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('K-Means Clustering')
        st.pyplot(plt)
