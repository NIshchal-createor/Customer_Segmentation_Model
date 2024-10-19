from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use the non-GUI backend
import matplotlib.pyplot as plt
import json

app = Flask(__name__)
model = pickle.load(open('kmeans_model.pkl', 'rb'))

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path, sep=",", usecols=lambda column: column != 'index')
    df = df.dropna()
    df['CustomerID'] = df['CustomerID'].astype(str)
    df['Amount'] = df['Quantity'] * df['UnitPrice']
    dfm_m = df.groupby('CustomerID')['Amount'].sum().reset_index()
    dfm_f = df.groupby('CustomerID')['InvoiceNo'].count().reset_index()
    dfm_f.columns = ['CustomerID', 'Frequency']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    max_date = max(df['InvoiceDate'])
    df['Diff'] = max_date - df['InvoiceDate']
    dfm_p = df.groupby('CustomerID')['Diff'].min().reset_index()
    dfm_p['Diff'] = dfm_p['Diff'].dt.days
    dfm = pd.merge(dfm_m, dfm_f, on= "CustomerID", how="inner")
    dfm = pd.merge(dfm, dfm_p, on = "CustomerID", how='inner')
    dfm.columns = ['CustomerID', 'Amount', 'Frequency', 'Recency']

    Q1 = dfm.Amount.quantile(0.05)
    Q3 = dfm.Amount.quantile(0.95)
    IQR = Q3-Q1
    dfm = dfm[(dfm.Amount >= Q1 - 1.5 *IQR) & (dfm.Amount <= Q3 + 1.5*IQR)]

    Q1 = dfm.Recency.quantile(0.05)
    Q3 = dfm.Recency.quantile(0.95)
    IQR = Q3 - Q1
    dfm = dfm[(dfm.Recency >= Q1-1.5 * IQR) & (dfm.Recency <= Q3 + 1.5*IQR)]

    Q1 = dfm.Frequency.quantile(0.05)
    Q3 = dfm.Frequency.quantile(0.95)
    IQR = Q3 - Q1
    dfm = dfm[(dfm.Frequency >= Q1-1.5 * IQR) & (dfm.Frequency <= Q3 + 1.5*IQR)]
    
    return dfm

def preprocess_data(file_path):
    dfm = load_and_clean_data(file_path)
    dfm_df = dfm[['Amount', 'Frequency', 'Recency']]
    scaler = StandardScaler()
    dfm_df_scaled = scaler.fit_transform(dfm_df)
    dfm_df_scaled = pd.DataFrame(dfm_df_scaled)
    dfm_df_scaled.columns = ['Amount', 'Frequency', 'Recency']

    return dfm, dfm_df_scaled

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    file = request.files['file']
    file_path = os.path.join(os.getcwd(), file.filename)
    file.save(file_path)
    df = preprocess_data(file_path)[1]
    results_df = model.predict(df)
    df_with_id = preprocess_data(file_path)[0]
    df_with_id['Cluster_Id'] = results_df

    sns.stripplot(x="Cluster_Id", y="Amount", data=df_with_id, hue="Cluster_Id")
    amount_img_path = 'static/ClusterId_Amount.png'
    plt.savefig(amount_img_path)
    plt.clf()

    sns.stripplot(x="Cluster_Id", y="Frequency", data=df_with_id, hue="Cluster_Id")
    freq_img_path = 'static/ClusterId_Frequency.png'
    plt.savefig(freq_img_path)
    plt.clf()

    sns.stripplot(x="Cluster_Id", y="Recency", data=df_with_id, hue="Cluster_Id")
    recency_img_path = 'static/ClusterId_Recency.png'
    plt.savefig(recency_img_path)
    plt.clf()

    response = {'amount_img': amount_img_path, 'freq_img': freq_img_path, 'recency_img': recency_img_path}
    return json.dumps(response)

if __name__ == "__main__":
    app.run(debug=True)
