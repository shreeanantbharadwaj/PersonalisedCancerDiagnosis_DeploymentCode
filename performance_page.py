import streamlit as st
import pandas as pd
import pickle
import base64
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score


@st.cache_resource
def load_data():
    df = pickle.load(open("act_pred_df.pkl", "rb"))
    return df


def calc_metrics(input_data):
    """
    This method calculates various performance metrics.
    """
    y_actual = input_data.iloc[:, 0]
    y_predicted = input_data.iloc[:, 1]
    precision = precision_score(y_actual, y_predicted, average='weighted')
    recall = recall_score(y_actual, y_predicted, average='weighted')
    f1 = f1_score(y_actual, y_predicted, average='weighted')
    precision_series = pd.Series(precision, name='Precision')
    recall_series = pd.Series(recall, name='Recall')
    f1_series = pd.Series(f1, name='F1_Score')
    df = pd.concat([precision_series, recall_series, f1_series], axis=1)

    return df


def plot_confusion_matrix(test_y, predict_y):
    """
    This method plots the confusion matrices given actual & predicted values.
    """
    c = confusion_matrix(test_y, predict_y)
    a = (c.T / (c.sum(axis=1)).T)
    b = (c / c.sum(axis=0))
    labels = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    plt.figure(figsize=(20, 7))
    sns.heatmap(c, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    st.divider()

    st.caption("## Precision matrix (Column Sum=1) :")
    plt.figure(figsize=(20, 7))
    sns.heatmap(b, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    st.divider()

    st.caption("## Recall matrix (Row sum=1) :")
    plt.figure(figsize=(20, 7))
    sns.heatmap(a, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    st.divider()


def file_download(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="performance_metrics.csv">Download CSV File</a>'
    return href


def show_performance_page():

    # Sidebar panel - Performance metrics
    performance_metrics = ['Precision', 'Recall', 'F1_Score']
    selected_metrics = st.sidebar.multiselect('Performance metrics', performance_metrics, performance_metrics)

    st.text("")
    st.text("")

    # Main panel
    st.caption('## Logistic Regression with Tf-Idf Vectorizer :')
    image = Image.open('lr_logloss.png')
    st.image(image)
    st.divider()

    input_df = load_data()

    metrics_df = calc_metrics(input_df)
    selected_metrics_df = metrics_df[selected_metrics]
    st.caption('## Actual & Predicted values :')
    st.write(input_df)
    st.divider()

    st.caption('## Confusion matrix :')
    y_actual = input_df.iloc[:, 0]
    y_predicted = input_df.iloc[:, 1]
    confusion_matrix_df = plot_confusion_matrix(y_actual, y_predicted)
    st.write(confusion_matrix_df)

    st.caption('## Performance metrics :')
    st.write(selected_metrics_df)

    st.markdown(file_download(selected_metrics_df), unsafe_allow_html=True)
