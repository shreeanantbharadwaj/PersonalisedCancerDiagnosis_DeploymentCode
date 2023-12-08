from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pickle
import streamlit as st

classes = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9}
class_labels = list(classes.values())


def tables():
    """
    This function displays the sample data in a tabular format.
    """

    st.text("")
    st.caption("### A CLOSER LOOK INTO THE DATA")
    st.text("")
    st.markdown("* training_variants :")

    df_var = pickle.load(open("df_var_sample", "rb"))
    fig = go.Figure(data=go.Table(columnwidth=[1, 1, 1, 1],
                                  header=dict(values=list(df_var[["ID", "Gene", "Variation", "Class"]].columns),
                                              fill_color="#FFBEBE", align="left"),
                                  cells=dict(values=[df_var.ID, df_var.Gene, df_var.Variation, df_var.Class],
                                             fill_color="#E5ECF9", align="left")))
    fig.update_layout(margin=dict(l=5, r=5, b=10, t=10))
    st.write(fig)

    st.markdown("* training_text :")

    df_var = pickle.load(open("df_txt_sample", "rb"))
    fig = go.Figure(data=go.Table(columnwidth=[1, 3],
                                  header=dict(values=list(df_var[["ID", "TEXT"]].columns),
                                              fill_color="#FFBEBE", align="left"),
                                  cells=dict(values=[df_var.ID, df_var.TEXT],
                                             fill_color="#E5ECF9", align="left")))
    fig.update_layout(margin=dict(l=5, r=5, b=10, t=10))
    st.write(fig)


@st.cache_resource
def load_data():
    # df = pickle.load(open('df.zip', 'rb'))
    # df = pd.read_csv("df.zip")
    df = pd.read_pickle("df.zip")
    return df


@st.cache_resource
def load_labels():
    # df = pickle.load(open('df.zip', 'rb'))
    # df = pd.read_csv("df.zip")
    df = pd.read_pickle("df.zip")
    labels = df['Class'].values
    return labels


def plot_distributions():
    """
    This function plots the distribution of y_i's in train, cv & test data.
    """

    data = load_data()
    data.Gene = data.Gene.str.replace('\s+', '_')
    data.Variation = data.Variation.str.replace('\s+', '_')
    y_true = load_labels()
    x_train, test_df, y_train, y_test = train_test_split(data, y_true, stratify=y_true, test_size=0.2)
    train_df, cv_df, y_train, y_cv = train_test_split(x_train, y_train, stratify=y_train, test_size=0.2)

    train_class_distribution = train_df['Class'].value_counts().sort_index()
    test_class_distribution = test_df['Class'].value_counts().sort_index()
    cv_class_distribution = cv_df['Class'].value_counts().sort_index()

    st.text("")
    st.text("")

    sns.barplot(class_labels, train_class_distribution, palette="Set2")
    plt.title("Distribution of y_i's in Train Data")
    plt.xlabel('Class')
    plt.ylabel('Data points per Class')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    # -(train_class_distribution.values): the minus sign will give us in decreasing order
    sorted_yi = np.argsort(-train_class_distribution.values)
    for i in sorted_yi:
        st.markdown("Number of data points in **class {}** : {}  ( {} %)"
                    .format(i + 1, train_class_distribution.values[i],
                            np.round((train_class_distribution.values[i] / train_df.shape[0] * 100), 3)))

    st.divider()

    sns.barplot(class_labels, test_class_distribution, palette="Set2")
    plt.title("Distribution of y_i's in Test Data")
    plt.xlabel('Class')
    plt.ylabel('Data points per Class')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    # -(train_class_distribution.values): the minus sign will give us in decreasing order
    sorted_yi = np.argsort(-test_class_distribution.values)
    for i in sorted_yi:
        st.markdown("Number of data points in **class {}** : {}  ( {} %)"
                    .format(i + 1, test_class_distribution.values[i],
                            np.round((test_class_distribution.values[i] / test_df.shape[0] * 100), 3)))

    st.divider()

    sns.barplot(class_labels, cv_class_distribution, palette="Set2")
    plt.title("Distribution of y_i's in Cross Validation Data")
    plt.xlabel('Class')
    plt.ylabel('Data points per Class')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    # -(train_class_distribution.values): the minus sign will give us in decreasing order
    sorted_yi = np.argsort(-train_class_distribution.values)
    for i in sorted_yi:
        st.markdown("Number of data points in **class {}** : {}  ( {} %)"
                    .format(i + 1, cv_class_distribution.values[i],
                            np.round((cv_class_distribution.values[i] / cv_df.shape[0] * 100), 3)))

    st.divider()

    st.markdown("**Observation :** Classes 3, 8 & 9 are heavily imbalanced.")

    st.divider()


def uni_variate_analysis_gene():
    """
    This function performs uni-variate analysis on Gene feature.
    """

    data = load_data()
    data.Gene = data.Gene.str.replace('\s+', '_')
    data.Variation = data.Variation.str.replace('\s+', '_')
    y_true = load_labels()
    x_train, test_df, y_train, y_test = train_test_split(data, y_true, stratify=y_true, test_size=0.2)
    train_df, cv_df, y_train, y_cv = train_test_split(x_train, y_train, stratify=y_train, test_size=0.2)
    unique_genes = train_df['Gene'].value_counts()
    st.markdown("There are **{}** different categories of Genes in training data.".format(unique_genes.shape[0]))

    st.text("")
    st.text("")

    # PDF of Genes
    s = sum(unique_genes.values)
    h = unique_genes.values / s
    plt.plot(h, label="PDF of Genes")
    plt.xlabel('Index of a Gene')
    plt.ylabel('Number of Occurrences')
    plt.legend()
    plt.grid()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    st.text("")
    st.text("")

    # CDF of Genes
    c = np.cumsum(h)
    plt.plot(c, label='CDF of Genes')
    plt.grid()
    plt.legend()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    st.divider()


def uni_variate_analysis_variation():
    """
    This function performs uni-variate analysis on Variation feature.
    """

    data = load_data()
    data.Gene = data.Gene.str.replace('\s+', '_')
    data.Variation = data.Variation.str.replace('\s+', '_')
    y_true = load_labels()
    x_train, test_df, y_train, y_test = train_test_split(data, y_true, stratify=y_true, test_size=0.2)
    train_df, cv_df, y_train, y_cv = train_test_split(x_train, y_train, stratify=y_train, test_size=0.2)
    unique_variations = train_df['Variation'].value_counts()
    st.markdown("There are **{}** different categories of Variations in training data."
                .format(unique_variations.shape[0]))

    st.text("")
    st.text("")

    # PDF of Variations
    s = sum(unique_variations.values)
    h = unique_variations.values / s
    plt.plot(h, label="PDF of Variations")
    plt.xlabel('Index of a Variation')
    plt.ylabel('Number of Occurrences')
    plt.legend()
    plt.grid()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    st.text("")
    st.text("")

    # CDF of Genes
    c = np.cumsum(h)
    plt.plot(c, label='CDF of Variations')
    plt.grid()
    plt.legend()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
