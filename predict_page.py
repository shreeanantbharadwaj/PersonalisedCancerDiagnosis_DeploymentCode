import re
from sklearn.preprocessing import normalize
from scipy.sparse import hstack
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st

classes = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9}
class_labels = list(classes.values())


def nlp_preprocessing_gene(gene):
    """
    This function replaces spaces with underscores in gene.
    """

    gene = " ".join([word for word in gene.split()]).replace(" ", "_")

    return gene


def nlp_preprocessing_variation(variation):
    """
    This function replaces spaces with underscores in variation.
    """

    variation = " ".join([word for word in variation.split()]).replace(" ", "_")

    return variation


@st.cache_resource
def nlp_preprocessing_text(total_text):
    """
    This function cleans the Clinical Literature text.
    """

    stop_words = pickle.load(open('stop_words.pkl', 'rb'))

    if type(total_text) is not int:
        string = ""

        # replace every special char with space
        total_text = re.sub('[^a-zA-Z0-9\n]', ' ', total_text)

        # replace multiple spaces with single space
        total_text = re.sub('\s+', ' ', total_text)

        # converting all the chars into lower-case.
        total_text = total_text.lower()

        q = "".join([string + word + " " for word in total_text.split() if word not in stop_words])

    return q


@st.cache_resource
def one_hot_encoding_gene(gene):
    """
    This function return One Hot Encoded features of gene.
    """

    gene = nlp_preprocessing_gene(gene)
    gene_ohe = pickle.load(open('gene_ohe.pkl', 'rb'))
    gene_query = gene_ohe.transform([gene])

    return gene_query


@st.cache_resource
def one_hot_encoding_variation(variation):
    """
    This function return One Hot Encoded features of variation.
    """

    variation = nlp_preprocessing_variation(variation)
    variation_ohe = pickle.load(open('variation_ohe.pkl', 'rb'))
    variation_query = variation_ohe.transform([variation])

    return variation_query


@st.cache_resource
def one_hot_encoding_text(text):
    """
    This function return One Hot Encoded features of text.
    """

    text = nlp_preprocessing_text(text)
    text_ohe = pickle.load(open('text_ohe.pkl', 'rb'))
    text_query = text_ohe.transform([text])
    text_query = normalize(text_query, axis=0)

    return text_query


def query_point_creator(gene, variation, text):
    """
    This function creates the final query data point.
    """

    # One Hot Encoding of features
    gene = one_hot_encoding_gene(gene)
    variation = one_hot_encoding_variation(variation)
    text = one_hot_encoding_text(text)

    # Stack gene, variation and text features of query point.
    gene_var = hstack((gene, variation))
    test_one_hot_encoding = hstack((gene_var, text)).tocsr()

    return test_one_hot_encoding


@st.cache_resource
def predict(_arr):
    """
    This function returns the predicted class and probabilities of the datapoint belonging to each class.
    """

    # Load the model
    model = pickle.load(open('lr_model.pkl', 'rb'))

    # Return predicted class as well as class probabilities
    predicted_cls = model.predict(_arr)[0]
    probabilities = model.predict_proba(_arr)[0]

    return predicted_cls, probabilities


@st.cache_resource
def predict_class(gene, variation, text, no_features):
    """
    This function prints the predicted class and displays probability of each class.
    """

    data = query_point_creator(gene, variation, text)
    result, probs = predict(data)

    clf = pickle.load(open('clf.pkl', 'rb'))
    indices = np.argsort(-1 * abs(clf.coef_))[:, :no_features]

    st.markdown("This Genetic Mutation belongs to **class {}.**".format(result))
    get_impfeature_names(indices[0], text, gene, variation, no_features)

    st.text("")
    st.text("")
    st.text("")
    st.text("")

    probs = [np.round(x, 6) for x in probs]
    ax = sns.barplot(probs, class_labels, palette="winter", orient='h')
    ax.set_yticklabels(class_labels, rotation=0)

    plt.title("Probabilities of the Data belonging to each class")

    for index, value in enumerate(probs):
        plt.text(value, index, str(value))

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()


@st.cache_resource
def get_impfeature_names(indices, text, gene, var, no_features):
    """
    This function returns feature importance.
    """

    gene_vec = pickle.load(open('gene_vec.pkl', 'rb'))
    var_vec = pickle.load(open('var_vec.pkl', 'rb'))
    text_vec = pickle.load(open('text_vec.pkl', 'rb'))

    fea1_len = len(gene_vec.get_feature_names_out())
    fea2_len = len(var_vec.get_feature_names_out())
    word_present = 0

    for i, v in enumerate(indices):
        if v < fea1_len:
            word = gene_vec.get_feature_names_out()[v]
            yes_no = True if word == gene else False
            if yes_no:
                word_present += 1
                print(i, "Gene feature [{}] present in test data point [{}]".format(word, yes_no))
        elif v < fea1_len + fea2_len:
            word = var_vec.get_feature_names_out()[v - fea1_len]
            yes_no = True if word == var else False
            if yes_no:
                word_present += 1
                print(i, "variation feature [{}] present in test data point [{}]".format(word, yes_no))
        else:
            word = text_vec.get_feature_names_out()[v - (fea1_len + fea2_len)]
            yes_no = True if word in text.split() else False
            if yes_no:
                word_present += 1
                print(i, "Text feature [{}] present in test data point [{}]".format(word, yes_no))

    # return "Out of the top {} features, {} are present in the query point.".format(no_features, word_present)
    return st.markdown("Out of the top **{}** features, **{}** are present in the query point."
                       .format(no_features, word_present))
