import streamlit as st
from PIL import Image


def show_home_page():
    """
    This function displays the Home Page.
    """

    st.header("Personalised Cancer Diagnosis")
    st.text("")
    st.text("")
    img = Image.open("cancer_home.png")
    st.image(img, caption="Credits: time.com")
    st.text("")

    st.caption("**OVERVIEW :**")
    st.write("""The future of cancer diagnosis will rely on machine learning, because biology is too complex for
                humans to understand. Our genes are made of DNA which is present in nearly every cell of our bodies.
                Each gene is usually present in two copies, one from each parent. The DNA provides the instructions 
                to make different proteins which are responsible for the various functions in the body. For example, 
                proteins are important for hair and eye color, growth, brain development, movement and metabolism - 
                the conversion of food into energy or body fat and muscle.
              """)
    st.text("")
    st.write("""The process of building proteins begins with DNA. Cells first read the instructions contained in DNA
                which are written using the genetic alphabet of A, C, G and T, to make a copy called RNA, which is
                written using a similar genetic alphabet. This process is called transcription. Then, cells read the
                instructions contained in the RNA to make proteins, using a process called translation.
                You can think of RNA as the molecule that carries the message between the DNA and the protein.
             """)
    st.text("")
    st.write("""Many rare diseases are caused by changes to the letters in the DNA, called genetic mutations or
                variations. Mutations in the DNA get transferred to the RNA and this can lead to a protein that
                doesn’t work properly, or can even prevent a protein from being made at all. Once
                sequenced, a cancer tumor can have thousands of genetic mutations. But the challenge is
                distinguishing the mutations that contribute to tumor growth (drivers) from the neutral mutations
                (passengers). Currently this interpretation of genetic mutations is being done manually. This is a
                very time-consuming task where a Geneticist has to manually review and classify every single genetic
                mutation based on evidence from text-based clinical literature.
             """)
    st.divider()

    st.caption("**OBJECTIVE :**")
    st.markdown("""For every gene, there is a variation associated with it along with an evidence from a 
                text-based Clinical Literature. **Now with the help of a given Genetic Variation/Mutation and its 
                Clinical Literature, we have to classify which class it belongs to**. Only some classes belong to 
                cancer. Since it's a medical use case, we need to predict the probability of a data-point
                belonging to each of the 9 classes just to be really sure.
                """)
    st.divider()

    st.caption("**BUSINESS PROBLEM :**")
    st.markdown("""Modeling the clinical evidence is a non trivial task because analyzing clinical evidence
                is very challenging even for a Geneticist. A Geneticist picks a set of genetic variations of
                interest that need to be analyzed. After finalizing the variations, he/she studies all kinds of
                medical literature which are associated with it and searches for evidence that justifies the
                variation being cancerous. In the final stage, the Geneticist spends a tremendous amount of time
                analyzing the evidence related to each of the variations to classify them. **We need to replace the
                final stage with a machine learning model so that it will reduce the workload of a Geneticist.**
              """)
    st.divider()

    st.caption("**BUSINESS CONSTRAINTS :**")
    st.markdown("""* **Interpretability of the model** is consequential because a Geneticist must be convinced as to
                   why the model has given a particular class to a genetic variation so that he can further explain
                   it to the patient.
                """)
    st.markdown("""* There are **no low-latency requirements** but at the same time, we also don't want our
                   latency to be in several minutes.
                """)
    st.markdown("* Since it's a matter of life and death, **errors can prove extremely fatal.**")
    st.markdown("""* The model should return a **probability of the data point belonging to a class** rather than
                   simply returning a class.
                """)
    st.divider()

    st.caption("**MAPPING THE PROBLEM TO A MACHINE LEARNING PROBLEM :**")
    st.markdown("""* As we already mentioned, we need to classify genetic variations. Therefore, it is a
                   classification problem. Since there are 9 classes, it is a **Multi-class classification problem.**
                """)
    st.markdown("""* **Performance metric :** Multi class Log-Loss, Confusion matrix (We have chosen Log-Loss
                   because it uses the actual probability values which is our business constraint.)
                """)
    st.divider()

    st.caption("**DATA DESCRIPTION :**")
    st.write("Dataset : [link](https://www.kaggle.com/competitions/msk-redefining-cancer-treatment/data)")
    st.markdown("""We have 2 data files **"training_text"** and
                **"training_variants"**.
                """)
    st.markdown("Data file's information :")
    st.markdown("""* **training_variants :** A comma separated file containing the description of the genetic
                mutations used for training. Fields are 'ID' (the id of the row used to link the mutation to the
                clinical evidence), 'Gene' (the gene where this genetic mutation is located), 'Variation' (the 
                aminoacid change for this mutations), 'Class' (one of the 9 classes this genetic mutation has been
                classified on)
                """)
    st.markdown("""* **training_text :** A double pipe (||) delimited file that contains the clinical evidence
                (text) used to classify genetic mutations. Fields are 'ID' (the id of the row used to link the
                 clinical evidence to the genetic mutation), 'TEXT' (the clinical evidence used to classify the
                genetic mutation)
                """)
    st.write("""Both these data files are linked via an 'ID' field. Therefore, the genetic mutation (row) with ID=15
                in the file "training_variants", is classified using the clinical evidence (text) from the row with
                ID=15 in the file "training_text".
             """)
    st.divider()

    st.caption("""**LIBRARIES : `base64`, `matplotlib`, `numpy`, `pandas`, `pickle`, `PIL`,
                `re`, `scikit-learn`, `scipy`, `seaborn`, `streamlit`**""")
