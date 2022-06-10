import streamlit as st
import altair as alt
from PIL import Image
import pandas as pd
import numpy as np
from langdetect import detect
import matplotlib.pyplot as plt
import scipy as sp
import shap

# core Pytorch tensorflow ML Pkgs
import torch
from transformers import CamembertTokenizer # DistilBertTokenizerFast

# core neo4j DATABASE
from py2neo import Graph, Node, Relationship

# app title, header, logo
title = "Text Classication  📊"
lang =  "Konvo NLP Project  🇫🇷"
image = Image.open('./dsti-logo.jpg')

col1, col2, col3 = st.columns([0.2, 0.1, 0.2])

with col1:
    st.markdown("<h1 style='text-align: left;'>"+title+"</h1>",
                                        unsafe_allow_html=True)

with col2:
    col2.image(image, use_column_width=True) # width=150)

with col3:
    st.markdown("<h1 style='text-align: right;'>"+lang+"</h1>",
                                        unsafe_allow_html=True)
    #col3.image(impytorch, use_column_width=True)

# Globabe Variables
predictions_info = "👇🏼 \t Two Options To Submit A Text Into Prediction \t 👇🏼"
usage_options = """
    <div id="block1" style="float:left;">
        <h3>Option A</h3>
        <ul>
            <li>Select a text to predict</li>
            <li>Click on predict</li>
        </ul>
    </div>
    <div id="block2" style="float:right;">
        <h3>Option B</h3">
        <ul>
            <li>Type your own text</li>
            <li>Click on predict</li>
        </ul>
    </div>
"""

# Show Description
st.markdown("<h3 style='text-align: center;'>"+predictions_info+"</h3>",
                                            unsafe_allow_html=True)

st.markdown(usage_options, unsafe_allow_html=True)

# load trained models
# sentimental model
model_smt = torch.load("./sentimental_model_cmbert.pth",
                    map_location=torch.device('cpu'))

# emotional model
model_emt = torch.load("./emotional_model_cmbert.pth",
                    map_location=torch.device('cpu'))

# tokenier camemBERT --> french bert version
tokenizer = CamembertTokenizer.from_pretrained(
                        "camembert-base",
                        problem_type="multi_class_classification")

# plotting shap values with streamlit
import streamlit.components.v1 as components
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils import capture

def st_plot_text_shap(shap_val, height=None):
    InteractiveShell().instance()
    with capture.capture_output() as cap:
        shap.plots.text(shap_val)
    components.html(cap.outputs[1].data['text/html'],
                        height=height, scrolling=True)

# no warning
st.set_option('deprecation.showPyplotGlobalUse', False)

# 5 examples from dotissimo.fr
text_0 = "Je me sens bien maintenant. Je suis bien guéri et je retrouve le goût de la vie. Merci à toutes et à tous pour le soutien dans ce forum."
text_1 = "Je pense que mon diabète est très bien régulé."
text_2 = "Je me sens très mal entre chaque repas ..."
text_3 = "Je n'ai pas eu de vertige depuis longtemps."
text_4 = "J'aimerais revoir mon médecin pour adapter mon traitement."
text_5 = "J'ai eu un problème hier et je me suis fait mordre les seins on va dire. Mon sein a une croute et des bleus sur les mamelons. Ils me font super mal. Les tétons pointent alors qu'ils ne pointent jamais autant. Je ne sais pas dans quel catégorie je pouvais mettre ça. Et je ne sais surtout pas si il y a un risque de quelque chose en sachant que c'était un homme."
text_6 = """ Bonjour à toutes et tous!

Me voilà, enceinte (de 10 semaines), découvrant un diabète gestationnel, et probablement un diabète pré existant!

Du coup là je suis en période de « test », pour voir la suite des choses. Régime assez strict, et vérification de la glycémie juste avant repas puis 2h après.

Je vais voir une infirmière diabetologue puis un diabetologue d’ici peu. Mais je me permets de vous solliciter, vous qui vivez cette maladie au quotidien, car il y a des choses que je ne comprends pas :

- J’ai toujours une glycémie élevée le matin au réveil (autour de 1.30), alors que je suis à jeun depuis en moyenne une dizaine ou une douzaine d’heures… est-ce qu’il y a une explication à ça?

- Est-ce que vous avez, au fil du temps, constaté une amélioration de votre glycémie? Je sais que ma question est un peu bête, mais je me demande si une régulation est vraiment possible, et surtout au bout de combien de temps elle commence à apparaître.

Je me doute qu’il n’y a pas de règle universelle, mais avoir votre expérience m’eclairerait.

Merci beaucoup de m’avoir lue 😄."""
text_7 = "Voici un topic qui va déprimer ceux qui se battent avec leur HbA1C. Même si ce n'est pas le but, mais y'en a beaucoup qui en passant ici vont culpabiliser de leur 7,5 ou de leur 8. Passants, croisez votre chemin. Je plaisante (...mais pas tant que ça), 7.2 pour moi la dernière fois ! A+"

# warring when text is not in french
lang_error = "⛔️ Please provide a text in 🇫🇷 !\n Predictions may not be accurate!!"


# select box option
st.markdown("<h6 style='text-align: center;'>"+"Choose a text"+"</h6>",
                                                unsafe_allow_html=True)
option = st.selectbox('',('',
                          text_0,
                          text_1,
                          text_2,
                          text_3,
                          text_4,
                          text_5,
                          text_6,
                          text_7))

label_desc="""
                <div id="block2" style="float:right;">
                    <h5>Predict sentiment in a text as:</h5">
                    <ul>
                        <li>Highly negative,</li>
                        <li>Moderately negative,</li>
                        <li>Neutral,</li>
                        <li>Moderately positive,</li>
                        <li>Highly positive.</li>
                    </ul>
                </div>
            """
label_desc2="""
                <div id="block2" style="float:right;">
                    <h5>Predict emotion in text as:</h5">
                    <ul>
                        <li>SAD,</li>
                        <li>NEUTRAL,</li>
                        <li>FEAR,</li>
                        <li>HAPPY.</li>
                    </ul>
                </div>
            """


# Menu list
menu = ["Sentiment", "Emotion"]
code = 'fr' # language_code
choice = st.sidebar.selectbox('Menu', menu)

# define predictor function
def predictor(text):
    """
    predoctor function that make inference :
        params :
            intputs: list of str & model weights
            outpout: numpy array list of score
    """
    # text max_length
    max_len = 512
    text_list = [text]

    # tokenize the input text
    encodings = tokenizer(text_list,
                          max_length=max_len,
                          truncation=True,
                          padding=True, return_tensors="pt").to('cpu')

    # predict with one model
    if choice == "Sentiment":
        outputs = model_smt(**encodings)
    else:
        outputs = model_emt(**encodings)

    # transform to array with probabilities
    sm = torch.nn.Softmax(dim=1)
    pred = sm(outputs.logits)
    # convert to numpy array
    return pred.cpu().detach().numpy()

# define a predict function used by shap explainer
# @st.experimental_singleton
def output_explainer(x):
    tv = torch.tensor([tokenizer.encode(v, padding='max_length', max_length=512,
                                        truncation=True) for v in x])#.cuda()
    if choice == "Sentiment":
        outputs = model_smt(tv)[0]
    else:
        outputs = model_emt(tv)[0]
    sm = torch.nn.Softmax(dim=1)
    outputs = sm(outputs).detach().cpu().numpy()
    scores = (outputs.T / outputs.sum(-1)).T
    val = sp.special.logit(scores[:,1]) # use one vs rest logit units
    return val

# build an explainer using a token masker
explainer = shap.Explainer(output_explainer, tokenizer)

# choose sentimental model
if choice == "Sentiment":
    #st.subheader("Sentiment Model")
    with st.form(key='dlform'):
        col1, col2 = st.columns([2, 1])
        with col1:
            comment = st.text_area("The input text goes here:")
            submit_comment = st.form_submit_button(label='Predict')

            if submit_comment:
                if len(comment) != 0:
                    language_code = detect(comment)
                    if language_code != code:
                        st.markdown("<h5 style='color: red;'> "+lang_error+"</h5>",
                                                            unsafe_allow_html=True)
                        comment = ""
                        st.info("Text not submitted")
                else:
                    comment = option
                    st.info("Text submitted")

        with col2:
            app_dsc = "Sentiment Analysis:"
            st.markdown("<h3 style='text-align: left;'>"+app_dsc+"</h3>",
                                                        unsafe_allow_html=True)
            st.markdown(label_desc, unsafe_allow_html=True)

        if submit_comment and len(comment) != 0:
            # labels list
            col = ['HighlyNeg', 'ModNeg', 'Neutral', 'ModPos', 'HighlyPos']
            # get results of prediction
            result = predictor(comment)

            # convert to data frame
            predictions = pd.DataFrame(result,
                                       columns=col).rename(index={0: 'scores'})

            # get the label
            label = np.argmax(predictions)-2
            target = {-2: "Highly Negative 😔", -1: "Moderately Negative 😔",
                       0: "Neutral 😐", 1: "Moderately Positive  😂",
                       2: "Highly Positive 🤗"}


            # results
            col1, col2 = st.columns([2, 1])

            # left column
            with col1:
                st.info('Scores Visualiazation:')
                # bar plot
                st.bar_chart(predictions.T,
                            use_container_width=True)

                # plot the first sentence's explanation
                # explain the model's predictions on konvo comment
                shap_values = explainer([comment], fixed_context=1)
                st.header('One vs all strategy')
                color_exp = 'Positive impact in red vs Negative impact in blue'
                st.markdown("<h4 style='text-align: center;'>"+color_exp+"</h4>",
                                                          unsafe_allow_html=True)
                st.success("Sentence's shap explanation:")
                st_plot_text_shap(shap_values, 200)
                st.success('Token impact shap values:')
                st.pyplot(shap.plots.bar(shap_values[0]))

            # right column
            with col2:
                st.success("Sentiment predicted:")
                result_label = f"{target[label]} ({(np.argmax(result)-2)})"
                st.markdown("<h3 style='text-align:center;'>"+result_label+"</h3>",
                                                            unsafe_allow_html=True)

                # highling greatest score
                st.info("Prediction best score highlighted:")
                st.dataframe(predictions.T.style.highlight_max(axis=0))

                # details infos
                st.info("Original text:")
                st.write(comment)
                st.info("Token maximum shap values:")
                st.pyplot(shap.plots.bar(shap_values[0].abs))
                st.info('Shap values in summary')
                st.pyplot(shap.summary_plot(shap_values.values,
                                            shap_values[0].data))
                score = np.round(float(predictions.max(axis=1)), 6)
                # st.write(score)
        else:
            st.warning('Type your text or select one above please ?')

# choose motional model
else:
    #st.subheader("Emotion Analysis")
    with st.form(key='dlform'):
        col1, col2 = st.columns([2, 1])
        with col1:
            comment = st.text_area("The input text goes here:").strip()
            submit_comment = st.form_submit_button(label='Predict')

            if submit_comment:
                if len(comment) != 0:
                    language_code = detect(comment)
                    if language_code != code :
                        st.markdown("<h5 style='color:red;'> "+lang_error+"</h5>",
                                                            unsafe_allow_html=True)
                        comment = ""
                        st.info("Text submitted")
                else:
                    comment = option
                    st.info("Text submitted")

        with col2:
            app_dsc = "Emotion Analysis:"
            st.markdown("<h3 style='text-align: left;'>"+app_dsc+"</h3>",
                                                        unsafe_allow_html=True)
            st.markdown(label_desc2, unsafe_allow_html=True)

        if submit_comment and len(comment) != 0:
            # labels list
            col = ["SAD", "NEUTRAL", "FEAR", "HAPPY"]
            # get results of prediction
            result = predictor(comment)

            # convert to data frame
            predictions = pd.DataFrame(result,
                                       columns=col).rename(index={0: 'scores'})

            # get the label
            label = np.argmax(predictions)
            target = {0: "SAD 😔",  1: "NEUTRAL 😐",
                      2: "FEAR 😨", 3: "HAPPY  😂"}

            # results
            col1, col2 = st.columns([2, 1])

            # left column
            with col1:
                st.info('Scores Visualiazation:')
                # bar plot
                st.bar_chart(predictions.T,
                             use_container_width=True)

                # plot the first sentence's explanation
                # explain the model's prediction on konvo comment text
                shap_values = explainer([comment], fixed_context=1)
                st.header('One vs all strategy')
                color_exp = 'Positive impact in red vs Pegative impact in blue'
                st.markdown("<h4 style='text-align:center;'>"+color_exp+"</h4>",
                                                         unsafe_allow_html=True)
                st.success("Sentence's explanation")
                st_plot_text_shap(shap_values, 200)
                st.success('Token impact shap values')
                st.pyplot(shap.plots.bar(shap_values[0]))

            # right column
            with col2:
                st.success("Emotion predicted:")
                result_label = f"{target[label]}  ({(np.argmax(result))})"
                st.markdown("<h3 style='text-align:center;'>"+result_label+"</h3>",
                                                            unsafe_allow_html=True)

                # highling greatest score
                st.info("Prediction best score highlighted:")
                st.dataframe(predictions.T.style.highlight_max(axis=0))

                # details infos
                st.info("Original text:")
                st.write(comment)
                st.info("Token maximum shap values:")
                st.pyplot(shap.plots.bar(shap_values[0].abs))
                st.info('Shap values in summary')
                st.pyplot(shap.summary_plot(shap_values.values,
                                            shap_values[0].data))
                score = np.round(float(predictions.max(axis=1)), 6)
                #st.write(score)

        else:
            st.warning('Type your text or select one above please ?')

######## SAVE TEXT AND CLASSES INT NEO4J DATABASE #####################
# connect to graph with credentials
g = Graph("bolt://localhost:7687", auth=("neo4j", "Konvo2002"))

# Creating the CONSTRAINTS // Create unique property constraint
# cons_person = "CREATE CONSTRAINT ON (p:Person) ASSERT p.message IS UNIQUE;"
# g.run(cons_person)

# remove existing before importing
# rem = 'MATCH (n) DETACH DELETE n'
# g.run(rem)

# let's create 3 nodes as follows
#    p:Person
#    e:Emotion
#    s:Sentiment

# let' create 2 relationship types as follows
#    re:TEXT_EMOTION_IS
#    rs:TEXT_SENTIMENT_IS

if submit_comment and len(comment) != 0:
    if choice == "Sentiment":
        text = Node("Person", message=comment)
        text.__primarylabel__ = "Person"
        text.__primarykey__ = "message"

        sentiment = Node("Sentiment",
                         label=target[label],
                         pred=int(label),
                         score=float(score))

        sentiment.__primarylabel__ = "Sentiment"
        sentiment.__primarykey__ = "label"

        SENTI = Relationship.type("TEXT_SENTIMENT_IS")
        g.merge(SENTI(text, sentiment))
    else:
        text = Node("Person", message=comment)
        text.__primarylabel__ = "Person"
        text.__primarykey__ = "message"

        emotion = Node("Emotion",
                       label=target[label],
                       pred=int(label),
                       score=float(score))

        emotion.__primarylabel__ = "Emotion"
        emotion.__primarykey__ = "label"

        EMO = Relationship.type("TEXT_EMOTION_IS")
        g.merge(EMO(text, emotion))
