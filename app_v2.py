import streamlit as st
import pandas as pd
import numpy as np

# Core Pytorch tensorflow ML Pkgs
import torch
from transformers import DistilBertTokenizerFast


# define predictor function # @st.cache(suppress_st_warning=True)
# @st.experimental_singleton(suppress_st_warning=True)
@st.experimental_memo(suppress_st_warning=False)
def predictor(text_list):
    """
    predoctor function that make inference :
        params :
            intputs: list of str
            outpout: numpy array list of score
    """

    # torch model with 100 epochs on cpu device
    model = torch.load("./model_100_pytorch",
                       map_location=torch.device('cpu'))

    # tokenier DISTILBERT
    tokenizer = DistilBertTokenizerFast.from_pretrained(
                    "distilbert-base-uncased",
                    problem_type="multi_label_classification")

    # max max_length
    max_len = 512

    # tokenize the text
    encodings = tokenizer(text_list,
                          max_length=max_len,
                          truncation=True,
                          padding=True, return_tensors="pt").to('cpu')

    # predict
    outputs = model(**encodings)

    # transform to array with probabilities
    sm = torch.nn.Softmax(dim=0)
    pred = sm(outputs.logits)
    # convert to numpy array
    # res = tf.nn.softmax(pred.cpu().detach().numpy(), axis=1).cpu().numpy()
    return pred.cpu().detach().numpy()


st.title("INFERENCE")
with st.form(key='dlform'):
    col1, col2 = st.columns([2, 1])
    with col1:
        comment = st.text_area("Text goes here:")
        submit_comment = st.form_submit_button(label='Predict')
        if submit_comment:
            st.info("Text submitted")
    with col2:
        st.markdown("#### Sentiment Analysis:")
        st.markdown("""_**Predict comment sentiment as**:
                    Highly negative,
                    Moderately negative,
                    Neutral,
                    Moderately positive,
                    Highly positive_
                            """)

if submit_comment and len(comment) != 0:
    # labels list
    col = ['HighlyNeg', 'ModNeg', 'Neutral', 'ModPos', 'HighlyPos']
    # get results of prediction
    result = predictor(list(comment))

    # convert to data frame
    predictions = pd.DataFrame(result, columns=col)
    max_val = predictions.apply(lambda x: x.max())

    # get the label
    label = np.argmax(np.amax(result, axis=0), axis=0)-2
    st.success("Sentiment Predicted:")
    target = {-2: "Highly negative", -1: "Moderately negative",
              0: "Neutral", 1: "Moderately positive", 2: "Highly positive"}

    st.write("Label predicted :",
             (np.argmax(max_val)-2),
             ' which means ',
             target[label])

    st.markdown(f"**{target[label]} : {(np.argmax(max_val)-2)}**",
                unsafe_allow_html=False)

    # results
    col1, col2 = st.columns([2, 1])
    with col2:
        st.info("Predictions best scores :")
        st.dataframe(max_val)
        st.info('Info memory usage')
        st.write(predictions.memory_usage(deep=True))
    with col1:
        st.info('Plotting results')
        st.bar_chart(predictions.max(axis=0),
                     use_container_width=True)
        st.info("Description statistics:")
        st.write(predictions.describe())

    # details infos
    st.info("Original text:")
    st.write(comment)
    st.info("Details about probabilities:")
    st.line_chart(predictions)
    st.info("Maximum predicted probabilities highlighted :")
    st.dataframe(predictions.style.highlight_max(axis=0))
else:
    st.warning('Add your text in text area please ?')
