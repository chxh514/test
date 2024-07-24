# Core Pkgs
import streamlit as st
st.set_page_config(page_title="NLP Web App", page_icon="👍", layout="centered", initial_sidebar_state="auto")

# Viz Pkgs
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from wordcloud import WordCloud

# NLP Pkgs
from textblob import TextBlob
import neattext as nt
import spacy

from collections import Counter
import re
  
def main():
   """NLP web app with Streamlit"""

title_template = """
  <div style="background-color:blue; padding:8px;">
  <h1 style="color:cyan">NLP Web App</h1>
  </div>
  """

st.markdown(title_template, unsafe_allow_html=True)

subheader_template = """
  <div style="background-color:cyan; padding:8px;">
  <h3 style="color:blue">Powered by Streamlit</h1>
  </div>
  """

st.markdown(subheader_template, unsafe_allow_html=True)

st.sidebar.image("nlp.jpg", use_column_width=True)

activity = ["Text Analysis", "Translation", "Sentiment Analysis", "About"]
choice = st.sidebar.selectbox("Menu", activity)

if choice == "Text Analysis":
    st.subheader("Text Analysis")
    st.write("")

    raw_text = st.text_area("Write something", "Enter a text in English...", height=350)

    if st.button("Analyze"):
      if len(raw_text) == 0:
        st.warning("Enter a text...")
      else:
        #blob = TextBlob(raw_text)
        st.info("Basic Functions")

        col1, col2 = st.columns(2)

        with col1:
          with st.expander("Basic Info"):
            st.info("Text Stats")
            word_desc = nt.TextFrame(raw_text).word_stats()
            result_desc = {"Length of Text":word_desc['Length of Text'],
                          "Num of Vowels":word_desc['Num of Vowels'],
                          "Num of Consonants":word_desc['Num of Consonants'],
                          "Num of Stopwords":word_desc['Num of Stopwords']}
            st.write(result_desc)

          with st.expander("Stopwords"):
            st.success("Stop Words List")
            stop_w = nt.TextExtractor(raw_text).extract_stopwords()
            st.error(stop_w)

        with col2:
          with st.expander("Processed Text"):
            st.success("Stopwords Excluded Text")
            processed_text = str(nt.TextFrame(raw_text).remove_stopwords())
            st.write(processed_text)

          with st.expander("Plot Wordcloud"):
            st.success("Wordcloud")
            wordcloud = WordCloud().generate(processed_text)
            fig = plt.figure(1, figsize=(20,10))
            plt.imshow(wordcloud, interpolation = 'bilinear')
            plt.axis('off')
            st.pyplot(fig)

        st.write("")
        st.write("")
        st.info("Advanced Features")
            
        col3, col4 = st.columns(2)

        with col3:
          with st.expander("Tokens&Lemmas"):
            st.write("T&K")
            processed_text_mid = str(nt.TextFrame(raw_text).remove_stopwords())
            processed_text_mid = str(nt.TextFrame(processed_text_mid).remove_puncts())
            processed_text_fin = str(nt.TextFrame(processed_text_mid).remove_special_characters())
            tandl = text_analyzer(processed_text_fin)
            st.json(tandl)

        with col4:
          with st.expander("Summarize"):
            st.success("Summarize")
            summary = summarize_text(raw_text)
            st.success(summary)
if choice == "Translation":
    st.subheader("Translation")
    st.write("")
    st.write("")
    raw_text = st.text_area("Original Text", "Write something to be translated...", height=200)
    if len(raw_text) < 3:
      st.warning("Please provide a text with at least 3 characters...")
    else:
      target_lang = st.selectbox("Target Language", ["German", "Spanish", "French", "Italian"])
      if target_lang == "German":
        target_lang = "de"
      elif target_lang == "Spanish":
        target_lang = "es"
      elif target_lang == "French":
        target_lang = "fr"      
      else:
        target_lang = "it" 
      
      if st.button("Translate"):
        translator = GoogleTranslator(source='auto', target=target_lang)  # set source and target languages
        translated_text = translator.translate(raw_text)
        st.write(translated_text)
      

if choice == "Sentiment Analysis":
    st.subheader("Sentiment Analysis")
    st.write("")
    st.write("")
    raw_text = st.text_area("Text to analyse", "Enter a text here....", height=200)
    if st.button("Evaluate"):
      if len(raw_text) == 0:
        st.warning("Enter a text...")
      else:
        blob = TextBlob(raw_text)
        st.info("Sentiment Analysis")
        st.write(blob.sentiment)
        st.write("")


if choice == "About":
    st.subheader("About")
    st.write("")
    st.markdown("""
    ### NLP Web App made with Streamlit
    
    for info:
    - [streamlit](https://streamlit.io)
    """)


if __name__ == "__main__":
  main()
