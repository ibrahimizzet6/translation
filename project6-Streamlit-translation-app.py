import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
import torch

st.set_page_config(page_title="Italian → English Translator",page_icon="🌍")

st.title("🇮🇹 ➡️ 🇬🇧 Italian to English Translator")
st.write("Enter an Italian sentence and click **Translate**.")

# Modeli cache'le (her butonda tekrar yüklenmesin)
@st.cache_resource
def load_model():
    model_name = "Helsinki-NLP/opus-mt-it-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Input alanı
text = st.text_area("Italian text:",height=150,placeholder="Ciao, come stai oggi?")

# Translate butonu
if st.button("Translate"):
    if text.strip() == "":
        st.warning("Please enter some Italian text.")
    else:
        with st.spinner("Translating..."):
            inputs = tokenizer(text,return_tensors="pt",padding=True,truncation=True)
            translated = model.generate( **inputs,max_length=128,num_beams=4)
            output = tokenizer.decode(translated[0],skip_special_tokens=True)

            st.success(output)
