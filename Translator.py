import streamlit as st
from transformers import MarianMTModel, MarianTokenizer, pipeline
from PyPDF2 import PdfReader

# Load translation model
@st.cache_resource
def load_translation_model():
    model_name = "Helsinki-NLP/opus-mt-en-fr"  # English to French
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

# Load sentiment analysis and summarization pipelines
@st.cache_resource
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis")

@st.cache_resource
def load_summarization_pipeline():
    return pipeline("summarization")

# Translation function
def translate_text(input_text, tokenizer, model):
    tokens = tokenizer.prepare_seq2seq_batch([input_text], return_tensors="pt")
    translated = model.generate(**tokens)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Streamlit app layout
st.title("🌍 Multilingual Translator App")
st.sidebar.title("Features")

# Load models
tokenizer, model = load_translation_model()
sentiment_pipeline = load_sentiment_pipeline()
summarization_pipeline = load_summarization_pipeline()

# Sidebar navigation
feature = st.sidebar.radio("Select a feature:", ["Text Translation", "Sentiment Analysis", "Text Summarization", "PDF Translation"])

if feature == "Text Translation":
    st.header("📜 Text Translation")
    input_text = st.text_area("Enter text to translate (English to French):")
    if st.button("Translate"):
        if input_text:
            translated_text = translate_text(input_text, tokenizer, model)
            st.success(f"Translated Text: {translated_text}")
        else:
            st.warning("Please enter text to translate.")

elif feature == "Sentiment Analysis":
    st.header("😊 Sentiment Analysis")
    input_text = st.text_area("Enter text to analyze sentiment:")
    if st.button("Analyze Sentiment"):
        if input_text:
            sentiment_result = sentiment_pipeline(input_text)
            st.json(sentiment_result)
        else:
            st.warning("Please enter text to analyze sentiment.")

elif feature == "Text Summarization":
    st.header("📝 Text Summarization")
    input_text = st.text_area("Enter text to summarize:")
    if st.button("Summarize"):
        if input_text:
            summary = summarization_pipeline(input_text, max_length=130, min_length=30, do_sample=False)
            st.success(f"Summary: {summary[0]['summary_text']}")
        else:
            st.warning("Please enter text to summarize.")

elif feature == "PDF Translation":
    st.header("📄 PDF Translation")
    uploaded_file = st.file_uploader("Upload a PDF file:", type=["pdf"])
    if uploaded_file:
        text = extract_text_from_pdf(uploaded_file)
        st.text_area("Extracted Text:", text, height=300)
        if st.button("Translate PDF to French"):
            if text:
                translated_text = translate_text(text, tokenizer, model)
                st.success(f"Translated PDF Text: {translated_text}")
            else:
                st.warning("No text found in the uploaded PDF.")
