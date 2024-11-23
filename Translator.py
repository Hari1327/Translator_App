import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import PyPDF2

# Load the SeamlessM4T model
model_name = "meta/seamlessm4t"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load additional pipelines
summarizer = pipeline("summarization")
sentiment_analyzer = pipeline("sentiment-analysis")

# App Header
st.title("🌍 Multilingual Translator App")
st.subheader("Powered by Meta's SeamlessM4T")

# Sidebar Features
st.sidebar.title("App Features")
enable_summarization = st.sidebar.checkbox("Enable Summarization")
enable_sentiment_analysis = st.sidebar.checkbox("Enable Sentiment Analysis")
file_upload_enabled = st.sidebar.checkbox("Enable File Uploads")

# Language Selection
target_language = st.selectbox(
    "Select Target Language", ["English", "French", "Spanish", "German"]
)
lang_map = {"English": "en", "French": "fr", "Spanish": "es", "German": "de"}

# Input Section
source_text = st.text_area("Enter Text to Translate", placeholder="Type here...")

# File Upload (Optional)
uploaded_file = None
if file_upload_enabled:
    uploaded_file = st.file_uploader("Upload a text or PDF file", type=["txt", "pdf"])

    if uploaded_file:
        if uploaded_file.name.endswith(".txt"):
            source_text = uploaded_file.read().decode("utf-8")
        elif uploaded_file.name.endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            source_text = " ".join([page.extract_text() for page in pdf_reader.pages])

# Translation Button
if st.button("Translate"):
    if source_text.strip():
        # Translate the text
        inputs = tokenizer(source_text, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[lang_map[target_language]]
        )
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.success("Translation:")
        st.write(translated_text)

        # Summarization
        if enable_summarization:
            summary = summarizer(translated_text, max_length=50, min_length=25, do_sample=False)
            st.info("Summary:")
            st.write(summary[0]['summary_text'])

        # Sentiment Analysis
        if enable_sentiment_analysis:
            sentiment = sentiment_analyzer(translated_text)
            st.info("Sentiment Analysis:")
            st.write(f"Sentiment: {sentiment[0]['label']}, Score: {sentiment[0]['score']:.2f}")

    else:
        st.warning("Please enter text or upload a file to translate.")

# Footer
st.caption("Built with ❤️ using Streamlit and SeamlessM4T.")
