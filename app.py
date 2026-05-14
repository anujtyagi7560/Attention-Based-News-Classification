import streamlit as st
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification

# Page configuration
st.set_page_config(page_title="AI News Classifier", page_icon="📰")

st.title("📰 News Category Classifier")
st.markdown("""
This app uses a **Fine-tuned BERT model** to classify news headlines into four categories: 
**World, Sports, Business, and Sci/Tech**.
""")

# 1. Load Model and Tokenizer
@st.cache_resource
def load_model():
    # Path to your organized BERT checkpoints
    model_path = "./checkpoints/bert" 
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

tokenizer, model = load_model()
# Class names from your config
class_names = ["World", "Sports", "Business", "Sci/Tech"]

# 2. User Interface
user_input = st.text_area("Enter News Headline:", placeholder="e.g., NASA's rover finds signs of ancient water on Mars...")

if st.button("Classify News"):
    if user_input.strip():
        # Preprocessing and Inference
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=128, padding="max_length")
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][prediction].item()

        # Display Results
        st.success(f"### Result: **{class_names[prediction]}**")
        st.info(f"**Confidence Score:** {confidence:.2%}")
    else:
        st.warning("Please enter a headline to analyze.")