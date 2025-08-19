import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pdfplumber

# Load model once
@st.cache_resource
def load_model():
    model_name = "Hello-SimpleAI/ChatGPT-Detector-RoBERTa"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# PDF text extraction
def extract_text_by_page(pdf_file):
    pages = []
    with pdfplumber.open(pdf_file) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                pages.append((i + 1, text.strip()))
    return pages

# Detection for each chunk
def detect_chunk_ai(text, threshold=0.5):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        ai_prob = probs[0][1].item()
    
    print(f">>> [DEBUG] AI Prob = {ai_prob:.3f}, Threshold = {threshold}")
    label = "AI" if ai_prob >= threshold else "Human"
    return label, ai_prob

# Full document analysis
def analyze_pdf_advanced(pdf_file, threshold):
    pages = extract_text_by_page(pdf_file)
    all_data = []

    for page_num, full_text in pages:
        chunks = [full_text[i:i+512] for i in range(0, len(full_text), 512)]
        for idx, chunk in enumerate(chunks):
            label, prob = detect_chunk_ai(chunk, threshold)
            all_data.append({
                "Page": page_num,
                "Chunk_Index": idx + 1,
                "Chunk_Text": chunk,
                "AI_Probability": round(prob, 3),
                "Label": label
            })
        print(f"[DEBUG] Page {page_num} | Chunk {idx+1} | Text: {chunk[:100]}")

    return pd.DataFrame(all_data)

