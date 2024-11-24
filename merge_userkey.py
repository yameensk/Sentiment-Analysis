import streamlit as st
import requests
import numpy as np
from huggingface_hub import InferenceClient
from deep_translator import GoogleTranslator

# Function to get or set the API key
def get_api_key():
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ''
    return st.session_state.api_key

# Sidebar for API key input
st.sidebar.title("Hugging Face API Key")
api_key = st.sidebar.text_input("Enter your API key:", type="password", value=get_api_key(), key="api_key_input")
st.session_state.api_key = api_key

# Define the API URLs for BERT models
bert_models = {
    "Twitter RoBERTa": "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment",
    "heBERT": "https://api-inference.huggingface.co/models/avichr/heBERT_sentiment_analysis",
    "Twitter XLM RoBERTa": "https://api-inference.huggingface.co/models/cardiffnlp/twitter-xlm-roberta-base-sentiment"
}

# Initialize the Hugging Face Inference Client for Mistral LLM
@st.cache_resource
def get_mistral_client():
    return InferenceClient(
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        token=st.session_state.api_key
    )

def query_bert(payload, model_url):
    headers = {"Authorization": f"Bearer {st.session_state.api_key}"}
    response = requests.post(model_url, headers=headers, json=payload)
    return response.json()

def get_bert_sentiment(text, model_url):
    output = query_bert({"inputs": text}, model_url)
    if isinstance(output, list) and len(output) > 0 and isinstance(output[0], list):
        scores = output[0]
        result = {}
        for item in scores:
            if item['label'] in ['LABEL_0', 'negative']:
                result['negative'] = item['score']
            elif item['label'] in ['LABEL_1', 'neutral']:
                result['neutral'] = item['score']
            elif item['label'] in ['LABEL_2', 'positive']:
                result['positive'] = item['score']
        return result
    else:
        return {"error": "Unexpected output format"}

def get_mistral_sentiment(text):
    client = get_mistral_client()
    prompt = f"""Analyze the sentiment of the following text and provide confidence scores (accurate up to 4th decimal place) for positive, negative, and neutral sentiments. Also, identify phrases in the text that correspond to each sentiment.

Text: {text}

Output the result in the following format:
Positive: [confidence score]
Negative: [confidence score]
Neutral: [confidence score]
Positive phrases: [list of positive phrases]
Negative phrases: [list of negative phrases]
Neutral phrases: [list of neutral phrases]
"""
    response = client.text_generation(prompt, max_new_tokens=500, temperature=0.2)
    return response

def parse_mistral_response(response):
    lines = response.split('\n')
    scores = {}
    phrases = {'Positive': [], 'Negative': [], 'Neutral': []}
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            if key in ['Positive', 'Negative', 'Neutral'] and not phrases[key]:
                scores[key.lower()] = float(value)
            elif key.endswith('phrases'):
                sentiment = key.split()[0]
                phrases[sentiment] = [phrase.strip() for phrase in value.strip('[]').split(',')]
    return scores, phrases

def translate_to_hebrew(text):
    return GoogleTranslator(source='en', target='hebrew').translate(text)

st.title("Sentiment Analysis App")

# Sidebar for model selection
st.sidebar.title("Select Model")
model_options = list(bert_models.keys()) + ["Combined BERT", "Mistral LLM"]
selected_model = st.sidebar.radio("Choose a model:", model_options)

# Text input
user_input = st.text_area("Enter text for sentiment analysis:", "")

if st.button("Analyze Sentiment"):
    if not st.session_state.api_key:
        st.error("Please enter your Hugging Face API key in the sidebar.")
    elif not user_input:
        st.warning("Please enter some text for analysis.")
    else:
        if selected_model == "Mistral LLM":
            with st.spinner("Analyzing sentiment with Mistral LLM..."):
                response = get_mistral_sentiment(user_input)
                scores, phrases = parse_mistral_response(response)
                st.subheader("Sentiment Scores:")
                for sentiment, score in scores.items():
                    st.write(f"{sentiment.capitalize()}: {score:.4f}")
                st.subheader("Sentiment Phrases:")
                for sentiment, phrase_list in phrases.items():
                    st.write(f"{sentiment} phrases: {', '.join(phrase_list)}")
        elif selected_model == "Combined BERT":
            results = {}
            for model_name, model_url in bert_models.items():
                text_to_analyze = translate_to_hebrew(user_input) if model_name == "heBERT" else user_input
                sentiment_scores = get_bert_sentiment(text_to_analyze, model_url)
                results[model_name] = sentiment_scores
                
                st.write(f"{model_name}:")
                st.write(f"  Positive: {sentiment_scores.get('positive', 'N/A') if isinstance(sentiment_scores.get('positive', 'N/A'), str) else sentiment_scores.get('positive'):.4f}")
                st.write(f"  Neutral: {sentiment_scores.get('neutral', 'N/A') if isinstance(sentiment_scores.get('neutral', 'N/A'), str) else sentiment_scores.get('neutral'):.4f}")
                st.write(f"  Negative: {sentiment_scores.get('negative', 'N/A') if isinstance(sentiment_scores.get('negative', 'N/A'), str) else sentiment_scores.get('negative'):.4f}")
                st.write("")

            # Calculate average scores
            avg_positive = np.mean([r.get('positive', 0) for r in results.values()])
            avg_neutral = np.mean([r.get('neutral', 0) for r in results.values()])
            avg_negative = np.mean([r.get('negative', 0) for r in results.values()])

            st.write("Combined Average Scores:")
            st.write(f"  Positive: {avg_positive:.4f}")
            st.write(f"  Neutral: {avg_neutral:.4f}")
            st.write(f"  Negative: {avg_negative:.4f}")
        else:
            text_to_analyze = translate_to_hebrew(user_input) if selected_model == "heBERT" else user_input
            sentiment_scores = get_bert_sentiment(text_to_analyze, bert_models[selected_model])
            st.write("Sentiment Scores:")
            st.write(f"  Positive: {sentiment_scores.get('positive', 'N/A') if isinstance(sentiment_scores.get('positive', 'N/A'), str) else sentiment_scores.get('positive'):.4f}")
            st.write(f"  Neutral: {sentiment_scores.get('neutral', 'N/A') if isinstance(sentiment_scores.get('neutral', 'N/A'), str) else sentiment_scores.get('neutral'):.4f}")
            st.write(f"  Negative: {sentiment_scores.get('negative', 'N/A') if isinstance(sentiment_scores.get('negative', 'N/A'), str) else sentiment_scores.get('negative'):.4f}")

st.sidebar.info("Your API key is used for all models and is not stored permanently.")
