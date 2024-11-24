import streamlit as st
from huggingface_hub import InferenceClient

# Initialize the Hugging Face Inference Client
client = InferenceClient(
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    token="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
)

def get_sentiment(text):
    prompt = f"""Analyze the sentiment of the following text and provide confidence scores (accurate upto 4th decimal place) for positive, negative, and neutral sentiments. Also, identify phrases in the text that correspond to each sentiment.

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

def parse_sentiment_response(response):
    lines = response.split('\n')
    scores = {}
    phrases = {'Positive': [], 'Negative': [], 'Neutral': []}
    
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            if key in ['Positive', 'Negative', 'Neutral'] and not phrases[key]:
                scores[key] = float(value)
            elif key.endswith('phrases'):
                sentiment = key.split()[0]
                phrases[sentiment] = [phrase.strip() for phrase in value.strip('[]').split(',')]
    
    return scores, phrases

st.title("Sentiment Analysis with Mistral LLM")

user_input = st.text_area("Enter your text for sentiment analysis:", height=150)

if st.button("Analyze Sentiment"):
    if user_input:
        with st.spinner("Analyzing sentiment..."):
            response = get_sentiment(user_input)
            scores, phrases = parse_sentiment_response(response)
            
            st.subheader("Sentiment Scores:")
            for sentiment, score in scores.items():
                st.write(f"{sentiment}: {score:.2f}")
            
            st.subheader("Input Text:")
            st.write(user_input)
            
            st.subheader("Sentiment Phrases:")
            for sentiment, phrase_list in phrases.items():
                st.write(f"{sentiment} phrases: {', '.join(phrase_list)}")
    else:
        st.warning("Please enter some text to analyze.")
