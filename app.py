%%writefile app.py
import streamlit as st
from groq import Groq

# 1. Setting up clean enterprise layout boundaries
st.set_page_config(page_title="AI Chat Assistant", page_icon="🤖", layout="centered")
st.title("🤖 AI Chat Assistant")
st.write("Welcome! This assistant is powered by the latest active architecture to process your requests.")

# 2. Secure connection setup (Insert your production Groq Key inside the quotes)
GROQ_API_KEY = "GROQ_API_KEY = YAHAN_APNI_GROQ_KEY_PASTE_KAREIN"
client = Groq(api_key=GROQ_API_KEY)

# 3. Generating a clean User Interface input text container
user_message = st.text_input("Enter your query below:", placeholder="Type your prompt here...")

# 4. Processing the observation and execution loop upon input submission
if user_message:
    with st.spinner("Analyzing request and processing reasoning loop..."):
        # Triggering secure payload delivery utilizing the active model ID
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": user_message}],
            model="openai/gpt-oss-20b",
        )
        
        # FIX: Added choices[0] array index to read list schema parameters safely
        ai_response = chat_completion.choices[0].message.content
        st.write("---")
        st.subheader("System Response:")
        st.write(ai_response)
