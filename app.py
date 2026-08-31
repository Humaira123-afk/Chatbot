%%writefile app.py
import streamlit as st
from groq import Groq

# 1. Page Config Layout Framework 
st.set_page_config(page_title="AI RAG Core Matrix", page_icon="🧠", layout="centered")

st.title("🧠 Enterprise RAG System")
st.write("Upload your document layout below to unlock conversational context analytics.")

# 2. Secure connection setup (Insert your production Groq Key inside the quotes)
GROQ_API_KEY = "Grok Key here"
client = Groq(api_key=GROQ_API_KEY)

# 3. Streamlit Sidebar Module for Ingestion
with st.sidebar:
    st.header("📄 Ingestion Module")
    uploaded_file = st.file_uploader("Upload reference text source:", type=["txt"])
    st.write("---")
    st.info("System partitions context arrays directly into localized data nodes.")

# 4. Initializing permanent chat memory arrays across window reruns
if "messages" not in st.session_state:
    st.session_state.messages = []

# Processing file payload bytes when document metadata populates
if uploaded_file is not None:
    custom_data = uploaded_file.read().decode("utf-8")
    
    # Custom light-weight text slicer logic into semantic chunks
    all_chunks = [chunk.strip() for chunk in custom_data.split(".") if len(chunk.strip()) > 5]
    
    # Render older persistent messaging blocks from session tracking loop
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
    # 5. Rendering the Modern Interactive Arrow Text Box Input Container
    user_query = st.chat_input("Ask a question about the uploaded document...")
    
    if user_query:
        # Commit query into memory array instantly and print layout
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)
        
        # Generating response container blocks
        with st.chat_message("assistant"):
            with st.spinner("Executing structural retrieval mapping index..."):
                
                # Smart Keyword Retrieval matrix builder
                relevant_context = ""
                query_words = [word.lower() for word in user_query.split() if len(word) > 2]
                
                for chunk in all_chunks:
                    if any(word in chunk.lower() for word in query_words):
                        relevant_context += chunk + ". "
                
                if not relevant_context:
                    relevant_context = custom_data[:1000]
                    
                # Constructing the exact augmented strict prompt template layout
                augmented_prompt = f"""
                You are a strict data analyst system. Answer the user question based on the provided Context rules only.
                Context: {relevant_context}
                Question: {user_query}
                Return an exact, clear response. Do not use external data.
                """
                
                # Dispatching data core tokens safely to the cloud server
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": augmented_prompt}],
                    model="openai/gpt-oss-20b",
                )
                
                # FIX: Added list mapping schema parser index [0] to read dynamic tokens safely
                ai_response = chat_completion.choices[0].message.content
                st.write(ai_response)
                
        # Commit response records securely into the session state tracker
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
else:
    st.warning("⚠️ Access Blocked: Please ingest a valid reference text document from the sidebar module first.")
