import streamlit as st
import json
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.vectorstores import InMemoryVectorStore
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
)

# Load the FAQ data
with open("FAQ.json", "r") as f:
    faq_data = json.load(f)['questions']

questions = [elem['question'] for elem in faq_data]
vectorstore = InMemoryVectorStore.from_texts(
    questions,
    embedding=embeddings,
)

# Initialize the chat model
model = init_chat_model("gpt-4o-mini", model_provider="openai")

# Streamlit app
st.title("Chat Assistant")

query = st.text_input("Enter your question:")

if query:
    threshold = 0.9
    results = vectorstore.similarity_search_with_score(query, k=1)
    doc, score = results[0]

    if score > threshold:
        st.write("### Answer:")
        st.write(faq_data[questions.index(doc.page_content)]['answer'])
    else:
        prompt = (f"Please provide an answer to the question: {query} in a user-friendly format. "
                  "If you don't know the answer, indicate clearly that you don't know it.\n\n")
        response = model.invoke(prompt)
        st.write("### Answer:")
        st.write(response.content)