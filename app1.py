import streamlit as st
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document

# Function to load the documents and setup LangChain models
def setup_langchain():
    # Load Excel data
    df = pd.read_excel('extracted_job_d.xlsx')  # Adjust the path as necessary

    # Convert the 'text' column to a list of LangChain Document objects
    documents = [Document(page_content=text) for text in df['text'].tolist()]
    
    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # Set up embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model='models/embedding-001',
        google_api_key='AIzaSyAUH70gKFSmR52QAbZq4fJFM3WSbTYCHp8',  # Replace with your API key
        task_type="retrieval_query"
    )

    # Create the vector store
    vectordb = Chroma.from_documents(documents=texts, embedding=embeddings)

    # Setup prompt template
    prompt_template = """
    ## Safety and Respect Come First!
    You are programmed to be a helpful and harmless AI. You will not answer requests that promote harmful behavior.
    **How to Use You:**
    1. Provide context on a topic.
    2. Ask a specific question.
    Context: \n {context}
    Question: \n {question}
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

    # Create the Chat Google model
    chat_model = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        google_api_key='AIzaSyAUH70gKFSmR52QAbZq4fJFM3WSbTYCHp8',  # Replace with your API key
        temperature=0.3
    )

    # Create the QA chain
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=vectordb.as_retriever(search_kwargs={"k": 5}),
        llm=chat_model
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_model,
        retriever=retriever_from_llm,
        return_source_documents=True,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain

# Streamlit app
def main():
    st.title("Job Role Finder with LangChain and Google GenAI")

    # Input for user question
    question = st.text_input("Ask a question about job roles:")

    if question:
        # Setup LangChain
        qa_chain = setup_langchain()

        # Run the query and get the response
        response = qa_chain.invoke({question})
        
        # Display the response
        st.write("Answer:", response['result'])

# Run the app
if __name__ == "__main__":
    main()