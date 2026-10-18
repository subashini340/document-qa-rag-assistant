import os
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Document Q&A Assistant",
    page_icon="📚",
    layout="centered"
)

st.title("📚 Document Q&A Assistant")
st.write("Upload a PDF and ask questions based on its content.")

# Get Gemini API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("GEMINI_API_KEY is missing. Check your .env file.")
    st.stop()

# Upload PDF
uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"]
)

if uploaded_file is not None:

    try:
        # Read PDF
        reader = PdfReader(uploaded_file)
        documents = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            if text.strip():
                documents.append({
                    "text": text,
                    "page": page_number
                })

        if not documents:
            st.error(
                "No readable text found. "
                "Please upload a text-based PDF."
            )
            st.stop()

        # Split text into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )

        chunks = []

        for document in documents:
            for chunk in splitter.split_text(document["text"]):
                chunks.append({
                    "text": chunk,
                    "page": document["page"]
                })

        st.info("Processing PDF with local embeddings...")

        # Local embeddings - no Gemini embedding API needed
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Create vector database
        vector_store = Chroma.from_texts(
            texts=[chunk["text"] for chunk in chunks],
            embedding=embeddings,
            metadatas=[
                {"page": chunk["page"]}
                for chunk in chunks
            ]
        )

        st.success(
            f"PDF processed successfully! "
            f"{len(chunks)} text chunks created."
        )

        # Ask a question
        question = st.text_input(
            "Ask a question about your PDF"
        )

        if question:

            # Retrieve relevant chunks
            results = vector_store.similarity_search(
                question,
                k=min(4, len(chunks))
            )

            context = "\n\n".join(
                f"[Page {item.metadata['page']}]\n"
                f"{item.page_content}"
                for item in results
            )

            # Prompt template
            prompt = ChatPromptTemplate.from_template("""
You are a helpful document question-answering assistant.

Answer using ONLY the provided document context.
Do not invent facts.

If the answer is not in the context, say:
"The uploaded document does not contain enough information."

Mention the relevant page numbers in your answer.

Document context:
{context}

Question:
{question}

Answer:
""")

            # Gemini generates the answer
            try:
                llm = ChatGoogleGenerativeAI(
                    model="gemini-3.1-flash-lite-preview",
                    google_api_key=api_key,
                    temperature=0
                )

                chain = prompt | llm

                with st.spinner("Generating answer..."):
                    response = chain.invoke({
                        "context": context,
                        "question": question
                    })

                st.subheader("Answer")
                st.write(response.content)

                # Show retrieved source pages
                st.subheader("Retrieved source pages")

                pages = sorted({
                    item.metadata["page"]
                    for item in results
                })

                st.write(", ".join(map(str, pages)))

            except Exception as error:
                st.error(
                    "Could not generate the answer. "
                    f"Please check your Gemini API quota and model. "
                    f"Details: {error}"
                )

    except Exception as error:
        st.error(f"PDF processing failed: {error}")