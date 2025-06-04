import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma  # ✅ using Chroma instead of FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

# Step 1: PDF Upload
def load_pdf(path_to_pdf):
    loader = PyPDFLoader(path_to_pdf)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from PDF.")
    return documents

# Step 2: Chunk text
def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

# Step 3: Vector DB setup
def create_vector_db(chunks, persist_dir="story_vectorstore/chroma_store"):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory=persist_dir)
    vectordb.persist()
    return vectordb

# Step 4: Load LLM
def load_llm():
    return ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.7,
        max_tokens=700
    )

# Step 5: Prompt Template for Children’s Author Rewrite
def get_custom_prompt(author_name, style_instruction):
    return PromptTemplate(
        input_variables=["context", "question"],
        template=f"""
You are a magical AI storyteller who rewrites classics in the voice of {author_name}.
The style should be: {style_instruction}

Use the given context and rewrite the selected chapter in this style.

Context: {{context}}
User's Request: {{question}}

Respond ONLY with the rewritten chapter, in a warm and child-friendly tone.
"""
    )

# Step 6: Create RAG Chain
def create_qa_chain(vectordb, prompt):
    return RetrievalQA.from_chain_type(
        llm=load_llm(),
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt}
    )

# ✨ MASTER FLOW ✨
def run_story_rewriter():
    pdf_path = input("Drop the path to the classic PDF file: ").strip()
    docs = load_pdf(pdf_path)
    chunks = create_chunks(docs)
    vectordb = create_vector_db(chunks)

    print("\nChapters Available:")
    for i, doc in enumerate(docs):
        print(f"Page {i+1}: {doc.metadata['page']} - {doc.page_content[:80]}...")

    chapter_page = int(input("\nEnter the PAGE number you want rewritten: ").strip()) - 1
    selected_content = docs[chapter_page].page_content

    author_name = input("\nEnter the children's author (e.g., Roald Dahl, Dr. Seuss, Mo Willems): ").strip()
    style_instruction = input("Describe the tone/style you want (e.g., rhyming, silly, calm): ").strip()

    prompt = get_custom_prompt(author_name, style_instruction)
    chain = create_qa_chain(vectordb, prompt)

    print("\n🪄 Generating the rewritten chapter...\n")
    result = chain.invoke({
        "query": f"Rewrite the following in the style of {author_name}:\n{selected_content}"
    })
    print(result["result"])

# Run the full pipeline
if __name__ == "__main__":
    run_story_rewriter()
