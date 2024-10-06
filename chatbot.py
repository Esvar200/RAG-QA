from pymongo import MongoClient
from langchain.docstore.document import Document
from langchain_core.prompts import PromptTemplate 
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup

# Configure Google API Key
GOOGLE_API_KEY = 'AIzaSyAwlfiHJXlS-vSJ7_2TrEdyW-dDNnQ7oG4'
genai.configure(api_key=GOOGLE_API_KEY)

# MongoDB Configuration
client = MongoClient("mongodb+srv://pesvarramkumar:Esvar2040@cluster0.031ktrg.mongodb.net/")  # Adjust MongoDB connection URI as necessary
db = client["Vadhandhi"]  # Database name
collection = db["QA_Embeddings"]  # Collection name

def chatbot(user_input1, user_input2):
    # Load data from the provided URL
    response = requests.get(user_input1)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text(separator='\n')
    cleaned_text = text.strip()
    #print(cleaned_text)
    # Convert the text to LangChain's `Document` format
    docs = [Document(page_content=cleaned_text, metadata={"source": "local"})]

    # Initialize Google Embeddings
    gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

    # Generate embeddings for the document text
    for doc in docs:
        embedding = gemini_embeddings.embed_documents([doc.page_content])[0]  # Generate embedding for the document

        # Store embedding and metadata in MongoDB
        embedding_data = {
            "embedding": embedding,
            "metadata": doc.metadata,
            "content": doc.page_content
        }
        collection.insert_one(embedding_data)

    # Retrieve embeddings from MongoDB for search
    def retrieve_from_mongo(query):
        # Fetch the most relevant document based on similarity (simplified for demo purposes)
        result = collection.find_one({"content": {"$regex": query}})
        return result

    # Function to format documents retrieved from MongoDB
    # def format_docs_from_mongo(docs):
    #     return docs["content"]

    # Simulate retrieval (For simplicity, using the same document as context)
    retrieved_doc = retrieve_from_mongo(user_input2)
    retrieved_doc = RunnablePassthrough()
    # Initialize Google Generative AI model
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3, top_p=0.85, google_api_key=GOOGLE_API_KEY)

    # Prompt template for querying Gemini
    llm_prompt_template = """You are an assistant for question-answering tasks.
    Use the following context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use five sentences maximum and keep the answer concise.
    Provide answer in a neat format if it has points.\n
    Question: {question} \nContext: {context} \nAnswer:"""

    llm_prompt = PromptTemplate.from_template(llm_prompt_template)

    # RAG Chain Configuration using MongoDB stored data
    rag_chain = (
        {"context": retrieved_doc, "question": RunnablePassthrough()}
        | llm_prompt
        | llm
        | StrOutputParser()
    )

    res = rag_chain.invoke(user_input2)
    result = collection.delete_many({})
    return res


