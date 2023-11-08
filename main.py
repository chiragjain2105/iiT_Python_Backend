import io

import uvicorn
from PyPDF2 import PdfReader
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import async_timeout
import os
from langchain import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
import pinecone
from langchain.vectorstores import FAISS
from db import database, users
from models import User, Status
from passlib.context import CryptContext
import PyPDF2

import nltk
from nltk.corpus import stopwords
# nltk.download("stopwords")
# nltk.download("punkt")
# nltk.download("wordnet")
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

pinecone.init(api_key="a7173290-0fd4-474b-85c8-140bdf29fb2b", environment="gcp-starter")

app = FastAPI()
# app.max_request_size = 1024 * 1024 * 10
origins = ["http://localhost:5173"]  # Replace with your React app's URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_db():
    await database.connect()


@app.on_event("shutdown")
async def shutdown_db():
    await database.disconnect()


# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# Dependency to get the database connection
async def get_db():
    async with database.transaction():
        yield database


@app.post("/register", response_model=User, status_code=status.HTTP_201_CREATED)
async def register(user: User, db: database = Depends(get_db)):
    # Hash the user's password
    hashed_password = pwd_context.hash(user.password)
    # Create a query to insert user data into the database
    query = users.insert().values(username=user.username, password=hashed_password)
    # Execute the query and get the ID of the last inserted record
    last_record_id = await db.execute(query)
    # Return the ID and the user's data as the response
    return {"id": last_record_id, **user.dict()}


@app.post("/login", response_model=Status)
async def login(user: User, db: database = Depends(get_db)):
    # Create a query to retrieve user data based on the provided username
    query = users.select().where(users.c.username == user.username)
    # Execute the query to get the user data
    db_user = await db.fetch_one(query)
    # Check if the username exists
    if db_user is None:
        raise HTTPException(status_code=200, detail="Username not found")
    # Verify the provided password against the hashed password in the database
    if not pwd_context.verify(user.password, db_user['password']):
        raise HTTPException(status_code=200, detail="Incorrect password")
    # Return the username, password as the response
    # return {"username": user.username,"password":user.password}
    return {"loginStatus": True}


class InputData(BaseModel):
    inputData: str


class QuestionData(BaseModel):
    question: str


# class pdfInput(BaseModel):
#     pdf: UploadFile = File(...)


# class pdfInput(BaseModel):
#     pdfData: str


@app.post('/api/endpoint')
async def process_input_data(data: InputData):
    global qa
    input_data = data.inputData
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.create_documents([input_data])
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    docsearch = Pinecone.from_documents(texts, embeddings, index_name="project-index")
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever()
    )
    print("uploaded")
    return 1


@app.post('/api/question')
async def process_question(data: QuestionData):
    global result
    global qa
    ques = data.question
    result = qa({"query": ques})
    print(result['result'])

    return result


@app.post("/uploadfiles/")
async def create_upload_files(files: list[UploadFile]):
    global qa
    global documents
    global pdf_documents
    # text_contents = []
    pdf_documents = []
    data = {}
    main_text = []
    documents = []
    for uploaded_file in files:
        main_text = []
        if uploaded_file.filename.endswith('.pdf'):
            pdf_data = await uploaded_file.read()
            pdf = PdfReader(io.BytesIO(pdf_data))
            text1 = ""
            for page_number in range(len(pdf.pages)):
                text1 += pdf.pages[page_number].extract_text()
                text = pdf.pages[page_number].extract_text()
                main_text.append(text)
                data[page_number] = text
            documents.append(data)
            # text_contents.append({"filename": uploaded_file.filename, "text": text1})
            pdf_documents.append(uploaded_file.filename)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    docs = text_splitter.create_documents(main_text)
    embeddings = OpenAIEmbeddings()  ## error is encountering at this line "module 'openai' has no attribute 'Embedding'"
    vectorstore = FAISS.from_documents(docs, embeddings)
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=vectorstore.as_retriever())
    # query = "Give me the gist of Transformers in 3 sentences"
    # res = qa.run(query)
    # print(res)
    return documents


def text_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text2, text1])
    similarity = cosine_similarity(vectors)
    return similarity


@app.get("/pdf_sources/")
def get_sources():
    global result
    global documents
    global pdf_documents
    max_score = float('-inf')
    source_page = None
    source_pdf = None
    for i in range(len(documents)):
        pdf = documents[i]
        for key in pdf:
            txt = pdf[key]
            score = text_similarity(result['result'], txt)
            if score[0][1] > max_score:
                max_score = score[0][1]
                source_page = key
                source_pdf = pdf_documents[i]

    # return 1
    return {"pdf": source_pdf, "page": source_page+1}


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
