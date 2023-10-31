from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import async_timeout
import os
from langchain import OpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
import pinecone
from langchain.vectorstores import FAISS
from pypdf import PdfReader
from db import database, users
from models import User,Status
from passlib.context import CryptContext

pinecone.init(api_key="", environment="")

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
    return {"loginStatus":True}

class InputData(BaseModel):
    inputData: str


class QuestionData(BaseModel):
    question: str

class pdfInput(BaseModel):
    pdfData: str

@app.post('/api/endpoint')
async def process_input_data(data: InputData):
    global qa
    input_data = data.inputData
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.create_documents([input_data])
    # print(pdfData)
    # if pdfData!="":
    #     embeddings = OpenAIEmbeddings()
    #     vectorstore = FAISS.from_documents(texts, embeddings)
    #     qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=vectorstore.as_retriever())
    #     return 2

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    docsearch = Pinecone.from_documents(texts, embeddings, index_name="project-index")
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever()
    )

    return 1


@app.post("/uploadpdf/")
async def upload_pdf(pdf_file: UploadFile):
    with open(pdf_file.filename, "wb") as f:
        f.write(pdf_file.file.read())

    reader = PdfReader(pdf_file.file)
    num_page = len(reader.pages)

    text=""
    for i in range(num_page):
        page = reader.pages[i]
        text+=page.extract_text()
    # print(text)
    # pdfText=text

    return {'data':text}


@app.post('/api/question')
async def process_question(data: QuestionData):
    global qa
    ques = data.question

    result = qa({"query": ques})
    print(result)

    return result





