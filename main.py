from fastapi import FastAPI, File, UploadFile
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

# pdfData=""

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
