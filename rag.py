from functools import partial

import gradio
from dotenv import load_dotenv
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from openai import OpenAI

load_dotenv()
openai = OpenAI()

loader = partial(TextLoader, encoding="utf-8")
dir_loader = DirectoryLoader("assets/rag", loader_cls=loader)
docs = dir_loader.load()

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

embedding = OpenAIEmbeddings()

vectorstore = FAISS.from_documents(chunks, embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    callbacks=[StdOutCallbackHandler()],
    return_source_documents=True
)


def chat(prompt, history):
    result = chain.invoke({"question": prompt})
    print("Src doc:", [doc.metadata.get("source", "") for doc in result["source_documents"]])
    print("Memory:", memory.chat_memory.messages)
    return result["answer"]


gradio.ChatInterface(fn=chat).launch()
