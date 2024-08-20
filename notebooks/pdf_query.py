import warnings

from langchain.llms.ollama import Ollama
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

class PDFQuery:
    def __init__(self, file_path:str):
        warnings.filterwarnings("ignore")
        self.file_path = file_path
        self.embeddings = None
        self.docSearch = None
        self.llm_model = None
        self.qa = None

    def setup_embeddings(self, model_name="nomic-embed-text"):
        self.embeddings = OllamaEmbeddings(model=model_name)

    def load_and_process_pdf(self):
        loader = PyPDFLoader(file_path=self.file_path)
        pages = loader.load_and_split()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(pages)
        self.docSearch = Chroma.from_documents(texts, embedding=self.embeddings)

    def setup_llm(self, model_name="gemma2:2b"):
        self.llm_model = Ollama(model=model_name)

    def setup_qa(self):
        prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Answer:"""
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm_model,
            chain_type="stuff",
            retriever=self.docSearch.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

    def chat_with_paper(self, question: str):
        result = self.qa({"query": question})
        return result["result"], result["source_documents"]

    def infer_model(self, model_name="gemma2:2b"):
        # This is a placeholder function. In a real-world scenario,
        # we might want to implement logic to infer the best model based on the PDF content.
        return model_name