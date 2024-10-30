import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import shutil

class ResumeBot:
    def __init__(self, resume_path, persist_directory="db"):
        """
        Initialize the Resume Bot with a resume file and Ollama model.
        
        Args:
            resume_path (str): Path to the resume PDF file
            persist_directory (str): Directory to persist the vector store
            model_name (str): Name of the Ollama model to use
        """
        if not resume_path.lower().endswith('.pdf'):
            raise ValueError("Resume file must be a PDF")
            
        self.resume_path = resume_path
        self.persist_directory = persist_directory
        self.model_name = "llama3.2"
        self.qa_chain = None
        
        # If the index is corrupted, remove the persist directory
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
            
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
    def load_and_split_resume(self):
        """Load and split the PDF resume into chunks."""
        try:
            # Load the PDF resume
            loader = PyPDFLoader(self.resume_path)
            documents = loader.load()
            
            # Split the text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            texts = text_splitter.split_documents(documents)
            
            if not texts:
                raise ValueError("No text could be extracted from the PDF")
                
            return texts
            
        except Exception as e:
            raise Exception(f"Error processing PDF file: {str(e)}")
        
    def create_vector_store(self):
        """Create and persist the vector store."""
        texts = self.load_and_split_resume()
        
        # Create vector store
        self.vector_store = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        self.vector_store.persist()
        
    def setup_qa_chain(self):
        """Set up the question-answering chain."""
        # Initialize Ollama
        llm = Ollama(
            model=self.model_name,
            temperature=0.1
        )
        
        # Create retriever
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 3}
        )
        
        # Create prompt template
        prompt_template = """
        You are an AI assistant specializing in answering questions about a person's resume. 
        Use the following pieces of resume context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Don't try to make up an answer.
        
        Resume context: {context}
        
        Question: {question}
        
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": PROMPT
            }
        )
        
    def initialize(self):
        """Initialize the bot by setting up vector store and QA chain."""
        print("Loading and processing resume...")
        self.create_vector_store()
        print("Setting up QA system...")
        self.setup_qa_chain()
        print("Initialization complete!")
        
    def answer_question(self, question):
        """
        Answer a question about the resume.
        
        Args:
            question (str): Question about the resume
            
        Returns:
            dict: Answer and source documents
        """
        if not self.qa_chain:
            raise ValueError("Bot not initialized. Please call initialize() first.")
            
        try:
            result = self.qa_chain({"query": question})
            return {
                "answer": result["result"],
                # "sources": [doc.page_content for doc in result["source_documents"]]
            }
        except ConnectionError:
            return {
                "answer": "Error: Cannot connect to Ollama. Please make sure Ollama is running (ollama serve)",
                # "sources": []
            }
        except Exception as e:
            return {
                "answer": f"Error: {str(e)}",
                # "sources": []
            }

def main():
    # Example usage
    resume_path = "Scott Lai's Resume.pdf"  # Make sure this points to your PDF file
    
    try:
        # Initialize bot
        bot = ResumeBot(resume_path)
        bot.initialize()
        
        # Interactive question answering
        print("\nResume Q&A Bot initialized! Ask questions about the resume (type 'quit' to exit)")
        while True:
            question = input("\nQuestion: ").strip()
            if not question:
                continue
            if question.lower() == 'quit':
                break
                
            try:
                result = bot.answer_question(question)
                if "Error" in result["answer"]:
                    print("\n" + result["answer"])
                else:
                    print("\nAnswer:", result["answer"])
                    # print("\nSources:")
                    # for i, source in enumerate(result["sources"], 1):
                    #     print(f"\nSource {i}:", source)
            except Exception as e:
                print(f"\nError: {str(e)}")
                
    except Exception as e:
        print(f"Error initializing the bot: {str(e)}")

if __name__ == "__main__":
    main()