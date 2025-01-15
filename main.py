import PyPDF2
import os
import models
from typing import List, Dict
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

data = "D:\\Tasks\\UpSkill\\Additional_projects\\MyGitRepo\\BasicRag\\nke-10k-2023.pdf"

class PDFProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len, #have to check this what they are up to
                add_start_index=True,
            )


    def extract_text_from_pdf(self, pdf_path: str) -> str:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
                
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_content = []
                for page in pdf_reader.pages:
                    text_content.append(page.extract_text())
                    
            return "\n".join(text_content) # return extracted content

    def create_chunks(self, text: str) -> List[str]:
            """
            Returns:
                List[str]: List of text chunks
            """
            return self.text_splitter.split_text(text)
        
    def create_vector_store(self, chunks: List[str]) -> FAISS:
            """
            Returns:
                FAISS: Vector store containing the embeddings
            """
            embeddings = models.get_OllamaEmbedding(model_name="nomic-embed-text:latest")
            vector_store = FAISS.from_texts(chunks, embeddings)
            return vector_store

    def process_pdf(self, pdf_path: str) -> Dict:
            """
            Returns:
                Dict containing the extracted text, chunks, and vector store
            """
            # Extract text from PDF
            text = self.extract_text_from_pdf(pdf_path)
            
            # Create chunks
            chunks = self.create_chunks(text)
            
            # Create vector store
            vector_store = self.create_vector_store(chunks)
            
            return {
                "raw_text": text,
                "chunks": chunks,
                "vector_store": vector_store
            }


def main():

    processor = PDFProcessor(chunk_size=1000, chunk_overlap=200)
    
    pdf_path = "D:\\Tasks\\UpSkill\\Additional_projects\\MyGitRepo\\BasicRag\\nke-10k-2023.pdf"

    try:
        result = processor.process_pdf(pdf_path)
        
        # Print some statistics
        print(f"Total characters in document: {len(result['raw_text'])}")
        print(f"Number of chunks created: {len(result['chunks'])}")
        
        query = "nike details"
        similar_chunks = result['vector_store'].similarity_search(query)
        print(similar_chunks)

    except Exception as e:
        print(f"Error processing PDF: {str(e)}")


if __name__ == "__main__":
    main()