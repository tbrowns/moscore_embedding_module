import os
from dotenv import load_dotenv

from supabase import create_client, Client
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

# Load environment variables
load_dotenv()

class TextEmbedder:
    def __init__(self, supabase_url, supabase_key, chunk_size=1000, chunk_overlap=100):
        # Initialize Supabase client
        self.supabase: Client = create_client(supabase_url, supabase_key)

        # Initialize the text splitter with default or provided parameters
        separators = ["\n\n", "\n", " ", ""]
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators
        )
        # Initialize the embeddings model
        self.embeddings_model = OpenAIEmbeddings()
    
    def generate_embeddings(self, cluster_uuid):
        """
        Split the text into chunks and generate embeddings for each chunk.
        Then store the embeddings in Supabase.
        """
        # Retrieve the text from the files table for the given cluster UUID
        text = self.retrieve_text(cluster_uuid)
        
        # Split the text into chunks
        text_chunks = self.chunk_text(text)
        
        # Create Document objects from the text chunks
        documents = [Document(page_content=chunk) for chunk in text_chunks]
        
        # Generate embeddings for each document (chunk)
        embeddings = self.embeddings_model.embed_documents([doc.page_content for doc in documents])
        
        # Store the embeddings in Supabase
        self.store_embeddings(cluster_uuid, text_chunks, embeddings)
        
        return embeddings
    
    def retrieve_text(self, cluster_uuid):
        """
        Retrieve the text from the files table for the given cluster UUID.
        """
        response = self.supabase.table("files").select("content").eq("cluster_id", cluster_uuid).execute()
        data = response.data
        text = ""
        for item in data:
            text += item['content']
        return text
    
    def chunk_text(self, text):
        """
        Split the input text into chunks.
        """
        return self.text_splitter.split_text(text)

    def store_embeddings(self, cluster_uuid, text_chunks, embeddings):
        """
        Store the generated embeddings into Supabase for a specific cluster.
        """
        for chunk_id, (chunk, embedding) in enumerate(zip(text_chunks, embeddings)):
            
            # Prepare the data to insert into the Supabase table
            data = {
                "cluster_id": cluster_uuid,  # Use cluster UUID to reference the specific cluster
                "text_chunk": chunk,  # Store the text chunk as well
                "embedding": embedding  # Ensure the embedding is stored as an array
            }
            
            # Insert into Supabase
            try:
                response = self.supabase.table("embeddings").insert(data).execute()
                print(f"Chunk {chunk_id} embeddings successfully inserted.")
                
            except Exception as exception:
                print(f"Error inserting chunk {chunk_id}: {exception}")

# Example usage
if __name__ == "__main__":
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    
    # Initialize the TextEmbedder with Supabase connection
    text_embedder = TextEmbedder(SUPABASE_URL, SUPABASE_KEY)
    
    # Generate embeddings and store them in Supabase for a specific cluster
    cluster_uuid = "c7b65b44-9c60-413b-8434-af3b58dd2c76"  # Replace this with your actual cluster UUID
    embeddings = text_embedder.generate_embeddings(cluster_uuid)
    
    print("Embeddings generation and storage complete.")
