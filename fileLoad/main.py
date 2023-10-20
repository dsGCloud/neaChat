from langchain.document_loaders import GCSDirectoryLoader
from langchain.embeddings import VertexAIEmbeddings
from langchain.document_loaders import GCSFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import MatchingEngine
from cloudevents.http import CloudEvent
from google.cloud import aiplatform


import functions_framework

#function to load documents from GCS bucket
def load_docs(projectID, bucketName):
    loader = GCSDirectoryLoader(project_name=projectID, bucket=bucketName)
    documents = loader.load()
    return documents

#function to split documents into chunk size
def split_docs(documents,chunk_size=500,chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

# Triggered by a change in a storage bucket
@functions_framework.cloud_event
def file_load(cloud_event: CloudEvent) -> tuple:

    data = cloud_event.data

    aiplatform.init(
    project='nom-llm-001',

    # the Vertex AI region you will use
    location='us-central1',

    # Google Cloud Storage bucket in same region as location
    # used to stage artifacts
    staging_bucket='gs://neachat-artifacts',
    )

    # Define Text Embeddings model
    embedding = VertexAIEmbeddings()

    # Define Matching Engine as Vector Store  
    me = MatchingEngine.from_components(
        project_id='nom-llm-001',
        region='us-central1',
        gcs_bucket_name=f'gs://neachat-index',
        embedding=embedding,
        index_id='nea-ME-index',
        endpoint_id='neaEndpoint'
    )

    # Get the index
    index = me.get_index()

    # Check if the index exists
    if index is None:
        me = MatchingEngine.create_index(
            project_id='nom-llm-001',
            region='us-central1',
            gcs_bucket_name='gs://neachat-index',
            embedding=embedding,
            index_id='nea-ME-index'
        )
    else:
        # Define Cloud Storage file loader to read a document
        documents = load_docs(me.project_id, me.gcs_bucket_name)

        # Split document into chunks
        doc_splits = split_docs(documents)

        # Add embeddings of document chunks to Matching Engine
        texts = [doc.page_content for doc in doc_splits]
        metadatas = [
            [
                {"namespace": "source", "allow_list": [doc.metadata["source"]]},
                {"namespace": "document_name", "allow_list": [doc.metadata["document_name"]]},
                {"namespace": "chunk", "allow_list": [str(doc.metadata["chunk"])]},
            ]
            for doc in doc_splits
        ]
        doc_ids = me.add_texts(texts=texts, metadatas=metadatas)

        print(me.similarity_search("¿Quiéres conocer cuánto vas a pagar de cuota mensual con Toperty?", k=2))