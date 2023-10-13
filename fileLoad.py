from langchain.embeddings import VertexAIEmbeddings
from langchain.document_loaders import GCSFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import MatchingEngine


from cloudevents.http import CloudEvent

import functions_framework


# Triggered by a change in a storage bucket
@functions_framework.cloud_event
def hello_gcs(cloud_event: CloudEvent) -> tuple:

    data = cloud_event.data

    # Define Text Embeddings model
    embedding = VertexAIEmbeddings()

    # Define Matching Engine as Vector Store 
    me = MatchingEngine.from_components(
        project_id='nom-llm-001',
        region='us-central1',
        gcs_bucket_name=f'gs://neachat',
        embedding=embedding,
        index_id='neaMatchE',
        endpoint_id='neaEndpoint'
    )

    # Define Cloud Storage file loader to read a document
    loader = GCSFileLoader(project_name=me.project_id,
        bucket=url.split("/")[2],
        blob='/'.join(url.split("/")[3:])
    )

    # Split document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    doc_splits = text_splitter.split_documents(data["name"])

    # Add embeddings of document chunks to Matching Engine
    texts = [doc.page_content for doc in doc_splits]
    me.add_texts(texts=texts)