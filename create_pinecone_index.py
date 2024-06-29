import os
from pinecone import Pinecone, PodSpec

from dotenv import load_dotenv

config = load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_DIMENSION = os.environ.get("PINECONE_DIMENSION")

pc = Pinecone(api_key=PINECONE_API_KEY)

pc.create_index(
    name="images",
    dimension=int(PINECONE_DIMENSION),
    metric="cosine",
    spec=PodSpec(environment="gcp-starter", pod_type="starter")
)
