import argparse
import logging
import socket
from contextlib import asynccontextmanager
from typing import List

import numpy as np
import uvicorn
from diskannpy import StaticDiskIndex
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
loaded_index = None
embeddings = None
qid_mapping = dict()

# Configuration variables (will be set from command line arguments)
NODE_ID = None
INDEX_DIR = None
PORT = None


def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='CW22 Search Node Service')
    parser.add_argument('--node-id', type=str, required=True,
                        help='Node ID for this search service instance')
    parser.add_argument('--index-dir', type=str, required=True,
                        help='Directory path for the search index')
    parser.add_argument('--port', type=int, required=True,
                        help='Port number for the service')
    parser.add_argument('--num-threads', type=int, default=4,
                        help='Number of threads for index operations')
    parser.add_argument('--num-nodes-to-cache', type=int, default=30000,
                        help='Number of nodes to cache in memory')
    parser.add_argument('--cache-mechanism', type=int, default=1,
                        help='Cache mechanism to use')
    parser.add_argument('--dimensions', type=int, default=1024,
                        help='Vector dimensions')
    parser.add_argument('--index-prefix', type=str, default="cw22_b_en",
                        help='Prefix for index files')
    
    return parser.parse_args()


def get_ip_address():
    """
    Get the machine's IP address that can be used to access the service
    """
    try:
        # Create a temporary socket to determine the IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Connect to an external server (doesn't actually send anything)
        s.connect(("8.8.8.8", 80))
        # Get the IP address used for this connection
        ip_address = s.getsockname()[0]
        s.close()
        return ip_address
    except Exception as e:
        logger.warning(f"Could not determine IP address: {str(e)}")
        # Fall back to hostname
        return socket.gethostbyname(socket.gethostname())


def init_index(args):
    """
    Initialize the vector search index with configurable parameters
    """
    global loaded_index

    try:
        index = StaticDiskIndex(index_directory=args.index_dir,
                                num_threads=args.num_threads,
                                num_nodes_to_cache=args.num_nodes_to_cache,
                                cache_mechanism=args.cache_mechanism,
                                distance_metric='mips',
                                vector_dtype=np.float32,
                                dimensions=args.dimensions,
                                index_prefix=args.index_prefix)
        loaded_index = index
        logger.info(f"-----------------Index shard {args.node_id} loaded-----------------")
        logger.info(f"Index directory: {args.index_dir}")
    except Exception as e:
        logger.error(f"Error initializing index: {str(e)}")
        raise


def query_index(q_emb: np.ndarray, k: int = 10, complexity: int = 50) -> tuple:
    """
    Query the vector index with the given embedding file

    Args:
        q_emb: The file name of the encoded query embedding vector, like "q_001.npy"
        k: Number of nearest neighbors to retrieve
        complexity: Search complexity parameter

    Returns:
        Tuple of (raw_doc_ids, distances)
    """
    global loaded_index

    try:
        # Search the index
        raw_doc_ids, distances = loaded_index.search(query=q_emb,
                                                     k_neighbors=k,
                                                     complexity=complexity,
                                                     beam_width=10)

        logger.debug(f"Labels: {raw_doc_ids.tolist()}")
        logger.debug(f"Distances: {distances.tolist()}")
        return raw_doc_ids.tolist(), distances.tolist()
    except Exception as e:
        logger.error(f"Error in query_index: {str(e)}")
        raise


# Request and response models
class SearchRequest(BaseModel):
    q_emb: List[float]
    k: int
    complexity: int


# Response model for type hints and documentation
class SearchResponse(BaseModel):
    indices: List[int]
    distances: List[float]


def create_app(args):
    """
    Create FastAPI application with lifespan management
    """
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan context manager for FastAPI application"""
        # Initialize the index on startup
        init_index(args)

        # Display service information with actual IP
        ip_address = get_ip_address()
        logger.info(f"Node {args.node_id} service is accessible at: http://{ip_address}:{args.port}")
        logger.info(f"API documentation available at: http://{ip_address}:{args.port}/docs")

        yield
        # Clean up resources if needed on shutdown
        logger.info(f"Shutting down Node {args.node_id} service...")

    # Initialize FastAPI app with lifespan
    app = FastAPI(
        title=f"Vector Search API - Node {args.node_id}",
        description=f"API for vector similarity search using diskannpy on Node {args.node_id}",
        version="1.0.0",
        lifespan=lifespan
    )

    @app.post("/search", response_model=SearchResponse)
    async def search(request: SearchRequest):
        """
        Search endpoint that returns nearest neighbors for a given query embedding

        Args:
            q_emb: The query embedding vector
            k: Number of results to return
            complexity: Search complexity parameter

        Returns:
            JSON with raw document IDs and distances
        """
        try:
            q_emb = np.array(request.q_emb, dtype=np.float32)
            
            raw_doc_ids, distances = query_index(q_emb, request.k, request.complexity)
            return {"indices": raw_doc_ids, "distances": distances}
        except Exception as e:
            logger.error(f"Error in search endpoint: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        if loaded_index is None:
            raise HTTPException(status_code=503, detail="Index not loaded")
        return {
            "status": "healthy",
            "node_id": args.node_id,
            "index_dir": args.index_dir,
            "port": args.port
        }

    # Node info endpoint
    @app.get("/info")
    async def node_info():
        """Get information about this node"""
        return {
            "node_id": args.node_id,
            "index_dir": args.index_dir,
            "port": args.port,
            "status": "running" if loaded_index is not None else "not_ready"
        }

    return app


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Get the machine's IP address
    ip_address = get_ip_address()

    # Print the service information before starting
    print(f"\n======== Node {args.node_id} Service Information ========")
    print(f"Starting search service on HPC cluster")
    print(f"Index Directory: {args.index_dir}")
    print(f"The service will be accessible at: http://{ip_address}:{args.port}")
    print(f"API documentation will be available at: http://{ip_address}:{args.port}/docs")
    print(f"======================================\n")

    # Create the FastAPI app
    app = create_app(args)

    # Run the server with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")