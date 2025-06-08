import base64
import logging
import socket
import csv
import os
from contextlib import asynccontextmanager
from typing import Optional, Dict

import uvicorn
from fastapi import FastAPI, Query, Depends, HTTPException, Header
from pydantic import BaseModel
from tqdm import tqdm

from utils.cw22_api import ClueWeb22Api
from auth.auth_db import *

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
url_to_docid: Dict[str, str] = {}
MAP_FILE = '/bos/tmp6/jmcoelho/cweb22-b-en/map_id_url.csv'


def get_ip_address():
    """Get the machine's IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip_address = s.getsockname()[0]
        s.close()
        return ip_address
    except Exception as e:
        logger.warning(f"Could not determine IP address: {str(e)}")
        return socket.gethostbyname(socket.gethostname())


def get_base64(json_bytes):
    """Convert bytes to base64 encoded string"""
    return base64.b64encode(json_bytes).decode()


# Define lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI application"""
    global url_to_docid

    logger.info("Initializing auth db...")
    init_auth()
    
    # Load URL to document ID mapping
    logger.info("Loading URL to document ID mapping...")
    if os.path.exists(MAP_FILE):
        with open(MAP_FILE, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for doc_id, url in tqdm(reader):
                url_to_docid[url] = doc_id
        logger.info(f"Loaded {len(url_to_docid)} URL mappings")
    else:
        logger.error(f"Mapping file not found at {MAP_FILE}")
        raise FileNotFoundError(f"Mapping file not found at {MAP_FILE}")

    # Display service information
    ip_address = get_ip_address()
    port = 51001
    logger.info(f"Fetch service is accessible at: http://{ip_address}:{port}")
    logger.info(f"API documentation available at: http://{ip_address}:{port}/docs")

    yield

    # Clean up resources if needed on shutdown
    logger.info("Shutting down fetch service...")


# Initialize FastAPI app
app = FastAPI(
    title="ClueWeb22 Fetch API",
    description="API for fetching clean text from ClueWeb22 documents",
    version="1.0.0",
    lifespan=lifespan
)


def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Verify the API key from the X-API-Key header"""
    if not verify_api_key_exists(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return True


@app.get("/fetch")
async def fetch(
        url: str = Query(..., description="The URL to fetch the clean text for"),
        with_outlink: Optional[bool] = Query(False, description="Whether with document outlink"),
        api_key_valid: bool = Depends(verify_api_key)
):
    """
    Fetch clean text for a given URL

    Args:
        url: The URL to fetch the clean text for
        with_outlink: Whether to include outlinks

    Returns:
        JSON with clean text and optionally outlinks
    """
    global url_to_docid
    
    clean_url = url.rstrip("\n")
    doc_id = url_to_docid.get(clean_url)
    
    if doc_id is None:
        logger.warning(f"URL not found in mapping: {clean_url}")
        raise HTTPException(status_code=404, detail="URL not found.")
    
    logger.info(f"Fetching content for URL: {clean_url}, doc_id: {doc_id}")
    
    try:
        clueweb_api = ClueWeb22Api(doc_id)
        clean_txt_b64 = get_base64(clueweb_api.get_clean_text())

        if with_outlink:
            outlink_b64 = get_base64(clueweb_api.get_outlinks())
            return {"clean_text": clean_txt_b64, "outlink": outlink_b64}
        
        return {"clean_text": clean_txt_b64}
        
    except Exception as e:
        logger.error(f"Error fetching content for URL {clean_url}, doc_id {doc_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch content: {str(e)}")


@app.get("/fetch/health")
async def health_check():
    """Health check endpoint"""
    global url_to_docid
    
    if not url_to_docid:
        raise HTTPException(status_code=503, detail="URL mapping not loaded")
    
    return {
        "status": "healthy",
        "service": "fetch",
        "url_mappings_loaded": len(url_to_docid)
    }


if __name__ == "__main__":
    # Get the machine's IP address
    ip_address = get_ip_address()
    port = 51005

    # Print the service information before starting
    print(f"\n======== Fetch Service Information ========")
    print(f"Starting ClueWeb22 fetch service")
    print(f"The service will be accessible at: http://{ip_address}:{port}")
    print(f"API documentation will be available at: http://{ip_address}:{port}/docs")
    print(f"==========================================\n")

    # Run the server with uvicorn
    uvicorn.run("__main__:app", host="0.0.0.0", port=port, log_level="info")