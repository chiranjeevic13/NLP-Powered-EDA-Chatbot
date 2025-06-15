from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import os
import logging
from typing import Dict, Any
from app.modules.eda_module import EDAModule
from app.modules.opena_module import OpenQAModule
import shutil
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NLP-Powered EDA Chatbot API",
              description="API for natural language data querying and advanced EDA",
              version="1.0.0")

# Directory for uploaded datasets
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# In-memory storage for session data (replace with database in production)
sessions: Dict[str, pd.DataFrame] = {}

@app.post("/upload_dataset", summary="Upload a CSV dataset")
async def upload_dataset(file: UploadFile = File(...)) -> Dict[str, str]:
    """
    Upload a CSV dataset and return a session ID.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        
        # Save uploaded file
        file_path = os.path.join(DATA_DIR, f"{session_id}.csv")
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Load and store DataFrame
        df = pd.read_csv(file_path)
        sessions[session_id] = df
        
        logger.info(f"Dataset uploaded successfully. Session ID: {session_id}")
        return {"session_id": session_id, "message": "Dataset uploaded successfully"}
    except Exception as e:
        logger.error(f"Error uploading dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading dataset: {str(e)}")

@app.post("/generate_eda/{session_id}", summary="Generate EDA report")
async def generate_eda(session_id: str, formats: list[str] = ["json", "yaml", "txt"]) -> Dict[str, str]:
    """
    Generate an EDA report for the dataset associated with the session ID.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session ID not found")
    
    try:
        df = sessions[session_id]
        eda = EDAModule(df)
        eda.run_and_save(formats=formats)
        
        # Get the latest report directory
        report_dir = max([os.path.join("eda_reports", d) for d in os.listdir("eda_reports") 
                         if os.path.isdir(os.path.join("eda_reports", d))], 
                         key=os.path.getmtime)
        
        logger.info(f"EDA report generated for session {session_id}")
        return {"message": "EDA report generated", "report_directory": report_dir}
    except Exception as e:
        logger.error(f"Error generating EDA report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating EDA report: {str(e)}")

@app.get("/download_eda/{session_id}/{format}", summary="Download EDA report")
async def download_eda(session_id: str, format: str) -> FileResponse:
    """
    Download the EDA report in the specified format (json, yaml, txt, csv).
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session ID not found")
    if format not in ["json", "yaml", "txt", "csv"]:
        raise HTTPException(status_code=400, detail="Invalid format. Supported: json, yaml, txt, csv")
    
    try:
        # Find the latest report file
        report_dir = max([os.path.join("eda_reports", d) for d in os.listdir("eda_reports") 
                         if os.path.isdir(os.path.join("eda_reports", d))], 
                         key=os.path.getmtime)
        report_file = os.path.join(report_dir, f"eda_insights_{os.path.basename(report_dir)}.{format}")
        
        if not os.path.exists(report_file):
            raise HTTPException(status_code=404, detail=f"Report file not found for format: {format}")
        
        logger.info(f"Downloading EDA report for session {session_id} in {format}")
        return FileResponse(report_file, filename=f"eda_report.{format}")
    except Exception as e:
        logger.error(f"Error downloading EDA report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error downloading EDA report: {str(e)}")

@app.post("/query/{session_id}", summary="Process a natural language query")
async def process_query(session_id: str, query: Dict[str, str]) -> Dict[str, Any]:
    """
    Process a natural language query on the dataset associated with the session ID.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session ID not found")
    if "query" not in query:
        raise HTTPException(status_code=400, detail="Query field is required")
    
    try:
        df = sessions[session_id]
        qa_module = OpenQAModule(df)
        response = qa_module.process_query(query["query"])
        
        logger.info(f"Query processed for session {session_id}: {query['query']}")
        return {"query": query["query"], "response": response}
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/health", summary="Check API health")
async def health_check() -> Dict[str, str]:
    """
    Check the health of the API.
    """
    return {"status": "healthy"}