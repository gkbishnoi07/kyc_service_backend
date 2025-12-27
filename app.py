from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import json
from typing import Optional, Dict, Any
import shutil

from pipeline.run_pipeline import run_pipeline
from config import settings

app = FastAPI(
    title="KYC Verification Service",
    description="Automated KYC document verification with AI-powered extraction",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

@app.get("/", response_class=HTMLResponse)
async def get_demo_ui():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "index.html")
    with open(file_path, "r") as f:
        return f.read()

@app.post("/kyc/verify")
async def verify_kyc(
    aadhaar_front: UploadFile = File(...),
    aadhaar_back: UploadFile = File(...),
    driving_license: UploadFile = File(...),
    vehicle_plate_photo: UploadFile = File(...),
    selfie: UploadFile = File(...),
    rc: Optional[UploadFile] = File(None),
    rider_id: Optional[str] = Form(None),
    onboarding_id: Optional[str] = Form(None)
):
    """
    Verify KYC documents including Aadhaar, Driving License, and Vehicle Registration
    
    Returns verification status with confidence scores and detailed issues
    """
    try:
        # Create temporary directory for uploaded files
        temp_dir = tempfile.mkdtemp()
        
        # Save uploaded files
        docs = {}
        file_mappings = {
            "aadhaar_front": aadhaar_front,
            "aadhaar_back": aadhaar_back,
            "driving_license": driving_license,
            "vehicle_plate_photo": vehicle_plate_photo
        }
        # selfie is required so always include it
        file_mappings["selfie"] = selfie

        if rc and rc.filename:
            file_mappings["rc"] = rc
        
        for doc_type, uploaded_file in file_mappings.items():
            file_path = os.path.join(temp_dir, f"{doc_type}.jpg")
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(uploaded_file.file, buffer)
            docs[doc_type] = file_path
        
        # Run the verification pipeline
        result = run_pipeline(docs)
        
        # Add metadata
        result["metadata"] = {
            "rider_id": rider_id,
            "onboarding_id": onboarding_id
        }
        
        # Cleanup temporary files
        shutil.rmtree(temp_dir)
        
        return result
        
    except Exception as e:
        # Cleanup on error
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir)
        
        raise HTTPException(
            status_code=500,
            detail=f"Verification failed: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "kyc-verification"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)