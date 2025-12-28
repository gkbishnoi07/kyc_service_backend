from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import tempfile
import os
import shutil
from typing import Optional, Dict, Any

from pipeline.run_pipeline import run_pipeline
from pipeline.file_converter import convert_to_images
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


# ------------------------
# Demo UI
# ------------------------
@app.get("/", response_class=HTMLResponse)
async def get_demo_ui():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "index.html")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# ------------------------
# KYC Verification API
# ------------------------
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
    Verify KYC documents including Aadhaar, Driving License, Vehicle Plate and optional RC.
    Supports JPG / PNG / HEIC / PDF uploads.
    """
    temp_dir = None

    try:
        temp_dir = tempfile.mkdtemp(prefix="kyc_")

        docs: Dict[str, str] = {}

        file_mappings = {
            "aadhaar_front": aadhaar_front,
            "aadhaar_back": aadhaar_back,
            "driving_license": driving_license,
            "vehicle_plate_photo": vehicle_plate_photo,
            "selfie": selfie,
        }

        if rc and rc.filename:
            file_mappings["rc"] = rc

        # ------------------------
        # Save + convert uploads
        # ------------------------
        for doc_type, uploaded_file in file_mappings.items():
            if not uploaded_file or not uploaded_file.filename:
                continue

            raw_path = os.path.join(
                temp_dir, f"raw_{doc_type}_{uploaded_file.filename}"
            )

            # Save raw upload
            with open(raw_path, "wb") as buffer:
                shutil.copyfileobj(uploaded_file.file, buffer)

            # Convert to JPEG(s)
            converted_images = convert_to_images(
                input_path=raw_path,
                output_dir=os.path.join(temp_dir, doc_type)
            )

            if not converted_images:
                raise ValueError(f"No images produced for {doc_type}")

            # IMPORTANT:
            # For now we take the FIRST page/image
            # (Aadhaar/DL PDFs are typically single-page)
            docs[doc_type] = converted_images[0]

        # ------------------------
        # Run verification pipeline
        # ------------------------
        result = run_pipeline(docs)

        # Attach metadata
        result["metadata"] = {
            "rider_id": rider_id,
            "onboarding_id": onboarding_id,
        }

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"KYC verification failed: {str(e)}"
        )

    finally:
        # Cleanup temp files
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


# ------------------------
# Health Check
# ------------------------
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "kyc-verification"
    }


# ------------------------
# Local Dev Entry
# ------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)