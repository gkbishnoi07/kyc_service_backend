from typing import Dict, Any, List
import os

from .quality import ImageQualityGate
from .extractor import DocumentExtractor
from .checks import DocumentChecks
from .decision import DecisionEngine
from .face_match import llm_face_match

def run_pipeline(docs: Dict[str, str]) -> Dict[str, Any]:
    """
    Main pipeline function that orchestrates the entire KYC verification process
    
    Args:
        docs: Dictionary with document types as keys and file paths as values
              Required: aadhaar_front, aadhaar_back, driving_license, vehicle_plate_photo
              Optional: rc
    
    Returns:
        Standardized verification response with status, confidence, and details
    """
    
    # Initialize components
    quality_gate = ImageQualityGate()
    extractor = DocumentExtractor()
    checker = DocumentChecks()
    decision_engine = DecisionEngine()
    
    # Step 1: Quality assessment for all documents
    quality_results = {}
    reupload_required = []
    
    for doc_type, file_path in docs.items():
        if os.path.exists(file_path):
            quality_result = quality_gate.evaluate(file_path)
            quality_results[doc_type] = quality_result
            
            if quality_result["quality"] == "bad":
                # For optional RC, we don't block the pipeline, we just skip it
                if doc_type == "rc":
                    quality_result["signals"].append("Optional RC skipped due to quality")
                    quality_result["recommended_action"] = "ignore"
                else:
                    reupload_required.append(doc_type)
        else:
            quality_results[doc_type] = {
                "quality": "bad",
                "risk_score": 0.0,
                "signals": ["File not found"],
                "recommended_action": "reject"
            }
            if doc_type != "rc":
                reupload_required.append(doc_type)
    
    # Step 2: Extract information from documents
    extracted_data = {}
    
    for doc_type, file_path in docs.items():
        if os.path.exists(file_path):
            try:
                extracted = extractor.extract(file_path, doc_type)
                extracted_data[doc_type] = extracted
            except Exception as e:
                # Create empty structure on extraction failure
                extracted_data[doc_type] = {
                    "extraction_error": str(e),
                    "confidence": 0.0
                }
    
    # Step 3: Run all validation checks
    format_issues = checker.format_checks(extracted_data)
    intra_issues = checker.intra_document_consistency(extracted_data)
    cross_issues = checker.cross_document_consistency(extracted_data)
    
    # Step 4: Validate plate OCR specifically
    if "vehicle_plate_photo" in extracted_data:
        plate_validation = checker.plate_ocr_validation(
            extracted_data["vehicle_plate_photo"]
        )
        extracted_data["vehicle_plate_photo"].update(plate_validation)

    # Step 4.5: If selfie provided, perform face similarity check against driving license
    if "selfie" in docs and "driving_license" in docs and os.path.exists(docs.get("selfie")) and os.path.exists(docs.get("driving_license")):
        try:
            face_result = llm_face_match(
                dl_image_path=docs.get("driving_license"),
                selfie_image_path=docs.get("selfie")
            )
            # Store face match result under extracted_data
            extracted_data["face_match"] = face_result
        except Exception as e:
            extracted_data["face_match"] = {"error": str(e)}
    
    # Step 5: Make final decision
    final_result = decision_engine.make_decision(
        quality_results=quality_results,
        extracted_data=extracted_data,
        format_issues=format_issues,
        intra_issues=intra_issues,
        cross_issues=cross_issues
    )
    
    # Add additional metadata
    final_result["pipeline_metadata"] = {
        "documents_processed": list(docs.keys()),
        "quality_summary": {
            doc: result["quality"] for doc, result in quality_results.items()
        },
        "extraction_summary": {
            doc: "success" if "extraction_error" not in data else "failed"
            for doc, data in extracted_data.items()
        }
    }
    
    return final_result