from typing import List, Dict, Any, Tuple
import re
from datetime import datetime
from config import settings

class DecisionEngine:
    """
    Makes final verification decisions based on all checks and quality assessments
    Implements v1 policy rules
    """
    
    def __init__(self):
        self.quality_threshold_proceed = settings.QUALITY_THRESHOLD_PROCEED
        self.quality_threshold_caution = settings.QUALITY_THRESHOLD_CAUTION

    def mask_aadhaar(self, aadhaar: str) -> str:
        """Mask Aadhaar number showing only last 4 digits"""
        if not aadhaar:
            return None
        # Remove spaces for processing
        clean = aadhaar.replace(" ", "")
        if len(clean) != 12:
            return "INVALID_FORMAT"
        # Show only last 4 digits
        return f"XXXX XXXX {clean[-4:]}"

    def mask_driving_license(self, dl_number: str) -> str:
        """Mask driving license number"""
        if not dl_number:
            return None
        # Show first 2 and last 4 characters
        if len(dl_number) > 6:
            return f"{dl_number[:2]}XXXX{dl_number[-4:]}"
        return "XXXX"

    def mask_name(self, name: str) -> str:
        """Mask name showing only first character and last name"""
        if not name:
            return None
        parts = name.strip().split()
        if len(parts) == 1:
            return f"{parts[0][0]}XXXX"
        return f"{parts[0][0]}XXXX {parts[-1]}"

    def calculate_confidence(self, 
                           quality_scores: Dict[str, float],
                           extraction_confidences: Dict[str, Dict[str, float]],
                           issues: List[str]) -> float:
        """Calculate overall confidence score"""
        
        # Base confidence from image quality
        quality_values = list(quality_scores.values())
        avg_quality = sum(quality_values) / len(quality_values) if quality_values else 0
        
        # Penalty for issues
        issue_penalty = len(issues) * 0.1
        
        # Penalty for low extraction confidence
        extraction_penalties = []
        for doc_type, doc_conf in extraction_confidences.items():
            if isinstance(doc_conf, dict):
                # Handle dictionary of confidence scores (Aadhaar, DL)
                for field_conf in doc_conf.values():
                    if isinstance(field_conf, (int, float)) and field_conf < settings.MIN_EXTRACTION_CONFIDENCE:
                        extraction_penalties.append(0.1)
            elif isinstance(doc_conf, (int, float)):
                # Handle single float confidence (Plate, RC)
                # Use stricter threshold for plates if desired, or standard one
                threshold = settings.MIN_PLATE_CONFIDENCE if "plate" in doc_type or "rc" in doc_type else settings.MIN_EXTRACTION_CONFIDENCE
                if doc_conf < threshold:
                    extraction_penalties.append(0.1)
        
        total_penalty = issue_penalty + sum(extraction_penalties)
        
        # Calculate final confidence
        confidence = max(0, avg_quality - total_penalty)
        
        return round(confidence, 2)

    def make_decision(self,
                     quality_results: Dict[str, Dict[str, Any]],
                     extracted_data: Dict[str, Any],
                     format_issues: List[str],
                     intra_issues: List[str],
                     cross_issues: List[str]) -> Dict[str, Any]:
        """
        Make final decision based on v1 policy rules
        
        Rules v1:
        - bad quality / unreadable → REUPLOAD
        - format/intra issues → REUPLOAD (or NEEDS_REVIEW if minor)
        - cross mismatch (name/dob) → NEEDS_REVIEW (not reject)
        - expired DL → REJECT
        - all good → VERIFIED
        """
        
        all_issues = format_issues + intra_issues + cross_issues
        
        # Check for hard rejects first
        if "DL_EXPIRED_NT" in all_issues or "DL_EXPIRED_TR" in all_issues:
            return self._build_response(
                status="REJECT",
                confidence=0.0,
                reasons=["Driving license expired"],
                issues=all_issues,
                quality_results=quality_results,
                extracted_data=extracted_data,
                format_issues=format_issues,
                intra_issues=intra_issues,
                cross_issues=cross_issues
            )

        # Face-match mandatory check: reject if face match exists and confidence below threshold
        face_match = extracted_data.get("face_match")
        try:
            face_conf = None
            if isinstance(face_match, dict) and "confidence" in face_match:
                face_conf = float(face_match.get("confidence") or 0.0)
        except Exception:
            face_conf = None

        if face_conf is not None:
            if face_conf < settings.FACE_MIN_CONFIDENCE:
                return self._build_response(
                    status="REJECT",
                    confidence=0.0,
                    reasons=[f"Face match confidence below threshold ({face_conf})"],
                    issues=all_issues,
                    quality_results=quality_results,
                    extracted_data=extracted_data,
                    format_issues=format_issues,
                    intra_issues=intra_issues,
                    cross_issues=cross_issues
                )
        
        # Check for reupload requirements
        reupload_docs = []
        for doc_type, result in quality_results.items():
            # The RC document is optional; do not require reupload for it
            if doc_type == "rc":
                continue
            if result.get("quality") == "bad":
                reupload_docs.append(doc_type)
        
        if reupload_docs:
            return self._build_response(
                status="REUPLOAD",
                confidence=0.0,
                reasons=[f"Poor quality images: {', '.join(reupload_docs)}"],
                issues=all_issues,
                quality_results=quality_results,
                extracted_data=extracted_data,
                format_issues=format_issues,
                intra_issues=intra_issues,
                cross_issues=cross_issues
            )

        # Face-match explicit mismatch: ask user to reupload relevant documents
        face_match = extracted_data.get("face_match")
        same_person = None
        if isinstance(face_match, dict) and "same_person" in face_match:
            same_person = face_match.get("same_person")

        if same_person is False:
            # Suggest reupload of selfie and driving_license
            suggested = ["selfie", "driving_license"]
            return self._build_response(
                status="REUPLOAD",
                confidence=0.0,
                reasons=["Face mismatch between driving license and selfie"],
                issues=all_issues,
                quality_results=quality_results,
                extracted_data=extracted_data,
                format_issues=format_issues,
                intra_issues=intra_issues,
                cross_issues=cross_issues,
                suggested_reuploads=suggested
            )
        
        # Check for format/intra issues
        if format_issues or intra_issues:
            # Determine if it's minor or major
            critical_format = [
                "INVALID_AADHAAR_FORMAT", "INVALID_DL_FORMAT", 
                "INVALID_PINCODE", "AADHAAR_FRONT_BACK_MISMATCH"
            ]
            
            has_critical = any(issue in critical_format for issue in format_issues + intra_issues)
            
            if has_critical:
                return self._build_response(
                    status="REUPLOAD",
                    confidence=0.3,
                    reasons=["Document format or consistency issues"],
                    issues=all_issues,
                    quality_results=quality_results,
                    extracted_data=extracted_data,
                    format_issues=format_issues,
                    intra_issues=intra_issues,
                    cross_issues=cross_issues
                )
            else:
                return self._build_response(
                    status="NEEDS_REVIEW",
                    confidence=0.5,
                    reasons=["Minor document issues require review"],
                    issues=all_issues,
                    quality_results=quality_results,
                    extracted_data=extracted_data,
                    format_issues=format_issues,
                    intra_issues=intra_issues,
                    cross_issues=cross_issues
                )
        
        # Check for cross-document mismatches
        if cross_issues:
            return self._build_response(
                status="NEEDS_REVIEW",
                confidence=0.4,
                reasons=["Cross-document inconsistencies"],
                issues=all_issues,
                quality_results=quality_results,
                extracted_data=extracted_data,
                format_issues=format_issues,
                intra_issues=intra_issues,
                cross_issues=cross_issues
            )
        
        # Check for DL expiry not readable
        if "DL_EXPIRY_NOT_READABLE" in all_issues:
            return self._build_response(
                status="NEEDS_REVIEW",
                confidence=0.6,
                reasons=["Driving license expiry date not readable"],
                issues=all_issues,
                quality_results=quality_results,
                extracted_data=extracted_data,
                format_issues=format_issues,
                intra_issues=intra_issues,
                cross_issues=cross_issues
            )
        
        # All checks passed
        return self._build_response(
            status="VERIFIED",
            confidence=0.9,
            reasons=["All verification checks passed"],
            issues=[],
            quality_results=quality_results,
            extracted_data=extracted_data,
            format_issues=format_issues,
            intra_issues=intra_issues,
            cross_issues=cross_issues
        )

    def _build_response(self,
                        status: str,
                        confidence: float,
                        reasons: List[str],
                        issues: List[str],
                        quality_results: Dict[str, Dict[str, Any]],
                        extracted_data: Dict[str, Any],
                        format_issues: List[str] = None,
                        intra_issues: List[str] = None,
                        cross_issues: List[str] = None,
                        suggested_reuploads: List[str] = None) -> Dict[str, Any]:
        """Build standardized response with masked data"""
        
        # Mask sensitive data
        masked_extracted = self._mask_extracted_data(extracted_data)
        
        # Build plate info
        plate_info = {}
        if "vehicle_plate_photo" in extracted_data:
            plate_data = extracted_data["vehicle_plate_photo"]
            plate_info = {
                "plate_number": plate_data.get("vehicle_number"),
                "plate_valid": bool(plate_data.get("vehicle_number")),
                "confidence": plate_data.get("confidence", 0)
            }
        
        # Calculate quality scores
        quality_scores = {
            doc: result.get("risk_score", 0) 
            for doc, result in quality_results.items()
        }
        
        # Calculate extraction confidences
        extraction_confidences = {}
        for doc_type, data in extracted_data.items():
            if "confidence" in data:
                extraction_confidences[doc_type] = data["confidence"]
        
        # Calculate final confidence
        final_confidence = self.calculate_confidence(
            quality_scores, extraction_confidences, issues
        )
        
        return {
            "status": status,
            "confidence": final_confidence,
            "reasons": reasons,
            "signals": {
                "quality": quality_results,
                "issues": issues,
                "checks": {
                    "format": format_issues or [],
                    "intra": intra_issues or [],
                    "cross": cross_issues or []
                },
                "plate": plate_info,
                "face_match": extracted_data.get("face_match"),
                "suggested_reuploads": suggested_reuploads or []
            },
            "extracted": masked_extracted
        }

    def _mask_extracted_data(self, extracted: Dict[str, Any]) -> Dict[str, Any]:
        """Mask all sensitive extracted data"""
        masked = {}
        def _clean_str(val: str) -> str:
            if not isinstance(val, str):
                return val
            # Keep only the first line to avoid trailing model commentary
            first_line = val.splitlines()[0].strip()
            # Collapse multiple whitespace into single space
            return re.sub(r"\s+", " ", first_line)

        for doc_type, data in extracted.items():
            masked_data = {}
            # Clean all string fields first
            if isinstance(data, dict):
                for k, v in list(data.items()):
                    if isinstance(v, str):
                        data[k] = _clean_str(v)

            # Mask specific fields based on document type
            if doc_type in ["aadhaar_front", "aadhaar_back"]:
                if "name" in data:
                    masked_data["name"] = self.mask_name(data["name"])
                if "aadhaar_number" in data:
                    masked_data["aadhaar_number"] = self.mask_aadhaar(data["aadhaar_number"])
                # Keep non-sensitive fields as is
                for field in ["gender", "date_of_birth", "year_of_birth"]:
                    if field in data:
                        masked_data[field] = data[field]
            
            elif doc_type == "driving_license":
                if "name" in data:
                    masked_data["name"] = self.mask_name(data["name"])
                if "license_number" in data:
                    masked_data["license_number"] = self.mask_driving_license(data["license_number"])
                # Keep non-sensitive fields
                for field in ["date_of_birth", "issue_date", "validity_nt", "validity_tr", "issuing_authority"]:
                    if field in data:
                        masked_data[field] = data[field]
            
            elif doc_type in ["vehicle_plate_photo", "rc"]:
                # Vehicle numbers are not highly sensitive, but we can keep them
                masked_data = data.copy()
            
            elif doc_type == "face_match":
                # Keep face match results intact (contains booleans/numbers/short text)
                masked_data = data.copy() if isinstance(data, dict) else data
            
            else:
                # For unknown document types, mask all string fields
                for field, value in data.items():
                    if isinstance(value, str) and len(value) > 2:
                        masked_data[field] = f"{value[0]}XXXX"
                    else:
                        masked_data[field] = value
            
            masked[doc_type] = masked_data
        
        return masked