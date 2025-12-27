import re
from datetime import datetime, date
from typing import List, Dict, Any, Optional
from config import (
    AADHAAR_REGEX, DL_REGEX, PINCODE_REGEX, 
    INDIAN_PLATE_REGEX, DOCUMENT_CONFIGS
)

class DocumentChecks:
    """
    Performs various validation checks on extracted document data
    """
    
    def __init__(self):
        self.aadhaar_regex = re.compile(AADHAAR_REGEX)
        self.dl_regex = re.compile(DL_REGEX)
        self.pincode_regex = re.compile(PINCODE_REGEX)
        self.plate_regex = re.compile(INDIAN_PLATE_REGEX)

    def normalize_text(self, text: Optional[str]) -> str:
        """Normalize text for comparison"""
        if not text:
            return ""
        return re.sub(r"\s+", " ", text.strip().lower())

    def normalize_vehicle_number(self, number: Optional[str]) -> Optional[str]:
        """Normalize vehicle number by removing special characters"""
        if not number:
            return None
        return re.sub(r"[^A-Z0-9]", "", number.upper())

    def format_checks(self, extracted: Dict[str, Any]) -> List[str]:
        """Check format validity of all extracted fields"""
        issues = []

        # Aadhaar Front checks
        aadhaar_front = extracted.get("aadhaar_front", {})
        
        # Aadhaar number format
        aadhaar_number = aadhaar_front.get("aadhaar_number")
        if aadhaar_number and not self.aadhaar_regex.fullmatch(aadhaar_number):
            issues.append("INVALID_AADHAAR_FORMAT")

        # DOB format
        dob = aadhaar_front.get("date_of_birth")
        if dob:
            try:
                datetime.strptime(dob, "%d-%m-%Y")
            except ValueError:
                issues.append("INVALID_DOB_FORMAT")

        # Aadhaar Back checks
        aadhaar_back = extracted.get("aadhaar_back", {})
        
        # Pincode format
        pincode = aadhaar_back.get("pincode")
        if pincode and not self.pincode_regex.fullmatch(pincode):
            issues.append("INVALID_PINCODE")

        # Driving License checks
        dl = extracted.get("driving_license", {})
        
        # DL number format
        dl_number = dl.get("license_number")
        if dl_number:
            # Ignore masked values (contain Xs)
            if "X" in dl_number.upper():
                pass
            else:
                # Normalize: uppercase and remove spaces/hyphens for validation
                norm = re.sub(r"[\s-]+", "", dl_number.upper())
                # Accept either the configured spaced format or a compact version without space
                compact_pattern = re.compile(r"^[A-Z]{2}\d{13}$")
                if not (self.dl_regex.fullmatch(dl_number) or compact_pattern.fullmatch(norm)):
                    issues.append("INVALID_DL_FORMAT")

        # Check DL expiry
        expiry_issues = self._check_dl_expiry(dl)
        issues.extend(expiry_issues)

        # Vehicle plate checks
        plate = extracted.get("vehicle_plate_photo", {})
        plate_number = plate.get("vehicle_number")
        if plate_number:
            normalized_plate = self.normalize_vehicle_number(plate_number)
            if not self.plate_regex.fullmatch(normalized_plate):
                issues.append("INVALID_PLATE_FORMAT")

        # RC checks (if present)
        rc = extracted.get("rc", {})
        rc_number = rc.get("vehicle_number")
        if rc_number:
            normalized_rc = self.normalize_vehicle_number(rc_number)
            if not self.plate_regex.fullmatch(normalized_rc):
                issues.append("INVALID_RC_FORMAT")

        return issues

    def _check_dl_expiry(self, dl_data: Dict[str, Any]) -> List[str]:
        """Check if driving license is expired"""
        issues = []
        
        # Check both validity fields
        validity_nt = dl_data.get("validity_nt")
        validity_tr = dl_data.get("validity_tr")
        
        today = date.today()
        
        def is_expired(date_str: str) -> bool:
            if not date_str:
                return False
            try:
                expiry_date = datetime.strptime(date_str, "%d-%m-%Y").date()
                return expiry_date < today
            except ValueError:
                return False
        
        # Check non-transport validity
        if validity_nt:
            if is_expired(validity_nt):
                issues.append("DL_EXPIRED_NT")
        
        # Check transport validity
        if validity_tr:
            if is_expired(validity_tr):
                issues.append("DL_EXPIRED_TR")
        
        # If no expiry dates found, add review flag
        if not validity_nt and not validity_tr:
            issues.append("DL_EXPIRY_NOT_READABLE")
        
        return issues

    def intra_document_consistency(self, extracted: Dict[str, Any]) -> List[str]:
        """Check consistency within the same document type"""
        issues = []

        # Aadhaar number front vs back
        a_front = extracted.get("aadhaar_front", {}).get("aadhaar_number")
        a_back = extracted.get("aadhaar_back", {}).get("aadhaar_number")

        if a_front and a_back and a_front != a_back:
            issues.append("AADHAAR_FRONT_BACK_MISMATCH")

        return issues

    def cross_document_consistency(self, extracted: Dict[str, Any]) -> List[str]:
        """Check consistency across different document types"""
        issues = []

        # Name consistency: Aadhaar vs DL
        name_a = extracted.get("aadhaar_front", {}).get("name")
        name_d = extracted.get("driving_license", {}).get("name")

        if name_a and name_d:
            if self.normalize_text(name_a) != self.normalize_text(name_d):
                issues.append("NAME_MISMATCH")

        # DOB consistency: Aadhaar vs DL
        dob_a = extracted.get("aadhaar_front", {}).get("date_of_birth")
        dob_d = extracted.get("driving_license", {}).get("date_of_birth")

        if dob_a and dob_d and dob_a != dob_d:
            issues.append("DOB_MISMATCH")

        # Vehicle number consistency: Plate vs RC
        plate = extracted.get("vehicle_plate_photo", {}).get("vehicle_number")
        rc = extracted.get("rc", {}).get("vehicle_number")

        if plate and rc:
            normalized_plate = self.normalize_vehicle_number(plate)
            normalized_rc = self.normalize_vehicle_number(rc)
            
            if normalized_plate != normalized_rc:
                issues.append("PLATE_RC_MISMATCH")

        return issues

    def plate_ocr_validation(self, plate_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate extracted plate number with OCR confidence"""
        plate_number = plate_data.get("vehicle_number")
        confidence = plate_data.get("confidence", 0)
        
        if not plate_number:
            return {
                "plate_number": None,
                "plate_valid": False,
                "confidence": 0,
                "reason": "NO_PLATE_DETECTED"
            }
        
        normalized_plate = self.normalize_vehicle_number(plate_number)
        is_valid_format = self.plate_regex.fullmatch(normalized_plate)
        
        return {
            "plate_number": normalized_plate,
            "plate_valid": bool(is_valid_format),
            "confidence": confidence,
            "reason": "VALID" if is_valid_format else "INVALID_FORMAT"
        }