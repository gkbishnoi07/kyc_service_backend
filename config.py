from pydantic_settings import BaseSettings
from typing import Dict, Any

class Settings(BaseSettings):
    # OpenAI Configuration
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4.1-mini"
    # Model used specifically for face similarity scoring (responses API)
    FACE_MODEL: str = "gpt-4.1-mini"
    # Minimum face-match confidence below which the application is rejected
    FACE_MIN_CONFIDENCE: float = 0.6
    
    # Image Quality Thresholds
    MIN_IMAGE_WIDTH: int = 800
    MIN_IMAGE_HEIGHT: int = 600
    BLUR_THRESHOLD: float = 100
    MIN_BRIGHTNESS: int = 50
    MAX_BRIGHTNESS: int = 200
    MIN_CONTRAST: int = 30
    
    # Confidence Thresholds
    MIN_EXTRACTION_CONFIDENCE: float = 0.7
    MIN_PLATE_CONFIDENCE: float = 0.8
    
    # Decision Rules
    QUALITY_THRESHOLD_PROCEED: float = 0.8
    QUALITY_THRESHOLD_CAUTION: float = 0.4
    
    class Config:
        env_file = ".env"

settings = Settings()

# Document type configurations
DOCUMENT_CONFIGS = {
    "aadhaar_front": {
        "required_fields": ["name", "date_of_birth", "aadhaar_number"],
        "optional_fields": ["gender", "year_of_birth"]
    },
    "aadhaar_back": {
        "required_fields": ["address", "pincode"],
        "optional_fields": ["state", "aadhaar_number"]
    },
    "driving_license": {
        "required_fields": ["name", "license_number", "date_of_birth"],
        "optional_fields": ["issue_date", "validity_nt", "validity_tr", "issuing_authority"]
    },
    "vehicle_plate_photo": {
        "required_fields": ["vehicle_number"],
        "optional_fields": []
    },
    "rc": {
        "required_fields": ["vehicle_number"],
        "optional_fields": []
    }
}

# Indian vehicle number plate regex pattern
INDIAN_PLATE_REGEX = r"^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{1,4}$"

# Aadhaar number format
AADHAAR_REGEX = r"^\d{4}\s\d{4}\s\d{4}$"

# DL number format
DL_REGEX = r"^[A-Z]{2}\d{2}\s\d{11}$"

# Pincode format
PINCODE_REGEX = r"^\d{6}$"