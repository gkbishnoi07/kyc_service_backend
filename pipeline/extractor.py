import base64
import json
import re
from typing import Dict, Any, Optional
from openai import OpenAI
from config import settings, DOCUMENT_CONFIGS

class DocumentExtractor:
    """
    Extracts structured information from document images using OpenAI Vision API
    """
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL
        self.min_confidence = settings.MIN_EXTRACTION_CONFIDENCE

    def encode_image(self, image_path: str) -> str:
        """Encode image as base64 data URL"""
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def safe_json_parse(self, text: str) -> Dict[str, Any]:
        """Safely parse JSON from LLM response"""
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("No JSON found in model output")
        return json.loads(match.group())

    def get_extraction_prompt(self, doc_type: str) -> str:
        """Generate extraction prompt based on document type"""
        
        prompts = {
            "aadhaar_front": """
You are an Aadhaar card front extraction system.

Extract ALL readable information from this document.
Even if text is blurry or partially visible, infer carefully.

IMPORTANT DATE RULES:
- If a full date is visible, return it in DD-MM-YYYY format
- If ONLY the year is visible, return the year as YYYY
- If no date information is visible, return null
- DO NOT guess day or month if they are not visible

Return STRICT JSON only.

Expected format:
{
  "name": "string or null",
  "date_of_birth": "string or null",
  "year_of_birth": "string or null",
  "gender": "string or null",
  "aadhaar_number": "string or null",
  "confidence": {
    "name": 0.0-1.0,
    "aadhaar_number": 0.0-1.0
  }
}

Rules:
- Aadhaar number format: XXXX XXXX XXXX
- Dates in DD-MM-YYYY
- Confidence values between 0 and 1
- If field not visible, return null
""",
            
            "aadhaar_back": """
You are an Aadhaar card back extraction system.

Extract address and other details from this document.

Return STRICT JSON only.

Expected format:
{
  "address": "string or null",
  "pincode": "string or null",
  "state": "string or null",
  "aadhaar_number": "string or null",
  "confidence": {
    "address": 0.0-1.0
  }
}

Rules:
- Pincode must be 6 digits
- If field not visible, return null
""",
            
            "driving_license": """
You are a Driving License extraction system.

Extract license details from this document.

IMPORTANT DATE RULES:
- If a full date is visible, return it in DD-MM-YYYY format
- If ONLY the year is visible, return the year as YYYY
- If no date information is visible, return null

Return STRICT JSON only.

Expected format:
{
  "name": "string or null",
  "license_number": "string or null",
  "date_of_birth": "string or null",
  "issue_date": "string or null",
  "validity_nt": "string or null",
  "validity_tr": "string or null",
  "issuing_authority": "string or null",
  "confidence": {
    "license_number": 0.0-1.0
  }
}

Rules:
- License number format: XX00 00000000000
- Dates in DD-MM-YYYY
- Check expiry dates carefully
- If field not visible, return null
""",
            
            "vehicle_plate_photo": """
You are a vehicle number plate extraction system.

Extract the vehicle registration number from the number plate.

Rules:
- Follow Indian vehicle number formats (e.g. MH12AB1234)
- Ignore spaces, hyphens, font styles
- If the vehicle number is not clearly visible, return null
- DO NOT guess or hallucinate

Return STRICT JSON only.

Expected format:
{
  "vehicle_number": "string or null",
  "confidence": 0.0-1.0
}
""",
            
            "rc": """
You are a Registration Certificate (RC) extraction system.

Extract the vehicle registration number from the RC document.

Rules:
- Look for the registration number field
- Follow Indian vehicle number formats
- If the vehicle number is not clearly visible, return null
- DO NOT guess or hallucinate

Return STRICT JSON only.

Expected format:
{
  "vehicle_number": "string or null",
  "confidence": 0.0-1.0
}
"""
        }
        
        return prompts.get(doc_type, "")

    def extract(self, image_path: str, doc_type: str) -> Dict[str, Any]:
        """Extract information from document image"""
        
        image_url = self.encode_image(image_path)
        prompt = self.get_extraction_prompt(doc_type)
        
        if not prompt:
            raise ValueError(f"Unknown document type: {doc_type}")
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        }
                    ]
                }
            ],
            max_tokens=600,
            temperature=0
        )
        
        try:
            extracted = self.safe_json_parse(response.choices[0].message.content)
            return extracted
        except Exception as e:
            # Return empty structure on parsing error
            config = DOCUMENT_CONFIGS.get(doc_type, {})
            result = {field: None for field in config.get("required_fields", [])}
            result.update({field: None for field in config.get("optional_fields", [])})
            result["confidence"] = {}
            result["extraction_error"] = str(e)
            return result