import base64
import json
import re
from openai import OpenAI
from config import settings


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode()}"


def safe_json_parse(text: str):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found in model output")
    return json.loads(match.group())


def llm_face_match(dl_image_path: str, selfie_image_path: str, model: str = None) -> dict:
    """Run an LLM-based face similarity check between DL photo and selfie."""
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    model = model or getattr(settings, "FACE_MODEL", settings.OPENAI_MODEL)

    dl_image = encode_image(dl_image_path)
    selfie_image = encode_image(selfie_image_path)

    prompt = """
You are an identity verification assistant.

You will be given two images:
1. A photo from a government-issued driving license
2. A selfie taken by a user

Task:
Determine whether both images appear to show the SAME PERSON.

Consider:
- Facial structure
- Eyes, nose, mouth
- Face shape
- Relative age
- Hairline (ignore hairstyle differences)
- Ignore lighting, image quality, or background differences

Return STRICT JSON ONLY.

Format:
{
  "same_person": true/false,
  "confidence": 0.0-1.0,
  "risk_level": "low" | "medium" | "high",
  "reasoning_summary": "short explanation"
}
"""

    # Use the same chat completions interface as other modules in this repo
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": dl_image}},
                    {"type": "image_url", "image_url": {"url": selfie_image}}
                ]
            }
        ],
        max_tokens=600,
        temperature=0
    )

    # Try to extract text content from the response similarly to extractor.py
    try:
        text = response.choices[0].message.content
    except Exception:
        text = str(response)

    # Attempt to parse and normalize fields
    try:
        parsed = safe_json_parse(text)
    except Exception as e:
        return {
            "same_person": None,
            "confidence": 0.0,
            "risk_level": None,
            "reasoning_summary": f"parse_error: {str(e)}",
            "raw_output": text
        }

    def _to_bool(val):
        if isinstance(val, bool):
            return val
        if val is None:
            return None
        try:
            if isinstance(val, (int, float)):
                return bool(val)
            s = str(val).strip().lower()
            if s in ("true", "yes", "y", "1"):
                return True
            if s in ("false", "no", "n", "0"):
                return False
        except Exception:
            pass
        return None

    def _to_conf(val):
        if val is None:
            return 0.0
        try:
            if isinstance(val, (int, float)):
                v = float(val)
            else:
                s = str(val).strip().replace('%', '')
                v = float(s)
            # If user returned percentage like 95, convert to 0.95
            if v > 1:
                v = v / 100.0
            return max(0.0, min(1.0, v))
        except Exception:
            return 0.0

    normalized = {
        "same_person": _to_bool(parsed.get("same_person")) if isinstance(parsed, dict) else None,
        "confidence": _to_conf(parsed.get("confidence") if isinstance(parsed, dict) else None),
        "risk_level": (str(parsed.get("risk_level")) if isinstance(parsed, dict) and parsed.get("risk_level") is not None else None),
        "reasoning_summary": (str(parsed.get("reasoning_summary")) if isinstance(parsed, dict) and parsed.get("reasoning_summary") is not None else ""),
        "raw_output": text
    }

    return normalized
