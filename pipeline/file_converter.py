import os
import uuid
from typing import List
from PIL import Image
import pillow_heif
from pdf2image import convert_from_path

pillow_heif.register_heif_opener()

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic"}
PDF_EXT = ".pdf"


def convert_to_images(input_path: str, output_dir: str) -> List[str]:
    """
    Converts input file (image / HEIC / PDF) into JPEG images.
    Returns list of image paths.
    """
    ext = os.path.splitext(input_path)[1].lower()
    os.makedirs(output_dir, exist_ok=True)

    output_paths = []

    # -------- Case 1: Normal image or HEIC --------
    if ext in SUPPORTED_IMAGE_EXTS:
        img = Image.open(input_path).convert("RGB")
        out_path = os.path.join(output_dir, f"{uuid.uuid4().hex}.jpg")
        img.save(out_path, "JPEG", quality=95)
        return [out_path]

    # -------- Case 2: PDF --------
    if ext == PDF_EXT:
        pages = convert_from_path(input_path, dpi=300)
        for i, page in enumerate(pages):
            out_path = os.path.join(
                output_dir, f"{uuid.uuid4().hex}_page{i+1}.jpg"
            )
            page.convert("RGB").save(out_path, "JPEG", quality=95)
            output_paths.append(out_path)
        return output_paths

    raise ValueError(f"Unsupported file type: {ext}")