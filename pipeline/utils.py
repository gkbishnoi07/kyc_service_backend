import os
import requests
import tempfile
from typing import Optional
from urllib.parse import urlparse

def download_image_from_url(url: str, save_path: Optional[str] = None) -> str:
    """
    Download an image from URL and save to local path
    
    Args:
        url: Image URL to download
        save_path: Optional path to save the image. If None, creates temp file
    
    Returns:
        Local path to downloaded image
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        if save_path is None:
            # Create temporary file
            fd, save_path = tempfile.mkstemp(suffix='.jpg')
            os.close(fd)
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        return save_path
    
    except Exception as e:
        raise Exception(f"Failed to download image from {url}: {str(e)}")

def is_valid_url(url: str) -> bool:
    """Check if string is a valid URL"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def cleanup_temp_file(file_path: str) -> None:
    """Safely remove temporary file"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except:
        pass

def get_file_extension(filename: str) -> str:
    """Get file extension from filename"""
    return os.path.splitext(filename)[1].lower()

def is_image_file(filename: str) -> bool:
    """Check if file has image extension"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    return get_file_extension(filename) in image_extensions