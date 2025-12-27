import cv2
import numpy as np
from typing import Tuple, List, Dict, Any
from config import settings

class ImageQualityGate:
    """
    Evaluates image quality for KYC documents
    Returns quality assessment with actionable recommendations
    """
    
    def __init__(self):
        self.min_width = settings.MIN_IMAGE_WIDTH
        self.min_height = settings.MIN_IMAGE_HEIGHT
        self.blur_threshold = settings.BLUR_THRESHOLD
        self.min_brightness = settings.MIN_BRIGHTNESS
        self.max_brightness = settings.MAX_BRIGHTNESS
        self.min_contrast = settings.MIN_CONTRAST
        self.quality_threshold_proceed = settings.QUALITY_THRESHOLD_PROCEED
        self.quality_threshold_caution = settings.QUALITY_THRESHOLD_CAUTION

    def load_image(self, image_path: str) -> Tuple[np.ndarray, List[str]]:
        """Load image from file path"""
        img = cv2.imread(image_path)
        if img is None:
            return None, ["Image could not be loaded"]
        return img, []

    def check_resolution(self, img: np.ndarray) -> Tuple[bool, str]:
        """Check if image meets minimum resolution requirements"""
        h, w = img.shape[:2]
        if w < self.min_width or h < self.min_height:
            return False, f"Low resolution ({w}x{h})"
        return True, None

    def check_blur(self, img: np.ndarray) -> Tuple[bool, str]:
        """Check for image blur using Laplacian variance"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if score < self.blur_threshold:
            return False, f"Blur detected (score={score:.1f})"
        return True, None

    def check_brightness(self, img: np.ndarray) -> Tuple[bool, str]:
        """Check if image brightness is within acceptable range"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean = gray.mean()
        if mean < self.min_brightness:
            return False, f"Too dark (mean={mean:.1f})"
        if mean > self.max_brightness:
            return False, f"Too bright (mean={mean:.1f})"
        return True, None

    def check_contrast(self, img: np.ndarray) -> Tuple[bool, str]:
        """Check if image has sufficient contrast"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        std = gray.std()
        if std < self.min_contrast:
            return False, f"Low contrast (std={std:.1f})"
        return True, None

    def check_text_likelihood(self, img: np.ndarray) -> Tuple[bool, str]:
        """Estimate if image contains readable text using edge detection"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        density = edges.mean()
        if density < 2.0:
            return False, "Low text likelihood"
        return True, None

    def evaluate(self, image_path: str) -> Dict[str, Any]:
        """
        Evaluate image quality and return assessment
        Returns dict with quality, risk_score, signals, and recommended_action
        """
        img, failures = self.load_image(image_path)

        # Hard fail: image unreadable
        if img is None:
            return {
                "quality": "bad",
                "risk_score": 0.0,
                "signals": failures,
                "recommended_action": "reject"
            }

        # Hard fail: extremely small images
        h, w = img.shape[:2]
        if w < 300 or h < 300:
            return {
                "quality": "bad",
                "risk_score": 0.0,
                "signals": ["Extremely low resolution"],
                "recommended_action": "reject"
            }

        checks = [
            self.check_resolution,
            self.check_blur,
            self.check_brightness,
            self.check_contrast,
            self.check_text_likelihood
        ]

        passed = 0
        for check in checks:
            ok, msg = check(img)
            if ok:
                passed += 1
            else:
                failures.append(msg)

        risk_score = passed / len(checks)

        # Determine quality tier and action
        if risk_score >= self.quality_threshold_proceed:
            quality = "good"
            action = "proceed"
        elif risk_score >= self.quality_threshold_caution:
            quality = "risky"
            action = "proceed_with_caution"
        else:
            quality = "bad"
            action = "reject"

        return {
            "quality": quality,
            "risk_score": round(risk_score, 2),
            "signals": failures,
            "recommended_action": action
        }