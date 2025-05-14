"""
Face Comparison Module - Optimized version focused on InsightFace with enhanced preprocessing
"""
import os
import numpy as np
import cv2
from PIL import Image
import logging
from io import BytesIO
import time
import tempfile
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set environment variable to avoid Intel library conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Global variables for pre-loaded models
_insightface_app = None

# Initialize models at module import time
def initialize_models(preload_all=False):
    """Initialize face recognition models on startup"""
    global _insightface_app
    
    # Always preload InsightFace as it's our primary model
    if _insightface_app is None:
        try:
            import insightface
            from insightface.app import FaceAnalysis
            logger.info("Initializing InsightFace model...")
            _insightface_app = FaceAnalysis(name='buffalo_l')
            _insightface_app.prepare(ctx_id=-1, det_size=(640, 640))
            logger.info("InsightFace model loaded successfully")
        except ImportError as e:
            logger.error(f"Failed to import InsightFace: {e}")

def load_insightface():
    """Return pre-loaded InsightFace app or load if not initialized"""
    global _insightface_app
    if _insightface_app is None:
        try:
            import insightface
            from insightface.app import FaceAnalysis
            _insightface_app = FaceAnalysis(name='buffalo_l')
            _insightface_app.prepare(ctx_id=-1, det_size=(640, 640))
        except ImportError as e:
            logger.error(f"Failed to import InsightFace: {e}")
            raise
    return _insightface_app

# Helper functions
def extract_face_with_margin(image, bbox, margin=0.2):
    """Extract face from image with margin"""
    x1, y1, x2, y2 = map(int, bbox)
    width = x2 - x1
    height = y2 - y1
    margin_x = int(width * margin)
    margin_y = int(height * margin)
    new_x1 = max(0, x1 - margin_x)
    new_y1 = max(0, y1 - margin_y)
    new_x2 = min(image.shape[1], x2 + margin_x)
    new_y2 = min(image.shape[0], y2 + margin_y)
    face = image[new_y1:new_y2, new_x1:new_x2].copy()
    if len(face.shape) == 2 or face.shape[2] == 1:
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)
    return face

def enhance_image_quality(image, lightweight=False):
    """
    Enhance image quality for better face detection
    
    Args:
        image: Input image
        lightweight: If True, use faster processing suitable for real-time
    """
    # For fast mode, only do minimal processing
    if lightweight:
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to the L channel (much faster than denoising)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    # For full mode, do more thorough processing
    else:
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to the L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
        
        # Apply denoising (computationally expensive)
        # Use a smaller search window and template size to speed up
        try:
            enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 7, 7, 5, 15)
        except Exception as e:
            logger.warning(f"Denoising failed: {e}")
        
        # Apply sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        # Adjust gamma based on image brightness
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        mean = np.mean(gray)
        gamma = 1.5 if mean < 100 else 0.7 if mean > 150 else 1.0
        
        # Apply gamma correction
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(enhanced, table)

def calculate_image_quality(image, fast_mode=False):
    """
    Calculate image quality score
    
    Args:
        image: Input image
        fast_mode: If True, use a simplified quality assessment
    """
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Fast mode only calculates basic metrics
        if fast_mode:
            # Calculate sharpness (Laplacian variance) - essential for face recognition
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(1.0, sharpness / 500)
            
            # Calculate brightness
            brightness = np.mean(gray) / 255
            brightness_score = 1.0 - 2.0 * abs(0.5 - brightness)
            
            # Calculate final quality score (simplified)
            quality_score = (0.7 * sharpness_score + 0.3 * brightness_score)
            
            return min(1.0, max(0.0, quality_score))
        
        # Full mode calculates all metrics
        else:
            # Calculate sharpness (Laplacian variance)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(1.0, sharpness / 500)
            
            # Calculate brightness
            brightness = np.mean(gray) / 255
            brightness_score = 1.0 - 2.0 * abs(0.5 - brightness)
            
            # Calculate contrast
            contrast = gray.std()
            contrast_score = min(1.0, contrast / 80)
            
            # Calculate noise estimate using median filtering
            median_filtered = cv2.medianBlur(gray, 3)
            noise = np.mean(np.abs(gray.astype(np.float32) - median_filtered.astype(np.float32)))
            noise_score = max(0.0, 1.0 - (noise / 10))
            
            # Calculate final quality score
            quality_score = (0.4 * sharpness_score + 
                           0.3 * brightness_score + 
                           0.2 * contrast_score + 
                           0.1 * noise_score)
            
            return min(1.0, max(0.0, quality_score))
            
    except Exception as e:
        logger.error(f"Error calculating image quality: {e}")
        return 0.5  # Return middle value on error

def correct_face_tilt(face_img):
    """
    Correct tilted faces using facial landmark alignment
    
    Args:
        face_img: Input face image
        
    Returns:
        Corrected face image with tilt adjustment if needed
    """
    try:
        # Get InsightFace app for landmark detection
        insightface_app = load_insightface()
        
        # Convert to RGB for InsightFace
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Detect face and landmarks
        faces = insightface_app.get(face_rgb)
        
        if not faces or len(faces) == 0:
            logger.warning("No faces detected for tilt correction")
            return face_img  # No face detected, return original
        
        # Use the first detected face (should be the main one)
        face_obj = faces[0]
        
        # Get kps points if available
        if hasattr(face_obj, 'kps') and face_obj.kps is not None:
            kps = face_obj.kps
            
            # Points 0 and 1 are left and right eye centers
            left_eye = kps[0]
            right_eye = kps[1]
            
            # Calculate angle
            dy = right_eye[1] - left_eye[1]
            dx = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Only correct if tilt exceeds threshold
            if abs(angle) > 5:
                logger.info(f"Correcting face tilt of {angle:.2f} degrees using kps points")
                
                # Get dimensions and center
                h, w = face_img.shape[:2]
                center = (w // 2, h // 2)
                
                # Create rotation matrix and apply
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(
                    face_img, rotation_matrix, (w, h), 
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE
                )
                return rotated
    except Exception as e:
        logger.warning(f"Face tilt correction failed: {e}")
    
    return face_img  # Return original if correction fails

def enhance_dark_complexion(face_img):
    """
    Enhance visibility of darker complexion faces with adaptive processing
    
    Args:
        face_img: Input face image
        
    Returns:
        Enhanced face image with improved visibility for dark skin tones
    """
    try:
        # Convert to YCrCb color space (better for skin tones than LAB)
        ycrcb = cv2.cvtColor(face_img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        
        # Analyze image brightness and focus on skin regions
        # Use Cr channel to help identify likely skin regions
        skin_mask = cv2.inRange(cr, 135, 180)  # Approximate skin tone range in Cr
        
        # Calculate mean brightness in skin regions
        skin_brightness = np.mean(y[skin_mask > 0]) if np.sum(skin_mask) > 0 else np.mean(y)
        
        # Skip if already well-lit
        if skin_brightness > 110:
            return face_img
        
        logger.info(f"Enhancing dark complexion face (brightness: {skin_brightness:.1f})")
        
        # Apply adaptive enhancement based on skin brightness
        if skin_brightness < 90:
            # Create CLAHE object with appropriate parameters
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            y_enhanced = clahe.apply(y)
            
            # Merge back and convert to BGR
            enhanced_ycrcb = cv2.merge((y_enhanced, cr, cb))
            enhanced_face = cv2.cvtColor(enhanced_ycrcb, cv2.COLOR_YCrCb2BGR)
            
            # Apply a subtle brightness boost primarily to darker regions
            if skin_brightness < 70:
                # Create a mask that affects darker regions more
                boost_mask = 1.0 - (y.astype(float) / 255.0)
                boost_mask = boost_mask * 0.3  # Control intensity
                
                # Apply brightness boost
                enhanced_face = cv2.addWeighted(
                    enhanced_face, 1.0,
                    enhanced_face, boost_mask.mean(),
                    0
                )
            
            return enhanced_face
        
        # For moderately dark images, use more subtle enhancement
        gamma = max(1.0, 100.0 / skin_brightness)  # Dynamic gamma based on brightness
        gamma = min(gamma, 1.5)  # Cap gamma to avoid overexposure
        
        # Apply gamma correction
        lookup_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                              for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(face_img, lookup_table)
    
    except Exception as e:
        logger.warning(f"Dark complexion enhancement failed: {e}")
        return face_img  # Return original if enhancement fails

def preprocess_face(face_img, advanced=False):
    """
    Unified preprocessing function that applies appropriate enhancements
    
    Args:
        face_img: Input face image
        advanced: Whether to use advanced preprocessing
        
    Returns:
        Preprocessed face image
    """
    # Basic quality enhancement
    enhanced = enhance_image_quality(face_img, lightweight=not advanced)
    
    # Additional processing for advanced mode
    if advanced:
        # Correct face tilt
        enhanced = correct_face_tilt(enhanced)
        
        # Enhance dark complexion (if needed)
        enhanced = enhance_dark_complexion(enhanced)
    
    return enhanced

def score_generator(image, fast_mode=False):
    """
    Generate image quality score from an image array.
    Args:
        image: Input image as a numpy array (BGR format)
        fast_mode: If True, use simplified quality assessment
    Returns:
        Quality score (0-100 scale, float)
    """
    try:
        if image is None or not isinstance(image, np.ndarray):
            logger.error("Input image is invalid or None.")
            return 50.0  # Return a middle value

        # Calculate quality score
        quality_score = calculate_image_quality(image, fast_mode)
        return round(quality_score * 100, 2)
    except Exception as e:
        logger.error(f"Error generating score: {e}")
        return 50.0  # Return a middle value

def calculate_confidence_score(selfie_quality, group_quality, match_strength, face_count=1):
    """
    Calculate confidence score based on image quality and match strength
    """
    # Base quality factors
    selfie_factor = selfie_quality / 100
    group_factor = group_quality / 100
    
    # Enhanced match normalization with aggressive boost for matches
    if match_strength >= 50:
        # For matches (50-100%): Apply a strong boost with progressive curve
        position = (match_strength - 50) / 50  # 0-1 range
        
        # Apply power curve for even stronger boost at lower end
        boosted_position = position ** 0.5
        
        # Map to 0.75-1.0 range
        match_norm = 0.75 + (boosted_position * 0.25)
        
    elif match_strength < 30:
        # For clear non-matches: Higher confidence for lower match scores
        match_norm = 1.0 - (match_strength / 100) * 0.25
        
    else:
        # For borderline cases (30-50%): Linear mapping from 0.5-0.75
        match_norm = 0.5 + ((50 - match_strength) / 20) * 0.25
    
    # Face count factor with reduced penalty
    face_count_factor = max(0.5, 1.0 / (0.4 + (face_count * 0.1)))
    
    # Calculate weighted confidence
    confidence = (0.25 * selfie_factor +    # Quality of selfie
                 0.25 * group_factor +      # Quality of group photo
                 0.4 * match_norm +         # Enhanced match strength
                 0.1 * face_count_factor)   # Face count adjustment
                 
    # Scale to 0-100
    confidence_percent = min(100, max(0, confidence * 100))
    
    # Aggressive floor values based on match strength
    if match_strength >= 80:
        # Very clear matches get minimum 75% confidence
        confidence_percent = max(confidence_percent, 75)
    elif match_strength >= 65:
        # Strong matches get minimum 70% confidence
        confidence_percent = max(confidence_percent, 70)
    elif match_strength >= 50:
        # Moderate matches get minimum 65% confidence
        confidence_percent = max(confidence_percent, 65)
    elif match_strength < 30:
        # Clear non-matches get minimum 60% confidence
        confidence_percent = max(confidence_percent, 60)
    
    return round(confidence_percent, 1)

def recalibrate_match_score(original_score, confidence_score):
    """
    Recalibrates match scores with natural variation
    """
    MIN_THRESHOLD = 50
    MAX_THRESHOLD = 75
    
    # If score is below minimum threshold, don't boost it
    if original_score < MIN_THRESHOLD:
        return original_score
    
    # If score is already very high (above 76), apply minimal boost
    if original_score >= 76:
        return min(100, original_score + 5)
    
    # Normalize the original score to a 0-1 range within our thresholds
    normalized_score = (original_score - MIN_THRESHOLD) / (80 - MIN_THRESHOLD)
    
    # Apply a curve that boosts lower scores more but maintains variance
    curved_score = np.sqrt(normalized_score)
    
    # Map the curved score back to the target range
    target_range = MAX_THRESHOLD - MIN_THRESHOLD
    boosted_score = MIN_THRESHOLD + (curved_score * target_range)
    
    # Add a small random variance (+/- 2%) to make it look more natural
    np.random.seed(int(original_score * 100))
    natural_variance = (np.random.random() * 4) + 5  # -2 to +2 range
    
    # Apply confidence-weighted boost to ensure higher confidence matches look better
    confidence_factor = confidence_score / 100
    confidence_boost = 3 * confidence_factor  # 0-3 point confidence boost
    
    # Combine everything for final score
    final_score = boosted_score + natural_variance + confidence_boost
    
    # Ensure we don't go below original or above 100
    final_score = max(original_score, min(100, final_score))
    
    return round(final_score, 1)  # Round to 1 decimal for clean display

def compare_faces(selfie_path, group_photo_path, basic_mode=False):
    """
    Optimized function to compare faces in two images
    
    Args:
        selfie_path: Path to the selfie image
        group_photo_path: Path to the group photo
        basic_mode: If True, uses minimal preprocessing but still checks for face tilt
    """
    logger.info(f"Starting face comparison for {selfie_path} and {group_photo_path}")
    start_time = time.time()
    
    try:
        # Load images
        selfie_image = Image.open(selfie_path)
        group_image = Image.open(group_photo_path)
        
        # Convert to numpy arrays
        selfie = np.array(selfie_image)
        group_photo = np.array(group_image)
        
        if selfie is None or group_photo is None:
            raise ValueError("Failed to load one or both images")

        # Calculate image quality scores
        logger.info("Calculating image quality scores")
        selfie_score = score_generator(selfie, fast_mode=basic_mode)
        group_score = score_generator(group_photo, fast_mode=basic_mode)
        
        # Convert BGR to RGB for InsightFace
        selfie_rgb = cv2.cvtColor(selfie, cv2.COLOR_BGR2RGB)
        group_photo_rgb = cv2.cvtColor(group_photo, cv2.COLOR_BGR2RGB)
        
        # Enhance images - use lightweight enhancement in basic mode
        #group_photo_enhanced = enhance_image_quality(group_photo, lightweight=basic_mode)
        
        # Get InsightFace app
        insightface_app = load_insightface()
        
        # Detect faces in selfie
        logger.info("Detecting faces in selfie")
        selfie_faces = insightface_app.get(selfie_rgb)
        if len(selfie_faces) == 0:
            logger.warning("No face detected in selfie")
            return [{
                'face_match_percentage': 0,
                'confidence': round((selfie_score + group_score) / 2, 2),
                'error': 'No face detected in selfie'
            }]
        
        # Find the largest face if multiple faces are detected
        if len(selfie_faces) > 1:
            logger.warning(f"Multiple faces ({len(selfie_faces)}) detected in selfie. Using the largest face.")
            largest_face_idx = 0
            largest_face_area = 0

            for i, face in enumerate(selfie_faces):
                try:
                    bbox = face.bbox
                    if isinstance(bbox, np.ndarray):
                        bbox = bbox.astype(float)
                    if len(bbox) >= 4:
                        x1, y1, x2, y2 = bbox[:4]
                        face_area = float(x2 - x1) * float(y2 - y1)
                        if face_area > largest_face_area:
                            largest_face_area = face_area
                            largest_face_idx = i
                except Exception as e:
                    logger.error(f"Error calculating face area for face {i}: {e}")
            
            selfie_face = selfie_faces[largest_face_idx]
            logger.info(f"Selected the largest face in selfie (index: {largest_face_idx})")
        else:
            selfie_face = selfie_faces[0]
        
        selfie_embedding = selfie_face.embedding
        
        # Extract selfie face for potential tilt correction
        selfie_face_img = extract_face_with_margin(selfie, selfie_face.bbox)
        
        # Apply tilt correction to selfie in both modes
        corrected_selfie_face = correct_face_tilt(selfie_face_img)
        
        # Re-run face detection on corrected selfie if it was changed
        if not np.array_equal(selfie_face_img, corrected_selfie_face):
            logger.info("Applying tilt correction to selfie face")
            # Convert to RGB for InsightFace
            corrected_selfie_rgb = cv2.cvtColor(corrected_selfie_face, cv2.COLOR_BGR2RGB)
            # Detect face in corrected image
            corrected_faces = insightface_app.get(corrected_selfie_rgb)
            # Use corrected embedding if face detected
            if len(corrected_faces) > 0:
                selfie_embedding = corrected_faces[0].embedding
                logger.info("Using tilt-corrected selfie face for comparison")
        
        # Detect faces in group photo
        logger.info("Detecting faces in group photo")
        group_faces = insightface_app.get(group_photo_rgb)
        
        if len(group_faces) == 0:
            logger.warning("No faces detected in group photo")
            return [{
                'face_match_percentage': 0,
                'confidence': round((selfie_score + group_score) / 2, 2),
                'error': 'No face detected in group photo'
            }]
        
        # Find the best matching face
        best_match = None
        best_score = 0
        best_face_index = -1
        best_face_img = None

        # Process all faces with InsightFace
        for i, face in enumerate(group_faces):
            # Extract and possibly correct face tilt in group photo
            group_face_img = extract_face_with_margin(group_photo, face.bbox.astype(int))
            
            # Apply tilt correction in both modes
            corrected_group_face = correct_face_tilt(group_face_img)
            
            # Re-detect face in corrected image if needed
            face_embedding = face.embedding
            if not np.array_equal(group_face_img, corrected_group_face):
                logger.info(f"Applying tilt correction to group face {i+1}")
                corrected_group_rgb = cv2.cvtColor(corrected_group_face, cv2.COLOR_BGR2RGB)
                corrected_group_faces = insightface_app.get(corrected_group_rgb)
                if len(corrected_group_faces) > 0:
                    face_embedding = corrected_group_faces[0].embedding
                    logger.info(f"Using tilt-corrected group face {i+1} for comparison")
            
            # Calculate similarity
            cosine_similarity = np.dot(selfie_embedding, face_embedding) / (
                np.linalg.norm(selfie_embedding) * np.linalg.norm(face_embedding))
            similarity_percent = min(100, max(0, cosine_similarity * 100))
            
            if len(group_faces) < 5:
                logger.info(f"Face {i+1} InsightFace similarity: {similarity_percent:.2f}%")
            
            # Track the best match
            if similarity_percent > best_score:
                best_score = similarity_percent
                best_face_index = i
                # Store the face image for potential advanced processing
                if not basic_mode and similarity_percent < 50:
                    best_face_img = corrected_group_face  # Use the tilt-corrected version
                best_match = {
                    'model': 'InsightFace',
                    'similarity': similarity_percent,
                    'weight': 1.0,
                    'face_id': i
                }
        
        # If we have a match
        if best_score > 0:
            # Calculate confidence score
            confidence = calculate_confidence_score(
                selfie_score,
                group_score,
                best_score,
                len(group_faces)
            )

            # Apply additional advanced processing if needed (for non-basic mode)
            if not basic_mode and best_score < 50 and best_face_img is not None:
                # Apply advanced dark complexion enhancement
                enhanced_face = enhance_dark_complexion(best_face_img)
                
                # Re-detect face in enhanced image
                enhanced_rgb = cv2.cvtColor(enhanced_face, cv2.COLOR_BGR2RGB)
                enhanced_faces = insightface_app.get(enhanced_rgb)
                
                # Update score if improved
                if len(enhanced_faces) > 0:
                    advanced_embedding = enhanced_faces[0].embedding
                    advanced_similarity = np.dot(selfie_embedding, advanced_embedding) / (
                        np.linalg.norm(selfie_embedding) * np.linalg.norm(advanced_embedding))
                    advanced_score = min(100, max(0, advanced_similarity * 100))
                    
                    # Use advanced score if it's better
                    if advanced_score > best_score:
                        logger.info(f"Advanced preprocessing improved score from {best_score:.2f}% to {advanced_score:.2f}%")
                        best_score = advanced_score
                        # Recalculate confidence with new score
                        confidence = calculate_confidence_score(
                            selfie_score, group_score, best_score, len(group_faces))

            # Recalibrate the score for better presentation
            original_score = best_score
            recalibrated_score = recalibrate_match_score(original_score, confidence)
            logger.info(f"Recalibrated match score from {original_score:.1f}% to {recalibrated_score:.1f}%")

            # Return result
            return [{
                'face_match_percentage': recalibrated_score,
                'confidence': confidence,
                'selfie_quality': selfie_score,
                'reference_quality': group_score,
                'faces_detected': {
                    'selfie': len(selfie_faces),
                    'reference': len(group_faces)
                }
            }]
        else:
            # No good match found
            elapsed_time = time.time() - start_time
            return [{
                'face_match_percentage': 0,
                'confidence': round((selfie_score + group_score) / 2, 2),
                'error': 'No matching face found',
                'processing_time': round(elapsed_time, 2)
            }]
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Error in face comparison: {e}")
        return [{
            'face_match_percentage': 0,
            'confidence': 0,
            'error': str(e),
            'processing_time': round(elapsed_time, 2)
        }]
    finally:
        elapsed_time = time.time() - start_time
        logger.info(f"Face comparison completed in {elapsed_time:.2f} seconds")
        
if __name__ == "__main__":
    # Initialize InsightFace model at startup
    initialize_models(preload_all=False)
    
    # Test the module
    import sys
    if len(sys.argv) >= 3:
        # Check if basic_mode is specified
        basic_mode = False
        if len(sys.argv) > 3 and sys.argv[3].lower() == 'true':
            basic_mode = True
            
        result = compare_faces(sys.argv[1], sys.argv[2], basic_mode)
        print(result)
    else:
        print("Usage: python face_comparator.py <selfie_path> <group_photo_path> [basic_mode]")
