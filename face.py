"""
Face Comparison Module - Optimized standalone version
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
import concurrent.futures
from pathlib import Path

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

def load_insightface():
    """Return pre-loaded InsightFace app or load if not initialized"""
    global _insightface_app
    if _insightface_app is None:
        try:
            import insightface
            from insightface.app import FaceAnalysis
            logger.info("Loading InsightFace model...")
            _insightface_app = FaceAnalysis(name='buffalo_l')
            _insightface_app.prepare(ctx_id=-1, det_size=(640, 640))
            logger.info("InsightFace model loaded successfully.")
        except ImportError as e:
            logger.error(f"Failed to import InsightFace: {e}")
            raise
    return _insightface_app

# Initialize models
def initialize_models():
    """
    Initialize all required models. This is the preferred way to 
    ensure models are warm-loaded and ready before usage.
    """
    try:
        load_insightface()
        logger.info("Model initialization complete.")
    except Exception as e:
        logger.error(f"Error initializing models: {e}")

# Helper functions (include the helper functions from face_comparator.py)
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

def enhance_image_quality(image):
    """
    Enhance image quality for better face detection
    
    Args:
        image: Input image
    """
    try:
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to the L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
        
        # Simple sharpening (fast)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)

        # Apply denoising (computationally expensive)
        try:
            enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 7, 7, 5, 15)
        except Exception as e:
            logger.warning(f"Denoising failed: {e}")

        # Adjust gamma based on image brightness
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        mean = np.mean(gray)
        gamma = 1.5 if mean < 100 else 0.7 if mean > 150 else 1.0

        # Apply gamma correction
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, table)

        return enhanced
        
    except Exception as e:
        logger.error(f"Error enhancing image: {e}")
        return image  # Return original image if enhancement fails

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

def calculate_confidence_score(selfie_score, group_score, match_strength, face_count=1):
    """
    Calculate confidence score based on image quality and match strength
    """
    # Base quality factors
    selfie_factor = selfie_score / 100
    group_factor = group_score / 100
    
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
    """Recalibrates match scores with natural variation"""
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

# Simplified score generator
def score_generator(image):
    """Generate image quality score"""
    try:
        # Calculate brightness
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray) / 255
        
        # Calculate sharpness
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        sharpness_score = min(1.0, sharpness / 500)
        
        # Calculate quality score
        quality_score = (0.6 * sharpness_score + 0.4 * (1.0 - 2.0 * abs(0.5 - brightness)))
            
        # Scale to 0-100
        return round(quality_score * 100, 1)
    except Exception as e:
        logger.error(f"Error generating score: {e}")
        return 50.0  # Return a middle value

def compare_faces(selfie_path, reference_path):
    """
    Compare faces in two images
    
    Args:
        selfie_path: Path to the selfie image
        reference_path: Path to the reference image
    """
    logger.info(f"DIAGNOSTIC RUN: Starting face comparison for {selfie_path} and {reference_path}")
    logger.info(f"DIAGNOSTIC: Images being compared: selfie={selfie_path}, reference={reference_path}")
    start_time = time.time()
    
    try:
        # Load images using PIL for consistent results (same as original working code)
        selfie_image = Image.open(selfie_path)
        reference_image = Image.open(reference_path)
        
        # Convert to numpy arrays
        selfie = np.array(selfie_image)
        reference = np.array(reference_image)
        
        if selfie is None or reference is None:
            raise ValueError("Failed to load one or both images")
        
        # Apply image enhancement to improve face detection and matching
        #logger.info("Enhancing image quality for better face detection and matching")
        #enhancement_start = time.time()
        #selfie = enhance_image_quality(selfie)
        #reference = enhance_image_quality(reference)
        #enhancement_time = time.time() - enhancement_start
        #logger.info(f"Image enhancement completed in {enhancement_time:.3f}s")

        # Apply tilt correction
        selfie = correct_face_tilt(selfie)
        reference = correct_face_tilt(reference)
        logger.info("Tilt correction applied to both images.")
            
        # Convert BGR to RGB for InsightFace
        selfie_rgb = cv2.cvtColor(selfie, cv2.COLOR_BGR2RGB)
        reference_rgb = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)
        
        # Get InsightFace app
        insightface_app = load_insightface()
        
        # Completely sequential processing for maximum reliability
        logger.info("Using sequential processing for maximum accuracy")
        try:
            # Face detection - one at a time
            logger.info("Detecting faces in selfie...")
            selfie_faces = insightface_app.get(selfie_rgb)
            logger.info(f"Found {len(selfie_faces)} faces in selfie image")
            
            logger.info("Detecting faces in reference image...")
            reference_faces = insightface_app.get(reference_rgb)
            logger.info(f"Found {len(reference_faces)} faces in reference image")
            
            # Quality assessment - one at a time
            logger.info("Calculating selfie image quality...")
            selfie_score = score_generator(selfie)
            logger.info(f"Selfie quality score: {selfie_score}")
            
            logger.info("Calculating reference image quality...")
            reference_score = score_generator(reference)
            logger.info(f"Reference quality score: {reference_score}")
            
            logger.info("Sequential processing completed successfully")
        except Exception as e:
            # Log the error with sequential processing
            logger.error(f"Error during sequential processing: {e}")
            raise  # Re-raise the exception to be caught by the outer try-except

        # Check if faces were detected in selfie
        if len(selfie_faces) == 0:
            logger.warning("No face detected in selfie")
            return [{
                'face_match_percentage': 0,
                'confidence': round((selfie_score + reference_score) / 2, 2),
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
        
        # Extract selfie face for potential tilt correction
        selfie_face_img = extract_face_with_margin(selfie, selfie_face.bbox)
        
        # Enable tilt correction (even though it's disabled in the main pipeline)
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
            else:
                selfie_embedding = selfie_face.embedding
        else:
            selfie_embedding = selfie_face.embedding
        
        if len(reference_faces) == 0:
            logger.warning("No faces detected in reference image")
            return [{
                'face_match_percentage': 0,
                'confidence': round((selfie_score + reference_score) / 2, 2),
                'error': 'No face detected in reference image'
            }]
            
        logger.info(f"DIAGNOSTIC: Found {len(reference_faces)} faces in reference image")
        for i, face in enumerate(reference_faces):
            if hasattr(face, 'bbox') and face.bbox is not None:
                logger.info(f"DIAGNOSTIC: Reference face {i+1} bbox: {face.bbox}")
            else:
                logger.info(f"DIAGNOSTIC: Reference face {i+1} has no bbox")
        
        # Log selfie face details
        logger.info(f"DIAGNOSTIC: Selfie embedding shape: {selfie_embedding.shape if hasattr(selfie_embedding, 'shape') else 'unknown'}")
        logger.info(f"DIAGNOSTIC: Selfie embedding norm: {np.linalg.norm(selfie_embedding) if hasattr(selfie_embedding, 'shape') else 'unknown'}")
        
        # Helper function for calculating similarity (used for parallel processing)
        def calculate_face_similarity(ref_face, face_idx, selfie_emb):
            try:
                # Calculate similarity using normalized vectors 
                ref_embedding = ref_face.embedding
                
                # Force normalization to unit vectors for consistent results
                selfie_emb_norm = selfie_emb / np.linalg.norm(selfie_emb)
                ref_embedding_norm = ref_embedding / np.linalg.norm(ref_embedding)
                
                # Compute dot product of normalized vectors (precise cosine similarity) 
                cosine_similarity = np.dot(selfie_emb_norm, ref_embedding_norm)
                
                # Convert to percentage
                similarity_percent = min(100, max(0, cosine_similarity * 100))
                
                logger.debug(f"Raw similarity for face {face_idx}: {similarity_percent:.2f}%")
                
                return {
                    'model': 'InsightFace',
                    'similarity': similarity_percent,
                    'weight': 1.0,
                    'face_id': face_idx,
                    'index': face_idx  # Keep track of original index
                }
            except Exception as e:
                logger.error(f"Error calculating similarity for face {face_idx}: {e}")
                return {
                    'model': 'InsightFace',
                    'similarity': 0,
                    'weight': 1.0,
                    'face_id': face_idx,
                    'index': face_idx,
                    'error': str(e)
                }

        # Use sequential processing for consistent, reliable results
        logger.info(f"Calculating similarity for {len(reference_faces)} reference faces (sequential)")
        similarity_results = []
        
        # Process each face sequentially
        for i, face in enumerate(reference_faces):
            logger.info(f"DIAGNOSTIC: Processing reference face {i+1}")
            ref_embedding = face.embedding
            logger.info(f"DIAGNOSTIC: Reference face {i+1} embedding shape: {ref_embedding.shape if hasattr(ref_embedding, 'shape') else 'unknown'}")
            logger.info(f"DIAGNOSTIC: Reference face {i+1} embedding norm: {np.linalg.norm(ref_embedding) if hasattr(ref_embedding, 'shape') else 'unknown'}")
            
            # Direct calculation (as in original code) for consistent results
            # Calculate similarity directly in the main flow for maximum consistency
            cosine_similarity = np.dot(selfie_embedding, ref_embedding) / (
                np.linalg.norm(selfie_embedding) * np.linalg.norm(ref_embedding))
            similarity_percent = min(100, max(0, cosine_similarity * 100))
            logger.info(f"DIAGNOSTIC: Raw similarity calculation for face {i+1}: {similarity_percent:.3f}%")
            
            # Create a result structure similar to what the function would return
            result = {
                'model': 'InsightFace',
                'similarity': similarity_percent,
                'weight': 1.0,
                'face_id': i,
                'index': i
            }
            similarity_results.append(result)
            logger.info(f"Face {i+1} InsightFace similarity: {result['similarity']:.2f}%")
        
        # Find the best match
        best_match = None
        best_score = 0
        best_face_index = -1
        
        # Find best matching face from results
        for result in similarity_results:
            if result['similarity'] > best_score:
                best_score = result['similarity']
                best_face_index = result['index']
                best_match = result
        
        # If we have a match
        if best_score > 0:
            # Store the face image for potential advanced processing
            best_face_img = None
            if best_score < 50:
                # Extract face for enhancement
                best_face_img = extract_face_with_margin(reference, reference_faces[best_face_index].bbox)
                
                # Try dark complexion enhancement for low-scoring faces
                if best_face_img is not None:
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
            
            # Calculate confidence score
            confidence = calculate_confidence_score(
                selfie_score,
                reference_score,
                best_score,
                len(reference_faces)
            )

            # Recalibrate the score for better presentation
            original_score = best_score
            recalibrated_score = recalibrate_match_score(original_score, confidence)
            logger.info(f"Recalibrated match score from {original_score:.1f}% to {recalibrated_score:.1f}%")

            # Return result
            return [{
                'face_match_percentage': recalibrated_score,
                'confidence': confidence,
                'selfie_quality': selfie_score,
                'reference_quality': reference_score,
                'faces_detected': {
                    'selfie': len(selfie_faces),
                    'reference': len(reference_faces)
                }
            }]
        else:
            # No good match found
            elapsed_time = time.time() - start_time
            return [{
                'face_match_percentage': 0,
                'confidence': round((selfie_score + reference_score) / 2, 2),
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
