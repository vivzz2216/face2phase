"""
Facial analysis module for emotion detection and engagement tracking
"""
import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Optional DeepFace import
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DeepFace = None
    DEEPFACE_AVAILABLE = False
    logger.warning("DeepFace not available. Emotion analysis will be limited.")

from ...core.settings import FRAME_EXTRACTION_FPS, MIN_FACE_CONFIDENCE

from ...utils.report_utils import compute_tension_metrics, smooth_emotion_timeline

# Optional dlib import for enhanced facial analysis
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    dlib = None
    DLIB_AVAILABLE = False

# MTCNN for modern face detection
try:
    from mtcnn import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN = None
    MTCNN_AVAILABLE = False
    logger.warning("MTCNN not available. Install with: pip install mtcnn")

class FacialAnalyzer:
    """Analyzes facial expressions and engagement from video"""
    
    def __init__(self):
        self.face_detector = None
        self.landmark_predictor = None
        self._models_initialized = False  # Track if models are loaded
        self.use_mtcnn = False
        # LAZY LOADING: Don't initialize models at startup
        # self._initialize_models()  # Commented out for fast startup
        logger.info("FacialAnalyzer created (models will load on first use)")
    
    def _ensure_models_initialized(self):
        """Lazy load models on first use"""
        if self._models_initialized:
            return
        self._models_initialized = True
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize facial analysis models"""
        try:
            # Try MTCNN first (modern, more accurate)
            if MTCNN_AVAILABLE:
                try:
                    self.face_detector = MTCNN()
                    self.use_mtcnn = True
                    logger.info("MTCNN face detector initialized (modern, high accuracy)")
                except Exception as e:
                    logger.warning(f"Could not initialize MTCNN: {e}, falling back to OpenCV")
                    self.use_mtcnn = False
                    # Fallback to OpenCV
                    self.face_detector = cv2.CascadeClassifier(
                        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                    )
            else:
                # Fallback to OpenCV
                self.face_detector = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                self.use_mtcnn = False
                logger.info("Using OpenCV face detector (MTCNN not available)")
            
            # Try to initialize dlib for landmark detection
            if DLIB_AVAILABLE:
                try:
                    # Download the landmark predictor model if not present
                    predictor_path = "shape_predictor_68_face_landmarks.dat"
                    if not Path(predictor_path).exists():
                        logger.warning("Dlib landmark predictor not found. Eye contact analysis will be limited.")
                        self.landmark_predictor = None
                    else:
                        self.landmark_predictor = dlib.shape_predictor(predictor_path)
                        logger.info("Loaded dlib landmark predictor")
                except Exception as e:
                    logger.warning(f"Could not load dlib landmark predictor: {e}")
                    self.landmark_predictor = None
            else:
                logger.info("Dlib not available. Eye contact analysis will use basic methods.")
                self.landmark_predictor = None
            
            logger.info("Facial analysis models initialized")
            
        except Exception as e:
            logger.error(f"Error initializing facial analysis models: {e}")
            self.face_detector = None
            self.landmark_predictor = None
            self.use_mtcnn = False
    
    def extract_frames(self, video_path: Path) -> List[np.ndarray]:
        """
        Extract frames from video at specified FPS
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of extracted frames
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise Exception("Could not open video file")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(fps / FRAME_EXTRACTION_FPS)
            
            frames = []
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    frames.append(frame)
                
                frame_count += 1
            
            cap.release()
            logger.info(f"Extracted {len(frames)} frames from video")
            return frames
            
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            return []
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in a frame using MTCNN (if available) or OpenCV
        
        Args:
            frame: Image frame
            
        Returns:
            List of face bounding boxes (x, y, w, h)
        """
        try:
            if not self.face_detector:
                return []
            
            if self.use_mtcnn and hasattr(self, 'use_mtcnn') and self.use_mtcnn:
                # Use MTCNN (more accurate)
                try:
                    # MTCNN expects RGB, OpenCV uses BGR
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.face_detector.detect_faces(rgb_frame)
                    
                    faces = []
                    for result in results:
                        if result['confidence'] >= MIN_FACE_CONFIDENCE:
                            x, y, w, h = result['box']
                            faces.append((x, y, w, h))
                    
                    return faces
                except Exception as e:
                    logger.warning(f"MTCNN detection failed: {e}, falling back to OpenCV")
                    # Permanently switch to OpenCV for this session
                    self.use_mtcnn = False
                    self.face_detector = cv2.CascadeClassifier(
                        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                    )
                    # Now use OpenCV
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_detector.detectMultiScale(
                        gray, 
                        scaleFactor=1.1, 
                        minNeighbors=5, 
                        minSize=(30, 30)
                    )
                    return faces.tolist()
            else:
                # Use OpenCV (fallback)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_detector.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(30, 30)
                )
                return faces.tolist()
            
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []
    
    def analyze_emotions(self, frame: np.ndarray) -> Dict:
        """
        Analyze emotions in a frame using DeepFace
        
        Args:
            frame: Image frame
            
        Returns:
            Dictionary containing emotion analysis results
        """
        if not DEEPFACE_AVAILABLE or DeepFace is None:
            return {
                'emotions': {},
                'dominant_emotion': 'neutral',
                'face_detected': False
            }

        try:
            # DeepFace emotion analysis
            result = DeepFace.analyze(
                frame, 
                actions=['emotion'], 
                enforce_detection=False,
                detector_backend='opencv'
            )
            
            if isinstance(result, list):
                result = result[0]
            
            emotions = result.get('emotion', {})
            dominant_emotion = result.get('dominant_emotion', 'neutral')
            
            return {
                'emotions': emotions,
                'dominant_emotion': dominant_emotion,
                'face_detected': True
            }
            
        except Exception as e:
            logger.debug(f"Emotion analysis failed for frame: {e}")
            return {
                'emotions': {},
                'dominant_emotion': 'neutral',
                'face_detected': False
            }
    
    def analyze_eye_contact(self, frame: np.ndarray, face_box: Tuple[int, int, int, int]) -> Dict:
        """
        Analyze eye contact and gaze direction using pupil/iris position
        
        Args:
            frame: Image frame
            face_box: Face bounding box (x, y, w, h)
            
        Returns:
            Dictionary containing eye contact analysis
        """
        try:
            if not self.landmark_predictor:
                # Fallback: Use face position relative to frame to estimate gaze
                x, y, w, h = face_box
                frame_h, frame_w = frame.shape[:2]
                
                # Face centered = likely looking at camera
                face_center_x = x + w/2
                face_center_y = y + h/2
                frame_center_x = frame_w / 2
                frame_center_y = frame_h / 2
                
                # Calculate how centered the face is (0 = center, 1 = edge)
                horizontal_offset = abs(face_center_x - frame_center_x) / (frame_w / 2)
                vertical_offset = abs(face_center_y - frame_center_y) / (frame_h / 2)
                
                # If face is centered, assume better eye contact
                centering_score = 1.0 - (horizontal_offset + vertical_offset) / 2
                
                # Face size matters too - closer to camera = more engaged
                face_area = w * h
                frame_area = frame_w * frame_h
                size_ratio = face_area / frame_area
                
                # Combine centering and size for basic estimate
                basic_score = (centering_score * 0.7 + min(size_ratio * 10, 1.0) * 0.3)
                
                return {
                    'eye_contact_score': max(0.2, min(0.8, basic_score)),  # Clamp between 0.2-0.8
                    'gaze_direction': 'center' if horizontal_offset < 0.3 else 'away',
                    'estimation_method': 'face_position'
                }
            
            # Convert face box to dlib rectangle
            x, y, w, h = face_box
            dlib_rect = dlib.rectangle(x, y, x + w, y + h)
            
            # Get facial landmarks
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            landmarks = self.landmark_predictor(gray, dlib_rect)
            
            # Extract eye landmarks (indices 36-47 for both eyes)
            left_eye_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
            right_eye_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
            
            # Calculate eye regions
            left_eye_region = self._get_eye_region(gray, left_eye_points)
            right_eye_region = self._get_eye_region(gray, right_eye_points)
            
            # Detect pupil/iris position in each eye
            left_gaze = self._detect_gaze_direction(left_eye_region, left_eye_points)
            right_gaze = self._detect_gaze_direction(right_eye_region, right_eye_points)
            
            # Combine both eyes for average gaze
            if left_gaze and right_gaze:
                # Average the gaze ratios
                horizontal_ratio = (left_gaze['horizontal_ratio'] + right_gaze['horizontal_ratio']) / 2
                vertical_ratio = (left_gaze['vertical_ratio'] + right_gaze['vertical_ratio']) / 2
                
                # Classify gaze direction
                gaze_direction = 'center'  # Default to center (looking at camera)
                eye_contact_score = 1.0  # Start with perfect score
                
                # Horizontal gaze (left/right)
                if horizontal_ratio < 0.35:
                    gaze_direction = 'left'
                    eye_contact_score *= 0.3
                elif horizontal_ratio > 0.65:
                    gaze_direction = 'right'
                    eye_contact_score *= 0.3
                elif 0.4 <= horizontal_ratio <= 0.6:
                    # Looking at center horizontally
                    eye_contact_score *= 1.0
                else:
                    # Slightly off-center
                    eye_contact_score *= 0.7
                
                # Vertical gaze (up/down)
                if vertical_ratio < 0.35:
                    if gaze_direction == 'center':
                        gaze_direction = 'up'
                    eye_contact_score *= 0.4
                elif vertical_ratio > 0.65:
                    if gaze_direction == 'center':
                        gaze_direction = 'down'
                    eye_contact_score *= 0.4
                elif 0.4 <= vertical_ratio <= 0.6:
                    # Looking at center vertically
                    eye_contact_score *= 1.0
                else:
                    # Slightly off-center
                    eye_contact_score *= 0.8
                
                return {
                    'eye_contact_score': min(1.0, max(0.0, eye_contact_score)),
                    'gaze_direction': gaze_direction,
                    'horizontal_ratio': horizontal_ratio,
                    'vertical_ratio': vertical_ratio,
                    'estimation_method': 'pupil_tracking'
                }
            else:
                # Fallback if pupil detection fails
                return {'eye_contact_score': 0.5, 'gaze_direction': 'unknown', 'estimation_method': 'fallback'}
            
        except Exception as e:
            logger.debug(f"Eye contact analysis failed: {e}")
            # Fallback - return neutral score
            return {'eye_contact_score': 0.5, 'gaze_direction': 'unknown', 'estimation_method': 'error'}
    
    def _get_eye_region(self, gray_frame: np.ndarray, eye_points: np.ndarray) -> Optional[np.ndarray]:
        """Extract eye region from frame"""
        try:
            # Get bounding box around eye
            x_min = int(np.min(eye_points[:, 0]))
            x_max = int(np.max(eye_points[:, 0]))
            y_min = int(np.min(eye_points[:, 1]))
            y_max = int(np.max(eye_points[:, 1]))
            
            # Add padding
            padding = 5
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(gray_frame.shape[1], x_max + padding)
            y_max = min(gray_frame.shape[0], y_max + padding)
            
            return gray_frame[y_min:y_max, x_min:x_max]
        except:
            return None
    
    def _detect_gaze_direction(self, eye_region: Optional[np.ndarray], eye_points: np.ndarray) -> Optional[Dict]:
        """
        Detect gaze direction by finding pupil/iris position relative to eye corners
        
        Returns:
            Dictionary with horizontal_ratio and vertical_ratio (0-1, where 0.5 is centered)
        """
        if eye_region is None or eye_region.size == 0:
            return None
        
        try:
            # Apply threshold to find dark pupil region
            _, threshold = cv2.threshold(eye_region, 50, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Find largest contour (likely the pupil/iris)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get moments to find centroid
            moments = cv2.moments(largest_contour)
            if moments['m00'] == 0:
                return None
            
            # Pupil center relative to eye region
            pupil_x = int(moments['m10'] / moments['m00'])
            pupil_y = int(moments['m01'] / moments['m00'])
            
            # Eye region dimensions
            eye_width = eye_region.shape[1]
            eye_height = eye_region.shape[0]
            
            # Calculate ratios (0 = left/top, 1 = right/bottom, 0.5 = center)
            horizontal_ratio = pupil_x / eye_width if eye_width > 0 else 0.5
            vertical_ratio = pupil_y / eye_height if eye_height > 0 else 0.5
            
            return {
                'horizontal_ratio': float(horizontal_ratio),
                'vertical_ratio': float(vertical_ratio)
            }
        except:
            return None
    
    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """
        Complete analysis of a single frame
        
        Args:
            frame: Image frame
            
        Returns:
            Dictionary containing frame analysis results
        """
        try:
            # Detect faces
            faces = self.detect_faces(frame)
            
            if not faces:
                return {
                    'faces_detected': 0,
                    'emotions': {},
                    'dominant_emotion': 'neutral',
                    'eye_contact_score': 0,
                    'gaze_direction': 'unknown'
                }
            
            # Use the largest face (assuming main speaker)
            main_face = max(faces, key=lambda f: f[2] * f[3])
            
            # Analyze emotions
            emotion_result = self.analyze_emotions(frame)
            
            # Analyze eye contact
            eye_result = self.analyze_eye_contact(frame, main_face)
            
            return {
                'faces_detected': len(faces),
                'main_face_size': main_face[2] * main_face[3],
                'emotions': emotion_result.get('emotions', {}),
                'dominant_emotion': emotion_result.get('dominant_emotion', 'neutral'),
                'eye_contact_score': eye_result.get('eye_contact_score', 0.5),
                'gaze_direction': eye_result.get('gaze_direction', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"Error analyzing frame: {e}")
            return {
                'faces_detected': 0,
                'emotions': {},
                'dominant_emotion': 'neutral',
                'eye_contact_score': 0,
                'gaze_direction': 'unknown'
            }
    
    def smooth_emotions_temporal(self, emotion_sequence: List[Dict], window_size: int = 5) -> List[Dict]:
        """
        Apply temporal smoothing to emotion sequence to reduce jitter
        
        Args:
            emotion_sequence: List of emotion dictionaries from frames
            window_size: Size of smoothing window
            
        Returns:
            Smoothed emotion sequence
        """
        if not emotion_sequence or len(emotion_sequence) < 2:
            return emotion_sequence
        
        smoothed = []
        emotion_keys = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        for i in range(len(emotion_sequence)):
            # Get window of frames (weighted towards recent frames)
            window_start = max(0, i - window_size + 1)
            window = emotion_sequence[window_start:i+1]
            
            # Calculate weights (exponential decay - recent frames more important)
            weights = np.exp(np.linspace(-2, 0, len(window)))
            weights = weights / np.sum(weights)  # Normalize
            
            # Weighted average of emotion scores
            smoothed_emotions = {}
            for key in emotion_keys:
                values = [frame.get('emotions', {}).get(key, 0) for frame in window]
                smoothed_emotions[key] = float(np.average(values, weights=weights))
            
            # Determine dominant emotion from smoothed values
            dominant_emotion = max(smoothed_emotions.items(), key=lambda x: x[1])[0]
            
            # Create smoothed frame result
            smoothed_frame = emotion_sequence[i].copy()
            smoothed_frame['emotions'] = smoothed_emotions
            smoothed_frame['dominant_emotion'] = dominant_emotion
            
            smoothed.append(smoothed_frame)
        
        return smoothed
    
    def calculate_facial_confidence_score(self, frame_results: List[Dict]) -> float:
        """
        Calculate facial confidence score (0-100)
        
        Args:
            frame_results: List of frame analysis results
            
        Returns:
            Facial confidence score
        """
        try:
            if not frame_results:
                return 50.0
            
            # Calculate emotion stability
            emotions = [result.get('dominant_emotion', 'neutral') for result in frame_results]
            emotion_counts = {}
            for emotion in emotions:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            # Find dominant emotion
            dominant_emotion = max(emotion_counts, key=emotion_counts.get)
            emotion_stability = emotion_counts[dominant_emotion] / len(emotions)
            
            # Calculate average eye contact score
            eye_scores = [result.get('eye_contact_score', 0.5) for result in frame_results]
            avg_eye_contact = np.mean(eye_scores)
            
            # Calculate face detection consistency
            face_detection_rate = sum(1 for result in frame_results if result.get('faces_detected', 0) > 0) / len(frame_results)
            
            # Calculate overall score
            score = 100.0
            
            # Bonus for emotion stability
            score += emotion_stability * 20
            
            # Bonus for good eye contact
            score += avg_eye_contact * 30
            
            # Bonus for consistent face detection
            score += face_detection_rate * 20
            
            # Penalty for negative emotions
            negative_emotions = ['angry', 'sad', 'fear', 'disgust']
            negative_count = sum(1 for emotion in emotions if emotion in negative_emotions)
            if negative_count > len(emotions) * 0.3:  # More than 30% negative emotions
                score -= 15
            
            return max(0, min(100, score))
            
        except Exception as e:
            logger.error(f"Error calculating facial confidence score: {e}")
            return 50.0
    
    def analyze_video(self, video_path: Path) -> Dict:
        """
        Complete facial analysis of video
        
        Args:
            video_path: Path to video file
            
        Returns:
            Complete facial analysis results
        """
        try:
            # LAZY LOADING: Initialize models on first use
            self._ensure_models_initialized()
            
            logger.info(f"Starting facial analysis for: {video_path}")
            
            # Extract frames
            frames = self.extract_frames(video_path)
            
            if not frames:
                return {
                    'frame_results': [],
                    'facial_confidence_score': 0,
                    'emotion_distribution': {},
                    'dominant_emotion': 'neutral',
                    'avg_eye_contact': 0,
                    'face_detection_rate': 0,
                    'analysis_successful': False,
                    'error': 'No frames extracted'
                }
            
            # Get video FPS and duration for timestamp calculation
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or FRAME_EXTRACTION_FPS
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration = total_frames / fps if fps > 0 else 0
            cap.release()
            
            # Analyze each frame with precise timestamps
            frame_results = []
            emotion_timeline = []  # Track emotions with timestamps
            eye_contact_timeline = []  # Track eye contact with timestamps
            tension_moments = []  # Track tension/stress moments
            
            for i, frame in enumerate(frames):
                # Calculate precise timestamp for this frame
                frame_timestamp = (i / FRAME_EXTRACTION_FPS) if FRAME_EXTRACTION_FPS > 0 else 0
                
                result = self.analyze_frame(frame)
                result['timestamp'] = float(frame_timestamp)  # Add timestamp to result
                frame_results.append(result)
                
                # Track emotion with timestamp
                dominant_emotion = result.get('dominant_emotion', 'neutral')
                emotion_confidence = result.get('emotions', {}).get(dominant_emotion, 0)
                emotion_timeline.append({
                    'timestamp': float(frame_timestamp),
                    'emotion': dominant_emotion,
                    'confidence': float(emotion_confidence),
                    'all_emotions': result.get('emotions', {})
                })
                
                # Track eye contact with timestamp
                eye_contact_score = result.get('eye_contact_score', 0.5)
                gaze_direction = result.get('gaze_direction', 'unknown')
                eye_contact_timeline.append({
                    'timestamp': float(frame_timestamp),
                    'eye_contact_score': float(eye_contact_score),
                    'gaze_direction': gaze_direction,
                    'is_looking_away': eye_contact_score < 0.3
                })
                
                # Detect tension/stress moments
                # Tension indicators: fear, anger, high stress emotions, low eye contact
                is_tension = False
                tension_reasons = []
                
                if dominant_emotion in ['fear', 'angry', 'disgust']:
                    is_tension = True
                    tension_reasons.append(f"Emotion: {dominant_emotion}")
                
                if eye_contact_score < 0.3:
                    is_tension = True
                    tension_reasons.append("Low eye contact")
                
                # High stress: combination of negative emotion + low eye contact
                if dominant_emotion in ['fear', 'angry', 'sad'] and eye_contact_score < 0.4:
                    is_tension = True
                    tension_reasons.append("High stress (negative emotion + low eye contact)")
                
                if is_tension:
                    tension_moments.append({
                        'timestamp': float(frame_timestamp),
                        'emotion': dominant_emotion,
                        'eye_contact_score': float(eye_contact_score),
                        'reasons': tension_reasons,
                        'severity': 'high' if len(tension_reasons) >= 2 else 'medium'
                    })
                
                if i % 10 == 0:  # Log progress every 10 frames
                    logger.debug(f"Analyzed frame {i+1}/{len(frames)} at {frame_timestamp:.2f}s")
            
            # Apply temporal smoothing to emotions (reduces jitter)
            if len(frame_results) > 1:
                frame_results = self.smooth_emotions_temporal(frame_results, window_size=5)
                logger.info("Applied temporal smoothing to emotion sequence")
            
            # Calculate overall metrics
            facial_confidence = self.calculate_facial_confidence_score(frame_results)
            
            # Calculate emotion distribution
            all_emotions = [result.get('dominant_emotion', 'neutral') for result in frame_results]
            emotion_distribution = {}
            for emotion in all_emotions:
                emotion_distribution[emotion] = emotion_distribution.get(emotion, 0) + 1
            
            # Normalize emotion distribution
            total_frames = len(frame_results)
            for emotion in emotion_distribution:
                emotion_distribution[emotion] = emotion_distribution[emotion] / total_frames
            
            # Find overall dominant emotion
            dominant_emotion = max(emotion_distribution, key=emotion_distribution.get)
            
            # Calculate average eye contact
            eye_scores = [result.get('eye_contact_score', 0.5) for result in frame_results]
            avg_eye_contact = np.mean(eye_scores) if eye_scores else 0
            
            # Calculate face detection rate
            face_detection_rate = sum(1 for result in frame_results if result.get('faces_detected', 0) > 0) / total_frames
            
            # Calculate eye contact statistics with timestamps
            low_eye_contact_moments = [e for e in eye_contact_timeline if e['eye_contact_score'] < 0.3]
            high_eye_contact_moments = [e for e in eye_contact_timeline if e['eye_contact_score'] > 0.7]
            
            results = {
                'frame_results': frame_results,
                'facial_confidence_score': facial_confidence,
                'emotion_distribution': emotion_distribution,
                'dominant_emotion': dominant_emotion,
                'avg_eye_contact': avg_eye_contact,
                'face_detection_rate': face_detection_rate,
                'total_frames_analyzed': total_frames,
                'analysis_successful': True,
                
                # NEW: Precise timestamp tracking
                'emotion_timeline': emotion_timeline,  # All emotions with timestamps
                'eye_contact_timeline': eye_contact_timeline,  # Eye contact with timestamps
                'tension_moments': tension_moments,  # When user looked tense/stressed
                'low_eye_contact_moments': low_eye_contact_moments,  # Moments with low eye contact
                'high_eye_contact_moments': high_eye_contact_moments,  # Moments with good eye contact
                'tension_count': len(tension_moments),
                'tension_percentage': (len(tension_moments) / total_frames * 100) if total_frames > 0 else 0,
                'video_duration': float(video_duration),
                'fps': float(fps)
            }

            results['emotion_timeline_smoothed'] = smooth_emotion_timeline(emotion_timeline, window=5)
            results['tension_summary'] = compute_tension_metrics(results)
            
            logger.info(f"Facial analysis completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in facial analysis: {e}")
            return {
                'frame_results': [],
                'facial_confidence_score': 0,
                'emotion_distribution': {},
                'dominant_emotion': 'neutral',
                'avg_eye_contact': 0,
                'face_detection_rate': 0,
                'analysis_successful': False,
                'error': str(e)
            }

# Global facial analyzer instance
facial_analyzer = FacialAnalyzer()
