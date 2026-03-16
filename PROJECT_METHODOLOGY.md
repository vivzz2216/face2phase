# Face2Phase: AI-Powered Presentation Analysis System

## Methodology

### System Architecture
Face2Phase employs a **multi-modal deep learning pipeline** that analyzes video/audio presentations across three independent dimensions: **Voice Delivery**, **Visual Presence**, and **Narrative Clarity**. The system processes uploaded media files through parallel analysis streams, each utilizing specialized AI models optimized for real-time feedback generation.

### Analysis Pipeline

#### 1. **Audio Processing & Transcription**
- **Transcription Engine**: OpenAI Whisper (large-v3 model) with deterministic settings (temperature=0.0)
- **Feature Extraction**: Librosa library extracts pitch (fundamental frequency via PYIN algorithm), RMS energy, zero-crossing rate, and voice breaks
- **Speech-to-Text**: Whisper generates verbatim transcripts with word-level timestamps, capturing filler words (um, uh, ah) by lowering no-speech threshold to 0.4
- **Pause Detection**: Analyzes inter-segment gaps to identify short pauses (>0.3s), medium pauses (>1.0s), and long pauses (>2.0s)
- **Enhanced Transcript Generation**: Annotates transcripts with timing markers, pause indicators, voice quality tags (low/normal/high based on RMS energy), and stammering detection

#### 2. **Computer Vision & Facial Analysis**
- **Face Detection**: MTCNN (Multi-task Cascaded Convolutional Networks) for robust face localization across frames
- **Emotion Recognition**: DeepFace framework analyzes emotional states (neutral, happy, sad, angry, fear, surprise, disgust) with temporal smoothing (5-frame window) to reduce jitter
- **Eye Contact Estimation**: Gaze direction analysis using pupil/iris position relative to eye corners, calculating horizontal and vertical gaze ratios
- **Frame Sampling**: Extracts 1 frame per second (FPS=1) for computational efficiency while maintaining temporal resolution
- **Engagement Metrics**: Tracks face visibility percentage, emotion stability (dominant emotion consistency), and eye contact duration

#### 3. **Natural Language Processing (NLP)**
- **Linguistic Analysis**: Spacy (en_core_web_md) for part-of-speech tagging, named entity recognition, and dependency parsing
- **Vocabulary Metrics**:
  - **Vocabulary Richness**: Unique words / total words ratio
  - **Type-Token Ratio (TTR)**: Lexical diversity measurement
  - **Complex Word Analysis**: Syllable counting to identify words with 3+ syllables
  - **Compression Ratio**: GZIP compression to detect repetitive patterns
  - **Distinct-N Metrics**: Distinct-1 and Distinct-2 for n-gram diversity
- **Sentence Structure Analysis**:
  - Average sentence length and standard deviation
  - Sentence opener pattern detection (weak patterns: "it", "there is/are", "this", "I")
  - Subject-verb combination diversity using dependency parsing
  - Complex sentence identification (>15 words)
- **Content Quality**:
  - Topic coherence via sentence-to-document vector similarity (cosine similarity)
  - Keyword extraction from noun chunks with TF-IDF weighting
  - Semantic coherence scoring using spaCy word embeddings

#### 4. **AI-Enhanced Feedback Generation**
- **OpenAI Integration**: GPT-3.5-turbo generates contextual coaching feedback based on transcript analysis
- **Vocabulary Enhancement**: AI suggests sophisticated word replacements while preserving original meaning
- **Pronunciation Analysis**: Detects mispronunciations and unclear articulation patterns
- **Contextual Questions**: Generates follow-up questions based on detected topics, entities, and content gaps

### Scoring Algorithm

#### **Composite Score Formula** (Research-Backed)
```
Composite Score = (Voice Delivery × 0.32) + (Visual Presence × 0.28) + 
                  (Narrative Clarity × 0.26) + (Engagement × 0.14)
```

#### **Voice Delivery Score** (0-100)
- **Base Score**: 70 (generous baseline)
- **Positive Factors**:
  - Speaking rate 120-180 WPM: +15 points
  - Filler ratio <2%: +10 points
  - Pitch stability (std <30 Hz): +10 points
  - Voice consistency (breaks <10%): +10 points
- **Penalties**:
  - Filler ratio >8%: -15 points (capped)
  - Long pauses >3: -10 points (2 points per additional pause)
  - Silence >30%: -10 points (capped)
  - Stammering detected: -5 points

#### **Visual Presence Score** (0-100)
- **Emotion Stability**: Dominant emotion ratio × 20 points
- **Eye Contact**: Average eye contact percentage × 30 points
- **Face Detection Rate**: Frame visibility × 20 points
- **Negative Emotion Penalty**: -15 points if negative emotions >30%
- **Tension Penalty**: Logistic function penalizing facial tension

#### **Narrative Clarity Score** (0-92, capped)
- **Base Score**: 50 (average presentation)
- **Vocabulary Richness** (0-25 points):
  - Richness ≥0.8: +25 | 0.6-0.8: +18 | 0.5-0.6: +12 | 0.4-0.5: +6 | <0.3: -5
- **Type-Token Ratio** (0-15 points):
  - TTR ≥0.7: +15 | 0.6-0.7: +12 | 0.5-0.6: +8 | 0.4-0.5: +4 | <0.4: -5
- **Complex Words** (0-10 points):
  - Ideal range 20-30%: +10 | 15-20% or 30-40%: +6 | <10%: -5 | >50%: -10
- **Sentence Structure** (0-10 points):
  - Ideal length 12-18 words: +10 | Acceptable 8-12 or 18-25: +5
- **Content Coherence**: Coherence score × 15 points

#### **Calibration & Normalization**
- **Post-Calibration**: Applies research-backed calibration (Kolen & Brennan method) with contrast factor 1.22 and center point 57.0
- **Score Ranges**:
  - 0-30: Poor | 31-50: Below Average | 51-70: Average | 71-85: Good | 86-95: Excellent | 96-100: Exceptional

---

## Tools & Technology

### Backend Stack (Python 3.10.13)
- **Web Framework**: FastAPI 0.110.0, Uvicorn 0.27.1 (async ASGI server)
- **Deep Learning**:
  - **Whisper**: OpenAI's speech recognition model (large-v3)
  - **DeepFace** 0.0.97: Facial emotion analysis
  - **Transformers** 4.37.2: Hugging Face model hub integration
- **Computer Vision**:
  - **OpenCV** 4.8.1.78: Video frame extraction and processing
  - **MTCNN** 0.1.1: Face detection
  - **Pillow** 10.2.0: Image manipulation
- **Audio Processing**:
  - **Librosa** 0.9.2: Audio feature extraction (pitch, energy, ZCR)
  - **Soundfile** 0.12.1: Audio I/O
- **NLP**:
  - **Spacy** 3.7.4: Linguistic analysis and entity recognition
  - **NLTK** 3.8.1: Text preprocessing and tokenization
  - **SentencePiece** 0.1.99: Subword tokenization
- **Data Science**:
  - **NumPy** 1.23.5: Numerical computations
  - **Pandas** 2.0.3: Data manipulation
  - **Scikit-learn** 1.3.2: TF-IDF vectorization, cosine similarity
  - **SciPy** 1.11.4: Statistical functions
- **Video Processing**:
  - **MoviePy** 2.1.1: Video/audio separation
  - **ImageIO-FFmpeg** 0.6.0: FFmpeg integration
- **AI Integration**:
  - **OpenAI API** 1.12.0: GPT-3.5-turbo for feedback generation
- **Reporting**:
  - **ReportLab** 4.1.0: PDF generation
  - **Matplotlib** 3.7.4: Data visualization
  - **Seaborn** 0.13.2: Statistical plots
- **Database**: SQLAlchemy 2.0.27, Alembic 1.13.1 (SQLite)
- **Security**: Python-JOSE 3.3.0, Passlib 1.7.4 (JWT authentication)

### Frontend Stack
- **Framework**: React 18.2.0 with Vite 5.0.8 (build tool)
- **Routing**: React Router DOM 6.20.0
- **Animations**: Framer Motion 12.23.24, GSAP 3.12.5
- **State Management**: React Context API (AuthContext)
- **Styling**: Custom CSS with modern design patterns
- **Integration**: Firebase 12.4.0 (authentication/hosting)

### Development Tools
- **Testing**: Pytest 7.4.4, Pytest-Asyncio 0.21.1
- **Deployment**: Gunicorn 21.2.0 (production server)
- **Environment**: Python-dotenv 1.0.1 for configuration management

---

## Outcomes

### Performance Metrics

#### **Overall Score** (0-100)
Holistic presentation quality combining all analysis dimensions. Weighted average of voice, visual, narrative, and engagement scores with post-calibration normalization.

#### **Voice Confidence** (0-100)
- **Speaking Rate**: Words per minute (WPM) - Target: 120-180 WPM
- **Filler Word Ratio**: Percentage of speech consisting of fillers (um, uh, like, basically)
- **Pause Analysis**:
  - Total pauses detected
  - Long pauses (>2.0s) count
  - Silence percentage
- **Pitch Variation**: Standard deviation of fundamental frequency (Hz)
- **Volume Consistency**: RMS energy stability and voice break detection

#### **Facial Confidence** (0-100)
- **Eye Contact**: Percentage of time maintaining camera gaze (Target: >70%)
- **Emotion Stability**: Consistency of dominant facial expression (0-100%)
- **Face Detection Rate**: Percentage of frames with clear face visibility
- **Emotion Timeline**: Frame-by-frame emotion tracking with temporal smoothing

#### **Vocabulary Score** (30-92, capped)
- **Vocabulary Richness**: Unique word ratio (Target: >0.5)
- **Complex Word Usage**: Percentage of 3+ syllable words (Target: 20-30%)
- **Sentence Structure**:
  - Average sentence length (Target: 12-18 words)
  - Sentence variety (standard deviation)
  - Complex sentence ratio (>15 words)
- **Content Coherence**: Topic consistency score (0-100)
- **Keyword Coverage**: Keyword density and distribution

#### **Advanced Analytics**
- **Filler Trend**: Time-series analysis of filler word clusters (60-second buckets)
- **Pause Cadence**: Distribution of pause durations with timestamps
- **Opening Confidence**: First 30-second analysis for strong starts
- **Sentence Opener Diversity**: Pattern detection for weak openers (it, there is/are, this)
- **Subject-Verb Diversity**: Grammatical pattern variety
- **Compression Ratio**: Text and POS-sequence compression for repetition detection
- **Distinct-N Metrics**: Distinct-1 and Distinct-2 for vocabulary diversity

### Accuracy Parameters

#### **Transcription Accuracy**
- **Whisper Model**: Large-v3 for proper noun recognition
- **Language**: English (forced)
- **Temperature**: 0.0 (deterministic)
- **No-Speech Threshold**: 0.4 (captures filler sounds)
- **Compression Ratio Threshold**: 2.4 (detects hallucinations)

#### **Scoring Calibration**
- **Weighted Formula**: Voice (32%), Visual (28%), Narrative (26%), Engagement (14%)
- **Calibration Method**: Kolen & Brennan research-backed normalization
- **Contrast Factor**: 1.22 (amplifies score differences)
- **Center Point**: 57.0 (average presentation baseline)
- **Score Bounds**: Minimum 30, Maximum 100 (realistic distribution)

#### **Validation Thresholds**
- **Minimum Metric Confidence**: 0.65 (shows "Insufficient Data" if below)
- **Speaking Rate Bounds**: 30-300 WPM (physically plausible range)
- **Face Detection Confidence**: >0.5 (minimum for reliable analysis)

---

## Keywords

AI Coaching; Computer Vision; Multimodal Analysis; Natural Language Processing; Public Speaking; Voice Analysis
