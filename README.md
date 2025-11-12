# Face2Phase - AI Communication Feedback System

A comprehensive AI-powered communication analysis platform that provides detailed feedback on voice, speech, facial expressions, and vocabulary from audio/video recordings.

## Features

- 🎤 **Voice Analysis**: Analyzes tone, steadiness, energy, and detects voice breaks
- 🗣️ **Speech Clarity**: Identifies filler words, measures pause frequency, evaluates fluency
- 🧠 **Vocabulary Assessment**: Analyzes vocabulary richness and linguistic clarity
- 🎭 **Facial Analysis**: Detects emotions, eye contact, and engagement from video
- 📊 **Comprehensive Reports**: Generates detailed PDF/Word reports with analytics
- 🔒 **Privacy-First**: All processing happens locally - no data sent to external servers
- 🎨 **Modern Dashboard**: Dark theme with collapsible sidebar, minimal icons, and smooth animations
- ⚙️ **Score Boost**: Configurable score adjustment for analysis results

## Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **Whisper** - Open-source speech recognition
- **DeepFace** - Facial analysis
- **spaCy** - NLP and text analysis
- **SQLAlchemy** - Database ORM
- **OpenAI API** (optional) - Enhanced analysis

### Frontend
- **React** - UI framework
- **Vite** - Build tool
- **Firebase** - Authentication
- **Framer Motion** - Animations
- **React Router** - Routing
- **Font Awesome** - Icon library

## Installation

### Prerequisites
- Python 3.8+
- Node.js 16+
- 4GB RAM minimum (8GB recommended)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Face2Phase
   ```

2. **Set up Python environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   python -m spacy download en_core_web_md
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here  # Optional
   JWT_SECRET_KEY=your_jwt_secret_key_here  # Generate with: python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

4. **Set up frontend**
   ```bash
   npm install
   ```

### Running the Application

#### Option 1: Use the startup script (Windows)
```bash
start.bat
```

#### Option 2: Manual start

**Backend:**
```bash
venv\Scripts\activate
python app.py
```

**Frontend:**
```bash
npm run dev
```

The application will be available at:
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Implementation Details

### Dashboard UI Redesign

The dashboard was completely redesigned with a focus on compact, professional design:

#### Typography & Spacing
- Reduced all font sizes by 20-30% for better space utilization
- Reduced padding by 25-35% across all components
- Implemented CSS variables for consistent scaling:
  - `--font-xl: 1.5rem` (was 2rem)
  - `--font-lg: 1.25rem` (was 1.75rem)
  - `--spacing-xl: 1.5rem` (was 2rem)

#### Minimal Icon System
- Replaced all emoji icons with Font Awesome icons
- Icons are uncolored (neutral gray/white) with no gradients
- Consistent icon sizing (`1rem`) throughout
- Icon mapping:
  - Navigation: `fa-home`, `fa-chart-line`, `fa-clock`, `fa-gear`
  - Stats: `fa-chart-line`, `fa-star`, `fa-microphone`, `fa-face-smile`
  - Upload: `fa-upload`

#### Collapsible Sidebar
- Collapsed state (80px): Icon-only navigation
- Expanded state (280px): Full layout with labels
- Hover expansion: Collapsed sidebar expands on hover with smooth animation
- Toggle button: Manual expand/collapse control in header
- Guest filtering: History navigation hidden for guest users

#### Card Optimization
- Stats grid: `minmax(180px, 1fr)` (reduced from 240px)
- Activity grid: `minmax(280px, 1fr)` (reduced from 320px)
- Border radius: `12px` (reduced from 20px)
- Reduced gaps between cards for tighter layout

### Score Boost Feature

A configurable score boost system adds points to all analysis scores:

- **Configuration**: Set `SCORE_BOOST = 30` in `config.py` (backend) or `Dashboard.jsx` (frontend)
- **Implementation**: `applyScoreBoost()` helper function adds points to:
  - Overall score
  - Voice confidence
  - Facial confidence
  - Vocabulary score
- **Capping**: Scores are capped at 100 maximum
- **Application**: Boost is applied when:
  - Displaying results after analysis
  - Calculating average scores in stats
  - Storing scores in history

Example: If you get 39, it becomes 69 (39 + 30).

### Loading Screen Animation

Modern loading screen with engaging animations:

- **Pulsing Dots**: Three animated dots at the top with staggered pulse effect
- **Wave Effect**: Animated wave on progress bar with glow effect
- **Stage Indicators**: Three stages (Processing, Analyzing, Finalizing) with icons
- **Smooth Transitions**: All animations use Framer Motion for smooth transitions

### Security Enhancements

- **Environment Variables**: All secrets moved to `.env` file
- **JWT Authentication**: Secure token-based authentication
- **Password Hashing**: pbkdf2_sha256 for secure password storage
- **CORS Protection**: Restricted origins and headers
- **Input Validation**: Pydantic models for request validation
- **Authorization Headers**: Token passed in `Authorization` header instead of query params

### Authentication System

- **Firebase Integration**: Google Sign-In via Firebase
- **Guest Mode**: Users can continue without authentication
- **Session Management**: JWT tokens for authenticated sessions
- **Protected Routes**: History and analytics features require authentication

## Project Structure

```
Face2Phase/
├── app.py                 # Main FastAPI application
├── config.py             # Configuration settings (includes SCORE_BOOST)
├── models/               # AI model implementations
│   ├── audio_analyzer.py
│   ├── facial_analyzer.py
│   ├── text_analyzer.py
│   ├── enhanced_audio_analyzer.py
│   ├── report_generator.py
│   └── ...
├── utils/                # Utility modules
│   ├── file_handler.py
│   └── device_detector.py
├── src/                  # React frontend
│   ├── components/
│   │   ├── Dashboard.jsx      # Main dashboard with score boost
│   │   ├── Dashboard.css      # Dark theme styling
│   │   ├── LoadingScreen.jsx   # Modern loading animation
│   │   └── ...
│   ├── context/
│   │   └── AuthContext.jsx    # Firebase authentication
│   └── ...
├── templates/            # HTML templates
├── static/              # Static assets
├── uploads/             # Uploaded files (gitignored)
└── reports/             # Generated reports (gitignored)
```

## API Endpoints

### Authentication
- `POST /auth/register` - Register new user (body: username, email, password, full_name)
- `POST /auth/login` - User login (body: email, password)
- `GET /auth/logout` - Logout user (requires Authorization header)

### File Processing
- `POST /upload` - Upload audio/video file (multipart/form-data)
- `GET /status/{session_id}` - Check processing status
- `GET /api/report/{session_id}` - Get analysis report

### User Data
- `GET /user/profile` - Get user profile (requires Authorization header)
- `GET /user/analyses` - Get user's analysis history (requires Authorization header)

### Analysis
- `POST /analyze/enhanced` - Enhanced analysis with OpenAI (requires Authorization header)

### Reports
- `GET /download/{session_id}?format={pdf|word|json}` - Download report in various formats

### Health
- `GET /health` - Health check endpoint
- `GET /routes` - List all registered routes

See http://localhost:8000/docs for complete API documentation with interactive testing.

## Configuration

### Score Boost Setting

Edit `config.py` or `src/components/Dashboard.jsx` to adjust the score boost:

```python
# config.py
SCORE_BOOST = 30  # Add 30 points to every analysis score
```

```javascript
// src/components/Dashboard.jsx
const SCORE_BOOST = 30 // Add 30 points to every analysis score
```

### Other Settings

Edit `config.py` to customize:
- Model sizes (Whisper model: tiny, base, small, medium, large)
- File size limits (default: 500MB)
- Analysis thresholds (pause detection, filler words)
- Processing settings (frame extraction FPS, face confidence)

## Key Implementation Features

### Dark Theme Dashboard
- Deep navy background with gradient accents
- Glassmorphism effects on cards
- Smooth transitions and hover effects
- Professional color scheme with CSS variables

### Responsive Design
- Mobile-first approach
- Collapsible sidebar adapts to screen size
- Grid layouts adjust for tablet/mobile
- Touch-friendly interactions

### Error Handling
- Comprehensive error messages for connection issues
- Graceful degradation for missing data
- User-friendly alerts for API failures
- Proper validation and error states

### Performance Optimizations
- Lazy loading of components
- Efficient state management
- Optimized re-renders with React hooks
- Background processing for file analysis

## Security

- All API keys stored in environment variables (never committed)
- JWT-based authentication with secure token generation
- Secure password hashing (pbkdf2_sha256)
- CORS protection with restricted origins
- Input validation with Pydantic models
- Authorization headers instead of query parameters

## Development

### Code Quality
- Modular component structure
- Consistent naming conventions
- CSS variables for theming
- Reusable helper functions
- Comprehensive error handling

### Testing
- Backend API endpoints tested via FastAPI docs
- Frontend components tested manually
- Error scenarios handled gracefully

## License

[Your License Here]

## Support

For issues and questions, please open an issue on GitHub.
