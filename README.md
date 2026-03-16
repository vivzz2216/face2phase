# Face2Phase

AI-powered presentation analysis platform for analyzing video/audio presentations with comprehensive feedback on delivery, word choice, and advanced analytics.

## Features

- **Video/Audio Analysis**: Upload videos or audio files for comprehensive analysis
- **Delivery Analytics**: Pacing, eye contact, pause detection
- **Word Choice Analytics**: Filler words, weak words, vocabulary quality
- **Advanced Analytics**: Filler trends, pause cadence, opening confidence, emotion timeline
- **AI-Powered Feedback**: Detailed coaching feedback with contextual follow-up questions
- **Chatbot**: Interactive assistant for analysis questions

## Setup

### Backend (Python)

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run server
python run_server.py
```

### Frontend (Node.js)

```bash
# Install dependencies
npm install

# Run development server
npm run dev
```

## Tech Stack

- **Backend**: FastAPI, Python
- **Frontend**: React, Vite
- **AI/ML**: OpenAI Whisper, DeepFace, spaCy, Transformers
- **Database**: SQLAlchemy

## License

MIT
