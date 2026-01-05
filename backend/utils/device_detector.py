"""
Device detection and model initialization utilities
"""
import torch
import logging
from pathlib import Path
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    spacy = None
    SPACY_AVAILABLE = False
    print("spaCy not available - some text analysis features will be limited")
    
from ..core.settings import MODELS_DIR, SPACY_MODEL

logger = logging.getLogger(__name__)

class DeviceManager:
    """Manages device selection and model initialization"""
    
    def __init__(self):
        self.device = self._detect_device()
        self.spacy_model = None
        self._initialize_models()
    
    def _detect_device(self):
        """Detect available device (GPU/CPU)"""
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"GPU detected: {torch.cuda.get_device_name()}")
            logger.info(f"CUDA version: {torch.version.cuda}")
        else:
            device = "cpu"
            logger.info("No GPU detected, using CPU")
        
        return device
    
    def _initialize_models(self):
        """Initialize models - spaCy is now lazy loaded"""
        # spaCy is lazy loaded in get_spacy_model() to avoid import issues
        pass
    
    def get_device(self):
        """Get the current device"""
        return self.device
    
    def get_spacy_model(self):
        """Get the spaCy model (lazy loading)"""
        if not SPACY_AVAILABLE:
            return None

        if self.spacy_model is None and spacy is not None:
            try:
                # Try to load the model
                self.spacy_model = spacy.load(SPACY_MODEL)
                logger.info(f"Loaded spaCy model: {SPACY_MODEL}")
            except OSError:
                logger.warning(f"spaCy model {SPACY_MODEL} not found. Please install it with:")
                logger.warning(f"python -m spacy download {SPACY_MODEL}")
                self.spacy_model = None
            except Exception as e:
                logger.warning(f"Could not load spaCy model: {e}")
                self.spacy_model = None

        return self.spacy_model
    
    def is_gpu_available(self):
        """Check if GPU is available"""
        return self.device == "cuda"
    
    def get_device_info(self):
        """Get detailed device information"""
        info = {
            "device": self.device,
            "is_gpu": self.is_gpu_available(),
            "spacy_model_loaded": self.spacy_model is not None
        }
        
        if self.is_gpu_available():
            info.update({
                "gpu_name": torch.cuda.get_device_name(),
                "gpu_memory": torch.cuda.get_device_properties(0).total_memory,
                "cuda_version": torch.version.cuda
            })
        
        return info

# Global device manager instance
device_manager = DeviceManager()
