"""
Authentication and User Management
"""
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict
from fastapi import HTTPException, status

try:
    from jose import jwt
    JOSE_AVAILABLE = True
except ImportError:
    jwt = None
    JOSE_AVAILABLE = False

try:
    from passlib.context import CryptContext
    PASSLIB_AVAILABLE = True
except ImportError:
    CryptContext = None
    PASSLIB_AVAILABLE = False

from ..core.settings import JWT_SECRET_KEY, JWT_ALGORITHM, JWT_EXPIRATION_HOURS
from ..db.database import db_manager, User

# Password hashing - using pbkdf2_sha256 for better compatibility
if PASSLIB_AVAILABLE:
    pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
else:
    pwd_context = None

class AuthManager:
    """Authentication and user management"""
    
    def __init__(self):
        self.secret_key = JWT_SECRET_KEY
        self.algorithm = JWT_ALGORITHM
        self.expiration_hours = JWT_EXPIRATION_HOURS
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        if not PASSLIB_AVAILABLE or not pwd_context:
            raise HTTPException(status_code=500, detail="Password hashing not available. Install passlib.")
        # Ensure password is not too long for bcrypt
        if len(password) > 72:
            password = password[:72]
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        if not PASSLIB_AVAILABLE or not pwd_context:
            raise HTTPException(status_code=500, detail="Password verification not available. Install passlib.")
        return pwd_context.verify(plain_password, hashed_password)
    
    def create_access_token(self, data: dict) -> str:
        """Create JWT access token"""
        if not JOSE_AVAILABLE or not jwt:
            raise HTTPException(status_code=500, detail="JWT not available. Install python-jose.")
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(hours=self.expiration_hours)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[Dict]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    def register_user(self, username: str, email: str, password: str, full_name: str = None) -> Dict:
        """Register a new user"""
        session = db_manager.get_session()
        try:
            # Check if user already exists
            existing_user = session.query(User).filter(
                (User.username == username) | (User.email == email)
            ).first()
            
            if existing_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username or email already registered"
                )
            
            # Create new user
            hashed_password = self.hash_password(password)
            user = User(
                username=username,
                email=email,
                hashed_password=hashed_password,
                full_name=full_name or username
            )
            
            session.add(user)
            session.commit()
            session.refresh(user)
            
            # Create access token
            token_data = {"sub": str(user.id), "username": user.username}
            access_token = self.create_access_token(token_data)
            
            return {
                "user_id": user.id,
                "username": user.username,
                "email": user.email,
                "access_token": access_token,
                "token_type": "bearer"
            }
            
        except HTTPException:
            raise
        except Exception as e:
            session.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Registration failed: {str(e)}"
            )
        finally:
            session.close()
    
    def authenticate_user(self, username: str, password: str) -> Dict:
        """Authenticate user login"""
        session = db_manager.get_session()
        try:
            user = session.query(User).filter(User.username == username).first()
            
            if not user or not self.verify_password(password, user.hashed_password):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid username or password"
                )
            
            if not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Account is deactivated"
                )
            
            # Update last login
            user.last_login = datetime.utcnow()
            session.commit()
            
            # Create access token
            token_data = {"sub": str(user.id), "username": user.username}
            access_token = self.create_access_token(token_data)
            
            return {
                "user_id": user.id,
                "username": user.username,
                "email": user.email,
                "full_name": user.full_name,
                "is_premium": user.is_premium,
                "access_token": access_token,
                "token_type": "bearer"
            }
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Authentication failed: {str(e)}"
            )
        finally:
            session.close()
    
    def get_current_user(self, token: str) -> User:
        """Get current user from token"""
        payload = self.verify_token(token)
        user_id = payload.get("sub")
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        session = db_manager.get_session()
        try:
            user = session.query(User).filter(User.id == int(user_id)).first()
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found"
                )
            return user
        finally:
            session.close()
    
    def logout_user(self, session_token: str) -> bool:
        """Logout user (JWT tokens are stateless)"""
        # Since we're using JWT tokens, logout is handled client-side
        # by simply discarding the token
        return True

# Global auth manager instance
auth_manager = AuthManager()
