#gate-api/app.py
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
import jwt
import os
from datetime import datetime, timedelta
import secrets

app = FastAPI(title="Gate API", version="0.1")

# --- Security Configuration ---
# Use environment variable or generate secure random key
#SECRET_KEY = os.getenv("SECRET_KEY")
SECRET_KEY = os.getenv("SECRET_KEY", "default_secret")
if not SECRET_KEY:
    # Generate a secure random key if not provided
    SECRET_KEY = secrets.token_urlsafe(32)
    print(f"WARNING: Using auto-generated secret key. For production, set SECRET_KEY environment variable!")

ALGORITHM = "HS256"
TOKEN_EXPIRE_MINUTES = int(os.getenv("TOKEN_EXPIRE_MINUTES", "180"))

# User database (in a real-world app, this would be a proper database)
USERS_DB = {
    "user": {"password": "user_pass", "role": "user"},
    "admin": {"password": "admin_pass", "role": "admin"}
}

class LoginRequest(BaseModel):
    username: str
    password: str
# You'll need to add the verify_token dependency function
async def verify_token(authorization: str = Header(...)):
    """Dependency to verify JWT token"""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    
    token = authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    
@app.post("/login")
def login(request: LoginRequest):
    # Input validation
    if not request.username or not request.password:
        raise HTTPException(status_code=400, detail="Username and password required")
    
    user = USERS_DB.get(request.username)
    if not user or user["password"] != request.password:
        # Use consistent error messaging to avoid username enumeration
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Create JWT token with expiration
    expires = datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRE_MINUTES)
    to_encode = {
        "sub": request.username,  # Standard JWT subject claim
        "username": request.username, 
        "role": user["role"], 
        "exp": expires,
        "iat": datetime.utcnow()  # Issued at time
    }
    
    try:
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Token generation failed")

    return {
        "token": encoded_jwt, 
        "role": user["role"],
        "expires_in": TOKEN_EXPIRE_MINUTES * 160,  # seconds
        "token_type": "bearer"
    }

@app.get("/")
def root():
    return {
        "status": "API up and running",
        "version": "0.2",
        "endpoints": ["/login", "/health"],
    }

@app.get("/health")
def health():
    return {
        "status": "healthy", 
        "service": "gate-api",
        "authentication": "jwt",
        "users_configured": len(USERS_DB)
    }


# Add token validation endpoint for other services to verify tokens
@app.post("/validate-token")
async def validate_token(authorization: str = Header(...)):
    """Validate JWT token and return user info"""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    
    token = authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return {
            "valid": True,
            "username": payload["username"],
            "role": payload["role"],
            "expires": payload["exp"]
        }
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        raise HTTPException(status_code=401, detail="Token validation failed")

# Add user info endpoint for authenticated users
@app.get("/user-info")
async def get_user_info(user: dict = Depends(verify_token)):
    """Get information about the currently authenticated user"""
    return {
        "username": user["username"],
        "role": user["role"],
        "authenticated": True
    }

# You'll need to add the verify_token dependency function
async def verify_token(authorization: str = Header(...)):
    """Dependency to verify JWT token"""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    
    token = authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")