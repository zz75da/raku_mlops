# gate-api/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import jwt
import os
from datetime import datetime, timedelta

app = FastAPI(title="Gate API", version="0.1")

# Use a secure secret key from environment variables
SECRET_KEY = os.getenv("SECRET_KEY", "your-super-secret-key")
ALGORITHM = "HS256"

# User database (in a real-world app, this would be a database)
USERS_DB = {
    "user": {"password": "user_pass", "role": "user"},
    "admin": {"password": "admin_pass", "role": "admin"}
}

class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/login")
def login(request: LoginRequest):
    user = USERS_DB.get(request.username)
    if not user or user["password"] != request.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Create a JWT token
    expires = datetime.utcnow() + timedelta(minutes=60)
    to_encode = {"username": request.username, "role": user["role"], "exp": expires}
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    return {"token": encoded_jwt, "role": user["role"]}

@app.get("/health")
def health_check():
    return {"status": "healthy"}