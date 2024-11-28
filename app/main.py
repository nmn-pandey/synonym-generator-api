from fastapi import FastAPI
from app.api.routes import router

app = FastAPI()

# Include API routes
app.include_router(router)