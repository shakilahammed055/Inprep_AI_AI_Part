from fastapi import APIRouter
from .endpoints import resume

api_router = APIRouter()
api_router.include_router(resume.router, prefix="/resume", tags=["resume"])
