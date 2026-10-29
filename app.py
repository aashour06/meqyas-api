"""
MEQYAS AI — Vercel Serverless Application Entry Point
=====================================================
Exposes the unified FastAPI application instance `app` from `main.py`
for Vercel serverless functions without instantiating a duplicate application.
"""

from main import app

__all__ = ["app"]
