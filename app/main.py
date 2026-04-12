from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import health, ingest, query


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="Autonomous Document Intelligence Agent",
    description="Multi-agent RAG system with LangGraph and pgvector",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(ingest.router, prefix="/api")
app.include_router(query.router, prefix="/api")
