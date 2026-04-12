from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.routers import health, ingest, query


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="Autonomous Document Intelligence Agent",
    description="Multi-agent RAG system with LangGraph, Claude API, and pgvector",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(ingest.router, prefix="/api")
app.include_router(query.router, prefix="/api")
