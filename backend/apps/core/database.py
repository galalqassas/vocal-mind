"""Database configuration."""
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings

# Use SQLite for dev, can switch to Postgres via connection string in .env
# For SQLite, we need a slight tweak to the URL to make it async compatible if using aiosqlite
# But for now, let's assume valid async URL in settings or default to a local file
DATABASE_URL = "sqlite+aiosqlite:///./vocal_mind.db"

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting async database session."""
    async with AsyncSessionLocal() as session:
        yield session
