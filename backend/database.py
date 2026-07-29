from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base

# MySQL connection string
# Format: mysql+pymysql://username:password@localhost:3306/database_name
DATABASE_URL = "mysql+pymysql://physics_user:physics_pass123@localhost:3306/particle_physics"

# Create engine
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # Verify connections before using
    pool_recycle=3600,   # Recycle connections after 1 hour
    echo=False           # Set to True for SQL query logging
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create all tables
def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully!")

# Dependency for FastAPI routes
def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
