import os
import json
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, MetaData, Table, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import text, func

# Get database URL from environment variable
DATABASE_URL = os.environ.get("DATABASE_URL")

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Create a base class for declarative models
Base = declarative_base()

# Define database models
class Dataset(Base):
    """Model for storing uploaded datasets"""
    __tablename__ = 'datasets'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    data_type = Column(String(50), nullable=False)  # 'financial', 'generic'
    source_type = Column(String(50), nullable=False)  # 'yahoo', 'uploaded', 'sample'
    source_details = Column(JSON, nullable=True)  # For ticker, date range, etc.
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationship with Narrative model
    narratives = relationship("Narrative", back_populates="dataset")

class Narrative(Base):
    """Model for storing generated narratives"""
    __tablename__ = 'narratives'
    
    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey('datasets.id'))
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    narrative_type = Column(String(50), nullable=False)  # 'financial', 'generic_data'
    target_audience = Column(String(50), nullable=True)
    depth_level = Column(Integer, nullable=True)
    consistency_score = Column(Float, nullable=True)
    consistency_report = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationship with Dataset model
    dataset = relationship("Dataset", back_populates="narratives")

# Create all tables if they don't exist
Base.metadata.create_all(engine)

# Create a session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Database operations
def save_dataset(name, data_type, source_type, source_details=None, description=None):
    """Save dataset information to the database"""
    with SessionLocal() as session:
        dataset = Dataset(
            name=name,
            data_type=data_type,
            source_type=source_type,
            source_details=source_details,
            description=description
        )
        session.add(dataset)
        session.commit()
        session.refresh(dataset)
        return dataset.id

def save_narrative(dataset_id, title, content, narrative_type, consistency_score=None, consistency_report=None, target_audience=None, depth_level=None):
    """Save a generated narrative to the database"""
    with SessionLocal() as session:
        narrative = Narrative(
            dataset_id=dataset_id,
            title=title,
            content=content,
            narrative_type=narrative_type,
            consistency_score=consistency_score,
            consistency_report=consistency_report,
            target_audience=target_audience,
            depth_level=depth_level
        )
        session.add(narrative)
        session.commit()
        session.refresh(narrative)
        return narrative.id

def get_dataset(dataset_id):
    """Get dataset by ID"""
    with SessionLocal() as session:
        dataset = session.query(Dataset).filter(Dataset.id == dataset_id).first()
        return dataset

def get_narrative(narrative_id):
    """Get narrative by ID"""
    with SessionLocal() as session:
        narrative = session.query(Narrative).filter(Narrative.id == narrative_id).first()
        return narrative

def get_all_datasets(limit=100):
    """Get all datasets with optional limit"""
    with SessionLocal() as session:
        datasets = session.query(Dataset).order_by(Dataset.created_at.desc()).limit(limit).all()
        return datasets

def get_narratives_for_dataset(dataset_id, limit=10):
    """Get narratives for a specific dataset"""
    with SessionLocal() as session:
        narratives = session.query(Narrative).filter(Narrative.dataset_id == dataset_id).order_by(Narrative.created_at.desc()).limit(limit).all()
        return narratives

def get_recent_narratives(limit=10):
    """Get most recent narratives"""
    with SessionLocal() as session:
        narratives = session.query(Narrative).order_by(Narrative.created_at.desc()).limit(limit).all()
        return narratives

def dataset_to_dict(dataset):
    """Convert dataset model to dictionary"""
    return {
        'id': dataset.id,
        'name': dataset.name,
        'description': dataset.description,
        'data_type': dataset.data_type,
        'source_type': dataset.source_type,
        'source_details': dataset.source_details,
        'created_at': dataset.created_at.strftime('%Y-%m-%d %H:%M:%S') if dataset.created_at else None
    }

def narrative_to_dict(narrative):
    """Convert narrative model to dictionary"""
    return {
        'id': narrative.id,
        'dataset_id': narrative.dataset_id,
        'title': narrative.title,
        'content': narrative.content,
        'narrative_type': narrative.narrative_type,
        'target_audience': narrative.target_audience,
        'depth_level': narrative.depth_level,
        'consistency_score': narrative.consistency_score,
        'created_at': narrative.created_at.strftime('%Y-%m-%d %H:%M:%S') if narrative.created_at else None
    }

def search_narratives(search_term, limit=10):
    """Search narratives by content"""
    with SessionLocal() as session:
        narratives = session.query(Narrative).filter(
            Narrative.content.ilike(f'%{search_term}%')
        ).order_by(Narrative.created_at.desc()).limit(limit).all()
        return narratives
