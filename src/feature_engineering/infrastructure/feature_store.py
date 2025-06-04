"""
Feature Store Module

Provides services for storing, retrieving, and managing features and feature sets.
"""
import logging
import json
import uuid
import pandas as pd
from typing import Dict, List, Optional, Set, Union, Any
from datetime import datetime, timedelta

# For database operations
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, String, Float, DateTime, Boolean, Integer, Text, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.future import select
from sqlalchemy import and_, or_, desc, func

# For feature entity conversions
from src.feature_engineering.domain.entities.feature import (
    Feature, FeatureType, FeatureCategory, FeatureScope, FeatureTimeframe, FeatureMetadata
)
from src.feature_engineering.domain.entities.feature_set import (
    FeatureSet, FeatureSetType, FeatureSetStatus, FeatureSetMetadata
)

logger = logging.getLogger(__name__)

# Define SQLAlchemy Base
Base = declarative_base()


class FeatureModel(Base):
    """SQLAlchemy model for Feature entity."""
    __tablename__ = 'features'
    
    id = Column(UUID(as_uuid=True), primary_key=True)
    name = Column(String(255), nullable=False)
    symbol = Column(String(50), nullable=False)
    type = Column(String(50), nullable=False)
    value = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    lookback_periods = Column(Integer, nullable=False, default=0)
    dependencies = Column(JSONB, nullable=True)
    confidence = Column(Float, nullable=True)
    
    # Metadata columns
    metadata_description = Column(Text, nullable=False)
    metadata_category = Column(String(50), nullable=False)
    metadata_scope = Column(String(50), nullable=False)
    metadata_timeframe = Column(String(50), nullable=False)
    metadata_is_experimental = Column(Boolean, nullable=False, default=False)
    metadata_created_at = Column(DateTime, nullable=False)
    metadata_updated_at = Column(DateTime, nullable=True)
    metadata_version = Column(String(20), nullable=False)
    metadata_tags = Column(JSONB, nullable=True)
    metadata_properties = Column(JSONB, nullable=True)
    
    # Foreign key to feature set
    feature_set_id = Column(UUID(as_uuid=True), ForeignKey('feature_sets.id'), nullable=True)
    
    # Indexes
    __table_args__ = (
        {'schema': 'feature_engineering'}
    )


class FeatureSetModel(Base):
    """SQLAlchemy model for FeatureSet entity."""
    __tablename__ = 'feature_sets'
    
    id = Column(UUID(as_uuid=True), primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    type = Column(String(50), nullable=False)
    status = Column(String(50), nullable=False)
    symbols = Column(JSONB, nullable=False)
    start_timestamp = Column(DateTime, nullable=False)
    end_timestamp = Column(DateTime, nullable=False)
    
    # Metadata columns
    metadata_description = Column(Text, nullable=False)
    metadata_version = Column(String(20), nullable=False)
    metadata_created_at = Column(DateTime, nullable=False)
    metadata_updated_at = Column(DateTime, nullable=True)
    metadata_tags = Column(JSONB, nullable=True)
    metadata_properties = Column(JSONB, nullable=True)
    metadata_author = Column(String(255), nullable=True)
    
    # Indexes
    __table_args__ = (
        {'schema': 'feature_engineering'}
    )


class FeatureStore:
    """Repository for storing and retrieving features and feature sets."""
    
    def __init__(self, db_uri: str):
        """
        Initialize the feature store.
        
        Args:
            db_uri: Database connection URI
        """
        self.db_uri = db_uri
        self.engine = create_async_engine(db_uri, echo=False)
        self.async_session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )
    
    async def initialize(self):
        """Initialize the database schema."""
        try:
            # Create schema if it doesn't exist
            async with self.engine.begin() as conn:
                await conn.execute("CREATE SCHEMA IF NOT EXISTS feature_engineering")
            
            # Create tables
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            logger.info("Feature store initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing feature store: {str(e)}")
            raise
    
    async def save_feature(self, feature: Feature) -> Feature:
        """
        Save a feature to the store.
        
        Args:
            feature: Feature object to save
            
        Returns:
            Saved Feature object
        """
        try:
            async with self.async_session() as session:
                # Check if feature already exists
                stmt = select(FeatureModel).where(FeatureModel.id == feature.id)
                result = await session.execute(stmt)
                feature_model = result.scalars().first()
                
                if feature_model:
                    # Update existing feature
                    feature_model.value = feature.value
                    feature_model.timestamp = feature.timestamp
                    feature_model.lookback_periods = feature.lookback_periods
                    feature_model.dependencies = feature.dependencies
                    feature_model.confidence = feature.confidence
                    
                    # Update metadata
                    feature_model.metadata_updated_at = datetime.utcnow()
                    feature_model.metadata_tags = feature.metadata.tags
                    feature_model.metadata_properties = feature.metadata.properties
                else:
                    # Create new feature model
                    feature_model = FeatureModel(
                        id=feature.id,
                        name=feature.name,
                        symbol=feature.symbol,
                        type=feature.type.name,
                        value=feature.value,
                        timestamp=feature.timestamp,
                        lookback_periods=feature.lookback_periods,
                        dependencies=feature.dependencies,
                        confidence=feature.confidence,
                        
                        # Metadata
                        metadata_description=feature.metadata.description,
                        metadata_category=feature.metadata.category.name,
                        metadata_scope=feature.metadata.scope.name,
                        metadata_timeframe=feature.metadata.timeframe.name,
                        metadata_is_experimental=feature.metadata.is_experimental,
                        metadata_created_at=feature.metadata.created_at,
                        metadata_updated_at=feature.metadata.updated_at,
                        metadata_version=feature.metadata.version,
                        metadata_tags=feature.metadata.tags,
                        metadata_properties=feature.metadata.properties
                    )
                    session.add(feature_model)
                
                await session.commit()
                
                logger.debug(f"Saved feature: {feature.name} for {feature.symbol}")
                return feature
        except Exception as e:
            logger.error(f"Error saving feature: {str(e)}")
            raise
    
    async def get_feature(self, feature_id: uuid.UUID) -> Optional[Feature]:
        """
        Retrieve a feature by ID.
        
        Args:
            feature_id: UUID of the feature to retrieve
            
        Returns:
            Feature object if found, None otherwise
        """
        try:
            async with self.async_session() as session:
                stmt = select(FeatureModel).where(FeatureModel.id == feature_id)
                result = await session.execute(stmt)
                feature_model = result.scalars().first()
                
                if not feature_model:
                    return None
                
                # Convert to domain entity
                return self._model_to_feature(feature_model)
        except Exception as e:
            logger.error(f"Error retrieving feature: {str(e)}")
            raise
    
    async def get_features_by_symbol(
        self,
        symbol: str,
        feature_types: Optional[Set[FeatureType]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Feature]:
        """
        Retrieve features by symbol.
        
        Args:
            symbol: Symbol to filter by
            feature_types: Optional set of feature types to filter by
            start_time: Optional start time filter
            end_time: Optional end time filter
            limit: Maximum number of features to retrieve
            
        Returns:
            List of Feature objects
        """
        try:
            async with self.async_session() as session:
                # Build query
                query = select(FeatureModel).where(FeatureModel.symbol == symbol)
                
                # Apply filters
                if feature_types:
                    type_names = [ft.name for ft in feature_types]
                    query = query.where(FeatureModel.type.in_(type_names))
                
                if start_time:
                    query = query.where(FeatureModel.timestamp >= start_time)
                
                if end_time:
                    query = query.where(FeatureModel.timestamp <= end_time)
                
                # Apply limit and order
                query = query.order_by(desc(FeatureModel.timestamp)).limit(limit)
                
                # Execute query
                result = await session.execute(query)
                feature_models = result.scalars().all()
                
                # Convert to domain entities
                features = [self._model_to_feature(model) for model in feature_models]
                
                return features
        except Exception as e:
            logger.error(f"Error retrieving features by symbol: {str(e)}")
            raise
    
    async def save_feature_set(self, feature_set: FeatureSet) -> FeatureSet:
        """
        Save a feature set and its features to the store.
        
        Args:
            feature_set: FeatureSet object to save
            
        Returns:
            Saved FeatureSet object
        """
        try:
            async with self.async_session() as session:
                async with session.begin():
                    # Check if feature set already exists
                    stmt = select(FeatureSetModel).where(FeatureSetModel.id == feature_set.id)
                    result = await session.execute(stmt)
                    feature_set_model = result.scalars().first()
                    
                    if feature_set_model:
                        # Update existing feature set
                        feature_set_model.name = feature_set.name
                        feature_set_model.type = feature_set.type.name
                        feature_set_model.status = feature_set.status.name
                        feature_set_model.symbols = list(feature_set.symbols)
                        feature_set_model.start_timestamp = feature_set.start_timestamp
                        feature_set_model.end_timestamp = feature_set.end_timestamp
                        
                        # Update metadata
                        feature_set_model.metadata_description = feature_set.metadata.description
                        feature_set_model.metadata_version = feature_set.metadata.version
                        feature_set_model.metadata_updated_at = datetime.utcnow()
                        feature_set_model.metadata_tags = feature_set.metadata.tags
                        feature_set_model.metadata_properties = feature_set.metadata.properties
                        feature_set_model.metadata_author = feature_set.metadata.author
                    else:
                        # Create new feature set model
                        feature_set_model = FeatureSetModel(
                            id=feature_set.id,
                            name=feature_set.name,
                            type=feature_set.type.name,
                            status=feature_set.status.name,
                            symbols=list(feature_set.symbols),
                            start_timestamp=feature_set.start_timestamp,
                            end_timestamp=feature_set.end_timestamp,
                            
                            # Metadata
                            metadata_description=feature_set.metadata.description,
                            metadata_version=feature_set.metadata.version,
                            metadata_created_at=feature_set.metadata.created_at,
                            metadata_updated_at=feature_set.metadata.updated_at,
                            metadata_tags=feature_set.metadata.tags,
                            metadata_properties=feature_set.metadata.properties,
                            metadata_author=feature_set.metadata.author
                        )
                        session.add(feature_set_model)
                    
                    # Save all features in the set
                    for feature in feature_set.features:
                        # Check if feature already exists
                        stmt = select(FeatureModel).where(FeatureModel.id == feature.id)
                        result = await session.execute(stmt)
                        feature_model = result.scalars().first()
                        
                        if feature_model:
                            # Update existing feature
                            feature_model.name = feature.name
                            feature_model.symbol = feature.symbol
                            feature_model.type = feature.type.name
                            feature_model.value = feature.value
                            feature_model.timestamp = feature.timestamp
                            feature_model.lookback_periods = feature.lookback_periods
                            feature_model.dependencies = feature.dependencies
                            feature_model.confidence = feature.confidence
                            
                            # Update metadata
                            feature_model.metadata_description = feature.metadata.description
                            feature_model.metadata_category = feature.metadata.category.name
                            feature_model.metadata_scope = feature.metadata.scope.name
                            feature_model.metadata_timeframe = feature.metadata.timeframe.name
                            feature_model.metadata_is_experimental = feature.metadata.is_experimental
                            feature_model.metadata_updated_at = datetime.utcnow()
                            feature_model.metadata_version = feature.metadata.version
                            feature_model.metadata_tags = feature.metadata.tags
                            feature_model.metadata_properties = feature.metadata.properties
                            
                            # Link to feature set
                            feature_model.feature_set_id = feature_set.id
                        else:
                            # Create new feature model
                            feature_model = FeatureModel(
                                id=feature.id,
                                name=feature.name,
                                symbol=feature.symbol,
                                type=feature.type.name,
                                value=feature.value,
                                timestamp=feature.timestamp,
                                lookback_periods=feature.lookback_periods,
                                dependencies=feature.dependencies,
                                confidence=feature.confidence,
                                
                                # Metadata
                                metadata_description=feature.metadata.description,
                                metadata_category=feature.metadata.category.name,
                                metadata_scope=feature.metadata.scope.name,
                                metadata_timeframe=feature.metadata.timeframe.name,
                                metadata_is_experimental=feature.metadata.is_experimental,
                                metadata_created_at=feature.metadata.created_at,
                                metadata_updated_at=feature.metadata.updated_at,
                                metadata_version=feature.metadata.version,
                                metadata_tags=feature.metadata.tags,
                                metadata_properties=feature.metadata.properties,
                                
                                # Link to feature set
                                feature_set_id=feature_set.id
                            )
                            session.add(feature_model)
                
                await session.commit()
                
                logger.info(f"Saved feature set: {feature_set.name} with {len(feature_set.features)} features")
                return feature_set
        except Exception as e:
            logger.error(f"Error saving feature set: {str(e)}")
            raise
    
    async def get_feature_set(self, feature_set_id: uuid.UUID) -> Optional[FeatureSet]:
        """
        Retrieve a feature set by ID.
        
        Args:
            feature_set_id: UUID of the feature set to retrieve
            
        Returns:
            FeatureSet object if found, None otherwise
        """
        try:
            async with self.async_session() as session:
                # Get feature set
                stmt = select(FeatureSetModel).where(FeatureSetModel.id == feature_set_id)
                result = await session.execute(stmt)
                feature_set_model = result.scalars().first()
                
                if not feature_set_model:
                    return None
                
                # Get associated features
                stmt = select(FeatureModel).where(FeatureModel.feature_set_id == feature_set_id)
                result = await session.execute(stmt)
                feature_models = result.scalars().all()
                
                # Convert to domain entities
                features = [self._model_to_feature(model) for model in feature_models]
                
                # Convert feature set model to domain entity
                return self._model_to_feature_set(feature_set_model, features)
        except Exception as e:
            logger.error(f"Error retrieving feature set: {str(e)}")
            raise
    
    async def get_feature_set_by_name(self, name: str) -> Optional[FeatureSet]:
        """
        Retrieve a feature set by name.
        
        Args:
            name: Name of the feature set to retrieve
            
        Returns:
            FeatureSet object if found, None otherwise
        """
        try:
            async with self.async_session() as session:
                # Get feature set
                stmt = select(FeatureSetModel).where(FeatureSetModel.name == name)
                result = await session.execute(stmt)
                feature_set_model = result.scalars().first()
                
                if not feature_set_model:
                    return None
                
                # Get associated features
                stmt = select(FeatureModel).where(FeatureModel.feature_set_id == feature_set_model.id)
                result = await session.execute(stmt)
                feature_models = result.scalars().all()
                
                # Convert to domain entities
                features = [self._model_to_feature(model) for model in feature_models]
                
                # Convert feature set model to domain entity
                return self._model_to_feature_set(feature_set_model, features)
        except Exception as e:
            logger.error(f"Error retrieving feature set by name: {str(e)}")
            raise
    
    async def get_feature_sets_by_symbol(
        self,
        symbol: str,
        status: Optional[FeatureSetStatus] = None,
        limit: int = 10
    ) -> List[FeatureSet]:
        """
        Retrieve feature sets by symbol.
        
        Args:
            symbol: Symbol to filter by
            status: Optional status to filter by
            limit: Maximum number of feature sets to retrieve
            
        Returns:
            List of FeatureSet objects
        """
        try:
            async with self.async_session() as session:
                # Build query for feature sets
                # We need to check if the symbol is in the symbols JSON array
                query = select(FeatureSetModel).where(
                    func.jsonb_exists(FeatureSetModel.symbols, symbol)
                )
                
                # Apply status filter if provided
                if status:
                    query = query.where(FeatureSetModel.status == status.name)
                
                # Apply limit and order
                query = query.order_by(desc(FeatureSetModel.metadata_created_at)).limit(limit)
                
                # Execute query
                result = await session.execute(query)
                feature_set_models = result.scalars().all()
                
                # Build list of feature sets with their features
                feature_sets = []
                
                for fs_model in feature_set_models:
                    # Get associated features
                    stmt = select(FeatureModel).where(FeatureModel.feature_set_id == fs_model.id)
                    result = await session.execute(stmt)
                    feature_models = result.scalars().all()
                    
                    # Convert to domain entities
                    features = [self._model_to_feature(model) for model in feature_models]
                    
                    # Convert feature set model to domain entity
                    feature_set = self._model_to_feature_set(fs_model, features)
                    feature_sets.append(feature_set)
                
                return feature_sets
        except Exception as e:
            logger.error(f"Error retrieving feature sets by symbol: {str(e)}")
            raise
    
    async def update_feature_set_status(
        self,
        feature_set_id: uuid.UUID,
        new_status: FeatureSetStatus
    ) -> Optional[FeatureSet]:
        """
        Update the status of a feature set.
        
        Args:
            feature_set_id: UUID of the feature set to update
            new_status: New status to set
            
        Returns:
            Updated FeatureSet object if found, None otherwise
        """
        try:
            async with self.async_session() as session:
                # Get feature set
                stmt = select(FeatureSetModel).where(FeatureSetModel.id == feature_set_id)
                result = await session.execute(stmt)
                feature_set_model = result.scalars().first()
                
                if not feature_set_model:
                    return None
                
                # Update status
                feature_set_model.status = new_status.name
                feature_set_model.metadata_updated_at = datetime.utcnow()
                
                await session.commit()
                
                # Get the updated feature set with its features
                return await self.get_feature_set(feature_set_id)
        except Exception as e:
            logger.error(f"Error updating feature set status: {str(e)}")
            raise
    
    async def delete_feature_set(self, feature_set_id: uuid.UUID) -> bool:
        """
        Delete a feature set and its associated features.
        
        Args:
            feature_set_id: UUID of the feature set to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            async with self.async_session() as session:
                # Delete associated features first
                stmt = select(FeatureModel).where(FeatureModel.feature_set_id == feature_set_id)
                result = await session.execute(stmt)
                feature_models = result.scalars().all()
                
                for feature_model in feature_models:
                    await session.delete(feature_model)
                
                # Delete feature set
                stmt = select(FeatureSetModel).where(FeatureSetModel.id == feature_set_id)
                result = await session.execute(stmt)
                feature_set_model = result.scalars().first()
                
                if not feature_set_model:
                    return False
                
                await session.delete(feature_set_model)
                
                await session.commit()
                
                logger.info(f"Deleted feature set with ID: {feature_set_id}")
                return True
        except Exception as e:
            logger.error(f"Error deleting feature set: {str(e)}")
            raise
    
    async def export_feature_set_to_dataframe(self, feature_set_id: uuid.UUID) -> pd.DataFrame:
        """
        Export a feature set to a pandas DataFrame.
        
        Args:
            feature_set_id: UUID of the feature set to export
            
        Returns:
            DataFrame containing the feature set data
        """
        try:
            # Get the feature set
            feature_set = await self.get_feature_set(feature_set_id)
            
            if not feature_set or not feature_set.features:
                return pd.DataFrame()
            
            # Group features by timestamp
            features_by_timestamp = {}
            for feature in feature_set.features:
                timestamp = feature.timestamp
                if timestamp not in features_by_timestamp:
                    features_by_timestamp[timestamp] = {}
                
                # Use feature name as column
                features_by_timestamp[timestamp][feature.name] = feature.value
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(features_by_timestamp, orient='index')
            
            # Sort by timestamp (index)
            df.sort_index(inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"Error exporting feature set to DataFrame: {str(e)}")
            raise
    
    def _model_to_feature(self, model: FeatureModel) -> Feature:
        """Convert a FeatureModel to a Feature domain entity."""
        # Create metadata
        metadata = FeatureMetadata(
            description=model.metadata_description,
            category=FeatureCategory[model.metadata_category],
            scope=FeatureScope[model.metadata_scope],
            timeframe=FeatureTimeframe[model.metadata_timeframe],
            is_experimental=model.metadata_is_experimental,
            created_at=model.metadata_created_at,
            updated_at=model.metadata_updated_at,
            version=model.metadata_version,
            tags=model.metadata_tags or [],
            properties=model.metadata_properties or {}
        )
        
        # Create feature
        return Feature(
            id=model.id,
            name=model.name,
            symbol=model.symbol,
            type=FeatureType[model.type],
            value=model.value,
            timestamp=model.timestamp,
            metadata=metadata,
            lookback_periods=model.lookback_periods,
            dependencies=model.dependencies or [],
            confidence=model.confidence
        )
    
    def _model_to_feature_set(self, model: FeatureSetModel, features: List[Feature]) -> FeatureSet:
        """Convert a FeatureSetModel to a FeatureSet domain entity."""
        # Create metadata
        metadata = FeatureSetMetadata(
            description=model.metadata_description,
            version=model.metadata_version,
            created_at=model.metadata_created_at,
            updated_at=model.metadata_updated_at,
            tags=model.metadata_tags or [],
            properties=model.metadata_properties or {},
            author=model.metadata_author
        )
        
        # Create feature set
        return FeatureSet(
            id=model.id,
            name=model.name,
            type=FeatureSetType[model.type],
            status=FeatureSetStatus[model.status],
            features=features,
            symbols=set(model.symbols),
            metadata=metadata,
            start_timestamp=model.start_timestamp,
            end_timestamp=model.end_timestamp
        )