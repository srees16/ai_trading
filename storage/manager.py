"""
Storage Manager Module.

Manages persistence of trading signals to Excel/CSV files
with support for append mode and deduplication.
"""

import logging

import pandas as pd
from pathlib import Path
from typing import List, Optional

from models import TradingSignal
from config import Config

logger = logging.getLogger(__name__)


class StorageManager:
    """Manages storage of trading signals to Excel/CSV files."""
    
    def __init__(self, output_file: str = None):
        """
        Initialize the storage manager.
        
        Args:
            output_file: Path to output file (Excel or CSV)
        """
        self.output_file = output_file or Config.OUTPUT_FILE
        self.file_path = Path(self.output_file)
    
    def save_signals(self, signals: List[TradingSignal], append: bool = True) -> Optional[str]:
        """
        Save trading signals to file.
        
        Args:
            signals: List of TradingSignal objects to persist
            append: If True, append to existing file; otherwise overwrite
            
        Returns:
            Full path to the saved file, or None if save failed
        """
        if not signals:
            logger.info("No signals to save")
            return None
        
        # Convert signals to DataFrame
        data = [signal.to_dict() for signal in signals]
        new_df = pd.DataFrame(data)
        
        # Check if file exists and append mode is enabled
        if append and self.file_path.exists():
            try:
                # Read existing data
                if self.file_path.suffix == '.xlsx':
                    existing_df = pd.read_excel(self.file_path)
                else:
                    existing_df = pd.read_csv(self.file_path)
                
                # Append new data
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                
                # Remove duplicates (same ticker, source, and title)
                combined_df = combined_df.drop_duplicates(
                    subset=['ticker', 'source', 'title'],
                    keep='last'
                )
                
                df_to_save = combined_df
                logger.info("Appended %d signals to existing file", len(new_df))
            
            except Exception as e:
                logger.error("Error reading existing file: %s", e)
                logger.info("Creating new file instead")
                df_to_save = new_df
        else:
            df_to_save = new_df
            logger.info("Saving %d signals to new file", len(new_df))
        
        # Save to file
        try:
            # Create directory if it doesn't exist
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            
            if self.file_path.suffix == '.xlsx':
                df_to_save.to_excel(self.file_path, index=False)
            else:
                df_to_save.to_csv(self.file_path, index=False)
            
            absolute_path = self.file_path.absolute()
            logger.info("Successfully saved data to %s", absolute_path)
            return str(absolute_path)
        
        except Exception as e:
            logger.error("Error saving to file: %s", e)
            return None
