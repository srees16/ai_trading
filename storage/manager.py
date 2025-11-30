"""
Storage module for saving trading signals to Excel/CSV.
"""

import pandas as pd
from pathlib import Path
from typing import List
from datetime import datetime
from models import TradingSignal
from config import Config


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
    
    def save_signals(self, signals: List[TradingSignal], append: bool = True) -> str:
        """
        Save trading signals to file.
        
        Args:
            signals: List of TradingSignal objects
            append: If True, append to existing file; otherwise overwrite
            
        Returns:
            Full path to the saved file, or None if save failed
        """
        if not signals:
            print("No signals to save.")
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
                print(f"Appended {len(new_df)} signals to existing file.")
            
            except Exception as e:
                print(f"Error reading existing file: {e}")
                print("Creating new file instead.")
                df_to_save = new_df
        else:
            df_to_save = new_df
            print(f"Saving {len(new_df)} signals to new file.")
        
        # Save to file
        try:
            # Create directory if it doesn't exist
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            
            if self.file_path.suffix == '.xlsx':
                df_to_save.to_excel(self.file_path, index=False)
            else:
                df_to_save.to_csv(self.file_path, index=False)
            
            absolute_path = self.file_path.absolute()
            print(f"Successfully saved data to {absolute_path}")
            return str(absolute_path)
        
        except Exception as e:
            print(f"Error saving to file: {e}")
            return None
    
    def load_signals(self) -> pd.DataFrame:
        """
        Load signals from file.
        
        Returns:
            DataFrame with historical signals
        """
        if not self.file_path.exists():
            print(f"File {self.file_path} does not exist.")
            return pd.DataFrame()
        
        try:
            if self.file_path.suffix == '.xlsx':
                df = pd.read_excel(self.file_path)
            else:
                df = pd.read_csv(self.file_path)
            
            print(f"Loaded {len(df)} signals from {self.file_path}")
            return df
        
        except Exception as e:
            print(f"Error loading file: {e}")
            return pd.DataFrame()
    
    def get_signals_by_ticker(self, ticker: str) -> pd.DataFrame:
        """
        Get all signals for a specific ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            DataFrame with signals for the ticker
        """
        df = self.load_signals()
        if df.empty:
            return df
        
        return df[df['ticker'] == ticker]
    
    def get_signals_by_date(self, date: datetime) -> pd.DataFrame:
        """
        Get all signals for a specific date.
        
        Args:
            date: Date to filter by
            
        Returns:
            DataFrame with signals for the date
        """
        df = self.load_signals()
        if df.empty:
            return df
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        date_str = date.strftime('%Y-%m-%d')
        
        return df[df['timestamp'].dt.strftime('%Y-%m-%d') == date_str]
