#!/usr/bin/env python3
"""
Satellite Object Classifier - Main Entry Point
A standalone application for classifying satellite objects using machine learning.
"""

import sys
import os
import tkinter as tk
from tkinter import messagebox
import traceback

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.main_window import SatelliteClassifierApp
from utils.logger import setup_logger

def main():
    """Main entry point for the Satellite Classifier application."""
    try:
        # Setup logging
        logger = setup_logger()
        logger.info("Starting Satellite Classifier Application")
        
        # Check if we can create a GUI window
        try:
            logger.info("Initializing GUI components...")
            
            # Create the main tkinter root window
            root = tk.Tk()
            logger.info("Tkinter root window created successfully")
            
            # Create and run the application
            logger.info("Creating SatelliteClassifierApp instance...")
            app = SatelliteClassifierApp(root)
            logger.info("SatelliteClassifierApp created successfully")
            
            logger.info("Starting application main loop...")
            app.run()
            
        except tk.TclError as e:
            # GUI not available, run in console mode
            logger.error(f"GUI not available: {str(e)}")
            print("GUI mode not available. Running in console mode.")
            print("This is a GUI application that requires a display.")
            print("For standalone executable creation, use: python build_exe.py")
            return
        
    except Exception as e:
        error_msg = f"Critical error starting application: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        
        # Try to show error in messagebox if tkinter is available
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Critical Error", f"Failed to start application:\n{str(e)}")
            root.destroy()
        except:
            pass
        
        sys.exit(1)

if __name__ == "__main__":
    main()
