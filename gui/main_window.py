"""
Main GUI Window for Satellite Classifier Application
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import sys
from datetime import datetime

from gui.progress_dialog import ProgressDialog
from gui.results_window import ResultsWindow
from preprocessing.image_processor import ImageProcessor
from features.feature_extractor import FeatureExtractor
from ml.classifier_ensemble import ClassifierEnsemble
from ml.data_manager import DataManager
from utils.logger import get_logger

class SatelliteClassifierApp:
    """Main application window for satellite classification."""
    
    def __init__(self, root):
        self.root = root
        self.logger = get_logger()
        self.setup_ui()
        self.selected_directory = None
        self.processing_thread = None
        self.is_processing = False
        
    def setup_ui(self):
        """Setup the main user interface."""
        self.root.title("Satellite Object Classifier v1.0")
        self.root.geometry("800x600")
        self.root.minsize(600, 500)
        
        # Configure styles
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Satellite Object Classifier", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Directory selection
        ttk.Label(main_frame, text="Select Dataset Directory:").grid(row=1, column=0, sticky=tk.W, pady=5)
        
        self.dir_var = tk.StringVar()
        self.dir_entry = ttk.Entry(main_frame, textvariable=self.dir_var, state='readonly')
        self.dir_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(10, 5), pady=5)
        
        self.browse_btn = ttk.Button(main_frame, text="Browse", command=self.browse_directory)
        self.browse_btn.grid(row=1, column=2, padx=(5, 0), pady=5)
        
        # Processing options frame
        options_frame = ttk.LabelFrame(main_frame, text="Processing Options", padding="10")
        options_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        options_frame.columnconfigure(1, weight=1)
        
        # Data split ratios
        ttk.Label(options_frame, text="Training Split:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.train_split = tk.DoubleVar(value=75.0)
        ttk.Entry(options_frame, textvariable=self.train_split, width=10).grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Label(options_frame, text="%").grid(row=0, column=2, sticky=tk.W)
        
        ttk.Label(options_frame, text="Validation Split:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.val_split = tk.DoubleVar(value=5.0)
        ttk.Entry(options_frame, textvariable=self.val_split, width=10).grid(row=1, column=1, sticky=tk.W, padx=5)
        ttk.Label(options_frame, text="%").grid(row=1, column=2, sticky=tk.W)
        
        # Checkboxes for processing steps
        self.apply_noise = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Apply Environmental Noise", 
                       variable=self.apply_noise).grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=2)
        
        self.remove_background = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Remove Background", 
                       variable=self.remove_background).grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=2)
        
        self.export_features = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Export Features to CSV", 
                       variable=self.export_features).grid(row=4, column=0, columnspan=3, sticky=tk.W, pady=2)
        
        # Action buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=3, pady=10)
        
        self.start_btn = ttk.Button(button_frame, text="Start Classification", 
                                   command=self.start_processing, style='Accent.TButton')
        self.start_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_btn = ttk.Button(button_frame, text="Stop", command=self.stop_processing, 
                                  state='disabled')
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.results_btn = ttk.Button(button_frame, text="View Results", 
                                     command=self.view_results, state='disabled')
        self.results_btn.pack(side=tk.LEFT)
        
        # Log output
        log_frame = ttk.LabelFrame(main_frame, text="Processing Log", padding="5")
        log_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, state='disabled')
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Initialize results storage
        self.results = None
        
    def log_message(self, message):
        """Add message to log display."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, formatted_message)
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')
        self.root.update_idletasks()
        
    def browse_directory(self):
        """Open directory browser dialog."""
        directory = filedialog.askdirectory(title="Select Dataset Directory")
        if directory:
            # Validate directory structure
            required_folders = ['Carrier Rockets', 'Satellites', 'Debris']
            missing_folders = []
            
            for folder in required_folders:
                if not os.path.exists(os.path.join(directory, folder)):
                    missing_folders.append(folder)
            
            if missing_folders:
                messagebox.showerror("Invalid Directory", 
                    f"Directory must contain folders: {', '.join(required_folders)}\n"
                    f"Missing: {', '.join(missing_folders)}")
                return
            
            self.selected_directory = directory
            self.dir_var.set(directory)
            self.log_message(f"Selected directory: {directory}")
            self.status_var.set("Directory selected - Ready to process")
            
    def start_processing(self):
        """Start the classification processing in a separate thread."""
        if not self.selected_directory:
            messagebox.showerror("Error", "Please select a dataset directory first.")
            return
            
        if self.is_processing:
            messagebox.showwarning("Warning", "Processing is already in progress.")
            return
        
        # Validate split ratios
        total_split = self.train_split.get() + self.val_split.get()
        if total_split >= 100:
            messagebox.showerror("Error", "Training + Validation split must be less than 100%")
            return
            
        # Update UI state
        self.is_processing = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.results_btn.config(state='disabled')
        self.browse_btn.config(state='disabled')
        
        # Clear previous results
        self.results = None
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_classification, daemon=True)
        self.processing_thread.start()
        
    def stop_processing(self):
        """Stop the current processing."""
        if self.is_processing:
            self.is_processing = False
            self.log_message("Stopping processing...")
            self.status_var.set("Stopping...")
            
    def process_classification(self):
        """Main processing pipeline."""
        try:
            self.log_message("Starting satellite classification pipeline...")
            self.status_var.set("Processing...")
            
            # Initialize components
            data_manager = DataManager()
            image_processor = ImageProcessor()
            feature_extractor = FeatureExtractor()
            classifier_ensemble = ClassifierEnsemble()
            
            # Step 1: Load and split data
            self.log_message("Loading and splitting dataset...")
            train_split = self.train_split.get() / 100.0
            val_split = self.val_split.get() / 100.0
            test_split = (100.0 - self.train_split.get() - self.val_split.get()) / 100.0
            
            train_data, val_data, test_data = data_manager.load_and_split_data(
                self.selected_directory, train_split, val_split, test_split)
            
            if not self.is_processing:
                return
                
            self.log_message(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
            
            # Step 2: Process images
            self.log_message("Processing images (grayscale, noise, denoising, background removal)...")
            
            all_data = train_data + val_data + test_data
            processed_images = []
            
            for i, (image_path, label) in enumerate(all_data):
                if not self.is_processing:
                    return
                    
                self.status_var.set(f"Processing image {i+1}/{len(all_data)}")
                
                try:
                    processed_img = image_processor.process_image(
                        image_path,
                        apply_noise=self.apply_noise.get(),
                        remove_background=self.remove_background.get()
                    )
                    processed_images.append((processed_img, label, image_path))
                except Exception as e:
                    self.log_message(f"Error processing {image_path}: {str(e)}")
                    continue
            
            if not self.is_processing:
                return
                
            # Step 3: Extract features
            self.log_message("Extracting features from processed images...")
            features_data = []
            
            for i, (image, label, path) in enumerate(processed_images):
                if not self.is_processing:
                    return
                    
                self.status_var.set(f"Extracting features {i+1}/{len(processed_images)}")
                
                try:
                    features = feature_extractor.extract_all_features(image)
                    features_data.append((features, label, path))
                except Exception as e:
                    self.log_message(f"Error extracting features from {path}: {str(e)}")
                    continue
            
            if not self.is_processing:
                return
                
            # Step 4: Export features to CSV if requested
            if self.export_features.get():
                self.log_message("Exporting features to CSV...")
                csv_path = os.path.join(os.path.dirname(self.selected_directory), "extracted_features.csv")
                feature_extractor.export_to_csv(features_data, csv_path)
                self.log_message(f"Features exported to: {csv_path}")
            
            # Step 5: Train classifiers
            self.log_message("Training machine learning classifiers...")
            
            # Split features back into train/val/test
            n_train = len(train_data)
            n_val = len(val_data)
            
            train_features = features_data[:n_train]
            val_features = features_data[n_train:n_train + n_val]
            test_features = features_data[n_train + n_val:]
            
            # Train ensemble
            self.status_var.set("Training classifiers...")
            results = classifier_ensemble.train_and_evaluate(
                train_features, val_features, test_features)
            
            if not self.is_processing:
                return
                
            # Store results
            self.results = results
            
            self.log_message("Classification completed successfully!")
            self.log_message(f"Best ensemble accuracy: {results['ensemble_accuracy']:.4f}")
            
            # Update UI
            self.root.after(0, self.processing_completed)
            
        except Exception as e:
            error_msg = f"Error during processing: {str(e)}"
            self.log_message(error_msg)
            self.logger.error(error_msg, exc_info=True)
            self.root.after(0, self.processing_error, str(e))
            
    def processing_completed(self):
        """Called when processing completes successfully."""
        self.is_processing = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.results_btn.config(state='normal')
        self.browse_btn.config(state='normal')
        self.status_var.set("Processing completed successfully")
        
        messagebox.showinfo("Success", "Classification completed successfully!\nClick 'View Results' to see the performance metrics.")
        
    def processing_error(self, error_msg):
        """Called when processing encounters an error."""
        self.is_processing = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.browse_btn.config(state='normal')
        self.status_var.set("Processing failed")
        
        messagebox.showerror("Processing Error", f"An error occurred during processing:\n{error_msg}")
        
    def view_results(self):
        """Open results window to display classification metrics."""
        if self.results:
            results_window = ResultsWindow(self.root, self.results)
        else:
            messagebox.showwarning("No Results", "No results available. Please run classification first.")
            
    def run(self):
        """Start the application main loop."""
        self.root.deiconify()  # Show the main window
        self.log_message("Satellite Classifier Application started")
        self.log_message("Please select a directory containing 'Carrier Rockets', 'Satellites', and 'Debris' folders")
        self.root.mainloop()
