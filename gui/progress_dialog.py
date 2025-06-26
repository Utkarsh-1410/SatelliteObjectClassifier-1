"""
Progress Dialog for showing processing status
"""

import tkinter as tk
from tkinter import ttk

class ProgressDialog:
    """Dialog window for showing processing progress."""
    
    def __init__(self, parent, title="Processing"):
        self.parent = parent
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("400x150")
        self.dialog.resizable(False, False)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center the dialog
        self.dialog.geometry("+%d+%d" % (
            parent.winfo_rootx() + 50,
            parent.winfo_rooty() + 50
        ))
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the progress dialog UI."""
        main_frame = ttk.Frame(self.dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Status label
        self.status_var = tk.StringVar(value="Initializing...")
        status_label = ttk.Label(main_frame, textvariable=self.status_var)
        status_label.pack(pady=(0, 10))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            main_frame, 
            variable=self.progress_var,
            maximum=100,
            length=300
        )
        self.progress_bar.pack(pady=(0, 10))
        
        # Detailed status
        self.detail_var = tk.StringVar(value="")
        detail_label = ttk.Label(main_frame, textvariable=self.detail_var, 
                                font=('Arial', 8))
        detail_label.pack(pady=(0, 10))
        
        # Cancel button
        self.cancel_btn = ttk.Button(main_frame, text="Cancel", 
                                    command=self.on_cancel)
        self.cancel_btn.pack()
        
        self.cancelled = False
        
    def update_progress(self, value, status="", detail=""):
        """Update progress bar and status."""
        self.progress_var.set(value)
        if status:
            self.status_var.set(status)
        if detail:
            self.detail_var.set(detail)
        self.dialog.update_idletasks()
        
    def set_indeterminate(self, status="Processing..."):
        """Set progress bar to indeterminate mode."""
        self.progress_bar.config(mode='indeterminate')
        self.progress_bar.start()
        self.status_var.set(status)
        
    def stop_indeterminate(self):
        """Stop indeterminate progress bar."""
        self.progress_bar.stop()
        self.progress_bar.config(mode='determinate')
        
    def on_cancel(self):
        """Handle cancel button click."""
        self.cancelled = True
        self.dialog.destroy()
        
    def is_cancelled(self):
        """Check if dialog was cancelled."""
        return self.cancelled
        
    def close(self):
        """Close the dialog."""
        self.dialog.destroy()
