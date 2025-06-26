"""
Results Window for displaying classification metrics and visualizations
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

class ResultsWindow:
    """Window for displaying classification results and metrics."""
    
    def __init__(self, parent, results):
        self.parent = parent
        self.results = results
        self.window = tk.Toplevel(parent)
        self.window.title("Classification Results")
        self.window.geometry("900x700")
        self.window.transient(parent)
        
        self.setup_ui()
        self.populate_results()
        
    def setup_ui(self):
        """Setup the results window UI."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.window)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Summary
        self.summary_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.summary_frame, text="Summary")
        
        # Tab 2: Individual Classifiers
        self.classifiers_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.classifiers_frame, text="Individual Classifiers")
        
        # Tab 3: Confusion Matrix
        self.confusion_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.confusion_frame, text="Confusion Matrix")
        
        # Tab 4: Feature Importance
        self.features_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.features_frame, text="Feature Importance")
        
        self.setup_summary_tab()
        self.setup_classifiers_tab()
        self.setup_confusion_tab()
        self.setup_features_tab()
        
    def setup_summary_tab(self):
        """Setup the summary tab."""
        # Title
        title_label = ttk.Label(self.summary_frame, text="Classification Results Summary", 
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=10)
        
        # Results frame
        results_frame = ttk.LabelFrame(self.summary_frame, text="Ensemble Results", padding="10")
        results_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Ensemble metrics will be populated in populate_results
        self.ensemble_text = scrolledtext.ScrolledText(results_frame, height=8, width=70)
        self.ensemble_text.pack(fill=tk.BOTH, expand=True)
        
        # Best classifier frame
        best_frame = ttk.LabelFrame(self.summary_frame, text="Best Individual Classifier", padding="10")
        best_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.best_text = scrolledtext.ScrolledText(best_frame, height=6, width=70)
        self.best_text.pack(fill=tk.BOTH, expand=True)
        
    def setup_classifiers_tab(self):
        """Setup the individual classifiers tab."""
        # Create treeview for classifier comparison
        columns = ('Classifier', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Weight')
        self.tree = ttk.Treeview(self.classifiers_frame, columns=columns, show='headings', height=15)
        
        # Define headings
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=120, anchor=tk.CENTER)
        
        # Scrollbar for treeview
        scrollbar = ttk.Scrollbar(self.classifiers_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack treeview and scrollbar
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(20, 0), pady=20)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 20), pady=20)
        
    def setup_confusion_tab(self):
        """Setup the confusion matrix tab."""
        # Create matplotlib figure for confusion matrix
        self.confusion_fig = Figure(figsize=(8, 6), dpi=100)
        self.confusion_canvas = FigureCanvasTkAgg(self.confusion_fig, self.confusion_frame)
        self.confusion_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
    def setup_features_tab(self):
        """Setup the feature importance tab."""
        # Create matplotlib figure for feature importance
        self.features_fig = Figure(figsize=(10, 8), dpi=100)
        self.features_canvas = FigureCanvasTkAgg(self.features_fig, self.features_frame)
        self.features_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
    def populate_results(self):
        """Populate the results tabs with data."""
        if not self.results:
            return
            
        # Populate summary
        self.populate_summary()
        
        # Populate classifiers table
        self.populate_classifiers()
        
        # Plot confusion matrix
        self.plot_confusion_matrix()
        
        # Plot feature importance
        self.plot_feature_importance()
        
    def populate_summary(self):
        """Populate the summary tab with ensemble results."""
        ensemble_info = f"""Ensemble Classification Results
{'='*50}

Overall Accuracy: {self.results['ensemble_accuracy']:.4f} ({self.results['ensemble_accuracy']*100:.2f}%)

Weighted Voting System:
- Uses performance-based weights for each classifier
- Combines predictions from multiple algorithms
- Robust against individual classifier weaknesses

Class-wise Performance:
"""
        
        if 'class_report' in self.results:
            ensemble_info += self.results['class_report']
            
        self.ensemble_text.insert(tk.END, ensemble_info)
        self.ensemble_text.config(state='disabled')
        
        # Best classifier info
        best_classifier = max(self.results['individual_results'].items(), 
                             key=lambda x: x[1]['accuracy'])
        
        best_info = f"""Best Individual Classifier: {best_classifier[0]}
{'='*40}

Accuracy: {best_classifier[1]['accuracy']:.4f} ({best_classifier[1]['accuracy']*100:.2f}%)
Precision: {best_classifier[1]['precision']:.4f}
Recall: {best_classifier[1]['recall']:.4f}
F1-Score: {best_classifier[1]['f1']:.4f}

Ensemble Weight: {self.results['weights'].get(best_classifier[0], 0.0):.4f}
"""
        
        self.best_text.insert(tk.END, best_info)
        self.best_text.config(state='disabled')
        
    def populate_classifiers(self):
        """Populate the classifiers comparison table."""
        for name, metrics in self.results['individual_results'].items():
            weight = self.results['weights'].get(name, 0.0)
            self.tree.insert('', tk.END, values=(
                name,
                f"{metrics['accuracy']:.4f}",
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{metrics['f1']:.4f}",
                f"{weight:.4f}"
            ))
            
    def plot_confusion_matrix(self):
        """Plot the confusion matrix."""
        if 'confusion_matrix' not in self.results:
            return
            
        ax = self.confusion_fig.add_subplot(111)
        cm = self.results['confusion_matrix']
        classes = ['Carrier Rockets', 'Satellites', 'Debris']
        
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes,
               yticklabels=classes,
               title='Confusion Matrix - Ensemble Classifier',
               ylabel='True Label',
               xlabel='Predicted Label')
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        self.confusion_fig.tight_layout()
        self.confusion_canvas.draw()
        
    def plot_feature_importance(self):
        """Plot feature importance if available."""
        if 'feature_importance' not in self.results:
            # Create a placeholder message
            ax = self.features_fig.add_subplot(111)
            ax.text(0.5, 0.5, 'Feature importance data not available\nfor ensemble classifier', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=14)
            ax.set_title('Feature Importance')
            self.features_canvas.draw()
            return
            
        feature_names = [
            'Solar Panel Area', 'Edge Density', 'Texture Contrast', 'Aspect Ratio',
            'Circularity', 'Solidity', 'LBP Uniformity', 'Hu Moments',
            'Object Area', 'Convex Hull Area', 'Eccentricity', 'Euler Number'
        ]
        
        importance = self.results['feature_importance']
        
        # Sort features by importance
        sorted_idx = np.argsort(importance)
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_importance = importance[sorted_idx]
        
        ax = self.features_fig.add_subplot(111)
        bars = ax.barh(range(len(sorted_features)), sorted_importance)
        ax.set_yticks(range(len(sorted_features)))
        ax.set_yticklabels(sorted_features)
        ax.set_xlabel('Feature Importance')
        ax.set_title('Feature Importance Analysis')
        
        # Color bars based on importance
        for i, bar in enumerate(bars):
            if sorted_importance[i] > np.mean(sorted_importance):
                bar.set_color('darkgreen')
            else:
                bar.set_color('lightblue')
        
        self.features_fig.tight_layout()
        self.features_canvas.draw()
