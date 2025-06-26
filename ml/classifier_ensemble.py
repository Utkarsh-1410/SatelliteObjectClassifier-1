"""
Ensemble classifier for satellite object classification
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib
from utils.logger import get_logger

class ClassifierEnsemble:
    """Ensemble of multiple classifiers with weighted voting."""
    
    def __init__(self):
        self.logger = get_logger()
        self.classifiers = {}
        self.scaler = StandardScaler()
        self.weights = {}
        self.class_names = ['Carrier Rockets', 'Satellites', 'Debris']
        
        # Initialize classifiers
        self._initialize_classifiers()
        
    def _initialize_classifiers(self):
        """Initialize all classifiers."""
        self.classifiers = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, random_state=42, max_depth=10, min_samples_split=5
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=42, max_depth=8, min_samples_split=10
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=5, weights='distance'
            ),
            'Support Vector Machine': SVC(
                kernel='rbf', random_state=42, probability=True, C=1.0, gamma='scale'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, random_state=42, learning_rate=0.1, max_depth=6
            ),
            'Naive Bayes': GaussianNB(),
            'Logistic Regression': LogisticRegression(
                random_state=42, max_iter=1000, C=1.0
            )
        }
        
    def train_and_evaluate(self, train_data, val_data, test_data):
        """
        Train all classifiers and evaluate using ensemble voting.
        
        Args:
            train_data: List of (features, label, path) tuples for training
            val_data: List of (features, label, path) tuples for validation
            test_data: List of (features, label, path) tuples for testing
            
        Returns:
            Dictionary with evaluation results
        """
        try:
            # Prepare data
            X_train, y_train = self._prepare_data(train_data)
            X_val, y_val = self._prepare_data(val_data)
            X_test, y_test = self._prepare_data(test_data)
            
            # Combine train and validation for final training
            X_train_full = np.vstack([X_train, X_val]) if len(X_val) > 0 else X_train
            y_train_full = np.hstack([y_train, y_val]) if len(y_val) > 0 else y_train
            
            # Fit scaler on training data
            self.scaler.fit(X_train_full)
            
            # Scale data
            X_train_scaled = self.scaler.transform(X_train_full)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train individual classifiers and calculate weights
            individual_results = {}
            predictions = {}
            
            for name, classifier in self.classifiers.items():
                self.logger.info(f"Training {name}...")
                
                # Train classifier
                classifier.fit(X_train_scaled, y_train_full)
                
                # Make predictions
                y_pred = classifier.predict(X_test_scaled)
                predictions[name] = y_pred
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division='warn')
                recall = recall_score(y_test, y_pred, average='weighted', zero_division='warn')
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division='warn')
                
                individual_results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }
                
                self.logger.info(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            
            # Calculate weights based on performance
            self._calculate_weights(individual_results)
            
            # Ensemble prediction using weighted voting
            ensemble_predictions = self._ensemble_predict(predictions)
            
            # Calculate ensemble metrics
            ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
            ensemble_precision = precision_score(y_test, ensemble_predictions, average='weighted', zero_division='warn')
            ensemble_recall = recall_score(y_test, ensemble_predictions, average='weighted', zero_division='warn')
            ensemble_f1 = f1_score(y_test, ensemble_predictions, average='weighted', zero_division='warn')
            
            # Generate classification report
            unique_labels = sorted(set(y_test))
            unique_pred_labels = sorted(set(ensemble_predictions))
            all_unique_labels = sorted(set(list(unique_labels) + list(unique_pred_labels)))
            
            try:
                if len(all_unique_labels) <= len(self.class_names):
                    # Map labels to class names
                    target_names = [self.class_names[i] for i in all_unique_labels if i < len(self.class_names)]
                    class_report = classification_report(y_test, ensemble_predictions, 
                                                       labels=all_unique_labels,
                                                       target_names=target_names)
                else:
                    # Use numeric labels if mapping fails
                    class_report = classification_report(y_test, ensemble_predictions)
            except (ValueError, IndexError):
                # Fallback to basic report without target names
                class_report = classification_report(y_test, ensemble_predictions)
            
            # Generate confusion matrix
            conf_matrix = confusion_matrix(y_test, ensemble_predictions)
            
            self.logger.info(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")
            
            return {
                'ensemble_accuracy': ensemble_accuracy,
                'ensemble_precision': ensemble_precision,
                'ensemble_recall': ensemble_recall,
                'ensemble_f1': ensemble_f1,
                'individual_results': individual_results,
                'weights': self.weights,
                'predictions': ensemble_predictions,
                'true_labels': y_test,
                'class_report': class_report,
                'confusion_matrix': conf_matrix,
                'feature_importance': self._get_feature_importance()
            }
            
        except Exception as e:
            self.logger.error(f"Error in training and evaluation: {str(e)}")
            raise
            
    def _prepare_data(self, data_list):
        """Prepare feature matrix and labels from data list."""
        if not data_list:
            return np.array([]).reshape(0, -1), np.array([])
        
        features = []
        labels = []
        
        # Create label mapping
        label_map = {'Carrier Rockets': 0, 'Satellites': 1, 'Debris': 2}
        
        for feature_dict, label, _ in data_list:
            # Convert feature dictionary to array
            feature_array = np.array(list(feature_dict.values()))
            features.append(feature_array)
            labels.append(label_map[label])
        
        return np.array(features), np.array(labels)
        
    def _calculate_weights(self, individual_results):
        """Calculate weights for ensemble voting based on performance."""
        # Use F1 score as the primary metric for weighting
        f1_scores = [results['f1'] for results in individual_results.values()]
        
        # Normalize F1 scores to get weights
        total_f1 = sum(f1_scores)
        
        if total_f1 > 0:
            self.weights = {
                name: results['f1'] / total_f1 
                for name, results in individual_results.items()
            }
        else:
            # Equal weights if all classifiers perform poorly
            num_classifiers = len(individual_results)
            self.weights = {
                name: 1.0 / num_classifiers 
                for name in individual_results.keys()
            }
            
    def _ensemble_predict(self, predictions):
        """Make ensemble predictions using weighted voting."""
        if not predictions:
            return np.array([])
        
        # Get number of samples
        sample_names = list(predictions.keys())
        n_samples = len(predictions[sample_names[0]])
        
        ensemble_preds = []
        
        for i in range(n_samples):
            # Count weighted votes for each class
            class_votes = {0: 0.0, 1: 0.0, 2: 0.0}  # 0: Carrier Rockets, 1: Satellites, 2: Debris
            
            for name, preds in predictions.items():
                weight = self.weights.get(name, 0.0)
                predicted_class = preds[i]
                class_votes[predicted_class] += weight
            
            # Select class with highest weighted vote
            predicted_class = max(class_votes.items(), key=lambda x: x[1])[0]
            ensemble_preds.append(predicted_class)
        
        return np.array(ensemble_preds)
        
    def _get_feature_importance(self):
        """Get feature importance from Random Forest classifier."""
        if 'Random Forest' in self.classifiers:
            rf_classifier = self.classifiers['Random Forest']
            if hasattr(rf_classifier, 'feature_importances_'):
                return rf_classifier.feature_importances_
        
        # Return uniform importance if Random Forest is not available
        return np.ones(24) / 24  # 24 features
        
    def predict(self, features):
        """Make prediction on new data."""
        try:
            # Prepare features
            if isinstance(features, dict):
                feature_array = np.array(list(features.values())).reshape(1, -1)
            else:
                feature_array = np.array(features).reshape(1, -1)
            
            # Scale features
            feature_scaled = self.scaler.transform(feature_array)
            
            # Get predictions from all classifiers
            predictions = {}
            for name, classifier in self.classifiers.items():
                pred = classifier.predict(feature_scaled)[0]
                predictions[name] = np.array([pred])
            
            # Ensemble prediction
            ensemble_pred = self._ensemble_predict(predictions)[0]
            
            # Convert back to class name
            class_names = ['Carrier Rockets', 'Satellites', 'Debris']
            return class_names[ensemble_pred]
            
        except Exception as e:
            self.logger.error(f"Error in prediction: {str(e)}")
            return 'Unknown'
            
    def save_model(self, filepath):
        """Save the trained ensemble model."""
        try:
            model_data = {
                'classifiers': self.classifiers,
                'scaler': self.scaler,
                'weights': self.weights,
                'class_names': self.class_names
            }
            joblib.dump(model_data, filepath)
            self.logger.info(f"Model saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            
    def load_model(self, filepath):
        """Load a pre-trained ensemble model."""
        try:
            model_data = joblib.load(filepath)
            self.classifiers = model_data['classifiers']
            self.scaler = model_data['scaler']
            self.weights = model_data['weights']
            self.class_names = model_data['class_names']
            self.logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
