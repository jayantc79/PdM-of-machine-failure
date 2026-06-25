================================================================================
PART 5: COMPREHENSIVE UTILITY FUNCTIONS & EVALUATION FRAMEWORK
================================================================================

This section provides helper functions, evaluation frameworks, and integration
guides for the three critical implementations.

================================================================================
SECTION 5.1: UTILITY FUNCTIONS FOR EVALUATION
================================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    classification_report, f1_score, precision_score, recall_score
)
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """
    Comprehensive model evaluation utility
    Handles multi-class and binary classification metrics
    """
    
    @staticmethod
    def generate_metrics_report(y_true, y_pred, y_pred_proba=None, 
                               labels=None, model_name="Model"):
        """
        Generate comprehensive metrics report
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        y_pred_proba : array-like, optional
            Predicted probabilities
        labels : list, optional
            Class labels for reporting
        model_name : str
            Name of model for reporting
        
        Returns:
        --------
        report : dict with comprehensive metrics
        """
        
        print("\n" + "="*70)
        print(f"METRICS REPORT: {model_name}")
        print("="*70)
        
        # Basic metrics
        accuracy = (y_pred == y_true).mean()
        
        # Macro-averaged metrics
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Weighted metrics
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        print(f"\nOverall Metrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"\nMacro-Averaged Metrics:")
        print(f"  F1-Score: {f1_macro:.4f}")
        print(f"  Precision: {precision_macro:.4f}")
        print(f"  Recall: {recall_macro:.4f}")
        print(f"\nWeighted-Averaged Metrics:")
        print(f"  F1-Score: {f1_weighted:.4f}")
        print(f"  Precision: {precision_weighted:.4f}")
        print(f"  Recall: {recall_weighted:.4f}")
        
        # Per-class metrics
        print(f"\nPer-Class Metrics:")
        class_report = classification_report(
            y_true, y_pred, target_names=labels, zero_division=0
        )
        print(class_report)
        
        # AUC if probabilities provided
        auc_macro = None
        if y_pred_proba is not None:
            try:
                auc_macro = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
                print(f"\nAUC Metrics:")
                print(f"  Macro-Averaged AUC: {auc_macro:.4f}")
            except:
                pass
        
        report = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_weighted': f1_weighted,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'auc_macro': auc_macro,
            'classification_report': class_report
        }
        
        return report
    
    @staticmethod
    def confusion_matrix_analysis(y_true, y_pred, labels=None, normalize=False):
        """
        Analyze confusion matrix with detailed insights
        """
        
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        print("\n" + "="*70)
        print("CONFUSION MATRIX ANALYSIS")
        print("="*70)
        
        if labels is None:
            labels = [f"Class {i}" for i in range(len(cm))]
        
        # Create DataFrame for better visualization
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        print("\nConfusion Matrix:")
        print(cm_df)
        
        # Analyze misclassifications
        print("\nMisclassification Analysis:")
        for i, label in enumerate(labels):
            misclassified = cm[i].sum() - cm[i, i]
            total = cm[i].sum()
            if total > 0:
                misclass_rate = (misclassified / total) * 100
                print(f"  {label}: {misclassified}/{total} misclassified ({misclass_rate:.1f}%)")
        
        return cm_df


class ThresholdAnalyzer:
    """
    Analyze and optimize probability thresholds
    """
    
    @staticmethod
    def analyze_threshold_range(y_true, y_proba, component_name="Component",
                               thresholds=None, optimize_for='f1'):
        """
        Analyze metrics across threshold range
        
        Parameters:
        -----------
        y_true : array-like
            True binary labels
        y_proba : array-like
            Predicted probabilities
        component_name : str
            Component name for reporting
        thresholds : array-like, optional
            Thresholds to test (default: 0.0-1.0 with 0.01 step)
        optimize_for : str
            Metric to optimize ('f1', 'precision', 'recall')
        
        Returns:
        --------
        results_df : DataFrame with threshold analysis
        optimal_threshold : float
        optimal_metrics : dict
        """
        
        if thresholds is None:
            thresholds = np.arange(0.0, 1.01, 0.01)
        
        print(f"\n{'='*70}")
        print(f"THRESHOLD ANALYSIS: {component_name}")
        print(f"{'='*70}")
        print(f"Analyzing {len(thresholds)} thresholds...")
        
        results = []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            # Calculate metrics
            if len(np.unique(y_pred)) > 1:
                tp = ((y_pred == 1) & (y_true == 1)).sum()
                fp = ((y_pred == 1) & (y_true == 0)).sum()
                tn = ((y_pred == 0) & (y_true == 0)).sum()
                fn = ((y_pred == 0) & (y_true == 1)).sum()
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            else:
                precision = recall = f1 = specificity = 0
            
            results.append({
                'Threshold': threshold,
                'Precision': precision,
                'Recall': recall,
                'F1': f1,
                'Specificity': specificity,
                'Positive_Predictions': (y_proba >= threshold).sum()
            })
        
        results_df = pd.DataFrame(results)
        
        # Find optimal threshold
        if optimize_for == 'f1':
            optimal_idx = results_df['F1'].idxmax()
        elif optimize_for == 'precision':
            optimal_idx = results_df['Precision'].idxmax()
        else:  # recall
            optimal_idx = results_df['Recall'].idxmax()
        
        optimal_threshold = results_df.loc[optimal_idx, 'Threshold']
        optimal_metrics = results_df.loc[optimal_idx].to_dict()
        
        # Print results
        print(f"\nOptimal Threshold ({optimize_for.upper()}):")
        print(f"  Threshold: {optimal_threshold:.3f}")
        print(f"  Precision: {optimal_metrics['Precision']:.4f}")
        print(f"  Recall: {optimal_metrics['Recall']:.4f}")
        print(f"  F1-Score: {optimal_metrics['F1']:.4f}")
        print(f"  Specificity: {optimal_metrics['Specificity']:.4f}")
        
        # Compare to default (0.5)
        default_row = results_df[results_df['Threshold'] == 0.5].iloc[0]
        print(f"\nComparison to Default (0.50):")
        print(f"  Precision improvement: {optimal_metrics['Precision']/default_row['Precision']:.2f}x")
        print(f"  F1-Score change: {optimal_metrics['F1'] - default_row['F1']:+.4f}")
        
        return results_df, optimal_threshold, optimal_metrics
    
    @staticmethod
    def plot_threshold_analysis(results_df, optimal_threshold, component_name="Component"):
        """
        Plot threshold analysis results
        """
        
        plt.figure(figsize=(12, 6))
        
        plt.plot(results_df['Threshold'], results_df['Precision'], marker='o', 
                label='Precision', linewidth=2, markersize=4)
        plt.plot(results_df['Threshold'], results_df['Recall'], marker='s',
                label='Recall', linewidth=2, markersize=4)
        plt.plot(results_df['Threshold'], results_df['F1'], marker='^',
                label='F1-Score', linewidth=2, markersize=4)
        plt.plot(results_df['Threshold'], results_df['Specificity'], marker='d',
                label='Specificity', linewidth=2, markersize=4)
        
        plt.axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='Default (0.50)')
        plt.axvline(optimal_threshold, color='red', linestyle='--', linewidth=2,
                   label=f'Optimal ({optimal_threshold:.2f})')
        
        plt.xlabel('Probability Threshold', fontsize=12, fontweight='bold')
        plt.ylabel('Metric Value', fontsize=12, fontweight='bold')
        plt.title(f'{component_name} Threshold Analysis', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        plt.xlim([0, 1.0])
        plt.ylim([0, 1.0])
        
        plt.tight_layout()
        return plt


================================================================================
SECTION 5.2: MODEL COMPARISON FRAMEWORK
================================================================================

class ModelComparison:
    """
    Framework for comparing multiple models and methods
    """
    
    def __init__(self):
        self.results = {}
    
    def add_result(self, method_name, model, X_test, y_test, y_pred, y_pred_proba=None):
        """
        Add model result to comparison
        """
        
        metrics = {
            'Method': method_name,
            'Accuracy': (y_pred == y_test).mean(),
            'F1_Macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'Precision_Macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'Recall_Macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
        }
        
        if y_pred_proba is not None:
            try:
                metrics['AUC_Macro'] = roc_auc_score(y_test, y_pred_proba, 
                                                     multi_class='ovr', average='macro')
            except:
                metrics['AUC_Macro'] = None
        
        self.results[method_name] = metrics
    
    def compare(self):
        """
        Generate comparison report
        """
        
        comparison_df = pd.DataFrame(self.results).T
        
        print("\n" + "="*70)
        print("MODEL COMPARISON RESULTS")
        print("="*70)
        print(comparison_df.to_string())
        
        # Find best method for each metric
        print("\n" + "-"*70)
        print("BEST METHODS BY METRIC:")
        print("-"*70)
        
        for col in comparison_df.columns:
            if col != 'Method':
                best_method = comparison_df[col].idxmax()
                best_value = comparison_df[col].max()
                print(f"  {col:20s}: {best_method:30s} ({best_value:.4f})")
        
        return comparison_df
    
    def plot_comparison(self):
        """
        Visualize model comparison
        """
        
        comparison_df = pd.DataFrame(self.results).T
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: F1 Scores
        ax = axes[0, 0]
        comparison_df['F1_Macro'].sort_values().plot(kind='barh', ax=ax, color='steelblue')
        ax.set_xlabel('F1-Score (Macro)', fontweight='bold')
        ax.set_title('F1-Score Comparison', fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Plot 2: Precision
        ax = axes[0, 1]
        comparison_df['Precision_Macro'].sort_values().plot(kind='barh', ax=ax, color='orange')
        ax.set_xlabel('Precision (Macro)', fontweight='bold')
        ax.set_title('Precision Comparison', fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Plot 3: Recall
        ax = axes[1, 0]
        comparison_df['Recall_Macro'].sort_values().plot(kind='barh', ax=ax, color='green')
        ax.set_xlabel('Recall (Macro)', fontweight='bold')
        ax.set_title('Recall Comparison', fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Plot 4: Accuracy
        ax = axes[1, 1]
        comparison_df['Accuracy'].sort_values().plot(kind='barh', ax=ax, color='red')
        ax.set_xlabel('Accuracy', fontweight='bold')
        ax.set_title('Accuracy Comparison', fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        return plt


================================================================================
SECTION 5.3: INTEGRATION WRAPPER CLASS
================================================================================

class PdMPipeline:
    """
    Complete integration wrapper for all three critical implementations
    
    Usage:
    ------
    pipeline = PdMPipeline()
    pipeline.fit(X_train, y_train, machine_ids_train)
    metrics = pipeline.evaluate(X_test, y_test, machine_ids_test)
    """
    
    def __init__(self, model_type='xgboost', random_state=42):
        """
        Initialize pipeline
        
        Parameters:
        -----------
        model_type : str
            'xgboost' or 'random_forest'
        random_state : int
            Random seed for reproducibility
        """
        
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.smote = None
        self.thresholds = {}
        self.metadata = {}
        
        print(f"\n{'='*70}")
        print(f"PdM PIPELINE INITIALIZED")
        print(f"{'='*70}")
        print(f"Model Type: {model_type}")
        print(f"Random State: {random_state}")
    
    def fit(self, X_train, y_train, machine_ids=None, apply_smote=True, 
           validate_with_gkfold=True, n_splits=5):
        """
        Fit model with all three critical improvements
        
        Parameters:
        -----------
        X_train : DataFrame or array
            Training features
        y_train : Series or array
            Training labels
        machine_ids : Series or array, optional
            Machine IDs for group k-fold validation
        apply_smote : bool
            Whether to apply SMOTE
        validate_with_gkfold : bool
            Whether to validate with group k-fold
        n_splits : int
            Number of folds for group k-fold
        """
        
        print(f"\n{'='*70}")
        print("FITTING MODEL")
        print(f"{'='*70}")
        
        # Step 1: Validate with Group K-Fold (optional)
        if validate_with_gkfold and machine_ids is not None:
            print("\n[1/3] Group K-Fold Validation...")
            self._validate_with_gkfold(X_train, y_train, machine_ids, n_splits)
        
        # Step 2: Apply SMOTE (optional)
        X_to_fit = X_train.copy()
        y_to_fit = y_train.copy()
        
        if apply_smote:
            print("\n[2/3] Applying SMOTE...")
            from imblearn.over_sampling import SMOTE
            
            self.smote = SMOTE(k_neighbors=5, random_state=2, sampling_strategy='auto')
            X_to_fit, y_to_fit = self.smote.fit_resample(X_train, y_train)
            print(f"  Original: {len(X_train):,} samples")
            print(f"  After SMOTE: {len(X_to_fit):,} samples")
        
        # Step 3: Train model
        print("\n[3/3] Training model...")
        
        from sklearn.preprocessing import StandardScaler
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_to_fit)
        
        if self.model_type == 'xgboost':
            from xgboost import XGBClassifier
            self.model = XGBClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=5,
                random_state=self.random_state, verbosity=0
            )
        else:
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=100, max_depth=7,
                random_state=self.random_state
            )
        
        self.model.fit(X_scaled, y_to_fit)
        print(f"  Model trained successfully")
        
        self.metadata['n_classes'] = len(np.unique(y_train))
        self.metadata['features'] = X_train.shape[1]
        print(f"\nModel ready with {self.metadata['n_classes']} classes")
    
    def _validate_with_gkfold(self, X, y, machine_ids, n_splits):
        """
        Validate with group k-fold (internal method)
        """
        
        class MachineGroupKFold:
            def __init__(self, n_splits=5, random_state=42):
                self.n_splits = n_splits
                self.random_state = random_state
            
            def split(self, X, y, groups):
                unique_machines = np.unique(groups)
                rng = np.random.RandomState(self.random_state)
                rng.shuffle(unique_machines)
                machines_per_fold = len(unique_machines) / self.n_splits
                
                for fold in range(self.n_splits):
                    start_idx = int(fold * machines_per_fold)
                    end_idx = int((fold + 1) * machines_per_fold)
                    test_machines = unique_machines[start_idx:end_idx]
                    test_mask = np.isin(groups, test_machines)
                    yield np.where(~test_mask)[0], np.where(test_mask)[0]
        
        kf = MachineGroupKFold(n_splits=n_splits, random_state=self.random_state)
        fold_scores = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X, y, machine_ids), 1):
            X_train_fold = X.iloc[train_idx] if isinstance(X, pd.DataFrame) else X[train_idx]
            X_test_fold = X.iloc[test_idx] if isinstance(X, pd.DataFrame) else X[test_idx]
            y_train_fold = y.iloc[train_idx] if isinstance(y, pd.Series) else y[train_idx]
            y_test_fold = y.iloc[test_idx] if isinstance(y, pd.Series) else y[test_idx]
            
            temp_model = RandomForestClassifier(n_estimators=50, random_state=self.random_state)
            temp_model.fit(X_train_fold, y_train_fold)
            y_pred_fold = temp_model.predict(X_test_fold)
            
            f1 = f1_score(y_test_fold, y_pred_fold, average='macro')
            fold_scores.append(f1)
            print(f"  Fold {fold_idx}: F1={f1:.4f}")
        
        mean_f1 = np.mean(fold_scores)
        print(f"  Average F1: {mean_f1:.4f} ± {np.std(fold_scores):.4f}")
        self.metadata['gkfold_mean_f1'] = mean_f1
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test set
        """
        
        print(f"\n{'='*70}")
        print("MODEL EVALUATION")
        print(f"{'='*70}")
        
        X_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_scaled)
        y_pred_proba = self.model.predict_proba(X_scaled)
        
        # Calculate metrics
        from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
        
        metrics = {
            'Accuracy': (y_pred == y_test).mean(),
            'F1_Macro': f1_score(y_test, y_pred, average='macro'),
            'Precision_Macro': precision_score(y_test, y_pred, average='macro'),
            'Recall_Macro': recall_score(y_test, y_pred, average='macro'),
            'AUC_Macro': roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
        }
        
        print("\nMetrics:")
        for metric, value in metrics.items():
            print(f"  {metric:20s}: {value:.4f}")
        
        return metrics, y_pred, y_pred_proba
    
    def optimize_thresholds(self, X_test, y_test, component_indices=None):
        """
        Optimize probability thresholds for specific components
        
        Parameters:
        -----------
        X_test : DataFrame or array
            Test features
        y_test : Series or array
            Test labels
        component_indices : list
            Which components to tune (e.g., [1, 3] for comp1 and comp3)
        """
        
        if component_indices is None:
            component_indices = []
        
        print(f"\n{'='*70}")
        print("THRESHOLD OPTIMIZATION")
        print(f"{'='*70}")
        
        X_scaled = self.scaler.transform(X_test)
        y_pred_proba = self.model.predict_proba(X_scaled)
        
        thresholds = {}
        
        for comp_idx in component_indices:
            y_binary = (y_test == comp_idx).astype(int)
            y_proba_comp = y_pred_proba[:, comp_idx]
            
            # Find optimal threshold
            best_f1 = 0
            best_thresh = 0.5
            
            for thresh in np.arange(0.0, 1.01, 0.01):
                y_pred_thresh = (y_proba_comp >= thresh).astype(int)
                if len(np.unique(y_pred_thresh)) > 1:
                    f1 = f1_score(y_binary, y_pred_thresh, zero_division=0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_thresh = thresh
            
            thresholds[f'comp{comp_idx}'] = best_thresh
            print(f"  Component {comp_idx}: Optimal threshold = {best_thresh:.2f}")
        
        self.thresholds = thresholds
        return thresholds
    
    def predict_with_thresholds(self, X):
        """
        Predict using optimized thresholds
        """
        
        X_scaled = self.scaler.transform(X)
        y_pred_proba = self.model.predict_proba(X_scaled)
        
        predictions = np.argmax(y_pred_proba, axis=1)
        
        # Apply component-specific thresholds if available
        for comp_name, thresh in self.thresholds.items():
            comp_idx = int(comp_name.replace('comp', ''))
            below_thresh = y_pred_proba[:, comp_idx] < thresh
            predictions[below_thresh & (predictions == comp_idx)] = 0  # Set to "None"
        
        return predictions, y_pred_proba
    
    def save_config(self, filepath):
        """
        Save pipeline configuration for production
        """
        
        import json
        import pickle
        
        config = {
            'model_type': self.model_type,
            'random_state': self.random_state,
            'n_classes': self.metadata.get('n_classes'),
            'n_features': self.metadata.get('features'),
            'thresholds': self.thresholds,
            'metadata': self.metadata
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"\n✓ Configuration saved to {filepath}")


================================================================================
SECTION 5.4: USAGE EXAMPLES
================================================================================

"""
EXAMPLE 1: Complete Pipeline with All Features
"""

# Generate or load data
df = generate_synthetic_pdm_data(n_samples=5000)
X = df.drop(['target', 'machine_id'], axis=1)
y = df['target']
machine_ids = df['machine_id']

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
machine_ids_train = machine_ids[:len(X_train)]

# Initialize and fit pipeline
pipeline = PdMPipeline(model_type='xgboost')
pipeline.fit(X_train, y_train, machine_ids=machine_ids_train, 
            apply_smote=True, validate_with_gkfold=True, n_splits=5)

# Evaluate
metrics, y_pred, y_pred_proba = pipeline.evaluate(X_test, y_test)

# Optimize thresholds for components 1 and 3
thresholds = pipeline.optimize_thresholds(X_test, y_test, component_indices=[1, 3])

# Make predictions with optimized thresholds
y_pred_optimized, _ = pipeline.predict_with_thresholds(X_test)

# Save configuration
pipeline.save_config('/mnt/user-data/outputs/pdm_pipeline_config.json')

print("\n✓ Pipeline ready for production deployment!")


"""
EXAMPLE 2: Detailed Model Comparison
"""

# Create comparison framework
comparison = ModelComparison()

# Method 1: Baseline (no SMOTE)
X_train_baseline, X_test_baseline, y_train_baseline, y_test_baseline = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model_baseline = RandomForestClassifier(n_estimators=100, random_state=42)
model_baseline.fit(X_train_baseline, y_train_baseline)
y_pred_baseline = model_baseline.predict(X_test_baseline)
y_proba_baseline = model_baseline.predict_proba(X_test_baseline)

comparison.add_result('Baseline (No SMOTE)', model_baseline, X_test_baseline, 
                     y_test_baseline, y_pred_baseline, y_proba_baseline)

# Method 2: With SMOTE
from imblearn.over_sampling import SMOTE

smote = SMOTE(k_neighbors=5, random_state=2)
X_train_smote, y_train_smote = smote.fit_resample(X_train_baseline, y_train_baseline)

model_smote = RandomForestClassifier(n_estimators=100, random_state=42)
model_smote.fit(X_train_smote, y_train_smote)
y_pred_smote = model_smote.predict(X_test_baseline)
y_proba_smote = model_smote.predict_proba(X_test_baseline)

comparison.add_result('With SMOTE', model_smote, X_test_baseline,
                     y_test_baseline, y_pred_smote, y_proba_smote)

# Compare
comparison_df = comparison.compare()
comparison.plot_comparison()

print("\n✓ Model comparison complete!")


"""
EXAMPLE 3: Detailed Threshold Analysis with Visualization
"""

# Analyze Component 1 threshold
y_test_comp1 = (y_test == 1).astype(int)
y_proba_comp1 = y_pred_proba[:, 1]

analyzer = ThresholdAnalyzer()
results_df, opt_thresh, opt_metrics = analyzer.analyze_threshold_range(
    y_test_comp1, y_proba_comp1, component_name='Component 1',
    optimize_for='precision'
)

# Visualize
analyzer.plot_threshold_analysis(results_df, opt_thresh, 'Component 1')

print("\n✓ Threshold analysis complete!")


