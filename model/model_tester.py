import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, Tuple, Optional
import pickle
import argparse
from pathlib import Path
import json


from utils.logging_manager import get_logger

logger = get_logger(__name__)


class GNNModelTester:
    """
    Comprehensive testing and analysis framework for GNN models
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.results = {}
        
    def train(
        self,
        data_dict: Dict,
        train_mask: torch.Tensor,
        val_mask: torch.Tensor,
        num_epochs: int = 200,
        lr: float = 0.01,
        weight_decay: float = 5e-4,
        patience: int = 50,
        verbose: bool = True,
        use_class_weights: bool = True
    ) -> Dict:
        """
        Train the model with early stopping
        
        Args:
            data_dict: Dictionary containing graph data
            train_mask: Boolean mask for training nodes
            val_mask: Boolean mask for validation nodes
            num_epochs: Maximum number of training epochs
            lr: Learning rate
            weight_decay: L2 regularization weight
            patience: Early stopping patience
            verbose: Print training progress
        
        Returns:
            Dictionary with training history
        """
        # Extract data
        x = data_dict['feature'].to(self.device)
        edge_index = data_dict['edge_index'].to(self.device)
        edge_curvature = data_dict['ollivier_curvature'].to(self.device)
        edge_attr = data_dict.get('edge_features', None)
        if edge_attr is not None:
            edge_attr = edge_attr.to(self.device)
        
        labels = data_dict['label'].long().to(self.device)
        train_mask = train_mask.to(self.device)
        val_mask = val_mask.to(self.device)
        
        # Calculate class weights to handle imbalance
        if use_class_weights:
            train_labels = labels[train_mask]
            class_counts = torch.bincount(train_labels)
            total_samples = len(train_labels)
            class_weights = total_samples / (len(class_counts) * class_counts.float())
            logger.info(f"Class distribution in training: {class_counts.cpu().numpy()}")
            logger.info(f"Using class weights: {class_weights.cpu().numpy()}")
        else:
            class_weights = None
        
        # Setup optimizer and loss
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': []
        }
        
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            optimizer.zero_grad()
            
            out = self.model(x, edge_index, edge_curvature, edge_attr)
            loss = criterion(out[train_mask], labels[train_mask])
            loss.backward()
            optimizer.step()
            
            # Calculate training metrics
            train_pred = out[train_mask].argmax(dim=1)
            train_acc = (train_pred == labels[train_mask]).float().mean()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                out = self.model(x, edge_index, edge_curvature, edge_attr)
                val_loss = criterion(out[val_mask], labels[val_mask])
                val_pred = out[val_mask].argmax(dim=1)
                val_acc = (val_pred == labels[val_mask]).float().mean()
                
                # Calculate F1 score
                val_f1 = f1_score(
                    labels[val_mask].cpu().numpy(),
                    val_pred.cpu().numpy(),
                    average='binary'
                )
            
            # Store history
            history['train_loss'].append(loss.item())
            history['train_acc'].append(train_acc.item())
            history['val_loss'].append(val_loss.item())
            history['val_acc'].append(val_acc.item())
            history['val_f1'].append(val_f1)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                if verbose:
                    logger.info(f"Early stopping at epoch {epoch}")
                break
            
            if verbose and (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"Train Loss: {loss.item():.4f} | "
                    f"Val Loss: {val_loss.item():.4f} | "
                    f"Val Acc: {val_acc.item():.4f} | "
                    f"Val F1: {val_f1:.4f}"
                )
                # Log prediction distribution
                with torch.no_grad():
                    train_pred_dist = torch.bincount(train_pred, minlength=2)
                    val_pred_dist = torch.bincount(val_pred, minlength=2)
                    logger.info(f"  Train predictions: {train_pred_dist.cpu().numpy()}")
                    logger.info(f"  Val predictions: {val_pred_dist.cpu().numpy()}")
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return history
    
    def evaluate(
        self,
        data_dict: Dict,
        test_mask: torch.Tensor,
        node_names: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Comprehensive evaluation of the model
        
        Args:
            data_dict: Dictionary containing graph data
            test_mask: Boolean mask for test nodes
            node_names: Optional array of node names for detailed analysis
        
        Returns:
            Dictionary with evaluation metrics and predictions
        """
        self.model.eval()
        
        # Extract data
        x = data_dict['feature'].to(self.device)
        edge_index = data_dict['edge_index'].to(self.device)
        edge_curvature = data_dict['ollivier_curvature'].to(self.device)
        edge_attr = data_dict.get('edge_features', None)
        if edge_attr is not None:
            edge_attr = edge_attr.to(self.device)
        
        labels = data_dict['label'].long().to(self.device)
        test_mask = test_mask.to(self.device)
        
        with torch.no_grad():
            out = self.model(x, edge_index, edge_curvature, edge_attr)
            probs = F.softmax(out, dim=1)
            preds = out.argmax(dim=1)
        
        # Get test set predictions
        y_true = labels[test_mask].cpu().numpy()
        y_pred = preds[test_mask].cpu().numpy()
        y_prob = probs[test_mask, 1].cpu().numpy()  # Probability of driver class
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0,
            'avg_precision': average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report
        report = classification_report(
            y_true, y_pred,
            target_names=['Non-Driver', 'Driver'],
            output_dict=True,
            zero_division=0
        )
        
        # Store results
        results = {
            'metrics': metrics,
            'confusion_matrix': cm,
            'classification_report': report,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
        
        # Add node-level predictions if names provided
        if node_names is not None:
            test_indices = torch.where(test_mask)[0].cpu().numpy()
            # Extract gene names for test nodes
            test_gene_names = [node_names[i] for i in test_indices]
            results['predictions_df'] = pd.DataFrame({
                'gene': test_gene_names,
                'true_label': y_true,
                'predicted_label': y_pred,
                'driver_probability': y_prob,
                'correct': y_true == y_pred
            })
        
        self.results = results
        return results
    
    def plot_training_history(self, history: Dict, save_path: Optional[str] = None):
        """Plot training history curves"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Loss
        axes[0].plot(history['train_loss'], label='Train Loss')
        axes[0].plot(history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(history['train_acc'], label='Train Accuracy')
        axes[1].plot(history['val_acc'], label='Val Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # F1 Score
        axes[2].plot(history['val_f1'], label='Val F1 Score', color='green')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('F1 Score')
        axes[2].set_title('Validation F1 Score')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved training history plot to {save_path}")
        plt.show()
    
    def plot_confusion_matrix(self, save_path: Optional[str] = None):
        """Plot confusion matrix heatmap"""
        cm = self.results['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Driver', 'Driver'],
            yticklabels=['Non-Driver', 'Driver']
        )
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to {save_path}")
        plt.show()
    
    def plot_roc_curve(self, save_path: Optional[str] = None):
        """Plot ROC curve"""
        y_true = self.results['y_true']
        y_prob = self.results['y_prob']
        
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = self.results['metrics']['roc_auc']
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved ROC curve to {save_path}")
        plt.show()
    
    def plot_precision_recall_curve(self, save_path: Optional[str] = None):
        """Plot Precision-Recall curve"""
        y_true = self.results['y_true']
        y_prob = self.results['y_prob']
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        avg_precision = self.results['metrics']['avg_precision']
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AP = {avg_precision:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved PR curve to {save_path}")
        plt.show()
    
    def analyze_predictions(self, top_k: int = 20):
        """
        Analyze predictions in detail
        
        Args:
            top_k: Number of top confident predictions to display
        """
        if 'predictions_df' not in self.results:
            logger.warning("No prediction DataFrame available")
            return
        
        df = self.results['predictions_df']
        
        print("\n" + "="*80)
        print("PREDICTION ANALYSIS")
        print("="*80)
        
        # Overall statistics
        print(f"\nTotal Test Genes: {len(df)}")
        print(f"Correctly Classified: {df['correct'].sum()} ({df['correct'].mean()*100:.2f}%)")
        print(f"Misclassified: {(~df['correct']).sum()} ({(~df['correct']).mean()*100:.2f}%)")
        
        # True positives, false positives, etc.
        tp = ((df['true_label'] == 1) & (df['predicted_label'] == 1)).sum()
        fp = ((df['true_label'] == 0) & (df['predicted_label'] == 1)).sum()
        tn = ((df['true_label'] == 0) & (df['predicted_label'] == 0)).sum()
        fn = ((df['true_label'] == 1) & (df['predicted_label'] == 0)).sum()
        
        print(f"\nTrue Positives (Driver → Driver): {tp}")
        print(f"False Positives (Non-Driver → Driver): {fp}")
        print(f"True Negatives (Non-Driver → Non-Driver): {tn}")
        print(f"False Negatives (Driver → Non-Driver): {fn}")
        
        # Check if model is learning
        if tp == 0 and fp == 0:
            print("\n" + "!"*80)
            print("WARNING: Model is predicting ALL genes as Non-Driver!")
            print("!"*80)
            print("\nPossible solutions:")
            print("1. Use class weights: Add --use_class_weights flag")
            print("2. Increase learning rate: Try --lr 0.05 or --lr 0.1")
            print("3. Reduce model complexity: Try --num_layers 2")
            print("4. Check if features contain useful signal")
            print("5. Try focal loss instead of cross-entropy")
            print("6. Use oversampling or SMOTE for the minority class")
            
            # Show probability distribution
            prob_stats = df['driver_probability'].describe()
            print(f"\nDriver Probability Distribution:")
            print(prob_stats)
            
            if prob_stats['std'] < 0.01:
                print("\nProbabilities are nearly constant - model output is not discriminative!")
                print("This suggests:")
                print("- Vanishing/exploding gradients")
                print("- Features not informative")
                print("- Learning rate too low")
        
        # Top confident driver predictions
        print(f"\n{'-'*80}")
        print(f"TOP {top_k} MOST CONFIDENT DRIVER PREDICTIONS")
        print(f"{'-'*80}")
        top_drivers = df.nlargest(top_k, 'driver_probability')
        print(top_drivers[['gene', 'true_label', 'predicted_label', 
                          'driver_probability', 'correct']].to_string(index=False))
        
        # Top confident non-driver predictions
        print(f"\n{'-'*80}")
        print(f"TOP {top_k} MOST CONFIDENT NON-DRIVER PREDICTIONS")
        print(f"{'-'*80}")
        top_non_drivers = df.nsmallest(top_k, 'driver_probability')
        print(top_non_drivers[['gene', 'true_label', 'predicted_label',
                               'driver_probability', 'correct']].to_string(index=False))
        
        # False positives (predicted driver but actually non-driver)
        fp_df = df[(df['true_label'] == 0) & (df['predicted_label'] == 1)]
        if len(fp_df) > 0:
            print(f"\n{'-'*80}")
            print(f"FALSE POSITIVES (Predicted Driver, Actually Non-Driver)")
            print(f"{'-'*80}")
            print(fp_df.nlargest(min(10, len(fp_df)), 'driver_probability')[
                ['gene', 'driver_probability']].to_string(index=False))
        
        # False negatives (predicted non-driver but actually driver)
        fn_df = df[(df['true_label'] == 1) & (df['predicted_label'] == 0)]
        if len(fn_df) > 0:
            print(f"\n{'-'*80}")
            print(f"FALSE NEGATIVES (Predicted Non-Driver, Actually Driver)")
            print(f"{'-'*80}")
            print(fn_df.nsmallest(min(10, len(fn_df)), 'driver_probability')[
                ['gene', 'driver_probability']].to_string(index=False))
        
        print("\n" + "="*80)
    
    def save_results(self, outdir: str):
        """Save all results to files"""
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        with open(outdir / 'metrics.json', 'w') as f:
            json.dump(self.results['metrics'], f, indent=2)
        
        # Save classification report
        with open(outdir / 'classification_report.json', 'w') as f:
            json.dump(self.results['classification_report'], f, indent=2)
        
        # Save predictions
        if 'predictions_df' in self.results:
            self.results['predictions_df'].to_csv(
                outdir / 'predictions.csv', index=False
            )
        
        # Save plots
        self.plot_confusion_matrix(outdir / 'confusion_matrix.png')
        self.plot_roc_curve(outdir / 'roc_curve.png')
        self.plot_precision_recall_curve(outdir / 'pr_curve.png')
        
        logger.info(f"All results saved to {outdir}")
    
    def diagnose_model(self, data_dict: Dict, sample_size: int = 100):
        """
        Diagnose model behavior by checking gradients and outputs
        """
        logger.info("\n" + "="*80)
        logger.info("MODEL DIAGNOSTICS")
        logger.info("="*80)
        
        self.model.train()
        
        x = data_dict['feature'][:sample_size].to(self.device)
        edge_index = data_dict['edge_index'].to(self.device)
        edge_curvature = data_dict['ollivier_curvature'].to(self.device)
        edge_attr = data_dict.get('edge_features', None)
        if edge_attr is not None:
            edge_attr = edge_attr.to(self.device)
        
        # Forward pass
        out = self.model(x, edge_index, edge_curvature, edge_attr)
        
        logger.info(f"\nOutput statistics (sample of {sample_size} nodes):")
        logger.info(f"  Mean: {out.mean().item():.4f}")
        logger.info(f"  Std: {out.std().item():.4f}")
        logger.info(f"  Min: {out.min().item():.4f}")
        logger.info(f"  Max: {out.max().item():.4f}")
        
        probs = F.softmax(out, dim=1)
        logger.info(f"\nProbability statistics:")
        logger.info(f"  Class 0 mean prob: {probs[:, 0].mean().item():.4f}")
        logger.info(f"  Class 1 mean prob: {probs[:, 1].mean().item():.4f}")
        
        # Check gradients
        loss = out.sum()
        loss.backward()
        
        logger.info(f"\nGradient statistics:")
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                logger.info(f"  {name}: {grad_norm:.6f}")
        
        logger.info("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Test and analyze curvature-constrained GNN'
    )
    parser.add_argument('--dataset_file', required=True,
                       help='Path to dataset pickle file')
    parser.add_argument('--outdir', default='./results',
                       help='Output directory for results')
    parser.add_argument('--model_type', default='adaptive',
                       choices=['basic', 'adaptive'],
                       help='Type of model to use')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='Number of GNN layers')
    parser.add_argument('--hidden_channels', type=int, default=128,
                       help='Hidden dimension size')
    parser.add_argument('--num_epochs', type=int, default=200,
                       help='Maximum training epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--use_class_weights', action='store_true',
                       help='Use class weights to handle imbalance')
    parser.add_argument('--diagnose', action='store_true',
                       help='Run model diagnostics before training')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load data
    logger.info(f"Loading dataset from {args.dataset_file}")
    with open(args.dataset_file, 'rb') as f:
        data_dict = pickle.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create train/val/test splits (60/20/20)
    num_nodes = data_dict['feature'].shape[0]
    indices = np.random.permutation(num_nodes)
    
    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True
    
    # Create model
    in_channels = data_dict['feature'].shape[1]
    out_channels = 2  # Binary classification
    
    if args.model_type == 'adaptive':
        model = AdaptiveCurvatureGNN(
            in_channels=in_channels,
            hidden_channels=args.hidden_channels,
            out_channels=out_channels,
            num_layers=args.num_layers,
            dropout=args.dropout,
            device=device
        )
    else:
        configs = [
            {'curvature_type': 'positive', 'hop_type': 'one_hop', 'use_attention': False},
            {'curvature_type': 'negative', 'hop_type': 'two_hop', 'use_attention': True}
        ]
        model = CurvatureConstrainedGNN(
            in_channels=in_channels,
            hidden_channels=args.hidden_channels,
            out_channels=out_channels,
            num_layers=args.num_layers,
            layer_configs=configs,
            dropout=args.dropout
        )
    
    logger.info(f"Created {args.model_type} model with {args.num_layers} layers")
    
    # Create tester
    tester = GNNModelTester(model, device)
    
    # Run diagnostics if requested
    if args.diagnose:
        tester.diagnose_model(data_dict)
    
    # Train model
    logger.info("Starting training...")
    history = tester.train(
        data_dict=data_dict,
        train_mask=train_mask,
        val_mask=val_mask,
        num_epochs=args.num_epochs,
        lr=args.lr,
        patience=50,
        verbose=True,
        use_class_weights=args.use_class_weights
    )
    
    # Plot training history
    tester.plot_training_history(
        history,
        save_path=Path(args.outdir) / 'training_history.png'
    )
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    results = tester.evaluate(
        data_dict=data_dict,
        test_mask=test_mask,
        node_names=data_dict.get('node_name', None)
    )
    
    # Print metrics
    print("\n" + "="*80)
    print("TEST SET METRICS")
    print("="*80)
    for metric, value in results['metrics'].items():
        print(f"{metric.upper()}: {value:.4f}")
    
    # Analyze predictions
    tester.analyze_predictions(top_k=20)
    
    # Save all results
    tester.save_results(args.outdir)
    
    logger.info("Testing and analysis complete!")


if __name__ == '__main__':
    main()