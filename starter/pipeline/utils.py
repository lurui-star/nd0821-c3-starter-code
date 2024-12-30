"""
Author: Rui Lu
Date: December, 2024
This script holds the plot and save functions
"""
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle


def plot_roc_curve(fpr, tpr, roc_auc, output_dir):
    """
    Plots the ROC curve given False Positive Rate (FPR), True Positive Rate (TPR), and AUC score.

    Parameters:
    ----------
    fpr : array-like
        False Positive Rate from ROC curve.
    
    tpr : array-like
        True Positive Rate from ROC curve.
    
    roc_auc : float
        Area Under the Curve (AUC) score.
    
    output_dir : str
        Directory to save the plot image.

    Returns:
    -------
    None
    """
    # Create the ROC curve plot
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line (random classifier)
    
    # Label the axes and set the title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    
    # Add a legend in the lower right corner
    plt.legend(loc="lower right")
    plt.grid(True)
    
    # Automatically adjust the layout to make the plot compact
    plt.tight_layout()

    # Save the plot as an image file with high resolution (DPI)
    if output_dir:
        roc_file = os.path.join(output_dir, 'roc_curve.png')
        plt.savefig(roc_file, dpi=300)  # Save with high DPI for clarity
        print(f"ROC curve saved as {roc_file}")
    
    # Show the plot (optional, can be skipped if you just want to save the plot)
    plt.show()
    
    # Close the plot to free memory
    plt.close()



def plot_feature_importance(model, X, output_dir, max_features=10):
    """
    Plots feature importance for the model, showing the top N features.

    Parameters:
    - model: trained model with a `feature_importances_` attribute (e.g., RandomForest, XGBoost)
    - X: DataFrame or array, features used for training the model (to extract feature names)
    - output_dir: str, Directory to save the plot image
    - max_features: int, the number of top features to display (default is 10)
    
    Returns:
    - None
    """
    # Extract the trained model from the pipeline (if applicable)
    model = model.named_steps['model'] if hasattr(model, 'named_steps') else model
    
    # Extract feature importances (if available)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        # Get feature names from the DataFrame if it has columns, or generate default names if ndarray
        feature_names = X.columns if isinstance(X, pd.DataFrame) else [f'Feature {i}' for i in range(X.shape[1])]
        
        # Create DataFrame to sort and plot feature importance
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).head(max_features)
        
        # Plot top feature importances
        plt.barh(importance_df['Feature'], importance_df['Importance'], color='purple')
        plt.xlabel('Importance')
        plt.title(f'Top {max_features} Feature Importances')
        plt.gca().invert_yaxis()  # Display most important feature at the top
        
        # Automatically adjust the layout to make the plot compact
        plt.tight_layout()

        # Save the plot as an image file with high resolution
        if output_dir:
            feature_importance_file = os.path.join(output_dir, 'feature_importance.png')
            plt.savefig(feature_importance_file, dpi=300)  # Save with high DPI for clarity
            print(f"Feature importance plot saved as {feature_importance_file}")
        
        # Show the plot (optional, can be skipped if you just want to save the plot)
        plt.show()

    else:
        print("No feature importance available for this model.")
    
    # Close the plot to free memory
    plt.close()

def save_model(model, model_dir):
    """
    Save the trained model to a .pkl file in the specified directory.
    """
    model_file = os.path.join(model_dir, 'best_model.pkl')
    with open(model_file, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved as {model_file}")



def check_file_exists(file_path, file_name):
    return os.path.exists(os.path.join(os.path.abspath(file_path), file_name))