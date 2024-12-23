import matplotlib.pyplot as plt
import shap
import numpy as np
import os

def generate_feature_importance_plot(processed_input, model):
    """
    Generate feature importance plot using SHAP values
    """
    # Initialize SHAP explainer
    explainer = shap.DeepExplainer(model.model, processed_input)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(processed_input)
    
    # Create feature importance plot
    plt.figure(figsize=(10, 6))
    feature_names = ['Loan Amount', 'Interest Rate', 'Rate Spread',
                    'Upfront Charges', 'Term', 'Property Value',
                    'Income', 'Credit Score']
    
    # Plot SHAP values
    shap.summary_plot(
        shap_values[0],
        processed_input,
        feature_names=feature_names,
        show=False
    )
    
    # Save plot
    plot_path = 'static/images/feature_importance.png'
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return plot_path