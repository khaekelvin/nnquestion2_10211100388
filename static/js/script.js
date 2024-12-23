document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('loan-form');
    const loadingIndicator = document.getElementById('loading');
    const resultContainer = document.getElementById('result');
    
    if (form) {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Show loading
            loadingIndicator.style.display = 'block';
            resultContainer.style.display = 'none';
            
            // Get form data
            const formData = new FormData(form);
            
            // Send prediction request
            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                // Hide loading
                loadingIndicator.style.display = 'none';
                resultContainer.style.display = 'block';
                
                // Update result
                resultContainer.innerHTML = `
                    <h2>Prediction Result</h2>
                    <div class="result ${data.default_probability > 0.5 ? 'default' : 'no-default'}">
                        <h3>${data.default_probability > 0.5 ? 'High Risk of Default' : 'Low Risk of Default'}</h3>
                        <p>Default Probability: ${(data.default_probability * 100).toFixed(2)}%</p>
                    </div>
                    <div class="feature-importance">
                        <h3>Key Factors</h3>
                        <img src="${data.feature_importance_plot}" alt="Feature Importance Plot">
                    </div>
                `;
            })
            .catch(error => {
                loadingIndicator.style.display = 'none';
                resultContainer.innerHTML = `
                    <div class="error">
                        An error occurred while processing your request. Please try again.
                    </div>
                `;
                console.error('Error:', error);
            });
        });
    }

    // Input validation
    const numericInputs = document.querySelectorAll('input[type="number"]');
    numericInputs.forEach(input => {
        input.addEventListener('input', function() {
            if (this.value < 0) {
                this.value = 0;
            }
        });
    });
});

function predict() {
    const form = document.getElementById('prediction-form');
    const formData = new FormData(form);
    
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'error') {
            throw new Error(data.error);
        }
        
        const resultDiv = document.getElementById('result');
        const probability = data.prediction[0];
        
        resultDiv.innerHTML = `
            <div class="prediction-result">
                <h2>Prediction Result</h2>
                <p>Default Probability: ${(probability * 100).toFixed(2)}%</p>
                <p>Risk Assessment: ${probability > 0.5 ? 'High Risk' : 'Low Risk'}</p>
            </div>
        `;
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error making prediction: ' + error.message);
    });
}