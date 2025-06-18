[Confusion_Matrix](https://github.com/Abhiudai12/Internship1/blob/main/confusion_matrix.png)
[Disease Prediction Example](intern_disease.png)
DISEASE PREDICTION FROM PATIENT SYMPTOMS:- 
This project utilizes a BERT-based model to predict diseases from patient-reported symptoms with 98% accuracy.
The system analyzes textual descriptions of symptoms and provides accurate disease predictions, potentially aiding in early diagnosis and treatment.

Features -
Utilizes BERT (Bidirectional Encoder Representations from Transformers) for natural language understanding
Achieves 98% accuracy in disease prediction
Processes patient-reported symptoms in text format
Handles a diverse range of medical conditions

Dataset -
The project uses the Symptom2Disease.csv dataset, which contains:
label: The disease diagnosis
text: Patient-reported symptoms in textual format
Model Architecture
Base model: BERT
Fine-tuned for the specific task of disease prediction

Implementation Details -
Language: Python
Main libraries:
PyTorch
Transformers (Hugging Face)
scikit-learn
pandas
numpy

Key Components -
Data Preprocessing:
Text cleaning and normalization
Label encoding for disease categories

Model Training:
Fine-tuning BERT on the symptom-disease dataset

Evaluation:
Cross-validation to ensure model robustness
Metrics: Accuracy, Precision, Recall, F1-score

Usage :
Install the required dependencies:
text
pip install torch transformers scikit-learn pandas numpy
Prepare your dataset in CSV format with 'label' and 'text' columns.
Run the Jupyter notebook to train and evaluate the model.

Results:
The model achieves 98% accuracy in predicting diseases from patient-reported symptoms, demonstrating its potential for assisting in medical diagnoses.

Disclaimer:
This tool is intended for research and educational purposes only. 
It should not be used as a substitute for professional medical advice, diagnosis, or treatment. 
Always consult with a qualified healthcare provider for medical concerns.
