# Intelligent Multi-Disease Medical Diagnosis System

CS-3310: Artificial Intelligence | Assignment 3 | Muhammad Ali Jinnah University

## Dataset
HAM10000 Skin Cancer Dataset (10,015 dermoscopic images, 7 disease categories)

## Models Implemented
- Random Forest (tabular baseline) - 38.69% accuracy
- XGBoost (tabular baseline) - 37.39% accuracy  
- ResNet50 Transfer Learning (image-based) - 77.53% accuracy
- Fusion Model (image + tabular) - 77.13% accuracy

## Key Features
- Transfer learning with frozen ResNet50 backbone
- Focal loss for class imbalance handling
- Grad-CAM explainability heatmaps
- Streamlit deployment app

## How to Run
1. conda activate skinai
2. python -m notebook (open skin-disease-ai.ipynb)
3. streamlit run app.py

## Project Structure
- skin-disease-ai.ipynb - Main source code (Tasks 2-4)
- app.py - Streamlit deployment application
- Task1_Literature_Review.docx - Literature review and Part B answers
- Final_Report.docx - Tasks 2-4 methodology and results
- *.png - All EDA, evaluation and explainability charts
