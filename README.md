# ChipSafe-Real-Time-Ingredient-Warnings-Using-OCR-and-Computer-Vision

This project develops a Computer Vision (CV)-based tool that extracts ingredients from food package labels and classifies them as **Safe** or **Risky** for individuals with health conditions like allergies, diabetes, heart disease, and more. 
It combines **OCR (Optical Character Recognition)** and **fine-tuned BERT classification** to assist students and consumers in making safer food choices.

## Project Overview

- **Problem**: Manual ingredient reading is difficult due to small fonts, complex chemical names, and hidden allergens.
- **Solution**: Automated system that scans ingredient lists and flags risky ingredients in real-time.
- **Tech Stack**: Python, Pytesseract OCR, Hugging Face Transformers, Gradio for UI.

## Model Details

- Model: `bert-base-uncased`
- Fine-tuned on a **custom dataset of 250+ ingredients**.
- Risk categories:
  - Allergies (e.g., nuts, shellfish)
  - Heart disease
  - Diabetes
  - Cholesterol
- Evaluation:
  - Accuracy: ~68%
  - F1 Score: ~66%
- Deployed with: **Gradio**

