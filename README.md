# ChipSafe Real Time Ingredient Warnings Using OCR and Computer-Vision

---

## Overview

Ever struggled to understand the tiny, scientific-sounding ingredients on packaged snacks like chips?
This project helps **students with allergies or chronic health conditions** make safer food choices — just by uploading a photo of the ingredients label.

Our system uses:
- **OCR** (Tesseract) to extract ingredients from images
- **BERT-based NLP model** to classify each ingredient as **Safe** or **Risky**

<p align="center">
  <img src="Sample output.png" alt="Demo Output" width="700">
</p>

---

## Key Features

- Upload snack/label images
- Auto-extract ingredients
- Risk classification: allergies, diabetes, cholesterol
- Real-time web interface using Gradio

---

## Motivation

Students often:
- Struggle to read tiny fonts
- Don’t recognise chemical names
- Miss critical allergy information

**We wanted to simplify this — using AI.**

---

## How It Works

| Step | Description |
|------|-------------|
| Input | Upload image of chip/snack ingredient list |
| OCR | Text extracted using Tesseract |
| Classifier | Ingredients passed through a fine-tuned BERT model |
| Output | Ingredients tagged as "Safe" 🟩 or "Risky" 🟥 |

---

## Tech Stack

- **Frontend**: Gradio
- **OCR**: Tesseract
- **Model**: BERT (fine-tuned on custom ingredient data)
- **Languages**: Python, PyTorch, Transformers

---

## 📊 Model Info

| Metric     | Score |
|------------|-------|
| Accuracy   | 68%   |
| F1 Score   | 66%   |
| Base Model | `bert-base-uncased` |
| Data       | 250+ custom-labeled ingredients |

---

## Project Files

| File | Description |
|------|-------------|
| `RiskDetector.py` | Main app with Gradio UI |
| `cis515project.ipynb` | Data prep, training, and EDA |
| `ingredient_classifier_final/` | Fine-tuned model files |
| `requirements.txt` | Python packages |
| `sample_images/` | Sample label images |

---

## Future Enhancements

- Improve OCR accuracy with denoising
- Add multilingual support
- Build mobile scanning app
- Expand to cosmetics & supplements

---

## References

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Kaggle Food Dataset](https://www.kaggle.com/datasets/uom190346a/food-ingredients-and-allergens)

## Installation
```bash
git clone https://github.com/YourUsername/ChipSafe-Real-Time-Ingredient-Warnings-Using-OCR-and-Computer-Vision.git
cd ChipSafe-Real-Time-Ingredient-Warnings-Using-OCR-and-Computer-Vision
pip install -r requirements.txt
python RiskDetector.py
