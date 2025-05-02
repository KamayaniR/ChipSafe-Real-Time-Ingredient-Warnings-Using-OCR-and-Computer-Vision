import gradio as gr
import pytesseract
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load Fine-tuned Model 
model_path = "ingredient_classifier_final"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# OCR + Classification Function 
def classify_ingredients(image):
    extracted_text = pytesseract.image_to_string(image).replace("\n", " ").lower()
    words = [i.strip() for i in extracted_text.split(",") if len(i.strip()) > 2]
    stopwords = {"and", "or", "is", "with", "the", "of", "in", "to", "a", "an"}
    ingredients = [word for word in words if word not in stopwords]
    
    if not ingredients:
        return "⚠️ No valid ingredients detected in the image. Please upload a clearer image."

    results = []
    for ingredient in ingredients:
        inputs = tokenizer(ingredient, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        pred_id = logits.argmax().item()
        pred_label = model.config.id2label[pred_id]
        color = "🟩" if "safe" in pred_label.lower() else "🟥"
        results.append((ingredient, pred_label, color))

    extracted_text_display = f"### 📝 Extracted Text:\n\n> {extracted_text}\n\n"
    classification_display = "### 🔍 Classification Results:\n\n"
    classification_display += "| Ingredient | Classification |\n"
    classification_display += "|:-----------|:----------------|\n"
    for ing, label, color in results:
        classification_display += f"| {ing} | {color} {label} |\n"

    return extracted_text_display + classification_display



#  Gradio Interface 
demo = gr.Interface(
    fn=classify_ingredients,
    inputs=gr.Image(type="pil", label="📷 Upload Chips Label Image"),
    outputs=gr.Markdown(),
    title="🧪 Ingredient Disease Risk Analyzer",
    description=(
        "Upload an image containing a list of ingredients. "
        "The system will extract the ingredients using OCR and classify each as 'Safe' or 'Risky'."
        "\n\n**💡 Tip:** Use clear, high-resolution images for best accuracy!"
    ),
    allow_flagging="never"
)

# Launch the App 
demo.launch()
