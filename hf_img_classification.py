import gradio as gr
from transformers import ViTImageProcessor, ViTForImageClassification

MODEL_NAME = "google/vit-base-patch16-224"

processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
model = ViTForImageClassification.from_pretrained(MODEL_NAME)


def classify_image(img):
    inputs = processor(images=img, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicated_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicated_class_idx]


with gr.Blocks() as ui:
    with gr.Row(equal_height=True):
        with gr.Column():
            img_input = gr.Image(type="pil", label="Animal Image")
        with gr.Column():
            with gr.Row():
                btn = gr.Button("Classify")
        with gr.Column():
            output = gr.Textbox(label="Class")
    btn.click(fn=classify_image, inputs=img_input, outputs=output)

ui.launch()
