from ultralytics import YOLO
import gradio as gr

model=YOLO("best (1).pt")
def pred_img(image):
    img=model.predict(image,conf=0.50)
    return img[0].plot()


app=gr.Interface(fn=pred_img,inputs="image",outputs="image")
app.launch(share=True)
