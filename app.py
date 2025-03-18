# 1
import gradio as gr
import os
import torch
from model import create_effnetb2
from timeit import default_timer as timer
from typing import Tuple, Dict
# setup class names
with open("class_names.txt", "r") as f:
    class_names = [food_name.strip() for food_name in f.readlines()]
# model and transforms
effnetb2,effnetb2_transforms=create_effnetb2(num_classes=101)
# load the save weights
effnetb2.load_state_dict(
    torch.load(f="food_big.pth",
               map_location=torch.device("cpu"))
)
# predict func
def predict(img):
  start_time=timer()
  img=effnetb2_transforms(img).unsqueeze(0) #add batch dim
  effnetb2.eval()
  with torch.inference_mode():
    pred_logit=effnetb2(img)
    pred_probs=torch.softmax(pred_logit,dim=1)
    pred_labels_and_probs={class_names[i]:float(pred_probs[0][i])for i in range(len(class_names)) }
    end_time=timer()
    pred_time=round(end_time-start_time,4)
    return pred_labels_and_probs,pred_time

# 4gradip app
import gradio as gr
title="FoodVision Big"
description="Classify images of food into 101 classes using an EfficientNet-B2 feature extractor. Quick, accurate, and perfect for showcasing computer vision in action!  "
# example list
# getting list of list

example_list=[["examples/"+example]for example in os.listdir("examples")]
# craete the gradient demo
demo=gr.Interface(fn=predict, #maps input to output
                  inputs=gr.Image(type="pil"),
                  outputs=[gr.Label(num_top_classes=5,label="Predictions"),
                           gr.Number(label="Prediction time(s)")],
                           examples=example_list,
                           title=title,
                           description=description

                  )
# launch it
demo.launch(
    debug=False
    
    # preints error locally? like in googlr collab
    # generate link publicaly like share with public
)
