import gradio as gr
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from transformers import GPT2Tokenizer
import importlib
res = importlib.import_module("train-echo")

# Assume the required variables are defined
model, mediator, loaded = res.load_models(res.args)
if not loaded:
    print("Load failed, exiting")
gpt2 = res.gpt2
tokenizer = res.tokenizer

def generate_text(input_text, guide_text, svd_bias, guide_weight, temperature, top_k, top_p, num_tokens):
    model.reset()
    mediator.reset()
    generated_text = res.sample_model(
        model, 
        input_text, 
        tokenizer,
        seq_length=res.seq_length,
        num_tokens=num_tokens,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        svd_bias=svd_bias,
        mediator=mediator,
        guide_text=guide_text,
        guide_weight=guide_weight
    )
    print("Got", generated_text)
    return generated_text

# Create Gradio interface
iface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.inputs.Textbox(lines=3, label="Input Text"),
        gr.inputs.Textbox(lines=1, label="Guide Text"),
        gr.inputs.Checkbox(label="Use Guided Text", default=True),
       

        gr.inputs.Slider(minimum=0.0, maximum=1.0, step=0.001, default=0.1, label="Guide weight"),
        gr.inputs.Slider(minimum=0.01, maximum=2.0, step=0.01, default=1.0, label="Temperature"),
        gr.inputs.Slider(minimum=1, maximum=1000, default=100, label="Top-k"),
        gr.inputs.Slider(minimum=0.1, maximum=1.0, step=0.01, default=1.0, label="Top-p"),
        gr.inputs.Slider(minimum=10, maximum=1000, default=100, label="Number of Tokens"),
    ],
    outputs=gr.outputs.Textbox(label="Generated Text"),
    title="Echo State Network Finetune",
    description="Proof of concept demo.",
    examples=[
        ["hello, i am", "friendly"],
        ["i have", "articulate"],
        ["Once upon a time", "happy"],
        ["In a galaxy far, far away", "sci-fi"],
    ],
)

# Launch the interface
iface.launch()

