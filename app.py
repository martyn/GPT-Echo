# app.py
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import eventlet
import importlib
res = importlib.import_module("train-echo")

eventlet.monkey_patch()

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret!"
socketio = SocketIO(app, async_mode="eventlet")
model, mediator, loaded = res.load_models(res.args)
if not loaded:
    print("Load failed, exiting")
gpt2 = res.gpt2
tokenizer = res.tokenizer
gpt2.cuda()
model.cuda()

def process_input(user_input):
    user_input = "<|prompter|>" + user_input + "<|endoftext|><|assistant|>"
    print("Added: " + user_input)
    num_tokens=128
    top_k=100
    top_p=0.9
    temperature=0.9
    svd_bias=False
    guide_text="tmp"
    guide_weight=0.4
    #while(generated_text != ">"):
    generated_text = res.sample_model(
        model, 
        user_input, 
        tokenizer,
        seq_length=res.seq_length,
        num_tokens=num_tokens,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        svd_bias=svd_bias,
        mediator=mediator,
        guide_text=guide_text,
        guide_weight=guide_weight,
        stopchar='>'
    )
    yield generated_text

@socketio.on("reset_model")
def handle_reset_model():
    model.reset()
    emit("model_reset")

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on("user_input")
def handle_user_input(data):
    user_input = data["user_input"]
    response_chunks = process_input(user_input)
    for chunk in response_chunks:
        emit("response_chunk", {"chunk": chunk, "end_of_line": False})
        eventlet.sleep(0.5)  # Adjust the delay between sending chunks as desired

    emit("response_chunk", {"chunk": "", "end_of_line": True})
if __name__ == "__main__":
    print("starting")
    socketio.run(app, host='0.0.0.0', port=8000)
    print("started")

