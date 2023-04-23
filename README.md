# GPT-Echo

GPT-Echo is an open source toolkit for using pretrained GPT models to generate embeddings which are fed into an echo state network (ESN).

The ESN acts as a contextualizer, preserving semantic information from the GPT embeddings to aid downstream tasks.

The readout layer of the ESN is then trained on target tasks like text generation, sentiment analysis, and topic modeling. Currently only text generation is supported.

## Installation

```bash
pip install gpt-echo
```  

Requirements:
- Python 3.6+
- PyTorch 1.0+
- pretrained GPT models

## Usage

### Initialization 
```python
import gpt_echo

# Load a pretrained GPT-2 model  
model = gpt_echo.GPT2Model("gpt2-medium")   

# Initialize an ESN  
esn = gpt_echo.ESN(input_dim=model.dim, ...)   

# Train readout layer for your task...  
```

## Experimental features

These are experimental. They work but do not guarantee better results and can slow down training.

1. `--usecot` - this trains 2 different ESN networks, a mediator and a generator. The mediator than then potentially be used to direct generator sampling.
2. `--forwardforward` - uses Hinton's forward-forward training on the readout layer. For negative samples it support either random uniform, a custom negative dataset, or sampling from the base model.

### Pretrained models

| Dataset | Model  | Parameters | Epochs | BLEU  | ROUGE | CIDEr ?|  
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| WikiText-103 | GPT-2 (small)  |  117M |  N/A (pre-trained) |  ?   | ? | ? |
| OpenWebText  | GPT-2 (medium) |  345M | N/A (pre-trained)|   ? |  ?|  ? |
| OpenBook  | GPT-2 (large)   |  774M | N/A (pre-trained)|  ? | ?  |? |  
| Common Crawl | GPT-3  |  175B | N/A (pre-trained)| ?|?|?|

### Examples
- [Text generation](examples/text_generation.ipynb)   

## References

- [Echo State Networks: A brief tutorial](https://haraldschilly.github.io/blog/ESNTutorial.html)
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)  
