# GPT-Echo

GPT-Echo is open source research which uses pretrained GPT models to generate embeddings that are then fed into an echo state network (ESN) for memory.

The ESN acts as a contextualizer, preserving semantic information from the GPT embeddings to aid downstream tasks. (thanks GPT4)

The only trainable layer is the readout layer which makes training costs potentially comparable to a fine-tune.

Currently only text generation is supported.

## Chatbot

[ Add screenshot ]

The chatbot interface.

## Installation

Download this repo.

`pip3 install -r requirements.txt`

## Running the chatbot

1. Grab the pretrained model.(Not ready yet)
2. run `python3 app.py -m '[foundation model]' -n '[pretrained_model]' -r [reservoir_size] -z [context_length]`


## Training

```
python3 train-echo.py -m 'cerebras/Cerebras-GPT-111M' -n test -e 10 -r 1024 -z 128 -t emoji_train.txt -v emoji_test.txt -lr 0.006
```

This will train the echo network and save `emoji.pth` after training for 10 epochs with a 1024 reservoir size, context length of 128 training with cross entropy loss.

## Scaling

This approach has not been scaled. In toy tasks larger foundation models do better.

If you do scale this make sure to do a grid search, a lot of options have to be just right.

### Grid search

Edit `search.py` and set your options.

Run it with the same arguments you'd use to train.

## Experimental features

These are experimental. They work but do not guarantee better results and can slow down training.

1. `--usecot` - this trains 2 different ESN networks, a mediator and a generator. The mediator than then potentially be used to direct generator sampling(needs more research).
2. `--forwardforward` - uses Hinton's forward-forward training the readout layer. For negative samples it support either random uniform, a custom negative dataset, or sampling from the base model.


### Pretrained models

| Dataset | Foundation Model | Download  | Reservoir size | Context Length | Epochs | Accuracy
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| [https://huggingface.co/datasets/OpenAssistant/oasst1](OpenAssistant/oasst1) | cerebras/Cerebras-GPT-111M | ... | 768 | 128 | 5 |  ?
| [https://huggingface.co/datasets/OpenAssistant/oasst1](OpenAssistant/oasst1) | OpenAssistant/stablelm-7b-sft-v7-epoch-3 | ... | 1024 |  128 |  0.1 |  ?


## Notes and contributing

This is an experimental and novel technique. There may be bugs. It may not perform as well as other methods. Further evaluation is required.

## License

MIT


## References

- [CoT: Cooperative Training for Generative Modeling of Discrete Data](https://proceedings.mlr.press/v97/lu19d.html)
- [The Forward-Forward Algorithm: Some Preliminary Investigations](https://arxiv.org/abs/2212.13345)  
- [Echo State Networks: A brief tutorial](https://haraldschilly.github.io/blog/ESNTutorial.html)
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)  
- [The''echo state''approach to analysing and training recurrent neural networks](https://www.semanticscholar.org/paper/The''echo-state''approach-to-analysing-and-training-Jaeger/8430c0b9afa478ae660398704b11dca1221ccf22)
