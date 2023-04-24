from torch.autograd import Variable
import random
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import GPTNeoXForCausalLM, AutoTokenizer, AutoModelForCausalLM
import io
import math
import matplotlib.pyplot as plt
import mmap
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('-n', '--name', required=True)
parser.add_argument('-m', '--modelname', required=True)

parser.add_argument('-t', '--trainfilepath', required=False, default="/ml/text/emojis.txt", type=str)
parser.add_argument('-v', '--testfilepath', required=False, default="/ml/text/emojis_test.txt", type=str)
parser.add_argument('-z', '--seqlength', required=False, default=8, type=int)
parser.add_argument('-train', '--train', action='store_true')
parser.add_argument('-nf', '--negfilepath', required=False, type=str, default=None)
parser.add_argument('-l', '--layers', required=False, default=1, type=int)
parser.add_argument('-r', '--reservoir', required=False, default=1024, type=int)
parser.add_argument('-e', '--epochs', required=False, default=2, type=int)
parser.add_argument('-sr', '--spectralradius', required=False, default=0.8, type=float)
parser.add_argument('-leak', '--leakrate', required=False, default=0.9, type=float)
parser.add_argument('-c', '--connectivity', required=False, default=0.2, type=float)
parser.add_argument('-lr', '--learnrate', required=False, default=2e-4, type=float)
parser.add_argument('-cot', '--usecot', action='store_true')
parser.add_argument('-s', '--save', action='store_true')
parser.add_argument('-a', '--attn', required=False, default=0, type=int)
parser.add_argument('-ah', '--attnheads', required=False, default=4, type=int)
parser.add_argument('-b', '--batchsize', required=False, default=32, type=int)
parser.add_argument('-llama', '--llamapath', default="/ml/ais", type=str)
parser.add_argument('-forwardforward', '--forwardforward', action='store_true')
args = parser.parse_args()

model_name = args.modelname
use_cot = args.usecot
use_forward_forward = args.forwardforward
use_cerebras = False
use_llama = False
model_path = f"{args.name}.pth"
med_model_path = f"{args.name}_med.pth"
#file_path="/ml/text/alpaca_data.txt"
#validation_path="/ml/text/alpaca_data_test.txt"
#file_path="/ml/text/curated.txt"
#validation_path="/ml/text/curated_test.txt"
neg_file_path=args.negfilepath
file_path=args.trainfilepath
validation_path=args.testfilepath
#model_path = "adv.pth"  # Replace with your desired file path
#med_model_path = "adv_med.pth"  # Replace with your desired file path
reservoir_size = args.reservoir
hidden_size=reservoir_size
epochs = args.epochs
seq_length = args.seqlength
batch_size = args.batchsize
num_workers = 8  # Adjust this based on your system's resources

if "llama" in model_name:
    from transformers import LlamaConfig, LlamaForCausalLM
    from transformers.models.llama.tokenization_llama import LlamaTokenizer

    llama_path = args.llamapath
    config = LlamaConfig.from_pretrained(llama_path, load_in_8_bit=True, device_map='auto')
    gpt2 = LlamaForCausalLM(config)
    tokenizer = LlamaTokenizer.from_pretrained(llama_path)
    use_llama = True
    vocab_size = tokenizer.vocab_size
elif "pythia" in model_name:
    gpt2 = GPTNeoXForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab_size = 50274
else:
    gpt2 = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab_size = tokenizer.vocab_size

gpt2.eval()
gpt2.cuda()

if use_llama:
    gpt2_embedding_layer = gpt2.base_model.embed_tokens
elif use_cerebras and hasattr(gpt2.base_model, 'wte'):
    use_cerebras = True
    gpt2_embedding_layer = gpt2.base_model.wte
else:
    gpt2_embedding_layer =  gpt2.base_model.embed_in

input_size = gpt2_embedding_layer.weight.shape[1]

def calculate_accuracy(output, target):
    _, predicted_tokens = torch.max(output, dim=1)
    correct_predictions = (predicted_tokens == target).sum().item()
    total_predictions = target.shape[0]
    accuracy = correct_predictions / total_predictions
    return accuracy

def calculate_perplexity(output, target):
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    losses = loss_fn(output, target)
    avg_loss = torch.mean(losses)
    perplexity = math.exp(avg_loss.item())
    return perplexity

class LargeTextDataset(Dataset):
    def __init__(self, file_path, tokenizer, seq_length):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.seq_length = seq_length

        self.chunk_size = 1024 * 1024  # 1 MB
        self.tokens = []

        self.tokenize_file()

    def tokenize_file(self):
        with open(self.file_path, 'rb') as file:
            with mmap.mmap(file.fileno(), 0, prot=mmap.PROT_READ) as mmapped_file:
                for text_chunk in self.read_text_chunks(mmapped_file):
                    token_chunk = self.tokenizer.encode(text_chunk)
                    self.tokens.extend(token_chunk)

    def read_text_chunks(self, mmapped_file):
        while True:
            text_chunk = mmapped_file.read(self.chunk_size)
            if not text_chunk:
                break

            yield text_chunk.decode('utf-8')

    def __len__(self):
        return len(self.tokens) - self.seq_length + 1

    def __getitem__(self, idx):
        start_idx = idx
        end_idx = start_idx + self.seq_length + 1

        input_sequence = self.tokens[start_idx:end_idx - 1]
        target_sequence = self.tokens[start_idx + 1:end_idx]

        # Pad the sequences if they are shorter than the seq_length
        padding_token = 0
        input_sequence.extend([padding_token] * (self.seq_length - len(input_sequence)))
        target_sequence.extend([padding_token] * (self.seq_length - len(target_sequence)))

        input_sequence = torch.tensor(input_sequence, dtype=torch.long)
        target_sequence = torch.tensor(target_sequence, dtype=torch.long)

        return input_sequence, target_sequence

class FilterModel(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate=0.5):
        super(FilterModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(self.input_size, self.hidden_size).cuda()
        self.bn1 = nn.BatchNorm1d(self.hidden_size).cuda()  # Add batch normalization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)  # Add dropout
        self.fc2 = nn.Linear(self.hidden_size, 1).cuda()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Assuming x has shape [batch_size, seq_len, input_size]
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size * seq_len, self.input_size)  # Flatten the input: [batch_size * seq_len, input_size]
        x = self.fc1(x)
        x = self.bn1(x)  # Apply batch normalization
        x = self.relu(x)
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        x = self.sigmoid(x)  # Apply sigmoid activation to the output

        # Reshape the output to match the input shape: [batch_size, seq_len]
        x = x.view(batch_size, seq_len)
        return x

class PositionWiseFeedForward(nn.Module):
    def __init__(self, reservoir_size, ff_hidden_size):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(reservoir_size, ff_hidden_size)
        self.fc2 = nn.Linear(ff_hidden_size, reservoir_size)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class MultiHeadAttention(nn.Module):
    def __init__(self, reservoir_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.reservoir_size = reservoir_size
        self.head_size = reservoir_size // num_heads
        
        self.attention_key = nn.Linear(reservoir_size, reservoir_size)
        self.attention_query = nn.Linear(reservoir_size, reservoir_size)
        self.attention_value = nn.Linear(reservoir_size, reservoir_size)
        
        self.sqrt_head_size = torch.sqrt(torch.tensor(self.head_size, dtype=torch.float32))

    def forward(self, prev_state, r):
        batch_size = prev_state.size(0)
        
        key = self.attention_key(r).view(batch_size, self.num_heads, self.head_size).transpose(1, 2)
        query = self.attention_query(prev_state).view(batch_size, self.num_heads, self.head_size).transpose(1, 2)
        value = self.attention_value(r).view(batch_size, self.num_heads, self.head_size).transpose(1, 2)
        
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self.sqrt_head_size
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, value).transpose(1, 2).contiguous().view(batch_size, self.reservoir_size)
        
        return attention_output

class CustomReservoir(nn.Module):
    def __init__(self, input_size, reservoir_size, connectivity=0.2, spectral_radius=0.9, leak_rate=0.8, num_heads=1, num_attention_layers=0):
        super(CustomReservoir, self).__init__()
        self.reservoir_size = reservoir_size
        self.connectivity = connectivity
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate

        self.win = nn.Parameter(torch.randn(reservoir_size, input_size))
        nn.init.xavier_uniform_(self.win)
        ff_hidden_size = reservoir_size * 2
        self.wres = nn.Parameter(self.init_sparse_weights(reservoir_size, connectivity), requires_grad=False)
        self.multi_head_attentions = nn.ModuleList([MultiHeadAttention(reservoir_size, num_heads) for _ in range(num_attention_layers)])
        self.feed_forwards = nn.ModuleList([PositionWiseFeedForward(reservoir_size, ff_hidden_size) for _ in range(num_attention_layers)])

        self.layer_norms = nn.ModuleList([nn.LayerNorm(reservoir_size) for _ in range(num_attention_layers)])

        self.init_weights()

    def activation(self, x):
        return nn.GELU()(x)

    def forward(self, x, prev_state):
        u = torch.matmul(x, self.win.t())
        r = torch.matmul(prev_state, self.wres)

        attention_output = r
        for multi_head_attention, layer_norm in zip(self.multi_head_attentions, self.layer_norms):
            mha = multi_head_attention(prev_state, layer_norm(attention_output))
            attention_output = attention_output + mha

        next_state = (1 - self.leak_rate) * prev_state + self.leak_rate * self.activation(u + attention_output)
        return next_state

    def init_sparse_weights(self, size, connectivity):
        indices = torch.nonzero(torch.rand(size, size) < connectivity)
        values = torch.empty(indices.shape[0]).normal_(0, 1 / math.sqrt(size))
        sparse_tensor = torch.sparse.FloatTensor(indices.t(), values, (size, size))
        return sparse_tensor

    def init_weights(self):
        with torch.no_grad():
            dense_wres = self.wres.to_dense()
            eigenvalues, _ = torch.linalg.eig(dense_wres)
            max_eigenvalue = torch.max(torch.abs(eigenvalues))
            self.wres._values().mul_(self.spectral_radius / max_eigenvalue)

class ReadAttention(nn.Module):
    def __init__(self, reservoir_size, memory_size, memory_vector_length, num_read_heads):
        super(ReadAttention, self).__init__()

        self.memory_size = memory_size
        self.num_read_heads = num_read_heads

        self.key_layer = nn.Linear(reservoir_size, memory_vector_length * num_read_heads)
        self.beta_layer = nn.Linear(reservoir_size, num_read_heads)

        # Initialize the layers with xavier_uniform_ for better convergence
        nn.init.xavier_uniform_(self.key_layer.weight)
        nn.init.xavier_uniform_(self.beta_layer.weight)

    def forward(self, reservoir_state, memory):
        batch_size = reservoir_state.size(0)

        # Compute keys and betas for each read head
        keys = self.key_layer(reservoir_state).view(batch_size, self.num_read_heads, -1)
        betas = torch.relu(self.beta_layer(reservoir_state))

        return self.calculate_attention_weights(keys, betas, memory)

    def calculate_attention_weights(self, keys, betas, memory):
        mem = memory.transpose(1, 2).unsqueeze(0)
        dot_products = torch.matmul(keys.unsqueeze(2), mem).squeeze(2)

        dot_products = dot_products / torch.sqrt(torch.sum(keys ** 2, dim=-1, keepdim=True))

        attention_weights = torch.softmax(betas.unsqueeze(1) * dot_products, dim=-1)
        return attention_weights

class WriteAttention(nn.Module):
    def __init__(self, reservoir_size, memory_size, memory_vector_length):
        super(WriteAttention, self).__init__()

        self.reservoir_size = reservoir_size
        self.memory_size = memory_size
        self.memory_vector_length = memory_vector_length

        # Define the linear layers for key, erase, and add vectors
        self.key_layer = nn.Linear(reservoir_size, memory_vector_length)
        self.erase_layer = nn.Linear(reservoir_size, memory_vector_length)
        self.add_layer = nn.Linear(reservoir_size, memory_vector_length)

        # Define the linear layer for the interpolation gate
        self.interpolation_gate_layer = nn.Linear(reservoir_size, 1)

    def forward(self, reservoir_state, memory):
        # Compute the key vector
        key = self.key_layer(reservoir_state)

        # Compute the erase and add vectors
        erase_vector = F.sigmoid(self.erase_layer(reservoir_state))
        add_vector = F.tanh(self.add_layer(reservoir_state))

        # Compute the interpolation gate
        interpolation_gate = F.sigmoid(self.interpolation_gate_layer(reservoir_state))

        # Calculate the content-based addressing weights
        memory_norm = F.normalize(memory, dim=-1)
        key_norm = F.normalize(key, dim=-1)
        content_weights = torch.matmul(memory_norm, key_norm.unsqueeze(-1)).squeeze(-1)

        # Compute the write weights using the interpolation gate
        write_weights = interpolation_gate * content_weights

        return write_weights, erase_vector, add_vector


class ReservoirComputing(nn.Module):
    def __init__(self, input_size, reservoir_size, output_size, num_layers=1, connectivity=0.3, spectral_radius=0.94, leak_rate=0.85, readout_layers=0, llama_scale=(1.65, 0.155), num_heads=1, num_attention_layers=0):
        super(ReservoirComputing, self).__init__()

        self.num_layers = num_layers
        self.llama_scale = llama_scale
        self.reservoir_layers = nn.ModuleList([
            CustomReservoir(input_size if i == 0 else reservoir_size, reservoir_size, connectivity, spectral_radius, leak_rate, num_heads=num_heads, num_attention_layers=num_attention_layers)
            for i in range(num_layers)
        ])

        if readout_layers == 0:
            self.mid_layers = nn.Identity()
        else:
            self.mid_layers = nn.Sequential(*([
                nn.Linear(reservoir_size, reservoir_size),
                nn.ReLU()
                ]*readout_layers))
        self.readout = nn.Linear(reservoir_size, output_size)
        self.memory_size=64
        self.memory_vector_length=reservoir_size

        self.hidden = [None] * num_layers

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        hidden_seq = []

        output = self.get_latent_space(x)
        output = self.mid_layers(output)
        output = self.readout(output)
        return output

    def get_latent_space(self, x):
        batch_size, seq_length, _ = x.size()
        hidden_seq = []

        if use_llama:
            if self.llama_scale[0] == -1:
                x = x / torch.norm(x, dim=1, keepdim=True)
            else:
                x = x / self.llama_scale[0] - self.llama_scale[1]

        x = x / torch.norm(x, dim=1, keepdim=True)

        for t in range(seq_length):
            layer_input = x[:, t, :]
            for i, reservoir in enumerate(self.reservoir_layers):
                if self.hidden[i] is None:
                    self.hidden[i] = torch.zeros(batch_size, reservoir.reservoir_size, device=x.device)
                    self.memory = torch.randn(batch_size, self.memory_size, self.memory_vector_length, device=x.device) * 0.01
                self.hidden[i] = reservoir(layer_input, self.hidden[i])
                layer_input = self.hidden[i]


            hidden_seq.append(self.hidden[-1])

        latent_space = torch.stack(hidden_seq, dim=1)
        return latent_space

    def get_output_from_latent_space(self, latent_space):
        output = self.readout(latent_space)
        return output

    def reset(self):
        self.hidden = [None] * self.num_layers

def evaluate_model(model, gpt2, tokenizer, data_loader, seq_length):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for input_batch, target_batch in data_loader:
            input_batch = input_batch.cuda()
            target_batch = target_batch.cuda()

            x, _ = extract_features(input_batch, gpt2)
            output = model(x)

            target = torch.clamp(target_batch, max=vocab_size - 1)
            loss = criterion(output, target[:, -1])

            total_loss += loss.item()
            total_correct += (output.argmax(dim=1) == target[:, -1]).sum().item()
            total_samples += input_batch.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_samples

    return avg_loss, accuracy

def sample_model(model, input_text, tokenizer, seq_length, num_tokens=50, top_k=100, top_p=1.0, temperature=1.0, svd_bias=True, mediator=None, guide_text="happy", guide_weight=0.005, alpha=0.5, stopchar=None):
    model.eval()
    generated_tokens = []
    current_input = tokenizer.encode(input_text, add_special_tokens=False)[-seq_length:]
    current_input = torch.tensor(current_input).unsqueeze(0).cuda()
    if svd_bias:
        # Prepare guide_text
        guide_text_emb = tokenizer.encode(guide_text, add_special_tokens=False)[-seq_length:]
        guide_text_emb = torch.tensor(guide_text_emb, dtype=torch.int).unsqueeze(0).cuda()
        guide_latent_space = mediator.get_latent_space(extract_features(guide_text_emb, gpt2)[0])

        # Calculate the average latent space across the sequence dimension for the guide_text
        guide_latent_space_avg = torch.mean(guide_latent_space, dim=1)
        guide_u, guide_s, guide_v = torch.svd(guide_latent_space_avg)

    with torch.no_grad():
        for _ in range(num_tokens):
            features, _ = extract_features(current_input, gpt2)

            output = model(features)[:, -1]
            next_token_logits = output[-1].unsqueeze(0)

            # Apply temperature
            next_token_logits = next_token_logits / temperature

            # Apply top-k sampling
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            next_token_logits[indices_to_remove] = float('-inf')

            # Apply top-p sampling
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p

            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = float('-inf')
            # SVD-based sampling
            if svd_bias:
                current_latent_space = mediator.get_latent_space(features)
                current_latent_space_avg = torch.mean(current_latent_space, dim=1)

                # Latent space interpolation
                interpolated_latent_space_avg = alpha * current_latent_space_avg + (1 - alpha) * guide_latent_space_avg
                projected_latent_space = torch.matmul(interpolated_latent_space_avg.t(), guide_v.t())

                guided_logits = mediator.get_output_from_latent_space(projected_latent_space.unsqueeze(1))
                next_token_logits = (1 - guide_weight) * next_token_logits + guide_weight * guided_logits
                next_token_logits = next_token_logits[-1]

            next_token_probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(next_token_probs, num_samples=1)

            next_token_item = next_token.item()
            # Check if the generated token is ">"
            if stopchar and tokenizer.decode([next_token_item]) == stopchar:
                break
            generated_tokens.append(next_token_item)

            # Reshape next_token to match the dimensions of current_input and concatenate
            next_token = next_token.view(1, 1)
            current_input = torch.cat([current_input[:, 1:], next_token], dim=1)

    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(generated_text)
    return generated_text

def embed_tokens(token_ids, gpt2):
    if use_llama:
        return gpt2.base_model.embed_tokens(token_ids).to(torch.float32)
    if use_cerebras:
        return gpt2.base_model.wte(token_ids).to(torch.float32)
    return gpt2.base_model.embed_in(token_ids)

def gumbel_softmax(logits, temperature=1.0):
    noise = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(noise + 1e-20) + 1e-20)
    return F.softmax((logits + gumbel_noise) / temperature, dim=-1)

def train_generator(generator, mediator, x, target, gen_optimizer, teacher_forcing_ratio=0.5):
    gen_optimizer.zero_grad()
    generator.reset()
    mediator.reset()
    output = generator(x)
    token_probs = F.softmax(output, dim=-1)
    token_probs_gumbel = gumbel_softmax(output)
    token_ids_gumbel = torch.argmax(token_probs_gumbel, dim=-1)
    token_ids = torch.argmax(token_probs_gumbel, dim=-1)

    teacher_force = torch.rand(1).item() < teacher_forcing_ratio
    if teacher_force:
        emb_out = embed_tokens(target, gpt2)
    else:
        emb_out = embed_tokens(token_ids, gpt2)

    output_score = mediator(emb_out)

    gen_loss_fn = nn.CrossEntropyLoss(reduction='mean')
    gen_loss = gen_loss_fn(output_score.view(-1, output_score.size(-1)), target.view(-1))
    gen_loss.backward()
    gen_optimizer.step()
    return gen_loss

def train_mediator(mediator, generator, target, x, med_optimizer):
    med_optimizer.zero_grad()

    mediator.reset()
    # Get the generator's output
    gen_output = generator(x)
    token_probs = F.softmax(gen_output, dim=-1)
    token_ids = torch.argmax(token_probs, dim=-1)

    # Embed the generator's output tokens
    emb_out = embed_tokens(token_ids, gpt2).detach()
    output_score = mediator(emb_out)
    
    med_loss_fn = nn.CrossEntropyLoss(reduction='mean')
    med_loss = med_loss_fn(output_score.view(-1, output_score.size(-1)), target.view(-1))
    med_loss.backward()
    med_optimizer.step()
    return med_loss

def replace_random_token(tensor):
    batch_size, seq_len, vocab_size = tensor.size()
    
    # Choose random indices for batch and sequence length
    rand_batch_idx = torch.randint(0, batch_size, (1,)).item()
    rand_seq_idx = torch.randint(0, seq_len, (1,)).item()
    
    # Get the current token index in the selected position
    current_token_idx = torch.argmax(tensor[rand_batch_idx, rand_seq_idx]).item()

    # Generate a new random token index that is different from the current one
    new_token_idx = current_token_idx
    while new_token_idx == current_token_idx:
        new_token_idx = torch.randint(0, vocab_size, (1,)).item()

    # Replace the selected token with the new random token
    tensor[rand_batch_idx, rand_seq_idx] = 0
    tensor[rand_batch_idx, rand_seq_idx, new_token_idx] = 1

    return tensor


def train_forward_forward(model, x_pos, x_neg, target):
    g_pos = model.forward(x_pos)
    vocab_len = g_pos.size(-1)
    if x_neg is None:
        g_neg = torch.randint(0, vocab_len - 1, target.size(), device=x_pos.device)  # [batch_size, seq_len]
        g_neg = F.one_hot(g_neg, num_classes=vocab_len).float()
        #g_neg = model.forward(rand_pos)
    else:
        g_neg = model.forward(x_neg)
    indices = torch.randint(x_pos.shape[1], size=(target.shape[0], 1, 1))
    alpha = torch.rand([1], device=x_pos.device)

    top_k = 50
    prob_threshold = 0.05
    entropy_threshold = 1.0
    #filter_mask = minimal_negative_filtering(g_neg, top_k, prob_threshold, entropy_threshold, target)

    # Get the probability of the correct next token for positive samples
    g_pos_correct = torch.gather(g_pos, -1, target.unsqueeze(-1)).squeeze(-1)  # [batch_size, seq_len]
    g_neg_incorrect = torch.gather(g_neg, -1, target.unsqueeze(-1)).squeeze(-1)  # [batch_size, seq_len]

    nll_pos = g_pos_correct
    nll_neg = g_neg_incorrect

    # The following loss pushes pos (neg) samples to
    # values larger (smaller) than the self.threshold.
    threshold = 0
    loss = torch.log(1 + torch.exp(torch.cat([
        -nll_pos + threshold,
        nll_neg - threshold]))).mean()
    return loss

def minimal_negative_filtering(probs, top_k, prob_threshold, entropy_threshold, targets):
    batch_size, seq_len, vocab_len = probs.size()

    # Sort probabilities and get indices
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)

    # Get the actual next token's probability
    target_probs = torch.gather(probs, -1, targets.unsqueeze(-1)).squeeze(-1)

    # Calculate entropy
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

    # 1. Candidate is not among the top K most likely predictions
    top_k_mask = sorted_indices[..., :top_k].unsqueeze(-1) != torch.arange(vocab_len, device=probs.device)
    top_k_mask = top_k_mask.all(dim=-2)

    # 2. Candidate's predicted probability is below a certain threshold
    prob_threshold_mask = probs < prob_threshold

    # 3. Candidate's prediction entropy is above a certain threshold
    entropy_threshold_mask = (entropy > entropy_threshold).unsqueeze(-1).expand_as(probs)

    # 4. Candidate is not the actual next token in the sequence
    target_mask = torch.eye(vocab_len, device=probs.device)[targets].bool()

    # Combine all criteria
    filter_mask = (top_k_mask & prob_threshold_mask & entropy_threshold_mask & target_mask).float()

    # Normalize the mask
    filter_mask /= filter_mask.sum(dim=-1, keepdim=True)

    return filter_mask


def train_model(model, gpt2, tokenizer, neg_data_loader, train_data_loader, val_data_loader, epochs, seq_length, lr=6e-3, weight_decay=1e-2, early_stop_patience=4, med = None, save_every_epoch=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.readout.parameters(), lr=lr, betas=(0.9, 0.99))
    #optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9,0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    if use_cot:
        med_optimizer = torch.optim.AdamW(med.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9,0.999))
        med_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(med_optimizer, T_max=epochs, eta_min=0)
        #med_optimizer = torch.optim.Adam(med.readout.parameters(), lr=lr)

    best_val_loss = float('inf')
    patience_counter = 0
    val_losses = []
    learning_rates = []
    training_losses = []
    val_perplexities = []
    val_accuracies = []

    total_iterations = len(train_data_loader) * epochs

    if neg_data_loader:
        neg_data = iter(neg_data_loader)
    for epoch in tqdm(range(epochs), desc="Epochs", ncols=100):
        # Training loop
        model.train()
        train_loss = 0

        for batch_idx, (input_batch, target_batch) in enumerate(tqdm(train_data_loader, desc=f"Epoch {epoch + 1}/{epochs}", ncols=100, leave=False)):
            model.reset()
            input_batch = input_batch.cuda()
            target_batch = target_batch.cuda()

            # Extract features from input_batch using GPT-2
            x, llm_target = extract_features(input_batch, gpt2)
            target = torch.clamp(target_batch, max=vocab_size-1)
            if use_forward_forward:
                #temperature = 0.8  # Adjust this value for desired variance (higher = more variance, lower = less variance)
                #logits = llm_target / temperature  # Divide logits by the temperature
                #probabilities = torch.softmax(logits, dim=-1)  # Convert logits to probabilities

                ## Reshape the probability tensor to be 2D (n, vocab_size), where n = batch_size * seq_length
                #probabilities_2d = probabilities.view(-1, probabilities.size(-1))

                ## Sample tokens from the probability distribution for each position in each sequence
                #tokens_2d = torch.multinomial(probabilities_2d, 1)

                ## Reshape the token tensor back to the original shape (batch_size, seq_length)
                #llm_gen = tokens_2d.view(probabilities.size(0), probabilities.size(1))
                #
                #x_neg, _ = extract_features(llm_gen, gpt2)
                if neg_data_loader:
                    try:
                        neg_batch = next(neg_data)[0]
                    except StopIteration:
                        neg_data = iter(data_loader)
                        neg_batch = next(neg_data)[0]
                    x_neg, _ = extract_features(neg_batch, gpt2)
                else:
                    x_neg = None

                loss = train_forward_forward(model, x, x_neg, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            else:

                if use_cot:
                    generator_loss = train_generator(model, med, x, target, optimizer)
                    # Train mediator
                    mediator_loss = train_mediator(med, model, target, x, med_optimizer)

                optimizer.zero_grad()
                # Train the reservoir computing model
                output = model(x)
                loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            if use_cot:
                tqdm.write(f'Step [{batch_idx + 1}/{len(train_data_loader)}], Loss: {loss.item():.4f}, GLoss {generator_loss.item():.4f}, MDloss {mediator_loss.item():.4f}', end='\r')
            else:
                tqdm.write(f'Step [{batch_idx + 1}/{len(train_data_loader)}], Loss: {loss.item():.4f}', end='\r')

            #if (batch_idx + 1) % 500 == 0:
                #model.reset()
                #print("Begin sample:")
                #sample_model(model, prompt, tokenizer, seq_length,mediator=med,svd_bias=None)
                #print("Saving")
                #torch.save(model.state_dict(), model_path)
                #if use_cot:
                #    torch.save(med.state_dict(), med_model_path)

        #else:
        scheduler.step()
        if use_cot:
            med_scheduler.step()
        training_losses.append(train_loss/len(train_data_loader))
        # Validation loop
        model.eval()
        val_loss = 0
        val_accuracy = 0
        val_perplexity = 0
        with torch.no_grad():
            for batch_idx, (input_batch, target_batch) in enumerate(tqdm(val_data_loader, desc=f"Epoch {epoch + 1}/{epochs}", ncols=100, leave=False)):
                input_batch = input_batch.cuda()
                target_batch = target_batch.cuda()
                model.reset()

                x, _ = extract_features(input_batch, gpt2)
                output = model(x)

                target = torch.clamp(target_batch, max=vocab_size-1)
                loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
                val_loss += loss.item()

                accuracy = calculate_accuracy(output.view(-1, output.size(-1)), target.view(-1))
                val_accuracy += accuracy

                perplexity = calculate_perplexity(output.view(-1, output.size(-1)), target.view(-1))
                val_perplexity += perplexity
                if (batch_idx + 1) % 10 == 0:
                    tqdm.write(f'Step [{batch_idx + 1}/{len(val_data_loader)}], Loss: {loss.item():.4f}', end='\r')

        val_loss /= len(val_data_loader)
        val_losses.append(val_loss)
        val_accuracy /= len(val_data_loader)
        val_accuracies.append(val_accuracy)
        val_perplexity /= len(val_data_loader)
        val_perplexities.append(val_perplexity)
        print(f'Epoch [{epoch + 1}/{epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation Perplexity: {val_perplexity:.4f}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if save_every_epoch:
            print("Saving after epoch...")
            torch.save(model.state_dict(), model_path)
            if use_cot:
                torch.save(med.state_dict(), med_model_path)
            print("Saved")
        if patience_counter >= early_stop_patience:
            print(f"Early stopping triggered. Stopping training at epoch {epoch + 1}.")
            epochs = epoch + 1
            break
        learning_rates.append(optimizer.param_groups[0]['lr'])
    plt.figure()
    epochs = range(1, epochs+1)
    plt.plot(epochs, training_losses, 'g', label='Training loss')
    plt.plot(epochs, val_losses, 'b', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("training_and_validation_loss.png")

    plt.figure()
    plt.plot(val_accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.savefig('accuracy.png')

    plt.figure()
    plt.plot(val_perplexities)
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Perplexity')
    plt.savefig('perplexity.png')


    plt.figure()
    plt.plot(learning_rates)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate')
    plt.savefig('learning_rates.png')
    return val_accuracy

def extract_features(x, model):
    """
    Extract features from the GPT-2 model.

    Args:
        x (torch.Tensor): Input tensor of shape (num_sequences, seq_length).
        model (nn.Module): The GPT-2 model to extract features from.

    Returns:
        last_hidden_state (torch.Tensor): Tensor containing the extracted features.
    """
    with torch.no_grad():
        features = model(x.cuda(), output_hidden_states=True)
        last_hidden_state = features.hidden_states[-1]

    return last_hidden_state.to(torch.float32), features.logits

def load_models(args):

    model = ReservoirComputing(input_size, reservoir_size, vocab_size, num_attention_layers=args.attn, num_layers=args.layers, spectral_radius=args.spectralradius, leak_rate=args.leakrate, connectivity=args.connectivity, num_heads=args.attnheads).cuda()
    med = ReservoirComputing(input_size, reservoir_size, vocab_size, num_attention_layers=args.attn, num_layers=args.layers, spectral_radius=args.spectralradius, leak_rate=args.leakrate, connectivity=args.connectivity, num_heads=args.attnheads).cuda()
    num_filters = 128
    loaded = False
    if os.path.exists(model_path):
        print("Model exists, loading")
        model.load_state_dict(torch.load(model_path))
        loaded = True
        if use_cot and os.path.exists(med_model_path):
            print("Mediator model exists, loading")
            med.load_state_dict(torch.load(med_model_path))
    else:
        print("No model exists, new model created.")
    return model, med, loaded


if __name__ == "__main__":
    dataset = LargeTextDataset(file_path, tokenizer, seq_length)
    test_dataset = LargeTextDataset(validation_path, tokenizer, seq_length)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True, shuffle=True, pin_memory=True, prefetch_factor=4)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, drop_last=True, shuffle=True, pin_memory=True)
    if neg_file_path:
        neg_dataset = LargeTextDataset(neg_file_path, tokenizer, seq_length)
        neg_data_loader = DataLoader(neg_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True, shuffle=True, pin_memory=True, prefetch_factor=4)
    else:
        neg_data_loader = None
    prompt = """emoji pizza="""

    model, med, loaded = load_models(args)
    model.reset()
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        #model = nn.DataParallel(model)
        #med = nn.DataParallel(med)
        model.to(torch.device("cuda"))
        med.to(torch.device("cuda"))

    if not loaded or args.train:
        size = torch.cuda.device_count()
        print("Training")
        train_model(model, gpt2, tokenizer, neg_data_loader, data_loader, test_data_loader, epochs, seq_length, med=med, save_every_epoch=args.save, lr=args.learnrate)
        torch.save(model.state_dict(), model_path)
        if use_cot:
            torch.save(med.state_dict(), med_model_path)
    for i in range(10):
        print(prompt, end='')
        model.reset()
        sample_model(model, prompt, tokenizer, seq_length,mediator=med, num_tokens=100)

