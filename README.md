 <div align="center">

# 🤖 Transformer-Based Next-Token Prediction Using Self-Attention

### A Mini Language Model Built with PyTorch

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Transformer](https://img.shields.io/badge/Architecture-Transformer-8A2BE2?style=for-the-badge)
![NLP](https://img.shields.io/badge/Domain-NLP-00A98F?style=for-the-badge)

**Building a language model that learns relationships between words, predicts the next token, and generates text using a Transformer architecture with causal self-attention.**

</div>

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Objectives](#-objectives)
3. [What Is Next-Token Prediction?](#-what-is-next-token-prediction)
4. [Dataset Description](#-dataset-description)
5. [Transformer Architecture](#-transformer-architecture)
6. [Self-Attention Mechanism](#-self-attention-mechanism)
7. [Technology Stack](#️-technology-stack)
8. [Installation and Setup](#-installation-and-setup)
9. [Project Workflow](#-project-workflow)
10. [Model Training](#-model-training)
11. [Next-Token Prediction](#-next-token-prediction)
12. [Text Generation](#-text-generation)
13. [Model Evaluation](#-model-evaluation)
14. [Training Visualization](#-training-visualization)
15. [Saving the Model](#-saving-the-model)
16. [Project Structure](#-project-structure)
17. [Limitations](#️-limitations)
18. [Future Improvements](#-future-improvements)
19. [Contributing](#-contributing)
20. [License](#-license)

---

## ✨ Project Overview

Next-token prediction is a fundamental task in Natural Language Processing (NLP). It involves predicting the next word or token based on the sequence of tokens that came before it.

This project implements a small Transformer-based language model using **PyTorch and a basic self-attention mechanism**. The model learns from a small, custom text corpus and uses the preceding words to predict the next token in a sequence.

Unlike traditional approaches that process text using fixed rules, the model learns token relationships through trainable embeddings, positional embeddings, causal self-attention, and feed-forward neural networks.

The project demonstrates the fundamental concepts behind Transformer language models, including tokenization, vocabulary construction, causal masking, training, next-token prediction, text generation, and evaluation.

### 🎯 Project Goal

To develop a simple Transformer language model that learns patterns in text and predicts subsequent tokens using basic self-attention.

### 🌟 Key Features

- Custom text corpus for language-model training.
- Word-level tokenization and vocabulary creation.
- Integer encoding of tokens.
- Training and validation data splitting.
- Token embeddings and positional embeddings.
- Causal self-attention with a lower-triangular mask.
- Transformer blocks with layer normalization and feed-forward networks.
- AdamW optimizer and cross-entropy loss.
- Gradient clipping and dropout regularization.
- Next-token prediction with top-k alternatives.
- Autoregressive text generation.
- Accuracy, loss, and perplexity evaluation.
- Training and validation loss visualization.
- Model checkpoint saving and downloading.

## 🎯 Objectives

The primary objectives of this project are:

1. Understand the fundamentals of Transformer-based language models.
2. Learn how text can be represented as numerical token IDs.
3. Build a vocabulary from a custom text corpus.
4. Prepare input sequences and their next-token targets.
5. Understand token embeddings and positional embeddings.
6. Implement the query, key, and value components of self-attention.
7. Apply causal masking to prevent access to future tokens.
8. Construct Transformer blocks using PyTorch.
9. Train a language model using cross-entropy loss.
10. Predict the most likely next token for a given prompt.
11. Generate text autoregressively.
12. Evaluate accuracy, loss, and perplexity.
13. Visualize training and validation loss.
14. Save the model parameters and vocabulary for future use.

## 🧠 What Is Next-Token Prediction?

Next-token prediction is a language-modeling task in which a model estimates the next token from the preceding context.

For example:

**Input:**

`the sun rises in`

**Possible prediction:**

`the east`

The model processes the input tokens and produces a probability distribution over the vocabulary. The token with the highest probability can be selected as the most likely prediction.

The actual prediction depends on the patterns learned from the training data.

### How Does the Model Learn?

The model is trained using input sequences and their corresponding next-token targets.

For example:

| Input Sequence | Target Sequence |
|---|---|
| the sun rises | sun rises in |
| sun rises in | rises in the |
| rises in the | in the east |

During training, the model predicts the next token at every position in a sequence. Its predictions are compared with the expected target tokens using cross-entropy loss.

The model then updates its parameters to reduce the prediction error.

## 📊 Dataset Description

This project uses a small, manually defined text corpus containing sentences about topics such as:

- The sun, moon, stars, and nature.
- Machine learning and deep learning.
- Artificial intelligence.
- Neural networks and Transformers.
- Programming and computers.
- Language models and self-attention.

The corpus is defined directly in the notebook, so an external dataset download is not required.

### Dataset Preparation

The text is converted to lowercase and split into words.

```python
tokens = text.lower().split()

vocab = sorted(set(tokens))

stoi = {
    word: i
    for i, word in enumerate(vocab)
}

itos = {
    i: word
    for word, i in stoi.items()
}
```

Here:

- `tokens` contains the word-level token sequence.
- `vocab` contains the unique words.
- `stoi` maps each word to its integer ID.
- `itos` maps each integer ID back to its word.

The complete token sequence is then converted into a PyTorch tensor.

### Dataset Configuration

| Parameter | Value |
|---|---|
| Dataset type | Custom text corpus |
| Tokenization | Word-level |
| Text preprocessing | Lowercase and whitespace splitting |
| Training split | 85% |
| Validation split | 15% |
| Context length | 8 tokens |
| Batch size | 16 |
| Vocabulary | Unique words in the corpus |
| External dataset | Not required |

The corpus is intentionally small to make the implementation easier to understand. It is suitable for learning Transformer fundamentals rather than producing general-purpose language-model behavior.

## 🏗️ Transformer Architecture

The project implements a custom Transformer language model using PyTorch.

### Architecture Overview

```text
              Input Text
                  |
                  ▼
          Word Tokenization
                  |
                  ▼
           Token IDs
                  |
                  ▼
          Token Embeddings
                  +
         Positional Embeddings
                  |
                  ▼
          Transformer Block 1
          ┌───────────────────┐
          │ Layer Normalization│
          │ Causal Self-Attention
          │ Residual Connection│
          │ Layer Normalization│
          │ Feed-Forward Network│
          │ Residual Connection│
          └───────────────────┘
                  |
                  ▼
          Transformer Block 2
          ┌───────────────────┐
          │ Layer Normalization│
          │ Causal Self-Attention
          │ Residual Connection│
          │ Layer Normalization│
          │ Feed-Forward Network│
          │ Residual Connection│
          └───────────────────┘
                  |
                  ▼
             Final Norm
                  |
                  ▼
          Linear Output Layer
                  |
                  ▼
       Vocabulary-Sized Logits
                  |
                  ▼
          Softmax Probabilities
                  |
                  ▼
        Next-Token Prediction
```

### Model Configuration

| Parameter | Value |
|---|---|
| Embedding dimension | 64 |
| Context length | 8 |
| Number of Transformer blocks | 2 |
| Attention heads | 1 |
| Feed-forward hidden dimension | 256 |
| Dropout | 0.1 |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.01 |
| Training epochs | 300 |
| Evaluation interval | 25 epochs |

The implementation uses a single self-attention operation rather than multi-head attention. It is therefore a simplified, educational Transformer architecture.

### 1. Token Embedding

Each token ID is mapped to a trainable vector of dimension 64.

These embeddings allow the model to represent words numerically and learn useful relationships between them.

### 2. Positional Embedding

Token embeddings alone do not explicitly represent the position of a token in a sequence.

The model adds learned positional embeddings to token embeddings so it can distinguish different positions within the context window.

### 3. Transformer Blocks

The model contains two Transformer blocks. Each block includes:

- Layer normalization.
- Causal self-attention.
- Residual connections.
- A feed-forward network.
- GELU activation.
- Dropout.

These components help the model learn relationships between tokens and transform the resulting representations.

### 4. Output Layer

The final linear layer maps the hidden representation to a vector whose length equals the vocabulary size.

Each output value is a logit associated with a possible next token. Applying softmax converts the logits into probabilities.

## 🔍 Self-Attention Mechanism

Self-attention allows a model to calculate how strongly each token should relate to other tokens in its context.

The implementation creates three projections:

- **Query (Q):** Represents what a token is looking for.
- **Key (K):** Represents information used to match against a query.
- **Value (V):** Contains the information combined using attention weights.

### Attention Calculation

The model computes attention scores using:

\[
S = \frac{QK^T}{\sqrt{d}}
\]

where:

- \(Q\) is the query matrix.
- \(K\) is the key matrix.
- \(d\) is the embedding dimension used for scaling.

The attention weights are calculated using softmax after masking future positions.

\[
A = \operatorname{softmax}(S + M)
\]

The final attention output is:

\[
O = AV
\]

Here, \(M\) represents the causal mask, which prevents tokens from attending to future positions.

### Why Is Causal Masking Important?

During next-token prediction, the model should use only the current token and preceding context when making each prediction.

The lower-triangular mask blocks access to future tokens. This prevents information leakage from later positions into earlier predictions.

### Feed-Forward Network

Each Transformer block contains a feed-forward network with:

- An expansion from 64 to 256 dimensions.
- GELU activation.
- A projection back to 64 dimensions.
- Dropout.

This network transforms the representations produced by the attention layer.

## ⚙️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Neural-network implementation and training |
| NumPy | Numerical operations |
| Matplotlib | Training-curve visualization |
| Google Colab | Notebook execution and file downloads |
| CUDA | Optional GPU acceleration when available |

The notebook automatically selects a CUDA GPU when available; otherwise, it runs on the CPU.

## 🚀 Installation and Setup

### Option 1: Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `Transformer_next_token_prediction.ipynb`.
3. Select **Runtime → Run all**.
4. Wait for the model to train.
5. Review the evaluation metrics and generated text.
6. Download the model checkpoint and loss graph from the final cell.

The notebook uses PyTorch, NumPy, and Matplotlib. If these packages are not available in the runtime, install them before executing the notebook.

### Option 2: Run Locally

Install Python and the required libraries:

```bash
python -m pip install torch numpy matplotlib jupyter
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open the notebook and execute the cells in order.

**Note:** The final download cell uses `google.colab.files`. When running locally, replace that cell with ordinary file access or copy the generated files from the working directory.

## 🔬 Project Workflow

### Step 1: Import Libraries

Import PyTorch, NumPy, Matplotlib, and the required neural-network utilities.

Set random seeds for Python, NumPy, and PyTorch to improve reproducibility.

The notebook automatically selects the available computing device.

### Step 2: Prepare the Text Corpus

Define the sample sentences and convert the text to lowercase.

Split the text into individual words and create a vocabulary containing unique tokens.

### Step 3: Encode Tokens

Map each word to an integer ID and convert the complete token sequence into a PyTorch tensor.

This representation allows the neural network to process text numerically.

### Step 4: Split Training and Validation Data

The notebook splits the encoded sequence chronologically:

- 85% for training.
- 15% for validation.

This preserves the order of the original token sequence.

### Step 5: Create Training Batches

The model uses a context length of eight tokens.

For each randomly selected starting position, it creates an input sequence and a target sequence shifted forward by one token.

```python
x = data[i:i + BLOCK_SIZE]

y = data[i + 1:i + BLOCK_SIZE + 1]
```

This shift creates the next-token prediction task.

### Step 6: Build the Transformer

Construct token embeddings, positional embeddings, two Transformer blocks, a final layer normalization, and the vocabulary output layer.

### Step 7: Configure Training

Use the AdamW optimizer with a learning rate of 0.001 and weight decay of 0.01.

Cross-entropy loss measures the difference between predicted token distributions and expected next-token targets.

### Step 8: Train the Model

Train for up to 300 epochs. The notebook estimates training and validation loss every 25 epochs.

Gradient clipping is used to limit the gradient norm to 1.0, helping control excessively large gradient updates.

### Step 9: Evaluate Performance

Calculate next-token accuracy, average cross-entropy loss, and perplexity on randomly sampled validation batches.

### Step 10: Predict and Generate Text

Enter a prompt to obtain the top next-token predictions or generate a continuation by repeatedly sampling from the model's predicted probability distribution.

### Step 11: Save the Model

Save the model's state dictionary, vocabulary mappings, and essential architecture configuration in a checkpoint file.

## 📈 Model Training

The training loop follows these steps:

1. Sample a batch of input sequences and next-token targets.
2. Pass the inputs through the Transformer.
3. Calculate cross-entropy loss.
4. Reset the previous gradients.
5. Backpropagate the loss.
6. Clip the gradient norm.
7. Update model parameters using AdamW.
8. Estimate training and validation loss at regular intervals.

The model uses the following training settings:

| Setting | Value |
|---|---|
| Epochs | 300 |
| Batch size | 16 |
| Context length | 8 |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.01 |
| Loss function | Cross-entropy |
| Gradient clipping | Maximum norm of 1.0 |
| Evaluation interval | Every 25 epochs |

The notebook does not implement early stopping. Training continues for the configured number of epochs.

## 🔮 Next-Token Prediction

The `predict_next_token()` function predicts likely continuations for a prompt and displays the top-k token candidates.

Example:

```python
predict_next_token("the sun rises in", top_k=5)
```

The function:

1. Converts the prompt to lowercase.
2. Splits it into word tokens.
3. Maps known words to token IDs.
4. Replaces unknown words with the fallback token ID for `"the"`.
5. Keeps only the most recent eight tokens.
6. Runs the model on the prompt.
7. Converts the final-position logits into probabilities.
8. Displays the top five candidate tokens.

### Understanding Top-k Predictions

Instead of displaying only one possible next token, top-k prediction shows several candidates with their estimated probabilities.

This helps users inspect the model's uncertainty and compare alternative continuations.

**Important:** The probabilities reflect the model's learned patterns, not guaranteed linguistic correctness.

## ✍️ Text Generation

The project also implements autoregressive text generation.

Example:

```python
generate_text(
    "the transformer",
    max_new_tokens=15,
    temperature=0.8
)
```

The model repeatedly predicts a new token and appends it to the existing sequence.

The process continues until the requested number of new tokens has been generated.

### Temperature

Temperature controls the randomness of the sampling process.

- **Lower temperature:** Makes the probability distribution more concentrated, generally favoring higher-probability tokens.
- **Higher temperature:** Makes the distribution flatter, allowing more varied selections.

The model uses a temperature of 0.8 by default.

Because the training corpus is small, generated text may be repetitive, grammatically incorrect, or unrelated to the prompt.

## 📊 Model Evaluation

The project evaluates the model using three principal metrics.

### 1. Next-Token Accuracy

Accuracy measures the percentage of predicted tokens that match the expected target tokens.

\[
\text{Accuracy} =
\frac{\text{Correct Predictions}}
{\text{Total Predictions}}
\times 100
\]

The notebook uses `argmax` to select the most likely token at each position and compares it with the corresponding target token.

### 2. Cross-Entropy Loss

Cross-entropy loss measures the error between the model's predicted probability distribution and the actual next-token labels.

A lower loss generally indicates that the model assigns greater probability to the correct targets.

### 3. Perplexity

Perplexity is calculated by exponentiating the average cross-entropy loss:

\[
\text{Perplexity} = e^{\text{Average Loss}}
\]

Lower perplexity generally indicates better next-token prediction on the evaluated data.

Perplexity must be interpreted in the context of the dataset, vocabulary, and evaluation method.

### Example of Previously Recorded Results

An earlier execution of this project reported the following results:

| Metric | Recorded value |
|---|---:|
| Correct predictions | 1,246 |
| Total predictions | 12,800 |
| Next-token accuracy | 9.73% |
| Average loss | 6.3573 |
| Perplexity | 576.6939 |

These values are execution-specific, not guaranteed results for every run. Re-run the notebook to obtain the metrics produced by your current environment.

### Interpreting the Results

The small corpus and vocabulary-learning setup limit the model's ability to generalize to unseen language.

A low next-token accuracy can result from the small amount of training text, the limited number of training updates, the model configuration, and the difficulty of predicting the exact next word.

Increasing the size and diversity of the corpus and improving the training process are useful next steps.

## 📉 Training Visualization

The notebook plots training loss and validation loss against the recorded epochs.

The graph is saved as:

`training_loss.png`

### Why Plot the Loss?

- To observe whether training loss decreases.
- To compare training and validation behavior.
- To identify potential overfitting.
- To understand whether the model continues to improve during training.

If training loss decreases while validation loss increases, the model may be overfitting.

Because validation loss is estimated from randomly sampled batches, small fluctuations can occur between evaluation points.

## 💾 Saving the Model

The notebook saves the checkpoint using:

```python
torch.save(
    checkpoint,
    "/content/transformer_next_token_model.pth"
)
```

The checkpoint contains:

- `model_state_dict`: Learned model parameters.
- `vocab`: Vocabulary list.
- `stoi`: Word-to-ID mapping.
- `itos`: ID-to-word mapping.
- `vocab_size`: Vocabulary size.
- `block_size`: Context length.
- `n_embd`: Embedding dimension.
- `n_layers`: Number of Transformer blocks.

This information is required to reconstruct the model and interpret its token IDs.

### Downloading the Files

The final notebook cell downloads:

- `transformer_next_token_model.pth`
- `training_loss.png`

These files can be retained for future use or included as project artifacts when appropriate.

### Loading the Checkpoint

To reload the model, recreate the `NextTokenTransformer` class in your Python environment and load the checkpoint:

```python
import torch

checkpoint = torch.load(
    "transformer_next_token_model.pth",
    map_location=device,
    weights_only=True
)

model = NextTokenTransformer(
    vocab_size=checkpoint["vocab_size"],
    n_embd=checkpoint["n_embd"],
    block_size=checkpoint["block_size"],
    n_layers=checkpoint["n_layers"],
    dropout=0.1
).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

vocab = checkpoint["vocab"]
stoi = checkpoint["stoi"]
itos = checkpoint["itos"]

model.eval()
```

Use a compatible PyTorch version and the same model implementation. The checkpoint contains the learned weights and vocabulary mappings, not the Python class definition itself.

## 📁 Project Structure

A recommended GitHub repository structure is:

```text
Transformer-Next-Token-Prediction/
│
├── Transformer_next_token_prediction.ipynb
│
├── README.md
│
├── transformer_next_token_model.pth
│
└── training_loss.png
```

### File Descriptions

- `Transformer_next_token_prediction.ipynb`: Complete notebook containing corpus preparation, Transformer implementation, training, evaluation, prediction, and text generation.
- `README.md`: Project documentation, architecture explanation, setup guide, and evaluation details.
- `transformer_next_token_model.pth`: Saved model checkpoint generated after training.
- `training_loss.png`: Training and validation loss graph generated by the notebook.

The checkpoint and graph are created when the relevant notebook cells run successfully. They are not automatically present in the repository.

## ⚠️ Limitations

- The model uses a small, manually created corpus.
- The tokenizer operates at the word level rather than using subword tokenization.
- Unknown words are replaced with the token ID for `"the"`, which may distort predictions.
- The context length is limited to eight tokens.
- The architecture uses a single causal self-attention operation rather than multi-head attention.
- The model is trained for a fixed number of epochs without early stopping.
- Evaluation uses randomly sampled batches rather than a complete deterministic pass over every validation token.
- The validation set is a contiguous section of the same short corpus.
- Generated text may lack grammatical consistency and semantic coherence.
- The model is intended for educational experimentation, not production-grade language generation.

## 🚀 Future Improvements

The following enhancements can make the project more capable and informative.

### 1. Expand the Dataset

Use a larger and more diverse text corpus to expose the model to additional vocabulary, sentence structures, and contextual relationships.

### 2. Improve Tokenization

Explore subword tokenization or character-level modeling to handle unknown words and reduce vocabulary limitations.

### 3. Increase Context Length

Experiment with longer input sequences to allow the model to learn relationships across a broader context.

### 4. Implement Multi-Head Attention

Replace the single attention operation with multiple attention heads to learn different relationships between tokens.

### 5. Improve Training

Experiment with learning rates, batch sizes, learning-rate scheduling, and more training examples. Monitor validation loss to avoid overfitting.

### 6. Add a Learning-Rate Scheduler

Reduce the learning rate as training progresses to explore more stable optimization.

### 7. Improve Evaluation

Evaluate all valid validation positions deterministically, report reproducible metrics, and compare results across multiple random seeds.

### 8. Add More Visualizations

Include accuracy curves, generated-text comparisons, and attention-weight heatmaps.

### 9. Build an Interactive Interface

Create a simple web interface where users can enter prompts, inspect next-token probabilities, and generate text.

### 10. Compare Against Existing Language Models

Compare the educational Transformer with simpler n-gram models or other small language-model architectures under a consistent evaluation setup.

## 🤝 Contributing

Contributions and suggestions are welcome.

To contribute:

1. Fork the repository.
2. Create a new branch.
3. Implement your changes.
4. Run the notebook and check the results.
5. Update the documentation when necessary.
6. Submit a pull request describing your improvements.

## 📄 License

No license is specified in the supplied notebook. Add an appropriate license file if you intend to publish the project and permit others to reuse or modify your code.

---

<div align="center">

### 🚀 Understanding Language Models, One Token at a Time

**Built with Python, PyTorch, and the fundamentals of Transformer self-attention.**

⭐ If you find this project useful, consider giving the repository a star on GitHub.

</div>
