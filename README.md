# Phrase-based Poem Generation

A deep learning project that generates creative poems based on input phrases using neural networks. The model employs bidirectional LSTM architecture to learn patterns from poetry datasets and create coherent, contextually relevant verses.

## 🎯 Project Overview

This project implements an AI-powered poetry generator that:
- Takes a user-provided phrase as a seed
- Generates complete poems with specified number of stanzas
- Uses advanced neural network architecture for natural language generation
- Maintains poetic structure and flow through learned patterns

## 🛠️ Technical Architecture

### Model Architecture
- **Embedding Layer**: 100-dimensional word embeddings
- **Bidirectional LSTM**: 150 units with sequence return capability
- **Dropout Layer**: 20% dropout for regularization
- **LSTM Layer**: 100 units for sequence processing
- **Dense Layers**: L2 regularized layers with ReLU and softmax activation
- **Output**: Categorical word prediction with vocabulary-sized softmax

### Key Features
- **Tokenization**: Advanced text preprocessing using Keras Tokenizer
- **Sequence Padding**: Uniform sequence length for efficient batch processing
- **Model Checkpointing**: Automatic saving of best-performing models
- **Learning Rate Scheduling**: Adaptive learning rate reduction
- **Training Visualization**: Real-time accuracy and loss plotting

## 📋 Requirements

```python
tensorflow >= 2.x
numpy
pandas
matplotlib
pickle
```

## 🚀 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Phrase-based-Poem-Generation
   ```

2. **Install dependencies**
   ```bash
   pip install tensorflow numpy pandas matplotlib
   ```

3. **Prepare your dataset**
   - Format: CSV file with 'Content' column containing poems
   - Place in accessible directory
   - Update file paths in the notebook

## 📊 Dataset Requirements

The model expects a CSV file with:
- **Column name**: 'Content'
- **Format**: Each row contains a complete poem
- **Preprocessing**: Poems are automatically split into lines and lowercased
- **Size**: Configurable (default: 600 poems for training)

## 🎨 Usage

### Training the Model

1. **Load the Jupyter notebook**
   ```bash
   jupyter notebook Phrase_based_Poem_Generation.ipynb
   ```

2. **Configure paths**
   ```python
   pth = "/your/dataset/path/"
   dest = "/temp/test.csv"
   ```

3. **Run training cells sequentially**
   - Data loading and preprocessing
   - Model architecture definition
   - Training with callbacks
   - Performance visualization

### Generating Poems

1. **Load trained model**
   ```python
   model = load_model('poem_generation.h5')
   tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
   ```

2. **Generate poems interactively**
   ```python
   # Input your seed phrase
   seed_text = "I dont know where he's stationed"
   
   # Specify number of stanzas
   next_words = 3
   
   # Generate complete poem
   generated_poem = generate_poem(seed_text, next_words)
   ```

## 🔧 Model Configuration

### Hyperparameters
- **Learning Rate**: 0.0001 (with adaptive reduction)
- **Epochs**: 200 (with early stopping via checkpoints)
- **Batch Processing**: Automatic batching with padding
- **Regularization**: L2 regularization (0.01) + Dropout (0.2)

### Training Callbacks
- **ModelCheckpoint**: Saves best model based on loss
- **ReduceLROnPlateau**: Reduces learning rate when loss plateaus
- **Monitoring**: Loss-based optimization with verbose logging

## 📈 Performance Monitoring

The project includes comprehensive training visualization:

```python
def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.show()

# Visualize training progress
plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')
```

## 📝 Output Format

Generated poems follow structured formatting:
- **Stanzas**: User-defined number of stanzas
- **Lines per stanza**: 8 lines per stanza
- **Word count**: 4 words per line (configurable)
- **Formatting**: Proper line breaks and spacing

## 🎯 Example Usage

```python
# Input
Enter phrase: I dont know where hes stationed, be it Cork or in Killarney
Enter number of stanzas: 2

# Output
Generated Poem:
I dont know where hes stationed, be it Cork or in Killarney
when the morning sun is shining on the hills
and the birds are singing sweetly in the trees
while the gentle breeze is blowing through the leaves
...
```

## 🔍 Technical Details

### Text Preprocessing Pipeline
1. **Data Loading**: CSV reading with selective column extraction
2. **Text Cleaning**: Lowercase conversion and line separation
3. **Tokenization**: Word-level tokenization with vocabulary building
4. **Sequence Generation**: N-gram sequence creation for training
5. **Padding**: Pre-padding sequences to uniform length

### Neural Network Training
1. **Data Preparation**: Feature-target splitting with categorical encoding
2. **Model Compilation**: Adam optimizer with categorical crossentropy loss
3. **Training Loop**: Epoch-based training with callback integration
4. **Model Persistence**: Automatic model and tokenizer saving

## 🚀 Advanced Features

- **Bidirectional Processing**: Captures both forward and backward context
- **Attention Mechanisms**: Enhanced sequence understanding
- **Regularization**: Prevents overfitting through dropout and L2
- **Adaptive Learning**: Dynamic learning rate adjustment
- **Batch Processing**: Efficient GPU utilization

## 🔧 Customization Options

### Model Architecture Modifications
```python
# Adjust embedding dimensions
model.add(Embedding(total_words, 200, input_length=max_sequence_len-1))

# Modify LSTM units
model.add(Bidirectional(LSTM(300, return_sequences=True)))

# Adjust dropout rate
model.add(Dropout(0.3))
```

### Training Parameters
```python
# Custom learning rate
optimizer = Adam(lr=0.001)

# Extended training
epochs = 500

# Custom batch size (implicit in fit method)
```

## 📊 Performance Metrics

The model tracks multiple performance indicators:
- **Training Accuracy**: Word prediction accuracy on training data
- **Loss Reduction**: Categorical crossentropy loss minimization
- **Convergence**: Learning rate adaptation and plateau detection
- **Generalization**: Poem quality through manual evaluation

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Enhanced model architectures (Transformer, GPT-style)
- Advanced preprocessing techniques
- Multi-language support
- Web interface development
- Performance optimization

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- TensorFlow/Keras community for deep learning frameworks
- Poetry datasets and literary corpus contributors
- Google Colab for accessible GPU computing
- Open source NLP research community

## 📚 References

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras LSTM Guide](https://keras.io/layers/recurrent/)
- [Natural Language Processing Research](https://nlp.stanford.edu/)
- [Poetry Generation Studies](https://arxiv.org/search/cs?query=poetry+generation)

---

*This project demonstrates practical applications of deep learning in creative AI, combining technical sophistication with artistic expression.*
