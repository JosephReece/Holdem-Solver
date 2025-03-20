import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Define the dataset (cycling from '0' to '9')
sequence = "01234567890123456789"  # Repeated to provide more context
char_to_idx = {char: idx for idx, char in enumerate("0123456789")}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

# Prepare input-output pairs
input_seq = [char_to_idx[c] for c in sequence[:-1]]  # All but last char
output_seq = [char_to_idx[c] for c in sequence[1:]]  # All but first char

# Convert to numpy arrays
X = np.array(input_seq)
y = np.array(output_seq)

# Reshape for embedding input
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

# Define Transformer model
class SimpleTransformer(keras.Model):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim):
        super(SimpleTransformer, self).__init__()
        self.embedding = layers.Embedding(vocab_size, embed_dim)
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(vocab_size)
        ])

    def call(self, inputs):
        x = self.embedding(inputs)
        attn_output = self.attention(x, x)
        x = x + attn_output  # Residual connection
        x = self.dense_proj(x)
        return x

# Hyperparameters
vocab_size = 10  # Digits 0-9
embed_dim = 16
num_heads = 2
ff_dim = 32

# Create and compile model
model = SimpleTransformer(vocab_size, embed_dim, num_heads, ff_dim)
model.compile(optimizer="adam", loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

# Train the model
model.fit(X, y, epochs=100, batch_size=2, verbose=2)

# Predict next character
input_char = "5"
input_idx = np.array([[char_to_idx[input_char]]])
predicted_logits = model.predict(input_idx)
predicted_idx = np.argmax(predicted_logits, axis=-1)[0][0]
predicted_char = idx_to_char[predicted_idx]

print(f"Predicted next character after '{input_char}': {predicted_char}")
