import tensorflow as tf

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        out1 = self.layernorm1(inputs + self.dropout1(attn_output, training=training))
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + self.dropout2(ffn_output, training=training))

@tf.keras.utils.register_keras_serializable()
class RegretNet(tf.keras.Model):
    def __init__(self, vocab_size, max_len, output_dim, embed_dim=32, num_heads=2, ff_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.embedding = tf.keras.layers.Embedding(vocab_size + 1, embed_dim)
        self.pos_encoding = tf.keras.layers.Embedding(input_dim=max_len, output_dim=embed_dim)
        self.transformer = TransformerBlock(embed_dim, num_heads, ff_dim)
        self.pooling = tf.keras.layers.GlobalAveragePooling1D()
        self.out_proj = tf.keras.layers.Dense(output_dim)

        self.max_len = max_len
        self.output_dim = output_dim
        self.vocab_size = vocab_size

    def call(self, x, training=False):
        positions = tf.range(start=0, limit=self.max_len, delta=1)
        x = self.embedding(x) + self.pos_encoding(positions)
        x = self.transformer(x, training=training)
        x = self.pooling(x)
        return self.out_proj(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "max_len": self.max_len,
            "output_dim": self.output_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable()
class StrategyNet(tf.keras.Model):
    def __init__(self, vocab_size, max_len, output_dim, embed_dim=32, num_heads=2, ff_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.embedding = tf.keras.layers.Embedding(vocab_size + 1, embed_dim)
        self.pos_encoding = tf.keras.layers.Embedding(input_dim=max_len, output_dim=embed_dim)
        self.transformer = TransformerBlock(embed_dim, num_heads, ff_dim)
        self.pooling = tf.keras.layers.GlobalAveragePooling1D()
        self.out_proj = tf.keras.layers.Dense(output_dim, activation="softmax")

        self.max_len = max_len
        self.output_dim = output_dim
        self.vocab_size = vocab_size

    def call(self, x, training=False):
        positions = tf.range(start=0, limit=self.max_len, delta=1)
        x = self.embedding(x) + self.pos_encoding(positions)
        x = self.transformer(x, training=training)
        x = self.pooling(x)
        return self.out_proj(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "max_len": self.max_len,
            "output_dim": self.output_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)