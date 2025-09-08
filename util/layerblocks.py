import tensorflow as tf

# === TRANSFORMER BLOCKS ===
def TransformerBlock(x, embed_dim=64, num_heads=4, ff_dim=256, rate=0.1, training=False):
    x0 = tf.keras.layers.Dense(embed_dim)(x)
    attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x0, x0, x0)
    attn_output = tf.keras.layers.Dropout(rate)(attn_output, training=training)
    out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x0 + attn_output)
    
    ffn = tf.keras.Sequential([
        tf.keras.layers.Dense(ff_dim, activation='relu'),
        tf.keras.layers.Dense(embed_dim),
    ])
    ffn_output = ffn(out1)
    ffn_output = tf.keras.layers.Dropout(rate)(ffn_output, training=training)
    return tf.keras.layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

def TransformerStack(x, num_blocks=4, embed_dim=64, num_heads=4, ff_dim=256, rate=0.1, training=False):
    for _ in range(num_blocks):
        x = TransformerBlock(x, embed_dim, num_heads, ff_dim, rate, training)
    return x

# === DENSE BLOCKS ===
def DenseBlock(x, embed_dim=128):
    x = tf.keras.layers.Dense(embed_dim, activation='relu')(x)
    x = tf.keras.layers.Dense(embed_dim, activation='relu')(x)
    x = tf.keras.layers.Dense(embed_dim, activation='relu')(x)
    return tf.keras.layers.Dense(embed_dim)(x)

def DenseStack(x, num_blocks=4, embed_dim=128):
    for _ in range(num_blocks):
        x = DenseBlock(x, embed_dim)
    return x

# === LSTM BLOCKS ===
def LSTMBlock(x, units=64, training=False):
    x = tf.keras.layers.LSTM(units, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(units, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(units, return_sequences=True)(x)
    return tf.keras.layers.LSTM(units, return_sequences=True)(x)

def LSTMStack(x, num_blocks=4, units=64, training=False):
    for _ in range(num_blocks):
        x = LSTMBlock(x, units, training)
    return x
