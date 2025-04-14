import tensorflow as tf

model_path = "models/model.keras"
max_sequence_length = 10

# Token set for No-Limit Hold'em GTO Solver
vocabulary = [
    # Special tokens
    "PAD",  # Padding Token
    "BOS",  # Beginning of sequence
    "EOS",  # End of sequence
    
    # Rank tokens (13)
    "RANK_2", "RANK_3", "RANK_4", "RANK_5", "RANK_6", 
    "RANK_7", "RANK_8", "RANK_9", "RANK_T", "RANK_J", 
    "RANK_K", "RANK_Q", "RANK_A",
    
    # Suit tokens (4)
    "SUIT_C",  # Clubs
    "SUIT_D",  # Diamonds
    "SUIT_H",  # Hearts
    "SUIT_S",  # Spades
    
    # Position tokens
    "BTN",  # Button
    "BB",   # Big Blind
    
    # Action tokens
    "FOLD",
    "CHECK",
    "POST",
    "CALL",
    "BET",
    "RAISE",
    "ALL_IN", # Use this instead of call, bet and raise when a player goes all in
    
    # Game state tokens
    "PREFLOP",
    "FLOP",
    "TURN",
    "RIVER",
    
    # Sizing tokens
    "STACK_SIZE",
    "POT_SIZE",
    "SB_SIZE",
    "BB_SIZE",
    
    # Numeric tokens
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
]

# Outputs for the network 
output_decisions = [
    'FOLD',
    'PASSIVE_ACTION', #Check or Call
    'BET_20', #Bet 20% of pot, or raise 20% of the pot
    'BET_100',
    'ALL_IN'
]

def load_model():
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except:
        model = create_model()
        return model

def create_model(vocabulary, output_decisions):
    # Adjustable parameters
    transformer_depth = 1
    feed_forward_units = 64
    attention_key_dim = 16
    attention_num_heads = 2
    embedding_size = 32

    inputs = tf.keras.layers.Input(shape=(max_sequence_length,), dtype=tf.int32)
    embedding = tf.keras.layers.Embedding(
        input_dim=len(vocabulary),
        output_dim=embedding_size
    )(inputs)
    
    def transformer_block(input_layer):
        # Multi-head attention
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=attention_num_heads,
            key_dim=attention_key_dim
        )(input_layer, input_layer)
        
        # Add & Norm
        add_norm1 = tf.keras.layers.LayerNormalization()(tf.keras.layers.Add()([input_layer, attention]))
        
        # Feed Forward Network
        feed_forward = tf.keras.layers.Dense(units=feed_forward_units, activation='relu')(add_norm1)
        feed_forward_output = tf.keras.layers.Dense(units=embedding_size)(feed_forward)
        
        # Add & Norm
        return tf.keras.layers.LayerNormalization()(tf.keras.layers.Add()([add_norm1, feed_forward_output]))
    
    transformer_output = embedding
    for _ in range(transformer_depth):
        transformer_output = transformer_block(transformer_output)
    
    global_avg_pool = tf.keras.layers.GlobalAveragePooling1D()(transformer_output)
    outputs = tf.keras.layers.Dense(units=len(output_decisions), activation='softmax')(global_avg_pool)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def save_model(model):
    model.save(model_path)

def card_as_tokens(card):
    return ["RANK_" + card.rank, "SUIT_" + card.suit[0]]
