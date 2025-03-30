import tensorflow as tf

model_path = "models/model.keras"
max_sequence_length = 64

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
    "ALL_IN",
    
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

# Note: The output decisions remain the same so that indices map consistently:
output_decisions = [
    'FOLD',
    'PASSIVE_ACTION',  # Check or Call
    'BET'
]

def load_model():
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except:
        model = create_model()
        return model

def create_model():
    # Adjustable parameters
    transformer_depth = 3          # Increased depth for better sequential understanding
    feed_forward_units = 256       # Increased units for better feature learning
    attention_key_dim = 32         # Increased key dimension for richer attention patterns
    attention_num_heads = 8        # Standard number of attention heads
    embedding_size = 128           # Larger embedding size for better token representation

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

def amount_as_tokens(amount):
    tokens = []
    octal_amount = oct(amount)[2:]
    for digit in octal_amount:
        tokens.append(digit)
    return tokens        
    
def encode_game_state(game, hero_idx):
    """
    Encode the current game state using the Token Set.
    """
    tokens = []
    
    # Start with BOS token
    tokens.append("BOS")
    
    # Determine positions
    opponent_idx = 1 - hero_idx
    if game.button_player_idx == hero_idx:
        hero_position = "BTN"
        opponent_position = "BB"
    else:
        hero_position = "BB"
        opponent_position = "BTN"
    
    tokens.append(hero_position)
    
    # Encode hero's cards
    for card in game.players[hero_idx].hand:
        tokens.extend(card_as_tokens(card))

    # Encode stack size (in octal)
    tokens.append("STACK_SIZE")
    hero_stack = game.players[hero_idx].stack
    tokens.extend(amount_as_tokens(hero_stack))
    
    street_cards = [0, 3, 1, 1]
    for i, (street, street_actions) in enumerate(game.history.items()):
        if len(street_actions) == 0:
            continue
        
        tokens.append(street.upper())
        
        if street != "preflop":
            tokens.append("POT_SIZE")
            tokens.extend(amount_as_tokens(game.pot))
        
        for j in range(street_cards[i]):
            tokens.extend(card_as_tokens(game.community_cards[j]))
        
        tokens.extend(encode_street_actions(street_actions, hero_idx, hero_position, opponent_position))
        
    tokens.append("EOS")
    
    return " ".join(tokens)

def pad_sequence(sequence):
    tokens = sequence.split()
    
    if len(tokens) >= max_sequence_length:
        tokens = tokens[:max_sequence_length]
    else:
        tokens += ['PAD'] * (max_sequence_length - len(tokens))
    return " ".join(tokens)
