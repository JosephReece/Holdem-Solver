import tensorflow as tf

model_path = "models/model.keras"

max_sequence_length = 256

vocabulary = [
    # Pad Token
    'PAD',
    
    # Special Tokens
    'EOS', 'BOS', 'SDM',
    
    # Card Ranks
    'RANK_2', 'RANK_3', 'RANK_4', 'RANK_5', 'RANK_6', 'RANK_7', 'RANK_8', 'RANK_9', 'RANK_T', 'RANK_J', 'RANK_Q', 'RANK_K', 'RANK_A',
    
    # Suits
    'SUIT_C', 'SUIT_D', 'SUIT_H', 'SUIT_S',
    
    # Octal Tokens
    '0', '1', '2', '3', '4', '5', '6', '7',
    
    # Player Positions
    'BTN', 'BB',
    
    # Actions
    'FOLD', 'CHECK', 'CALL', 'BET', 'RAISE', 'ALL_IN',
    
    # Stack Size Tokens
    'STACK_SIZE',
    
    # Pot Size Tokens
    'POT_SIZE',
    
    # Game State Tokens
    'PREFLOP', 'FLOP', 'TURN', 'RIVER'
]

output_tokens = [
    'FOLD',
    'PASSIVE_ACTION',  # Check or Call
    'START_AGGRESSIVE_ACTION',  # Begin Bet or Raise (Use OCTAL_X tokens until END_AGGRESSIVE_ACTION)
    'END_AGGRESSIVE_ACTION',
    'ALL_IN',
    'OCTAL_0',
    'OCTAL_1',
    'OCTAL_2',
    'OCTAL_3',
    'OCTAL_4',
    'OCTAL_5',
    'OCTAL_6',
    'OCTAL_7'
]

def load_model():
    return create_model()
    
    # try:
    #     model = tf.keras.models.load_model(model_path)
    #     return model
    # except:
    #     model = create_model()
    #     return model

def create_model():
    # inputs = tf.keras.layers.Input(shape=(None,), dtype=tf.int32)
    
    inputs = tf.keras.layers.Input(shape=(max_sequence_length,), dtype=tf.int32)
    embedding = tf.keras.layers.Embedding(
        input_dim=len(vocabulary),
        output_dim=512
    )(inputs)
    
    def transformer_block(input_layer):
        # Multi-head attention
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=8,
            key_dim=64
        )(input_layer, input_layer)
        
        # Add & Norm
        add_norm1 = tf.keras.layers.LayerNormalization()(tf.keras.layers.Add()([input_layer, attention]))
        
        # Feed Forward Network
        feed_forward = tf.keras.layers.Dense(units=1024, activation='relu')(add_norm1)
        feed_forward_output = tf.keras.layers.Dense(units=512)(feed_forward)
        
        # Add & Norm
        return tf.keras.layers.LayerNormalization()(tf.keras.layers.Add()([add_norm1, feed_forward_output]))
    
    transformer_output = embedding
    for _ in range(6):
        transformer_output = transformer_block(transformer_output)
    
    global_avg_pool = tf.keras.layers.GlobalAveragePooling1D()(transformer_output)
    outputs = tf.keras.layers.Dense(units=13, activation='softmax')(global_avg_pool)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='sparse_categorical_crossentropy',
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
    
    if len(tokens) > max_sequence_length:
        tokens = tokens[:max_sequence_length]
        return " ".join(tokens)
    
    return sequence + ' PAD' * (max_sequence_length - len(tokens))

def encode_street_actions(actions, hero_idx, hero_position, opponent_position):
    """Helper function to encode actions for a street."""
    tokens = []
    
    for action in actions:
        action_type = action["action"]
        if action_type == "POST":
            continue
        
        position = hero_position if action["player_idx"] == hero_idx else opponent_position
        tokens.append(position)
        
        tokens.append(action_type)
        if action_type in ["BET", "RAISE", "ALL_IN"]:
            tokens.extend(amount_as_tokens(action["amount"]))
    
    return tokens