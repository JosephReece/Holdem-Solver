import tensorflow as tf

model_path = "models/model.keras"

max_sequence_length = 64

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

def encode_game_state(game, hero_idx):
    """
    Encode the current game state using the GTO No-Limit Hold'em Transformer Token Set.
    
    Parameters:
        game (PokerGame): The current game state
        
    Returns:
        str: A space-separated string of tokens representing the game state
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
        # Encode rank
        rank_map = {
            '2': 'RANK_2', '3': 'RANK_3', '4': 'RANK_4', '5': 'RANK_5',
            '6': 'RANK_6', '7': 'RANK_7', '8': 'RANK_8', '9': 'RANK_9',
            'T': 'RANK_T', 'J': 'RANK_J', 'Q': 'RANK_Q', 'K': 'RANK_K', 'A': 'RANK_A'
        }
        
        suit_map = {
            'Clubs': 'SUIT_C', 'Diamonds': 'SUIT_D', 'Hearts': 'SUIT_H', 'Spades': 'SUIT_S'
        }
        
        tokens.append(rank_map.get(card.rank))
        tokens.append(suit_map.get(card.suit))

    # Encode stack size (in octal)
    hero_stack = game.players[hero_idx].stack
    tokens.append("STACK_SIZE")
    octal_stack = oct(hero_stack)[2:]  # Convert to octal string and remove "0o" prefix
    for digit in octal_stack:
        tokens.append(digit)
    
    # Encode game history by street
    if "preflop" in game.history and game.history["preflop"]:
        tokens.append("PREFLOP")
        tokens.extend(encode_street_actions(game.history["preflop"], hero_position, opponent_position))
    
    if "flop" in game.history and game.history["flop"]:
        tokens.append("FLOP")
        
        # Add pot size (in octal)
        tokens.append("POT_SIZE")
        octal_pot = oct(game.pot)[2:]
        for digit in octal_pot:
            tokens.append(digit)
        
        # Add community cards (first 3)
        if len(game.community_cards) >= 3:
            for i in range(3):
                card = game.community_cards[i]
                card_str = str(card)
                rank = card_str[0]
                suit = card_str[1].lower()
                
                tokens.append(rank_map.get(card.rank))
                tokens.append(suit_map.get(card.suit))
        
        tokens.extend(encode_street_actions(game.history["flop"], hero_position, opponent_position))
    
    if "turn" in game.history and game.history["turn"]:
        tokens.append("TURN")
        
        # Add pot size
        tokens.append("POT_SIZE")
        octal_pot = oct(game.pot)[2:]
        for digit in octal_pot:
            tokens.append(digit)
        
        # Add turn card
        if len(game.community_cards) >= 4:
            card = game.community_cards[3]
            card_str = str(card)
            rank = card_str[0]
            suit = card_str[1].lower()
            
            tokens.append(rank_map.get(card.rank))
            tokens.append(suit_map.get(card.suit))
        
        tokens.extend(encode_street_actions(game.history["turn"], hero_position, opponent_position))
    
    if "river" in game.history and game.history["river"]:
        tokens.append("RIVER")
        
        # Add pot size
        tokens.append("POT_SIZE")
        octal_pot = oct(game.pot)[2:]
        for digit in octal_pot:
            tokens.append(digit)
        
        # Add river card
        if len(game.community_cards) >= 5:
            card = game.community_cards[4]
            card_str = str(card)
            rank = card_str[0]
            suit = card_str[1].lower()
            
            tokens.append(rank_map.get(card.rank))
            tokens.append(suit_map.get(card.suit))
        
        tokens.extend(encode_street_actions(game.history["river"], hero_position, opponent_position))
    
    tokens.append("EOS")
    
    # Pad the sequence
    if len(tokens) > max_sequence_length:
        tokens = tokens[:max_sequence_length]
    tokens = tokens + ['PAD'] * (max_sequence_length - len(tokens))

    return " ".join(tokens)


def encode_street_actions(actions, hero_position, opponent_position):
    """Helper function to encode actions for a street."""
    tokens = []
    
    # Filter out RESULT actions and any other non-player actions
    player_actions = [action for action in actions if not action.startswith("RESULT:")]
    
    for action_str in player_actions:
        # Parse the action string
        parts = action_str.split(": ")
        if len(parts) != 2:
            continue
        
        player_name, action = parts
        position = hero_position if player_name == "Player 1" else opponent_position
        tokens.append(position)
        
        # Extract the action and amount if applicable
        action_parts = action.split()
        action_type = action_parts[0].upper()
        
        if action_type == "POSTS":
            # Skip blind postings as they're implied
            continue
        elif action_type == "FOLD":
            tokens.append("FOLD")
        elif action_type == "CHECK":
            tokens.append("CHECK")
        elif action_type == "CALL":
            tokens.append("CALL")
        elif action_type == "BET":
            tokens.append("BET")
            # Convert amount to octal and add digits
            if len(action_parts) > 1:
                amount = int(action_parts[1])
                octal_amount = oct(amount)[2:]
                for digit in octal_amount:
                    tokens.append(digit)
        elif action_type == "RAISE":
            tokens.append("RAISE")
            # Convert amount to octal and add digits
            if len(action_parts) > 2 and action_parts[1] == "TO":
                amount = int(action_parts[2])
                octal_amount = oct(amount)[2:]
                for digit in octal_amount:
                    tokens.append(digit)
        elif action_type == "ALL-IN":
            tokens.append("ALL_IN")
            # Extract amount if present
            if "(" in action and ")" in action:
                amount_str = action.split("(")[1].split(")")[0]
                if amount_str.isdigit():
                    amount = int(amount_str)
                    octal_amount = oct(amount)[2:]
                    for digit in octal_amount:
                        tokens.append(digit)
    
    return tokens