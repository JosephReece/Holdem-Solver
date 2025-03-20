import tensorflow as tf

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

# Output tokens
output_tokens = [
    'FOLD',
    'PASSIVE_ACTION',  # Check or Call
    'START_AGGRESSIVE_ACTION',  # Bet or Raise
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

def create_model():
    inputs = tf.keras.layers.Input(shape=(None,), dtype=tf.int32)
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

def encode_game_state(game, player_index):
    """
    Encodes the current poker game state into tokens for the transformer model input
    
    Args:
        game: The current poker game instance
        player_index: Index of the player whose perspective to encode (0 or 1)
        
    Returns:
        list: Array of tokens representing the game state
    """
    
    tokens = []
    hero = game.players[player_index]
    villain = game.players[(player_index + 1) % 2]
    
    # Start sequence
    tokens.append('BOS')
    
    # Player position (BTN or BB)
    tokens.append('BTN' if player_index == 0 else 'BB')
    
    # Hero's cards
    for card in hero.hand:
        tokens.append(f"RANK_{card.rank}")
        tokens.append(f"SUIT_{card.suit[0].upper()}")
    
    # Stack size in octal
    tokens.append('STACK_SIZE')
    stack_octal = convert_to_octal(hero.stack)
    for digit in stack_octal:
        tokens.append(digit)
    
    # Group action history by street
    preflop_actions = [action for action in game.action_history if action.street == 'preflop']
    flop_actions = [action for action in game.action_history if action.street == 'flop']
    turn_actions = [action for action in game.action_history if action.street == 'turn']
    river_actions = [action for action in game.action_history if action.street == 'river']
    
    # Add PREFLOP section with actions
    tokens.append('PREFLOP')
    encode_street_actions(preflop_actions, hero, villain, tokens)
    
    # If we've reached the flop or beyond
    if game.street in ['flop', 'turn', 'river', 'showdown']:
        tokens.append('FLOP')
        
        # Add pot size at the start of flop
        tokens.append('POT_SIZE')
        flop_pot_adjustment = 0 if game.street == 'flop' else get_total_bets_for_street(flop_actions)
        flop_pot_octal = convert_to_octal(game.pot - flop_pot_adjustment)
        for digit in flop_pot_octal:
            tokens.append(digit)
        
        # Add flop community cards (first 3)
        for i in range(min(3, len(game.community_cards))):
            card = game.community_cards[i]
            tokens.append(f"RANK_{card.rank}")
            tokens.append(f"SUIT_{card.suit[0].upper()}")
        
        # Add flop actions
        encode_street_actions(flop_actions, hero, villain, tokens)
    
    # If we've reached the turn or beyond
    if game.street in ['turn', 'river', 'showdown']:
        tokens.append('TURN')
        
        # Add pot size at the start of turn
        tokens.append('POT_SIZE')
        turn_pot_adjustment = 0 if game.street == 'turn' else get_total_bets_for_street(turn_actions)
        turn_pot_octal = convert_to_octal(game.pot - turn_pot_adjustment)
        for digit in turn_pot_octal:
            tokens.append(digit)
        
        # Add turn community card (4th card)
        if len(game.community_cards) >= 4:
            card = game.community_cards[3]
            tokens.append(f"RANK_{card.rank}")
            tokens.append(f"SUIT_{card.suit[0].upper()}")
        
        # Add turn actions
        encode_street_actions(turn_actions, hero, villain, tokens)
    
    # If we've reached the river or showdown
    if game.street in ['river', 'showdown']:
        tokens.append('RIVER')
        
        # Add pot size at the start of river
        tokens.append('POT_SIZE')
        river_pot_adjustment = 0 if game.street == 'river' else get_total_bets_for_street(river_actions)
        river_pot_octal = convert_to_octal(game.pot - river_pot_adjustment)
        for digit in river_pot_octal:
            tokens.append(digit)
        
        # Add river community card (5th card)
        if len(game.community_cards) >= 5:
            card = game.community_cards[4]
            tokens.append(f"RANK_{card.rank}")
            tokens.append(f"SUIT_{card.suit[0].upper()}")
        
        # Add river actions
        encode_street_actions(river_actions, hero, villain, tokens)
    
    # End sequence
    tokens.append('EOS')
    return tokens

def encode_street_actions(actions, hero, villain, tokens):
    """
    Helper function to encode actions for a specific street
    
    Args:
        actions: List of actions for the street
        hero: The hero player
        villain: The villain player
        tokens: The token list to append to
    """
    for action in actions:
        # Determine if action is from hero or villain
        is_hero_action = action.player == hero.name
        # Add position token
        if is_hero_action:
            tokens.append('BTN' if 'dealer' in hero.position else 'BB')
        else:
            tokens.append('BTN' if 'dealer' in villain.position else 'BB')
        
        # Add action type
        if action.action == 'fold':
            tokens.append('FOLD')
        elif action.action == 'check':
            tokens.append('CHECK')
        elif action.action == 'call':
            tokens.append('CALL')
        elif action.action == 'bet':
            tokens.append('BET')
            # Add bet size in octal
            bet_octal = convert_to_octal(action.amount)
            for digit in bet_octal:
                tokens.append(digit)
        elif action.action == 'raise':
            tokens.append('RAISE')
            # Add raise size in octal
            raise_octal = convert_to_octal(action.amount)
            for digit in raise_octal:
                tokens.append(digit)
        elif action.action == 'all-in':
            tokens.append('ALL_IN')

def get_total_bets_for_street(actions):
    """
    Calculates the total bets made on a specific street
    
    Args:
        actions: List of actions for the street
        
    Returns:
        int: The total amount bet on the street
    """
    total = 0
    for action in actions:
        if action.action in ['bet', 'raise', 'call', 'all-in']:
            total += action.amount
    return total

def convert_to_octal(decimal):
    """
    Converts a decimal number to its octal representation as a list of digits
    
    Args:
        decimal: The decimal number to convert
        
    Returns:
        list: The octal digits as strings
    """
    
    octal = oct(decimal)[2:]  # Remove '0o' prefix
    return list(octal)