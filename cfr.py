import random
import copy
import numpy as np
import tensorflow as tf
from card import Card, Deck, compare_hands, suits, rank_order
from model import load_model, save_model, vocabulary, max_sequence_length, card_as_tokens, pad_sequence

# Fixed bet amount for this simplified game.
BET_SIZE = 2

# Global replay buffer for advantage (regret) samples.
regret_buffer = []

BATCH_SIZE = 32
TRAIN_EPOCHS = 2

token_to_id = {token: idx for idx, token in enumerate(vocabulary)}

def tokenize(sequence):
    tokens = sequence.split()
    token_ids = [token_to_id[t] for t in tokens if t in token_to_id]
    # Pad to fixed length:
    while len(token_ids) < max_sequence_length:
        token_ids.append(0)
    return np.array(token_ids, dtype=np.int32)

def encode_state(state, current_player):
    """
    Encode an information set as a token sequence that includes:
      • The acting player's hole cards (placed immediately after BOS).
      • Per–street history. For each street in order, we output:
            [STREET_TOKEN] + board cards for that street + actions taken on that street.
      • Finally, the EOS token is appended.
      
    For board cards, we assume:
       - Preflop: no board cards.
       - Flop: the first three cards.
       - Turn: the fourth board card.
       - River: the fifth board card.
    """
    tokens = ["BOS"]
    
    # 1. Put acting player's hole cards right after BOS.
    acting_hole = state['p1_hole'] if current_player == 0 else state['p2_hole']
    for card in acting_hole:
        tokens.extend(card_as_tokens(card))
    
    # Fixed street ordering.
    street_order = ["preflop", "flop", "turn", "river"]
    for street in street_order:
        # Append street label and board cards for that street.
        if street == "flop" and len(state['board']) >= 3:
            tokens.append(street.upper())
            for card in state['board'][:3]:
                tokens.extend(card_as_tokens(card))
        elif street == "turn" and len(state['board']) >= 4:
            tokens.append(street.upper())
            tokens.extend(card_as_tokens(state['board'][3]))
        elif street == "river" and len(state['board']) >= 5:
            tokens.append(street.upper())
            tokens.extend(card_as_tokens(state['board'][4]))
        
        # Append the actions recorded on that street (if any).
        if street in state['history'] and state['history'][street]:
            tokens.extend(action.upper() for action in state['history'][street])
    
    tokens.append("EOS")
    token_sequence = " ".join(tokens)
    token_sequence = pad_sequence(token_sequence)
    return token_sequence

def available_actions(state):
    if state['phase'] == "act":
        return ["CHECK", "BET"]
    elif state['phase'] == "response":
        return ["CALL", "FOLD"]
    return []

def deep_result_state(state, action):
    """
    Given a state and an action, update the state.
    The action is recorded in the 'history' dictionary under the current street.
    """
    new_state = copy.deepcopy(state)
    current_street = new_state['street']
    
    # Ensure history for the current street.
    if 'history' not in new_state:
        new_state['history'] = {}
    if current_street not in new_state['history']:
        new_state['history'][current_street] = []
    
    if new_state['phase'] == "act":
        if action == "CHECK":
            new_state['history'][current_street].append("CHECK")
            new_state = advance_street(new_state)
        elif action == "BET":
            new_state['pot'] += BET_SIZE
            new_state['phase'] = "response"
            new_state['current_player'] = 1
            new_state['history'][current_street].append("BET")
    elif new_state['phase'] == "response":
        if action == "CALL":
            new_state['pot'] += BET_SIZE
            new_state['history'][current_street].append("CALL")
            new_state = advance_street(new_state)
        elif action == "FOLD":
            new_state['history'][current_street].append("FOLD")
            new_state['terminal'] = True
            new_state['folded_by'] = new_state['current_player']
    else:
        new_state['history'][current_street].append(action)
    
    return new_state

def advance_street(state):
    """
    Advance the game to the next street.
    When moving to the new street, initialize its history in the state.
    Also, add board cards as required.
    """
    new_state = copy.deepcopy(state)
    if state['street'] == 'preflop':
        new_state['street'] = 'flop'
        for _ in range(3):
            if new_state['deck']:
                new_state['board'].append(new_state['deck'].pop())
    elif state['street'] == 'flop':
        new_state['street'] = 'turn'
        if new_state['deck']:
            new_state['board'].append(new_state['deck'].pop())
    elif state['street'] == 'turn':
        new_state['street'] = 'river'
        if new_state['deck']:
            new_state['board'].append(new_state['deck'].pop())
    elif state['street'] == 'river':
        new_state['street'] = 'showdown'
        new_state['terminal'] = True
    
    new_state['phase'] = "act"
    new_state['current_player'] = 0
    # Initialize history for the new street if it does not yet exist.
    if new_state['street'] not in new_state['history']:
        new_state['history'][new_state['street']] = []
    return new_state

def terminal_utility(state):
    pot = state['pot']
    if 'folded_by' in state:
        return -pot/2.0 if state['folded_by'] == 0 else pot/2.0
    if state['street'] == 'showdown':
        p1_hand = state['p1_hole'] + state['board']
        p2_hand = state['p2_hole'] + state['board']
        winner = compare_hands(p1_hand, p2_hand)
        if winner == "Player 1":
            return pot/2.0
        elif winner == "Player 2":
            return -pot/2.0
        else:
            return 0
    return 0

def deep_cfr(state, p0, p1, net):
    if state.get('terminal', False) or state['street'] == 'showdown':
        return terminal_utility(state)
    
    current_player = state['current_player']
    phase = state['phase']
    actions = available_actions(state)
    
    # Encode the information set including complete per–street history.
    info_seq = encode_state(state, current_player)
    input_ids = np.expand_dims(tokenize(info_seq), axis=0)
    
    predictions = net(input_ids, training=False).numpy()[0]
    # With the softmax output, the network's predictions are already a probability distribution.
    # We use them to form the regret–matched strategy on the valid actions.
    
    if phase == "act":
        # Map network outputs: index 1 => CHECK, index 2 => BET.
        valid_indices = [1, 2]
        index_to_action = {1: "CHECK", 2: "BET"}
    elif phase == "response":
        # Map network outputs: index 0 => FOLD, index 1 => CALL.
        valid_indices = [0, 1]
        index_to_action = {0: "FOLD", 1: "CALL"}
    else:
        valid_indices = []
        index_to_action = {}
    
    # Compute strategy over valid actions by re‐normalizing the net’s probability predictions.
    valid_probs = [predictions[i] for i in valid_indices]
    normalizing_sum = sum(valid_probs)
    reg_strat = {}
    for i in valid_indices:
        reg_strat[i] = predictions[i] / normalizing_sum if normalizing_sum > 0 else 1.0 / len(valid_indices)
    
    # Recursively compute utilities for available actions.
    util = {}
    node_util = 0.0
    for i in valid_indices:
        action = index_to_action[i]
        next_state = deep_result_state(state, action)
        if current_player == 0:
            util[i] = -deep_cfr(next_state, p0 * reg_strat[i], p1, net)
        else:
            util[i] = -deep_cfr(next_state, p0, p1 * reg_strat[i], net)
        node_util += reg_strat[i] * util[i]
    
    # Instead of computing a regret signal, we now build a target probability distribution.
    # For the valid actions, the target is set to the regret–matched strategy.
    target = np.zeros(3, dtype=np.float32)
    for i in valid_indices:
        target[i] = reg_strat[i]
    
    regret_buffer.append((tokenize(info_seq), target))
    return node_util


def sample_episode(net):
    deck_obj = Deck()
    deck_obj.shuffle()
    deck = deck_obj.cards[:]
    p1_hole = [deck.pop(), deck.pop()]
    p2_hole = [deck.pop(), deck.pop()]
    state = {
        'street': 'preflop',
        'phase': 'act',
        'p1_hole': p1_hole,
        'p2_hole': p2_hole,
        'board': [],
        # Initialize history as a dictionary keyed by street.
        'history': {"preflop": []},
        'pot': 2,
        'current_player': 0,
        'deck': deck
    }
    return deep_cfr(state, 1.0, 1.0, net)

def train_network(net):
    if not regret_buffer:
        return
    X = np.array([sample[0] for sample in regret_buffer])
    Y = np.array([sample[1] for sample in regret_buffer])
    net.fit(X, Y, batch_size=BATCH_SIZE, epochs=TRAIN_EPOCHS, verbose=0)
    regret_buffer.clear()

def train_deep_cfr(iterations, episodes_per_iteration):
    net = load_model()
    util_sum = 0.0
    for it in range(iterations):
        for ep in range(episodes_per_iteration):
            util = sample_episode(net)
            util_sum += util
        train_network(net)
        if (it + 1) % 10 == 0:
            print(f"Iteration {it+1}\t Average game value: {util_sum/((it+1)*episodes_per_iteration):.3f}")
    print("\n========== Training Complete ==========\n")
    return net

def simulate_hand_with_strategy(net):
    deck_obj = Deck()
    deck_obj.shuffle()
    deck = deck_obj.cards[:]
    
    # Deal hole cards
    p1_hole = [deck.pop(), deck.pop()]
    p2_hole = [deck.pop(), deck.pop()]
    
    # Initialize game state
    state = {
        'street': 'preflop',
        'phase': 'act',
        'p1_hole': p1_hole,
        'p2_hole': p2_hole,
        'board': [],
        'history': {"preflop": []},
        'pot': 2,
        'current_player': 0,
        'deck': deck
    }
    
    print("\n" + "="*50)
    print("POKER HAND SIMULATION WITH TRAINED STRATEGY")
    print("="*50)
    print(f"Player 1 hole cards: {p1_hole[0]}, {p1_hole[1]}")
    print(f"Player 2 hole cards: {p2_hole[0]}, {p2_hole[1]}")
    print("-"*50)
    
    def print_state_info(state, action, predictions):
        print("\nSTATE INFO:")
        print("-"*30)
        print(f"Street: {state['street'].upper()}")
        print(f"Phase: {state['phase']}")
        print(f"Current pot: {state['pot']}")
        print(f"Board: {state['board'] if state['board'] else 'None'}")
        print(f"Current player: Player {state['current_player'] + 1}")
        print("\nACTION PROBABILITIES:")
        print("-"*30)
        if state['phase'] == "act":
            print(f"CHECK: {predictions[1]:.3f}")
            print(f"BET: {predictions[2]:.3f}")
        else:
            print(f"FOLD: {predictions[0]:.3f}")
            print(f"CALL: {predictions[1]:.3f}")
        print(f"\nChosen action: {action}")
        print("-"*50)
    
    # Play until terminal state
    while not state.get('terminal', False) and state['street'] != 'showdown':
        current_player = state['current_player']
        info_seq = encode_state(state, current_player)
        input_ids = np.expand_dims(tokenize(info_seq), axis=0)
        
        predictions = net(input_ids, training=False).numpy()[0]
        actions = available_actions(state)
        
        # Select action based on strategy
        if state['phase'] == "act":
            action = "CHECK" if predictions[1] > predictions[2] else "BET"
        else:
            action = "FOLD" if predictions[0] > predictions[1] else "CALL"
        
        print_state_info(state, action, predictions)
        state = deep_result_state(state, action)
        
    print("\nFINAL RESULT:")
    print("="*30)
    if 'folded_by' in state:
        print(f"Player {state['folded_by'] + 1} folded")
        print(f"Player {2 - state['folded_by']} wins {state['pot']} chips")
    else:
        print("Showdown!")
        print(f"Final board: {state['board']}")
        result = compare_hands(p1_hole + state['board'], p2_hole + state['board'])
        print(f"Winner: {result}")
    print("="*30)

def display_bet_response_grid():
    net = load_model()
    print("\nWhen P1 checks, P2 probabilities to fold vs call (red >0.5 fold):")
    print("   ", end="")
    for r in rank_order:
        print(f"  {r}  ", end="")
    print()
    
    for r1 in rank_order:
        print(f" {r1} ", end="")
        for r2 in rank_order[::-1]:
            # Create sample hand with first two cards
            state = {
                'street': 'preflop',
                'phase': 'response',
                'p1_hole': [Card(rank='2', suit='Hearts'), Card(rank='3', suit='Hearts')],
                'p2_hole': [Card(rank=r1, suit='Spades'), Card(rank=r2, suit='Clubs')],
                'board': [],
                'history': {'preflop': ['BET']},
                'pot': 2,
                'current_player': 1,
                'deck': []
            }
            
            info_seq = encode_state(state, 1)  # P2's perspective
            input_ids = np.expand_dims(tokenize(info_seq), axis=0)
            preds = net(input_ids, training=False).numpy()[0]
            fold_prob = preds[0]
            
            # Color output based on fold probability
            if fold_prob > 0.5:
                print("\033[91m{:0.2f}\033[0m ".format(fold_prob), end="")  # Red
            else:
                print("\033[92m{:0.2f}\033[0m ".format(fold_prob), end="")  # Green
        print()

if __name__ == '__main__':
    print("Training Deep CFR on a simplified NLHE game ...")
    display_bet_response_grid()
    deep_net = train_deep_cfr(iterations=1, episodes_per_iteration=1)
    simulate_hand_with_strategy(deep_net)
    # save_model(deep_net)
