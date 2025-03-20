import tensorflow as tf
import numpy as np
import copy

from card import Card, Deck, compare_hands
from model import create_model, encode_game_state, vocabulary, output_tokens

save_model = True
load_model = False
verbose = True
model_path = "./models/model.keras"

training_data = []
model = None

class Player:
    def __init__(self, name, stack, position):
        self.name = name
        self.stack = stack
        self.hand = []
        self.position = position
        self.has_acted = False
        self.has_folded = False
        self.current_bet = 0

    def reset_for_hand(self):
        self.hand = []
        self.has_acted = False
        self.has_folded = False
        self.current_bet = 0


class PokerGame:
    def __init__(self, starting_state=None):
        self.small_blind = 1
        self.big_blind = 2
        self.starting_stack = 100
        
        self.players = [
            Player("Player 1", self.starting_stack, "BTN"),
            Player("Player 2", self.starting_stack, "BB")
        ]
        
        self.deck = Deck()
        self.deck.shuffle()
        
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0
        
        if starting_state:
            # Restore game state from history
            self.street = starting_state["street"]
            self.community_cards = starting_state["community_cards"]
            self.pot = starting_state["pot"]
            self.current_bet = starting_state["current_bet"]
            self.action_history = starting_state["action_history"]
            
            for i, player_state in enumerate(starting_state["players"]):
                self.players[i].stack = player_state["stack"]
                self.players[i].hand = player_state["hand"]
                self.players[i].current_bet = player_state["current_bet"]
                self.players[i].has_acted = player_state["has_acted"] 
                self.players[i].has_folded = player_state["has_folded"]
                
            self.training_data = starting_state["training_data"]
        else:
            self.street = "preflop"
            self.action_history = []
            self.training_data = []
            
        self.start_hand(self.street)

    def get_game_state(self):
        return {
            "street": str(self.street),
            "deck": Deck([copy.copy(card) for card in self.deck.cards]),
            "community_cards": [copy.copy(card) for card in self.community_cards],
            "pot": int(self.pot),
            "current_bet": int(self.current_bet),
            "action_history": [copy.copy(action) for action in self.action_history],
            "training_data": [copy.copy(datum) for datum in self.training_data],
            "players": [{
                "stack": int(player.stack),
                "hand": [copy.copy(card) for card in player.hand],
                "current_bet": int(player.current_bet),
                "has_acted": bool(player.has_acted),
                "has_folded": bool(player.has_folded)
            } for player in self.players]
        }

    def start_hand(self, street = "preflop"):
        if verbose:
            print("\n=== STARTING HAND ===")
        self.pot = 0
        self.current_bet = 0
        self.community_cards = []
        self.action_history = []

        for player in self.players:
            player.reset_for_hand()

        if street == "preflop":
            self.post_blinds()
            self.deal_hands()
            
            self.run_betting_round()
            self.street = "flop"
            
            game_states = [PokerGame(self.get_game_state()) for i in range(1)]
            return
        
        if street == "flop":
            flop = self.deck.draw_cards(3)
            self.community_cards.extend(flop)
            
            if verbose:
                print("\n=== FLOP ===")
                print(', '.join(map(str, flop)))
            
            self.run_betting_round()
            self.street = "turn"
            
            game_states = [PokerGame(self.get_game_state()) for i in range(1)]
            return
        
        if street == "turn":
            turn = self.deck.draw_card()
            self.community_cards.append(turn)
            
            if verbose:
                print("\n=== TURN ===")
                print(turn)
            
            self.run_betting_round()
            self.street = "river"
            
            game_states = [PokerGame(self.get_game_state()) for i in range(1)]
            return
 
        if street == "river":
            river = self.deck.draw_card()
            self.community_cards.append(river)
            
            if verbose:
                print("\n=== RIVER ===")
                print(river)
            
            self.run_betting_round()
            self.street = "showdown"
        
        if verbose:
            print("\n=== SHOWDOWN ===")
        self.run_showdown()

    def post_blinds(self):
        btn, bb = self.players
        btn.stack -= self.small_blind
        btn.current_bet = self.small_blind
        self.pot += self.small_blind

        bb.stack -= self.big_blind
        bb.current_bet = self.big_blind
        self.current_bet = self.big_blind
        self.pot += self.big_blind

        if verbose:
            print(f"{btn.name} (BTN) posts small blind: ${self.small_blind}")
            print(f"{bb.name} (BB) posts big blind: ${self.big_blind}")

    def deal_hands(self):
        for player in self.players:
            player.hand = self.deck.draw_cards(2)
            if verbose:
                print(f"{player.name}'s hand: {', '.join(map(str, player.hand))}")
        
    def run_betting_round(self):
        for player in self.players:
            player.has_acted = False

        current_player_index = 0
        while not self.check_round_complete():
            player = self.players[current_player_index]
            if player.has_folded:
                break

            if verbose:
                print(f"{player.name}'s turn (AI decision)...")
            token = self.get_decision(player)
            
            state_tokens = encode_game_state(self, self.players.index(player))
            self.training_data.append({
                'state': state_tokens,
                'decision': token,
                'player_index': current_player_index
            })
            
            action, amount = None, 0
            if token == 'FOLD':
                action = 'fold'
            elif token == 'PASSIVE_ACTION':
                if self.current_bet > player.current_bet:
                    action = 'call'
                    amount = self.current_bet - player.current_bet
                else:
                    action = 'check'
            elif token == 'START_AGGRESSIVE_ACTION':
                if self.current_bet > player.current_bet:
                    action = 'raise'
                    legal_min = (self.current_bet * 2) - player.current_bet
                    amount = self.get_aggressive_amount(player, legal_min)
                else:
                    action = 'bet'
                    legal_min = max(int(self.pot * 0.2), 1)
                    amount = self.get_aggressive_amount(player, legal_min)
            elif token == 'ALL_IN':
                action = 'all-in'
                amount = player.stack

            # Declare hand dead with entire stack if invalid token used (eg OCTAL_4)
            available_actions = self.get_valid_actions(player)
            if action not in available_actions:
                action = "kill"
            
            self.process_action(player, action, amount)

            if any(p.has_folded for p in self.players):
                break

            current_player_index = (current_player_index + 1) % 2

    def get_decision(self, player):
        tokens = encode_game_state(self, self.players.index(player))
        token_indices = [vocabulary.index(t) for t in tokens]
        input_tensor = tf.constant([token_indices])
        
        prediction = model(input_tensor)
        output_probs = prediction.numpy()[0]
        
        return output_tokens[output_probs.argmax()]

    def get_aggressive_amount(self, player, default_min):
        digits = []
        max_chips = player.stack
        while True:
            tokens = encode_game_state(self, self.players.index(player)) + ["SDM"] + digits
            token_indices = [vocabulary.index(t) for t in tokens]
            input_tensor = tf.constant([token_indices])
            prediction = model(input_tensor)
            output_probs = prediction.numpy()[0]
            next_token = output_tokens[output_probs.argmax()]

            if next_token == "STOP_AGGRESSIVE_ACTION":
                break
            elif next_token in ['OCTAL_0', 'OCTAL_1', 'OCTAL_2', 'OCTAL_3', 'OCTAL_4', 'OCTAL_5', 'OCTAL_6', 'OCTAL_7']:
                digits.append(next_token)
                candidate = int("".join(digits), 8)
                if candidate > max_chips:
                    return max_chips
            else:
                break

        final_amount = int("".join(digits), 8) if digits else 0
        return max(default_min, min(final_amount, max_chips))

    def get_valid_actions(self, player):
        actions = []
        diff = self.current_bet - player.current_bet
        if diff > 0:
            actions.append("fold")
            if player.stack >= diff:
                actions.append("call")
            if player.stack >= diff * 2:
                actions.append("raise")
        else:
            actions.append("check")
            if player.stack > 0:
                actions.append("bet")
        if player.stack > 0:
            actions.append("all-in")
        return actions

    def process_action(self, player, action, amount):
        # Store the game state and action for training
        state_tokens = encode_game_state(self, self.players.index(player))
        
        player.has_acted = True
        if action == "fold":
            player.has_folded = True
        elif action == "kill": # Fold hand and forfeit entire stack
            player.has_folded = True
            self.pot += player.stack
            player.stack = 0
        elif action in ["call", "bet", "raise", "all-in"]:
            player.stack -= amount
            player.current_bet += amount
            self.pot += amount
            if player.current_bet > self.current_bet:
                self.current_bet = player.current_bet
                for p in self.players:
                    if p != player:
                        p.has_acted = False
        if verbose:
            print(f"{player.name} {action}s ${amount}. Pot: ${self.pot}")

    def check_round_complete(self):
        active = [p for p in self.players if not p.has_folded]
        return len(active) == 1 or all(p.has_acted and (p.current_bet == self.current_bet or p.stack == 0) for p in active)

    def run_showdown(self):
        active = [p for p in self.players if not p.has_folded]
        winner = None
        
        if len(active) == 1:
            winner = active[0]
            winner.stack += self.pot
            winner_index = self.players.index(winner)

            for data in self.training_data:
                if winner_index == data["player_index"]:
                    training_data.append({
                        'state': data['state'],
                        'decision': data['decision'],
                        'amount': self.pot // 2
                    })
                    
                    print("State: " + data['state'])
                    print("Decision: " + data['decision'])
                    print("Amount: " + str(self.pot // 2))
                else:
                    training_data.append({
                        'state': data['state'],
                        'decision': data['decision'],
                        'amount': -(self.pot // 2)
                    })
                    
                    print("State: " + str(data['state']))
                    print("Decision: " + str(data['decision']))
                    print("Amount: " + str(-self.pot // 2))

            if verbose:
                print(f"{winner.name} wins ${self.pot} uncontested.")
        else:
            winner_name = compare_hands(active[0].hand + self.community_cards, active[1].hand + self.community_cards)
            
            if winner_name == "Tie":
                player[0] += self.pot // 2
                player[1] += self.pot // 2
                
                for data in self.training_data:
                    training_data.append({
                        'state': data['state'],
                        'decision': data['decision'],
                        'amount': 0
                    })
                    
                if verbose:
                    print(f"Both players tie")
            else:
                winner = next(p for p in active if p.name == winner_name)
                winner.stack += self.pot
                winner_index = self.players.index(winner)
            
                for data in self.training_data:
                    if winner_index == data["player_index"]:
                        training_data.append({
                            'state': data['state'],
                            'decision': data['decision'],
                            'amount': self.pot // 2
                        })
                    else:
                        training_data.append({
                            'state': data['state'],
                            'decision': data['decision'],
                            'amount': -(self.pot // 2)
                        })
            
                if verbose:
                    print(f"{winner} wins ${self.pot}.")    
            

def train_batch(self):
    if not self.training_data:
        return
    
    # Prepare training data
    states = []
    labels = []
    
    # Find maximum sequence length
    max_len = max(len(data['state']) for data in self.training_data)
    
    for data in self.training_data:
        # Filter out losing actions
        if data["amount"] < 0:
            continue
        
        # Pad sequences with a special PAD token index (using 0)
        padded_state = [vocabulary.index(t) for t in data['state']]
        padded_state.extend([0] * (max_len - len(padded_state)))
        states.append(padded_state)
        
        # Create sparse categorical labe
        if data['stack_change'] > 0:  # Winning outcome
            if data['action'] in ['bet', 'raise']:
                label = output_tokens.index('START_AGGRESSIVE_ACTION')
            elif data['action'] in ['check', 'call']:
                label = output_tokens.index('PASSIVE_ACTION')
            else:
                label = output_tokens.index('ALL_IN')
        else:  # Losing outcome
            label = output_tokens.index('FOLD')
        
        labels.append(label)
    
    # Convert to tensors
    states = tf.convert_to_tensor(states, dtype=tf.int32)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)  # Changed to int32
    
    # Train the model
    if verbose:
        print("Training model with batch data...")
    model.fit(states, labels, epochs=1, verbose=2)
    
    if save_model:
        model.save(model_path)
        
        if verbose:
            print("Model saved successfully")
    
    # Clear training data
    self.training_data = []

def initialize_model():
        if verbose:
            print("Initializing AI model...")
        
        if load_model:
            return tf.keras.models.load_model(model_path)
            if verbose:
                print("Loaded existing model successfully")
        else:
            return create_model()
            if verbose:
                print("Created new model successfully")

if __name__ == "__main__":
    model = initialize_model()
    print(model)
    
    # Play 100 new starting-stack games
    for i in range(1):
        print(f"\nPlaying game {i+1}/100")
        game = PokerGame()
        
        if (i + 1) % 10 == 0:  # Train every 10 games
            if game.verbose:
                print("Training model with batch data...")
            game.train_batch()
    
    print("Completed training session")
