import random
import tensorflow as tf
import numpy as np
from card import Card, Deck, compare_hands, get_hand_type, hand_ranks
from model import encode_game_state, load_model, save_model, vocabulary, output_tokens

# Global model for AI decision-making
global_model = load_model()

class Player:
    def __init__(self, name, stack):
        self.name = name
        self.stack = stack
        self.reset_hand()
    
    def receive_card(self, card):
        self.hand.append(card)
    
    def reset_hand(self):
        self.hand = []
        self.is_folded = False
        self.current_bet = 0

class PokerGame:
    def __init__(self, verbose=True):
        self.deck = None
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0
        self.button_player_idx = 0  # 0 or 1, representing which player has the dealer button
        self.small_blind = 1
        self.big_blind = 2
        self.verbose = verbose  # Flag to control message output
        
        # Initialize history to track actions by street
        self.history = {
            "preflop": [],
            "flop": [],
            "turn": [],
            "river": []
        }
        # Track state-action-reward sequences for training
        self.training_data = []
        
        # Initialize players
        self.setup_players()
    
    def log(self, message):
        """
        Print message only if verbose mode is enabled
        """
        if self.verbose:
            print(message)
    
    def setup_players(self):
        self.log("\n===== AI TEXAS HOLD'EM POKER =====")
        self.log("\nSetting up AI players...")
        
        stack_size = 200  # Default stack size
        
        self.players = [
            Player("AI Player 1", stack_size),
            Player("AI Player 2", stack_size)
        ]
        self.log(f"Players created with {stack_size} chips each.")
    
    def switch_button(self):
        self.button_player_idx = 1 - self.button_player_idx
    
    def deal_hole_cards(self):
        # Create a new shuffled deck
        self.deck = Deck()
        
        # Deal two cards to each player
        for _ in range(2):
            for player in self.players:
                player.receive_card(self.deck.draw_card())
    
    def deal_community_cards(self, count):
        new_cards = self.deck.draw_cards(count)
        self.community_cards.extend(new_cards)
        return new_cards
    
    def determine_current_street(self):
        """
        Determine the current street based on community cards.
        Returns: 'preflop', 'flop', 'turn', or 'river'
        """
        if not self.community_cards:
            return "preflop"
        elif len(self.community_cards) == 3:
            return "flop"
        elif len(self.community_cards) == 4:
            return "turn"
        else:  # len(self.community_cards) == 5
            return "river"
    
    def record_action(self, street, player_name, action):
        """
        Record a player action to the history.
        
        Parameters:
            street (str): 'preflop', 'flop', 'turn', or 'river'
            player_name (str): Name of the player taking the action
            action (str): Description of the action taken
        """
        self.history[street].append(f"{player_name}: {action}")
    
    def get_ai_action(self, player_idx):
        """
        Get action from the AI model for the player using sampling instead of argmax.
        """
        player = self.players[player_idx]
        opponent = self.players[1 - player_idx]
        
        # Calculate how much the player needs to call
        call_amount = max(0, self.current_bet - player.current_bet)
        
        # If player has no chips, they can only check or fold
        if player.stack == 0:
            action = "check" if call_amount == 0 else "fold"
            # Determine current street for recording to history
            street = self.determine_current_street()
            self.record_action(street, player.name, action)
            return action
        
        # Encode the current game state
        encoded_state = encode_game_state(self, player_idx)
        self.log(f"Encoded state: {encoded_state}")
        
        # Store the state for training
        game_state = encoded_state
        
        # Convert encoded state to model input
        # This would involve tokenizing the string and converting to tensor
        token_to_id = {token: i for i, token in enumerate(vocabulary)}
        tokenized_input = [token_to_id.get(token, 0) for token in encoded_state.split()]
        model_input = tf.expand_dims(tf.constant(tokenized_input), 0)
        
        # Get prediction from model
        prediction = global_model.predict(model_input, verbose=0)[0]
        
        # Sample from the prediction probability distribution instead of argmax
        # Add a small constant to ensure no probability is exactly zero
        prediction = np.array(prediction) + 1e-8
        prediction = prediction / np.sum(prediction)  # Normalize to ensure sum = 1
        action_idx = np.random.choice(len(prediction), p=prediction)
        action_token = output_tokens[action_idx]
        
        # Determine current street for recording to history
        street = self.determine_current_street()
        
        # Convert model output to concrete action
        if action_token == 'FOLD':
            player.is_folded = True
            self.record_action(street, player.name, "fold")
            action_str = "fold"
            self.log(f"{player.name} folds")
        
        elif action_token == 'PASSIVE_ACTION':
            if call_amount == 0:
                self.record_action(street, player.name, "check")
                action_str = "check"
                self.log(f"{player.name} checks")
            else:
                # Call logic
                if player.stack <= call_amount:  # Call is actually all-in
                    self.pot += player.stack
                    self.current_bet = opponent.current_bet
                    player.current_bet = player.current_bet + player.stack
                    player.stack = 0
                    action_str = f"call {player.stack} (all-in)"
                    self.record_action(street, player.name, action_str)
                    self.log(f"{player.name} calls {player.stack} (all-in)")
                else:
                    self.pot += call_amount
                    player.stack -= call_amount
                    player.current_bet = self.current_bet
                    action_str = f"call {call_amount}"
                    self.record_action(street, player.name, action_str)
                    self.log(f"{player.name} calls {call_amount}")
        
        elif action_token == 'START_AGGRESSIVE_ACTION':
            # Look for octal digits until END_AGGRESSIVE_ACTION
            # For simplicity, let's use a random amount between 1/4 and 3/4 of the stack
            min_bet = self.big_blind
            if call_amount > 0:  # Raising
                min_bet = self.current_bet + self.big_blind
                
            max_bet = min(player.stack, self.current_bet + player.stack)
            bet_amount = random.randint(max(min_bet, player.stack // 4), 
                                        max(min_bet, min(player.stack * 3 // 4, max_bet)))
            
            if self.current_bet == 0:  # Betting
                self.pot += bet_amount
                player.stack -= bet_amount
                self.current_bet = bet_amount
                player.current_bet = bet_amount
                action_str = f"bet {bet_amount}"
                self.record_action(street, player.name, action_str)
                self.log(f"{player.name} bets {bet_amount}")
            else:  # Raising
                # Calculate how much more the player needs to put in
                additional_amount = bet_amount - player.current_bet
                self.pot += additional_amount
                player.stack -= additional_amount
                self.current_bet = bet_amount
                player.current_bet = bet_amount
                action_str = f"raise to {bet_amount}"
                self.record_action(street, player.name, action_str)
                self.log(f"{player.name} raises to {bet_amount}")
        
        elif action_token == 'ALL_IN':
            total_bet = player.current_bet + player.stack
            self.pot += player.stack
            if total_bet > self.current_bet:
                self.current_bet = total_bet
            player.current_bet = total_bet
            player.stack = 0
            action_str = f"all-in ({total_bet})"
            self.record_action(street, player.name, action_str)
            self.log(f"{player.name} goes all-in ({total_bet})")
        
        else:
            # Default to a safe action if something goes wrong
            if call_amount == 0:
                self.record_action(street, player.name, "check")
                action_str = "check"
                self.log(f"{player.name} checks (default action)")
            else:
                player.is_folded = True
                self.record_action(street, player.name, "fold")
                action_str = "fold"
                self.log(f"{player.name} folds (default action)")
        
        # Add this state-action pair to training data (reward will be determined at end of hand)
        self.training_data.append({
            'player_idx': player_idx,
            'state': game_state,
            'action': action_token,
            'stack_before': player.stack + (0 if 'call' not in action_str and 'bet' not in action_str and 'raise' not in action_str and 'all-in' not in action_str else int(action_str.split()[-1].strip('()').strip('all-in'))),
            'reward': None  # Will be filled in later
        })
        
        return action_str
    
    def betting_round(self, first_player_idx):
        """
        Run a betting round
        first_player_idx: Index of the player who acts first (0 or 1)
        Returns: Whether the hand is over (True if only one player left)
        """
        player_idx = first_player_idx
        players_acted = 0
        players_all_in = sum(1 for p in self.players if p.stack == 0)
        active_players = sum(1 for p in self.players if not p.is_folded)
        
        # If all active players are all-in, no betting needed
        if players_all_in == active_players:
            return active_players < 2
        
        # Reset current bet for the new round (if not preflop)
        if self.community_cards:
            self.current_bet = 0
            for player in self.players:
                player.current_bet = 0
        
        # Last raise amount (used for min-raise calculations)
        last_raise = 0
        
        while True:
            # Skip folded players
            if self.players[player_idx].is_folded:
                player_idx = 1 - player_idx
                continue
            
            # Skip all-in players
            if self.players[player_idx].stack == 0:
                player_idx = 1 - player_idx
                players_acted += 1
                if players_acted >= active_players:
                    break
                continue
                
            # Get action from the AI player
            action = self.get_ai_action(player_idx)
            
            # Process the action
            if "fold" in action:
                # Check if only one player left
                if sum(not p.is_folded for p in self.players) == 1:
                    return True
            
            elif "raise" in action or "bet" in action:
                # Update the last raise amount
                new_bet = self.players[player_idx].current_bet
                prev_bet = self.players[1 - player_idx].current_bet if "raise" in action else 0
                last_raise = new_bet - prev_bet
                
                # Reset players_acted to ensure the other player gets to act
                players_acted = 0
            
            player_idx = 1 - player_idx
            players_acted += 1
            
            # Check if betting round is complete
            if players_acted >= active_players and self.players[0].current_bet == self.players[1].current_bet:
                break
        
        return False
    
    def determine_winner(self):
        # Check if only one player is left (not folded)
        active_players = [p for p in self.players if not p.is_folded]
        if len(active_players) == 1:
            return active_players[0]
        
        # Compare hands for showdown
        hands = []
        for player in self.players:
            if not player.is_folded:
                best_hand = player.hand + self.community_cards
                hand_type = get_hand_type(best_hand)
                hands.append((player, best_hand, hand_type))
        
        # Show the showdown information
        self.log("\n=== SHOWDOWN ===")
        for player, hand, hand_type in hands:
            self.log(f"{player.name}: {[str(card) for card in player.hand]} - {hand_type}")
        self.log(f"Community cards: {[str(card) for card in self.community_cards]}")
        
        if len(hands) == 2:
            # Use compare_hands function to determine the winner
            result = compare_hands(hands[0][1], hands[1][1])
            if result == "Player 1":
                return hands[0][0]  # Player object
            elif result == "Player 2":
                return hands[1][0]  # Player object
            else:  # Tie
                return None
        
        return None  # Shouldn't happen in heads-up
    
    def display_hand_history(self):
        """
        Display the complete history of actions for the current hand.
        """
        self.log("\n=== HAND HISTORY ===")
        
        for street in ["preflop", "flop", "turn", "river"]:
            actions = self.history[street]
            if actions:
                self.log(f"--- {street.upper()} ---")
                for action in actions:
                    self.log(action)
        
        self.log("===================\n")
    
    def compute_rewards(self, winner):
        """
        Compute rewards for all decisions made in the hand.
        This uses a simplified reward system where winning = positive reward,
        losing = negative reward, with the magnitude based on pot size.
        """
        if winner is None:
            # Split pot - small positive reward for both
            for data in self.training_data:
                data['reward'] = self.pot / 4  # Small positive reward for tie
        else:
            winner_idx = self.players.index(winner)
            for data in self.training_data:
                # Decisions by winner get positive reward proportional to pot size
                if data['player_idx'] == winner_idx:
                    data['reward'] = self.pot / 2
                else:
                    # Decisions by loser get negative reward proportional to pot size
                    data['reward'] = -self.pot / 2
    
    def play_hand(self):
        # Reset for new hand
        self.pot = 0
        self.current_bet = 0
        self.community_cards = []
        for player in self.players:
            player.reset_hand()
        
        # Reset history for new hand
        self.history = {
            "preflop": [],
            "flop": [],
            "turn": [],
            "river": []
        }
        
        # Reset training data for new hand
        self.training_data = []
        
        # Determine positions
        button_idx = self.button_player_idx
        sb_idx = button_idx  # In heads-up, button is also SB
        bb_idx = 1 - button_idx
        
        self.log("\n" + "="*50)
        self.log(f"NEW HAND - {self.players[0].name}: {self.players[0].stack} chips, {self.players[1].name}: {self.players[1].stack} chips")
        self.log(f"{self.players[button_idx].name} has the button (small blind)")
        
        # Post blinds
        sb_amount = min(self.small_blind, self.players[sb_idx].stack)
        bb_amount = min(self.big_blind, self.players[bb_idx].stack)
        
        self.players[sb_idx].stack -= sb_amount
        self.players[sb_idx].current_bet = sb_amount
        self.players[bb_idx].stack -= bb_amount
        self.players[bb_idx].current_bet = bb_amount
        
        self.pot = sb_amount + bb_amount
        self.current_bet = bb_amount
        
        self.log(f"{self.players[sb_idx].name} posts small blind: {sb_amount}")
        self.log(f"{self.players[bb_idx].name} posts big blind: {bb_amount}")
        
        # Record blinds in history
        self.record_action("preflop", self.players[sb_idx].name, f"posts small blind {sb_amount}")
        self.record_action("preflop", self.players[bb_idx].name, f"posts big blind {bb_amount}")
        
        # Deal hole cards
        self.deal_hole_cards()
        
        # Show hole cards 
        for i, player in enumerate(self.players):
            self.log(f"\n{player.name} cards: {[str(card) for card in player.hand]}")
        
        # Pre-flop betting (SB acts first in heads-up)
        self.log("\n--- PRE-FLOP ---")
        if self.betting_round(sb_idx):
            # Hand is over - only one player left
            winner = self.determine_winner()
            self.compute_rewards(winner)
            self.award_pot(winner)
            self.display_hand_history()
            self.switch_button()
            return self.training_data
        
        # Flop
        self.log("\n--- FLOP ---")
        flop_cards = self.deal_community_cards(3)
        self.log(f"Flop: {[str(card) for card in flop_cards]}")
        
        # Flop betting (BB acts first)
        if self.betting_round(bb_idx):
            winner = self.determine_winner()
            self.compute_rewards(winner)
            self.award_pot(winner)
            self.display_hand_history()
            self.switch_button()
            return self.training_data
        
        # Turn
        self.log("\n--- TURN ---")
        turn_card = self.deal_community_cards(1)
        self.log(f"Turn: {[str(card) for card in turn_card]}")
        
        # Turn betting (BB acts first)
        if self.betting_round(bb_idx):
            winner = self.determine_winner()
            self.compute_rewards(winner)
            self.award_pot(winner)
            self.display_hand_history()
            self.switch_button()
            return self.training_data
        
        # River
        self.log("\n--- RIVER ---")
        river_card = self.deal_community_cards(1)
        self.log(f"River: {[str(card) for card in river_card]}")
        
        # River betting (BB acts first)
        if self.betting_round(bb_idx):
            winner = self.determine_winner()
            self.compute_rewards(winner)
            self.award_pot(winner)
            self.display_hand_history()
            self.switch_button()
            return self.training_data
        
        # Showdown
        winner = self.determine_winner()
        self.compute_rewards(winner)
        self.award_pot(winner)
        self.display_hand_history()
        self.switch_button()
        return self.training_data
    
    def award_pot(self, winner):
        if winner is None:
            # Split pot
            split_amount = self.pot // 2
            remainder = self.pot % 2
            
            self.players[0].stack += split_amount
            self.players[1].stack += split_amount
            
            # Give the odd chip to the button player
            if remainder:
                self.players[self.button_player_idx].stack += remainder
            
            self.log(f"\nTie game! Pot ({self.pot}) is split.")
            # Record the result in history
            street = self.determine_current_street()
            self.record_action(street, "RESULT", f"Pot ({self.pot}) split between players")
        else:
            winner.stack += self.pot
            self.log(f"\n{winner.name} wins pot of {self.pot} chips!")
            # Record the result in history
            street = self.determine_current_street()
            self.record_action(street, "RESULT", f"{winner.name} wins pot of {self.pot} chips")
    
    def play_game(self, num_hands=10, training=False):
        self.log(f"\nAI poker simulation starting with {num_hands} hands...")
        
        all_training_data = []
        
        for hand_num in range(1, num_hands + 1):
            self.log(f"\nPlaying hand {hand_num} of {num_hands}")
            
            if self.players[0].stack <= 0:
                self.log(f"\n{self.players[1].name} wins the game!")
                break
            elif self.players[1].stack <= 0:
                self.log(f"\n{self.players[0].name} wins the game!")
                break
            
            # Play the hand and get training data
            hand_data = self.play_hand()
            
            if training:
                all_training_data.extend(hand_data)
        
        self.log("\nFinal chip counts:")
        for player in self.players:
            self.log(f"{player.name}: {player.stack} chips")
        self.log("\nAI poker simulation complete!")
        
        return all_training_data

def train_model(epochs=5, hands_per_epoch=100, batch_size=32, learning_rate=0.001):
    """
    Train the global model using reinforcement learning on self-play games.
    
    Parameters:
        epochs (int): Number of training epochs
        hands_per_epoch (int): Number of hands to play in each epoch
        batch_size (int): Batch size for model training
        learning_rate (float): Learning rate for optimizer
    """
    print("Starting model training...")
    
    # Compile the model with an appropriate optimizer and loss function
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    global_model.compile(optimizer=optimizer, 
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
    
    # Token mappings
    token_to_id = {token: i for i, token in enumerate(vocabulary)}
    action_to_id = {action: i for i, action in enumerate(output_tokens)}
    
    # For tracking progress
    avg_rewards = []
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        training_data = []
        for hand in range(hands_per_epoch):
            # Play games to collect training data
            game = PokerGame(verbose=True)
            training_data += game.play_game(num_hands=1, training=True)
        
        if not training_data:
            print("No training data collected in this epoch.")
            continue
        
        # Process training data
        states = []
        actions = []
        rewards = []
        
        for data_point in training_data:
            # Skip if no reward was assigned (shouldn't happen but just in case)
            if data_point['reward'] is None:
                continue
                
            # Process state
            state_tokens = [token_to_id.get(token, 0) for token in data_point['state'].split()]
                
            # Get action ID
            action_id = action_to_id.get(data_point['action'], 0)
            
            # Append to lists
            states.append(state_tokens)
            actions.append(action_id)
            rewards.append(data_point['reward'])
        
        # Convert to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        
        # Normalize rewards
        if len(rewards) > 0:
            rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)
            avg_rewards.append(np.mean(rewards))
        
        # Train in batches
        num_samples = len(states)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            batch_states = states[batch_indices]
            batch_actions = actions[batch_indices]
            batch_rewards = rewards[batch_indices]
            
            # Custom training step with rewards as sample weights
            global_model.fit(
                batch_states,
                batch_actions,
                sample_weight=batch_rewards,
                epochs=1,
                verbose=0
            )
        
        print(f"Epoch {epoch+1} complete. Average reward: {np.mean(avg_rewards[-1] if avg_rewards else 0):.4f}")
    
    print("\nTraining complete!")
    save_model(global_model)
    print("Model saved")

# Start the game and training
if __name__ == "__main__":
    print("Initializing AI model...")
    
    # First play without training to establish a baseline
    print("\n=== BASELINE PERFORMANCE (BEFORE TRAINING) ===")
    game = PokerGame(verbose=True)
    game.play_game(num_hands=1)
    
    # Now train the model
    print("\n=== STARTING MODEL TRAINING ===")
    train_model(epochs=3, hands_per_epoch=10, batch_size=16)
    
    # Test the trained model
    print("\n=== EVALUATING TRAINED MODEL ===")
    game = PokerGame(verbose=True)
    game.play_game(num_hands=3)