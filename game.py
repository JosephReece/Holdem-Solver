import time
import random
import tensorflow as tf
import numpy as np
from card import Card, Deck, compare_hands, get_hand_type, hand_ranks
from model import encode_game_state, load_model, save_model, vocabulary, output_tokens, pad_sequence

# Global model for AI decision-making
global_model = load_model()

# Human-Readable tokens for printing
shorthand_output_tokens = ['Fold', 'Check/Call/End Betting', '0', '1', '2', '3', '4', '5', '6', '7']

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
    
    def record_action(self, street, player_idx, action, amount=None):
        self.history[street].append({
            'player_idx': player_idx,
            'action': action,
            'amount': amount
        })
    
    def get_ai_action(self, player_idx):
        """
        Get action from the AI model for the player with proper validation and resampling for invalid actions.
        Valid tokens are:
           - "FOLD"
           - "PASSIVE_ACTION" (represented by "Passive" in shorthand)
           - Any token starting with "OCTAL_" (including OCTAL_0)
        When an octal token is selected the aggressive action sequence begins. In that sequence,
        octal tokens (e.g. OCTAL_3, OCTAL_0, ...) are accumulated until a "PASSIVE_ACTION" is produced,
        which stops the betting sequence.
        """
        player = self.players[player_idx]
        opponent = self.players[1 - player_idx]
        
        # Calculate how much the player needs to call
        call_amount = max(0, self.current_bet - player.current_bet)
        
        # Variable to track actual amount bet for training data
        bet_amount = 0
        
        # If player has no chips, they can only check or fold
        if player.stack == 0:
            action = "check" if call_amount == 0 else "fold"
            street = self.determine_current_street()
            self.record_action(street, player_idx, action)
            return action
        
        # Encode the current game state
        encoded_state = encode_game_state(self, player_idx)
        
        # Store the state for training
        game_state = pad_sequence(encoded_state)
        
        # Convert encoded state to model input
        token_to_id = {token: i for i, token in enumerate(vocabulary)}
        tokenized_input = [token_to_id.get(token, 0) for token in game_state.split()]
        model_input = tf.expand_dims(tf.constant(tokenized_input), 0)
        
        # Get an initial action token with validation
        action_token = None
        max_attempts = 3
        attempts = 0
        
        while attempts < max_attempts:
            prediction = global_model.predict(model_input, verbose=0)[0]
            prediction = np.array(prediction) + 1/100  # smoothing factor
            prediction = prediction / np.sum(prediction)  # Normalize probabilities
            
            # For logging
            formatted_probs = []
            for i, (prob, token) in enumerate(zip(prediction, shorthand_output_tokens)):
                formatted_probs.append(f"{token} ({prob:.4f})")
            self.log(" • ".join(formatted_probs))
            
            action_idx = np.random.choice(len(prediction), p=prediction)
            action_token = output_tokens[action_idx]
            
            attempts += 1
        
        # Determine the current betting street
        street = self.determine_current_street()
        
        # Record the initial action in training data
        self.training_data.append({
            'player_idx': player_idx,
            'state': game_state,
            'action': action_token,
            'stack_before': player.stack,
            'reward': None  # will be filled later
        })
        
        # Process based on the token type
        if action_token == 'FOLD':
            player.is_folded = True
            self.record_action(street, player_idx, "FOLD")
            self.log(f"{player.name} folds")
            return "fold"
        
        elif action_token == 'PASSIVE_ACTION':
            # Passive action: if no bet is required then check; otherwise, call.
            if call_amount == 0:
                self.record_action(street, player_idx, "CHECK")
                self.log(f"{player.name} checks")
                return "check"
            else:
                if player.stack <= call_amount:  # call is effectively all-in
                    bet_amount = player.stack
                    self.pot += bet_amount
                    player.current_bet += bet_amount
                    player.stack = 0
                    
                    self.record_action(street, player_idx, "CALL", player.current_bet)
                    self.log(f"{player.name} calls {bet_amount} (all-in)")
                    return f"call {bet_amount} (all-in)"
                else:
                    bet_amount = call_amount
                    self.pot += bet_amount
                    player.stack -= bet_amount
                    player.current_bet = self.current_bet
                    
                    self.record_action(street, player_idx, "CALL", bet_amount)
                    self.log(f"{player.name} calls {bet_amount}")
                    return f"call {bet_amount}"
        
        elif action_token.startswith('OCTAL_'):
            # Begin an aggressive betting sequence.
            octal_value = ""
            # Process the first token (even if it is OCTAL_0)
            digit = action_token.split('_')[1]
            octal_value += digit
            
            # Update the state sequence (for training) with the chosen octal token.
            current_sequence = encoded_state + " SDM " + action_token
            
            # Now continue to sample tokens until we see a PASSIVE_ACTION,
            # which indicates that the aggressive sequence is complete.
            max_iterations = 10
            iterations = 0
            while iterations < max_iterations:
                tokenized_sequence = [token_to_id.get(token, 0) for token in current_sequence.split()]
                sequence_input = tf.expand_dims(tf.constant(tokenized_sequence), 0)
                next_prediction = global_model.predict(sequence_input, verbose=0)[0]
                next_prediction = np.array(next_prediction) + 1/100
                next_prediction = next_prediction / np.sum(next_prediction)
                
                # For logging next-token probabilities:
                formatted_probs = []
                for i, (prob, token) in enumerate(zip(next_prediction, shorthand_output_tokens)):
                    formatted_probs.append(f"{token} ({prob:.4f})")
                self.log(" • ".join(formatted_probs))
                
                next_token_idx = np.random.choice(len(next_prediction), p=next_prediction)
                next_token = output_tokens[next_token_idx]
                
                # Record the token prediction in training data.
                self.training_data.append({
                    'player_idx': player_idx,
                    'state': pad_sequence(current_sequence),
                    'action': next_token,
                    'stack_before': player.stack,
                    'reward': None
                })
                
                if next_token == 'PASSIVE_ACTION':
                    # PASSIVE_ACTION stops the aggressive sequence.
                    break
                elif next_token.startswith('OCTAL_'):
                    # Append the digit (even if it is a "0")
                    digit = next_token.split('_')[1]
                    octal_value += digit
                else:
                    # Decides to fold during betting
                    player.is_folded = True
                    self.record_action(street, player_idx, "FOLD")
                    self.log(f"{player.name} folds")
                    return "fold"
                
                current_sequence += " " + next_token
                iterations += 1
            
            # Calculate the minimum bet first.
            min_bet = self.big_blind
            if call_amount > 0:
                min_bet = call_amount + self.big_blind
            
            # Convert the octal string (if "0", then use min_bet)
            try:
                bet_as_int = int(octal_value, 8)
            except Exception:
                bet_as_int = 0
            if bet_as_int == 0:
                bet_amount = min_bet
            else:
                bet_amount = bet_as_int
                if bet_amount < min_bet:
                    bet_amount = min_bet
            bet_amount = min(bet_amount, player.stack)
            
            # Process the bet action
            if bet_amount == player.stack:
                total_bet = player.current_bet + bet_amount
                self.pot += bet_amount
                if total_bet > self.current_bet:
                    self.current_bet = total_bet
                player.current_bet = total_bet
                player.stack = 0
                action_str = "all-in"
                self.record_action(street, player_idx, "ALL_IN", total_bet)
                self.log(f"{player.name} goes all-in ({total_bet})")
            elif self.current_bet == 0:
                self.pot += bet_amount
                player.stack -= bet_amount
                self.current_bet = bet_amount
                player.current_bet = bet_amount
                action_str = "bet"
                self.record_action(street, player_idx, "BET", bet_amount)
                self.log(f"{player.name} bets {bet_amount}")
            else:
                self.pot += bet_amount
                player.stack -= bet_amount
                self.current_bet = player.current_bet + bet_amount
                player.current_bet = self.current_bet
                action_str = "raise"
                self.record_action(street, player_idx, "RAISE", self.current_bet)
                self.log(f"{player.name} raises to {self.current_bet}")
            
            return action_str
        
        # Fallback (should not happen)
        self.log("Defaulting to check")
        return "check"
    
    def betting_round(self, first_player_idx):
        """
        Run a betting round.
        first_player_idx: Index of the player who acts first (0 or 1)
        Returns: Whether the hand is over (True if only one player left)
        """
        player_idx = first_player_idx
        players_acted = 0
        players_all_in = sum(1 for p in self.players if p.stack == 0)
        active_players = sum(1 for p in self.players if not p.is_folded)
        
        # If all active players are all-in, no betting needed.
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
            # Skip folded players.
            if self.players[player_idx].is_folded:
                player_idx = 1 - player_idx
                continue
            
            # Skip all-in players.
            if self.players[player_idx].stack == 0:
                player_idx = 1 - player_idx
                players_acted += 1
                if players_acted >= active_players:
                    break
                continue
                
            # Get action from the AI player.
            action = self.get_ai_action(player_idx)
            
            # Process the action.
            if action == "fold":
                # Check if only one player left.
                if sum(not p.is_folded for p in self.players) == 1:
                    return True
            
            elif action == "raise" or action == "bet" or action == "all-in":
                # Update the last raise amount.
                new_bet = self.players[player_idx].current_bet
                prev_bet = self.players[1 - player_idx].current_bet if action == "raise" else 0
                last_raise = new_bet - prev_bet
                
                # Reset players_acted so the other player gets to act.
                players_acted = 0
            
            player_idx = 1 - player_idx
            players_acted += 1
            
            # Check if betting round is complete.
            if players_acted >= active_players and self.players[0].current_bet == self.players[1].current_bet:
                break
        
        return False
    
    def determine_winner(self):
        # Check if only one player is left (not folded).
        active_players = [p for p in self.players if not p.is_folded]
        if len(active_players) == 1:
            return active_players[0]
        
        # Compare hands for showdown.
        hands = []
        for player in self.players:
            if not player.is_folded:
                best_hand = player.hand + self.community_cards
                hand_type = get_hand_type(best_hand)
                hands.append((player, best_hand, hand_type))
        
        # Show the showdown information.
        self.log("\n=== SHOWDOWN ===")
        for player, hand, hand_type in hands:
            self.log(f"{player.name}: {[str(card) for card in player.hand]} - {hand_type}")
        self.log(f"Community cards: {[str(card) for card in self.community_cards]}")
        
        if len(hands) == 2:
            result = compare_hands(hands[0][1], hands[1][1])
            if result == "Player 1":
                return hands[0][0]
            elif result == "Player 2":
                return hands[1][0]
            else:
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
                for action_data in actions:
                    player_idx = action_data['player_idx']
                    action = action_data['action']
                    amount = action_data['amount']
                    player_name = self.players[player_idx].name
                    if amount is not None:
                        self.log(f"({player_name}): {action} {amount}")
                    else:
                        self.log(f"({player_name}): {action}")
        
        self.log("===================\n")
    
    def compute_rewards(self, winner):
        """
        Compute rewards for all decisions made in the hand.
        Instead of basing the reward on the change in the decision maker's own stack,
        we assign as reward the number of chips won from the opponent's stack.
        
        The hand starting stacks are recorded at the beginning of the hand.
        For a winning hand, reward for each decision made by the winner is:
            opponent_start_stack - opponent_final_stack
        For decisions made by the loser, the reward is the negative of that amount.
        For ties, rewards are 0.
        """
        if winner is None:
            for data in self.training_data:
                data['reward'] = 0
            return
        
        winner_idx = self.players.index(winner)
        loser_idx = 1 - winner_idx
        
        winning_reward = self.hand_starting_stacks[loser_idx] - self.players[loser_idx].stack
        losing_reward = -winning_reward
        
        for data in self.training_data:
            if data['player_idx'] == winner_idx:
                data['reward'] = winning_reward
            else:
                data['reward'] = losing_reward
    
    def play_hand(self):
        # Record each player's starting stack before any forced bets (blinds) are posted.
        self.hand_starting_stacks = [player.stack for player in self.players]
        
        # Reset for new hand.
        self.pot = 0
        self.current_bet = 0
        self.community_cards = []
        for player in self.players:
            player.reset_hand()
        
        # Reset history for new hand.
        self.history = {
            "preflop": [],
            "flop": [],
            "turn": [],
            "river": []
        }
        
        # Reset training data for new hand.
        self.training_data = []
        
        # Determine positions.
        button_idx = self.button_player_idx
        sb_idx = button_idx  # In heads-up, button is also small blind.
        bb_idx = 1 - button_idx
        
        self.log("\n" + "="*50)
        self.log(f"NEW HAND - {self.players[0].name}: {self.players[0].stack} chips, {self.players[1].name}: {self.players[1].stack} chips")
        self.log(f"{self.players[button_idx].name} has the button (small blind)")
        
        # Post blinds.
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
        
        # Record blinds in history.
        self.record_action("preflop", sb_idx, "POST", sb_amount)
        self.record_action("preflop", bb_idx, "POST", bb_amount)
        
        # Deal hole cards.
        self.deal_hole_cards()
        
        # Show hole cards.
        for i, player in enumerate(self.players):
            self.log(f"\n{player.name} cards: {[str(card) for card in player.hand]}")
        
        # Pre-flop betting (small blind acts first in heads-up).
        self.log("\n--- PRE-FLOP ---")
        if self.betting_round(sb_idx):
            winner = self.determine_winner()
            self.compute_rewards(winner)
            self.award_pot(winner)
            self.display_hand_history()
            self.switch_button()
            return self.training_data
        
        # Flop.
        self.log("\n--- FLOP ---")
        flop_cards = self.deal_community_cards(3)
        self.log(f"Flop: {[str(card) for card in flop_cards]}")
        
        # Flop betting (big blind acts first).
        if self.betting_round(bb_idx):
            winner = self.determine_winner()
            self.compute_rewards(winner)
            self.award_pot(winner)
            self.display_hand_history()
            self.switch_button()
            return self.training_data
        
        # Turn.
        self.log("\n--- TURN ---")
        turn_card = self.deal_community_cards(1)
        self.log(f"Turn: {[str(card) for card in turn_card]}")
        
        # Turn betting (big blind acts first).
        if self.betting_round(bb_idx):
            winner = self.determine_winner()
            self.compute_rewards(winner)
            self.award_pot(winner)
            self.display_hand_history()
            self.switch_button()
            return self.training_data
        
        # River.
        self.log("\n--- RIVER ---")
        river_card = self.deal_community_cards(1)
        self.log(f"River: {[str(card) for card in river_card]}")
        
        # River betting (big blind acts first).
        if self.betting_round(bb_idx):
            winner = self.determine_winner()
            self.compute_rewards(winner)
            self.award_pot(winner)
            self.display_hand_history()
            self.switch_button()
            return self.training_data
        
        # Showdown.
        winner = self.determine_winner()
        self.compute_rewards(winner)
        self.award_pot(winner)
        self.display_hand_history()
        self.switch_button()
        return self.training_data
    
    def award_pot(self, winner):
        if winner is None:
            split_amount = self.pot // 2
            remainder = self.pot % 2
            
            self.players[0].stack += split_amount
            self.players[1].stack += split_amount
            
            if remainder:
                self.players[self.button_player_idx].stack += remainder
            
            self.log(f"\nTie game! Pot ({self.pot}) is split.")
            street = self.determine_current_street()
        else:
            winner.stack += self.pot
            winner_idx = self.players.index(winner)
            self.log(f"\n{winner.name} wins pot of {self.pot} chips!")
            street = self.determine_current_street()
    
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
            
            hand_data = self.play_hand()
            
            if training:
                all_training_data.extend(hand_data)
        
        self.log("\nFinal chip counts:")
        for player in self.players:
            self.log(f"{player.name}: {player.stack} chips")
        self.log("\nAI poker simulation complete!")
        
        return all_training_data

def train_model(epochs=5, hands_per_epoch=100, batch_size=32, learning_rate=0.01):
    """
    Train the global model using reinforcement learning on self-play games.
    
    Parameters:
        epochs (int): Number of training epochs
        hands_per_epoch (int): Number of hands to play in each epoch
        batch_size (int): Batch size for model training
        learning_rate (float): Learning rate for optimizer
    """
    print("Starting model training...")
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    global_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    token_to_id = {token: i for i, token in enumerate(vocabulary)}
    action_to_id = {action: i for i, action in enumerate(output_tokens)}
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        print("\n=== BASELINE PERFORMANCE ===")
        game = PokerGame(verbose=True)
        game.play_game(num_hands=1)
        
        training_data = []
        for hand in range(hands_per_epoch):
            print(f"Hand {hand+1}/{hands_per_epoch}")
            game = PokerGame(verbose=False)
            training_data += game.play_game(num_hands=1, training=True)
        
        if not training_data:
            print("No training data collected in this epoch.")
            continue
        
        states = []
        actions = []
        rewards = []
        
        for data_point in training_data:
            if data_point['reward'] is None:
                continue
            state_tokens = [token_to_id.get(token, 0) for token in data_point['state'].split()]
            action_id = action_to_id.get(data_point['action'], 0)
            states.append(state_tokens)
            actions.append(action_id)
            rewards.append(data_point['reward'])
        
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        
        positive_rewards = rewards[rewards > 0]
        if len(positive_rewards) > 0:
            print("\n=== POSITIVE REWARDS STATISTICS ===")
            print(f"Total positive decisions: {len(positive_rewards)}")
            print(f"Mean reward: {np.mean(positive_rewards):.4f}")
            print(f"Median reward: {np.median(positive_rewards):.4f}")
            print(f"Min reward: {np.min(positive_rewards):.4f}")
            print(f"Max reward: {np.max(positive_rewards):.4f}")
            print(f"Std dev: {np.std(positive_rewards):.4f}")
            
            percentiles = [25, 50, 75, 90, 95, 99]
            print("\nReward percentiles:")
            for p in percentiles:
                print(f"{p}th percentile: {np.percentile(positive_rewards, p):.4f}")
                
            positive_actions = actions[rewards > 0]
            unique_actions, action_counts = np.unique(positive_actions, return_counts=True)
            print("\nAction distribution for winning decisions:")
            for action, count in zip(unique_actions, action_counts):
                action_name = output_tokens[action]
                percentage = (count / len(positive_actions)) * 100
                print(f"{action_name}: {count} ({percentage:.1f}%)")
        
        if len(rewards) > 0:
            rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)
        
        num_samples = len(states)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            batch_states = states[batch_indices]
            batch_actions = actions[batch_indices]
            batch_rewards = rewards[batch_indices]
            
            global_model.fit(
                batch_states,
                batch_actions,
                sample_weight=batch_rewards,
                epochs=1,
                verbose=2
            )
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        
        print(f"Epoch {epoch+1} complete in {epoch_duration:.2f} seconds")
    
    print("\nTraining complete!")
    save_model(global_model)
    print("Model saved")

if __name__ == "__main__":
    print("Initializing AI model...")
    
    print("\n=== STARTING MODEL TRAINING ===")
    train_model(epochs=5, hands_per_epoch=1000, batch_size=100)
    
    print("\n=== EVALUATING TRAINED MODEL ===")
    for i in range(3):
        game = PokerGame(verbose=True)
        game.play_game(num_hands=1)