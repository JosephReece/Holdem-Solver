import numpy as np
import random
from cfr import GameState, MonteCarloDeepCFRSolver

# ---------------------------------------------------------------------
# Kuhn Poker GameState updated for new Deep Solver interface
# ---------------------------------------------------------------------
class KuhnPokerState(GameState):
    CARDS = ["J", "Q", "K"]
    ACTIONS = ["ACTION_P", "ACTION_B"]

    def __init__(self, player_cards=None, history=""):
        # When player_cards is None, the state is a chance node.
        self.player_cards = player_cards  
        self.history = history

    def is_terminal(self):
        # Terminal only when cards are dealt and the betting history completes a round.
        if self.player_cards is None:
            return False
        tokens = self.history.split()
        if len(tokens) == 2:
            if tokens == ["ACTION_P", "ACTION_P"]:
                return True
            if tokens[0] == "ACTION_B":
                return True
        if len(tokens) == 3 and tokens[0] == "ACTION_P" and tokens[1] == "ACTION_B":
            return True
        return False

    def is_chance(self):
        # A chance node if cards have not been dealt.
        return self.player_cards is None

    def current_player(self):
        # Returns the player index whose turn it is; returns None for chance or terminal nodes.
        if self.player_cards is None or self.is_terminal():
            return None
        tokens = self.history.split()
        if len(tokens) == 0:
            return 0
        if len(tokens) == 1:
            return 1
        if len(tokens) == 2 and tokens[0] == "ACTION_P" and tokens[1] == "ACTION_B":
            return 0
        return None

    def legal_player_actions(self):
        # Only valid if cards are assigned and the state isn’t terminal.
        return self.ACTIONS if self.player_cards and not self.is_terminal() else []

    def next_state_from_chance(self):
        # This method is only called when no cards have been dealt (chance node).
        if not self.is_chance():
            raise ValueError("Called next_state_from_chance on a non-chance state.")
        # Uniformly sample one of the valid deals (cards for player 0 and player 1)
        possible_deals = [(c1, c2) for c1 in self.CARDS for c2 in self.CARDS if c1 != c2]
        deal = random.choice(possible_deals)
        return KuhnPokerState(player_cards=deal, history="")

    def next_state_from_action(self, action):
        # This method applies a player action, updating the betting history.
        if self.is_chance():
            raise ValueError("Called next_state_from_action on a chance state.")
        new_history = self.history + (" " if self.history else "") + action
        return KuhnPokerState(self.player_cards, new_history)

    def utility(self):
        # Computes the payoff if the game is over. Otherwise, returns (0, 0).
        if not self.is_terminal():
            return (0, 0)
        tokens = self.history.split()
        rank = {"J": 1, "Q": 2, "K": 3}
        card0, card1 = self.player_cards[0], self.player_cards[1]
        # Determine pot size: if both players passed, pot is 1; otherwise 2.
        payoff = 1 if tokens == ["ACTION_P", "ACTION_P"] else 2
        # Special cases based on betting sequence
        if tokens == ["ACTION_B", "ACTION_P"]:
            return (1, -1)
        if tokens == ["ACTION_P", "ACTION_B", "ACTION_P"]:
            return (-1, 1)
        # Showdown: higher card wins the pot.
        if rank[card0] > rank[card1]:
            return (payoff, -payoff)
        return (-payoff, payoff)

    def information_set(self):
        # Returns the information set for the current player (their own card and the betting history).
        cp = self.current_player()
        if cp is None:
            return ""
        return f"{self.player_cards[cp]} {self.history}".strip()

# ---------------------------------------------------------------------
# Training and Strategy Evaluation
# ---------------------------------------------------------------------
if __name__ == '__main__':
    input_vocabulary = ["PAD", "J", "Q", "K", "ACTION_P", "ACTION_B"]
    output_decisions = ["ACTION_P", "ACTION_B"]
    max_input_length = 3

    # Root state is a chance node (cards not yet dealt)
    deep_root = KuhnPokerState()
    deep_solver = MonteCarloDeepCFRSolver(
        root_state=deep_root,
        input_vocabulary=input_vocabulary,
        output_decisions=output_decisions,
        max_input_length=max_input_length
    )

    try:
        deep_solver.load("models/kuhn")
    except Exception as e:
        print("No existing model found, starting fresh training")
    
    # Training parameters
    sessions = 5
    iterations = 10
    traversals_per_iter = 1000
    batch_size = 512
    epochs = 5
    
    # Split training into sessions
    for session in range(sessions):
        print(f"\nTraining Session {session + 1}/{sessions}")
        deep_solver.train(iterations, traversals_per_iter, batch_size, epochs)
        deep_solver.save("models/kuhn")

        # Evaluate strategies after each session
        infosets = [
            "J", "Q", "K",
            "J ACTION_P", "Q ACTION_P", "K ACTION_P", 
            "J ACTION_B", "Q ACTION_B", "K ACTION_B",
            "J ACTION_P ACTION_B", "Q ACTION_P ACTION_B", "K ACTION_P ACTION_B"
        ]
        
        print(f"\nStrategy Comparison after Session {session + 1}:\n" + "-" * 50)
        for info_set in infosets:
            legal_actions = ["ACTION_P", "ACTION_B"]
            deep_strategy = deep_solver.get_average_strategy(info_set, legal_actions)
            print(f"InfoSet: {info_set:<20} | P={deep_strategy['ACTION_P']:.2f}, B={deep_strategy['ACTION_B']:.2f}")
        
        print("-" * 50)