import numpy as np
from cfr import GameState, CFRSolver, MonteCarloDeepCFRSolver

# ---------------------------------------------------------------------
# Kuhn Poker GameState
# ---------------------------------------------------------------------
class KuhnPokerState(GameState):
    CARDS = ["J", "Q", "K"]
    ACTIONS = ["ACTION_P", "ACTION_B"]

    def __init__(self, player_cards=None, history=""):
        self.player_cards = player_cards
        self.history = history

    def is_terminal(self):
        if self.player_cards is None:
            return False
        tokens = self.history.split()
        if len(tokens) == 2:
            if tokens[0] == "ACTION_P" and tokens[1] == "ACTION_P":
                return True
            if tokens[0] == "ACTION_B":
                return True
        if len(tokens) == 3 and tokens[0] == "ACTION_P" and tokens[1] == "ACTION_B":
            return True
        return False

    def current_player(self):
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

    def legal_chance_outcomes(self):
        if self.player_cards is None:
            return [(c1, c2) for c1 in self.CARDS for c2 in self.CARDS if c1 != c2]
        return []

    def legal_player_actions(self):
        return self.ACTIONS if self.player_cards and not self.is_terminal() else []

    def next_state(self, action):
        if self.player_cards is None:
            return KuhnPokerState({0: action[0], 1: action[1]}, "")
        new_history = self.history + (" " if self.history else "") + action
        return KuhnPokerState(self.player_cards, new_history)

    def utility(self):
        if not self.is_terminal():
            return (0, 0)
        tokens = self.history.split()
        rank = {"J": 1, "Q": 2, "K": 3}
        card0, card1 = self.player_cards[0], self.player_cards[1]
        payoff = 1 if tokens == ["ACTION_P", "ACTION_P"] else 2
        if tokens == ["ACTION_B", "ACTION_P"]:
            return (1, -1)
        if tokens == ["ACTION_P", "ACTION_B", "ACTION_P"]:
            return (-1, 1)
        if rank[card0] > rank[card1]:
            return (payoff, -payoff)
        return (-payoff, payoff)

    def information_set(self):
        cp = self.current_player()
        if cp is None:
            return ""
        return f"{self.player_cards[cp]} {self.history}".strip()

# ---------------------------------------------------------------------
# Training and comparison
# ---------------------------------------------------------------------
if __name__ == '__main__':
    input_vocabulary = ["PAD", "J", "Q", "K", "ACTION_P", "ACTION_B"]
    output_decisions = ["ACTION_P", "ACTION_B"]
    max_input_length = 3

    tabular_root = KuhnPokerState()
    tabular_solver = CFRSolver(tabular_root)
    tabular_solver.load("models/kuhn.pkl")

    deep_root = KuhnPokerState()
    deep_solver = MonteCarloDeepCFRSolver(
        root_state=deep_root,
        input_vocabulary=input_vocabulary,
        output_decisions=output_decisions,
        max_input_length=max_input_length
    )

    try:
        deep_solver.load("models")
    except:
        print("No existing model found, starting fresh training")
    
    # Training parameters
    sessions = 5
    iterations = 10
    traversals_per_iter = 1000
    batch_size = 256
    epochs = 10
    
    # Split training into sessions
    for session in range(sessions):
        print(f"\nTraining Session {session + 1}/{sessions}")
        deep_solver.train(iterations, traversals_per_iter, batch_size, epochs)
        deep_solver.save("models")

        # Evaluate strategies after each session
        infosets = [
            "J", "Q", "K",
            "J ACTION_P", "Q ACTION_P", "K ACTION_P", 
            "J ACTION_B", "Q ACTION_B", "K ACTION_B",
            "J ACTION_P ACTION_B", "Q ACTION_P ACTION_B", "K ACTION_P ACTION_B"
        ]

        print(f"\nStrategy Comparison after Session {session + 1}:\n" + "-" * 78)
        for info_set in infosets:
            legal_actions = ["ACTION_P", "ACTION_B"]
            deep_strategy = deep_solver.get_average_strategy(info_set, legal_actions)
            tabular_strategy = tabular_solver.get_average_strategy(info_set, legal_actions)
            print(f"InfoSet: {info_set:<20} | Deep: P={deep_strategy['ACTION_P']:.2f}, B={deep_strategy['ACTION_B']:.2f} | Tabular: P={tabular_strategy['ACTION_P']:.2f}, B={tabular_strategy['ACTION_B']:.2f}")
        print("-" * 78)
