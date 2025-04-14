# kuhn.py
import numpy as np
from cfr import GameState, CFRSolver, MonteCarloDeepCFRSolver
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Flatten, Dropout

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
# Model builder for Deep CFR
# ---------------------------------------------------------------------
def create_model(input_vocabulary, output_decisions):
    vocab_size = len(input_vocabulary)
    output_size = len(output_decisions)
    model = Sequential([
        Embedding(input_dim=vocab_size + 1, output_dim=16),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(output_size)
    ])
    return model

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
    # tabular_solver.train(iterations=1000)
    # tabular_solver.save("models/kuhn.pkl")


    deep_root = KuhnPokerState()
    deep_model = create_model(input_vocabulary, output_decisions)
    deep_solver = MonteCarloDeepCFRSolver(deep_root, deep_model, input_vocabulary, output_decisions, max_input_length)
    
    # deep_solver.load("models/model-APR-14.keras")
    deep_solver.train(iterations=50, simulations_per_iteration=1000, batch_size=256, epochs=10)
    deep_solver.save("models/model-APR-14.keras")

    # Evaluate strategies
    infosets = [
        "J", "Q", "K",
        "J ACTION_P", "Q ACTION_P", "K ACTION_P",
        "J ACTION_B", "Q ACTION_B", "K ACTION_B",
        "J ACTION_P ACTION_B", "Q ACTION_P ACTION_B", "K ACTION_P ACTION_B"
    ]

    print("\nComparison of Strategies (Deep CFR vs. Tabular CFR):\n" + "-" * 80)
    for info_set in infosets:
        legal_actions = ["ACTION_P", "ACTION_B"]
        deep_strategy = deep_solver.get_strategy(info_set, legal_actions)
        tabular_strategy = tabular_solver.get_average_strategy(info_set, legal_actions)
        print(f"InfoSet: {info_set:<25} | Deep: P={deep_strategy['ACTION_P']:.2f}, B={deep_strategy['ACTION_B']:.2f} | Tabular: P={tabular_strategy['ACTION_P']:.2f}, B={tabular_strategy['ACTION_B']:.2f}")
    print("-" * 80)