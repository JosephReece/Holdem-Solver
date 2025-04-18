# cfr.py

import random
import tensorflow as tf
from collections import deque
from model import RegretNet, StrategyNet

# ---------------------------------------------------------------------
# Abstract GameState Interface
# ---------------------------------------------------------------------
class GameState:
    def is_terminal(self):
        raise NotImplementedError

    def is_chance(self):
        raise NotImplementedError

    def current_player(self):
        raise NotImplementedError

    def legal_player_actions(self):
        raise NotImplementedError

    # New method replacing legal_chance_outcomes:
    def next_state_from_chance(self):
        raise NotImplementedError

    # Renamed method for player actions:
    def next_state_from_action(self, action):
        raise NotImplementedError

    def utility(self):
        raise NotImplementedError

    def information_set(self):
        raise NotImplementedError

# ---------------------------------------------------------------------
# Monte Carlo Deep CFR Solver
# ---------------------------------------------------------------------
class MonteCarloDeepCFRSolver:
    def __init__(self, root_state, input_vocabulary, output_decisions, max_input_length):
        self.root_state = root_state
        self.input_vocabulary = {token: i + 1 for i, token in enumerate(input_vocabulary)}
        self.output_decisions = output_decisions
        self.max_input_length = max_input_length
        self.output_dim = len(output_decisions)

        # Fill in blanks
        self.regret_nets = [
            RegretNet(
                vocab_size=len(self.input_vocabulary),
                max_len=self.max_input_length,
                output_dim=self.output_dim
            ) for _ in range(2)
        ]

        self.strategy_net = StrategyNet(
            vocab_size=len(self.input_vocabulary),
            max_len=self.max_input_length,
            output_dim=self.output_dim
        )

        # Compile the networks
        for net in self.regret_nets:
            net.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.MeanSquaredError())
        
        self.strategy_net.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.CategoricalCrossentropy())

        self.regret_memory = [deque(maxlen=100000) for _ in range(2)]
        self.strategy_memory = deque(maxlen=100000)

    def encode(self, info_set):
        tokens = info_set.split()
        indices = [self.input_vocabulary.get(tok, 0) for tok in tokens]
        indices = [0] * (self.max_input_length - len(indices)) + indices[-self.max_input_length:]
        return tf.convert_to_tensor([indices], dtype=tf.int32)  # int32 for embeddings

    def traverse(self, state, player, traversing_player, pi=1.0):
        if state.is_terminal():
            return state.utility()[player]

        if state.is_chance():
            next_state = state.next_state_from_chance()
            return self.traverse(next_state, player, traversing_player, pi)

        current_player = state.current_player()
        infoset = state.information_set()
        legal_actions = state.legal_player_actions()
        encoded = self.encode(infoset)

        if current_player == traversing_player:
            strategy = self.get_strategy(current_player, encoded, legal_actions)
            action_utilities = {}
            node_util = 0.0

            for action in legal_actions:
                ns = state.next_state_from_action(action)
                util = self.traverse(ns, player, traversing_player, pi * strategy[action])
                action_utilities[action] = util
                node_util += strategy[action] * util

            regrets = [0.0] * self.output_dim
            for i, action in enumerate(self.output_decisions):
                if action in action_utilities:
                    regrets[i] = action_utilities[action] - node_util

            # Store as a single sample in regret memory
            encoded_np = tf.squeeze(encoded).numpy()  # Convert to numpy here for storage
            self.regret_memory[traversing_player].append((encoded_np, regrets))
            return node_util

        else:
            strategy = self.get_strategy(current_player, encoded, legal_actions)
            action = random.choices(legal_actions, weights=[strategy[a] for a in legal_actions])[0]
            ns = state.next_state_from_action(action)
            return self.traverse(ns, player, traversing_player, pi)

    def get_strategy(self, player, encoded, legal_actions):
        # Use TensorFlow operations directly
        logits = self.regret_nets[player](encoded, training=False)
        positive_regrets = tf.maximum(logits, 0)
        positive_regrets_np = positive_regrets.numpy()[0]  # Convert to numpy only after TF ops
        
        strategy = {a: positive_regrets_np[i] for i, a in enumerate(self.output_decisions) if a in legal_actions}

        total = sum(strategy.values())
        if total > 0:
            return {a: strategy[a] / total for a in strategy}
        else:
            return {a: 1 / len(legal_actions) for a in legal_actions}

    def collect_strategy(self, state):
        if state.is_terminal():
            return

        if state.is_chance():
            next_state = state.next_state_from_chance()
            return self.collect_strategy(next_state)

        current_player = state.current_player()
        infoset = state.information_set()
        legal_actions = state.legal_player_actions()
        encoded = self.encode(infoset)

        strategy = self.get_strategy(current_player, encoded, legal_actions)
        label = [strategy.get(a, 0.0) for a in self.output_decisions]
        
        # Store as numpy for memory efficiency
        encoded_np = tf.squeeze(encoded).numpy()
        self.strategy_memory.append((encoded_np, label))

        action = random.choices(legal_actions, weights=[strategy[a] for a in legal_actions])[0]
        next_state = state.next_state_from_action(action)
        self.collect_strategy(next_state)

    def train(self, iterations, traversals_per_iter, batch_size, epochs):
        for i in range(iterations):
            for _ in range(traversals_per_iter):
                for player in range(2):
                    self.traverse(self.root_state, player, player)
                    self.collect_strategy(self.root_state)

            print(f"\nIteration {i+1}/{iterations}:")
            print(f"Memory sizes - Regret P0: {len(self.regret_memory[0])}, P1: {len(self.regret_memory[1])}, Strategy: {len(self.strategy_memory)}")

            for player in range(2):
                if self.regret_memory[player]:
                    X, y = zip(*self.regret_memory[player])
                    X = tf.convert_to_tensor(X, dtype=tf.float32)
                    y = tf.convert_to_tensor(y, dtype=tf.float32)
                    history = self.regret_nets[player].fit(X, y, batch_size=batch_size, epochs=epochs, verbose=0)
                    print(f"Player {player} Regret Net Loss: {history.history['loss'][-1]:.6f}")

            if self.strategy_memory:
                X, y = zip(*self.strategy_memory)
                X = tf.convert_to_tensor(X, dtype=tf.float32)
                y = tf.convert_to_tensor(y, dtype=tf.float32)
                history = self.strategy_net.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=0)
                print(f"Strategy Net Loss: {history.history['loss'][-1]:.6f}")

    def save(self, path):
        for i, net in enumerate(self.regret_nets):
            net.save(f"{path}/regret_net_player_{i}.keras")
        self.strategy_net.save(f"{path}/strategy_net.keras")

    def load(self, path):
        for i in range(2):
            self.regret_nets[i] = tf.keras.models.load_model(f"{path}/regret_net_player_{i}.keras")
        self.strategy_net = tf.keras.models.load_model(f"{path}/strategy_net.keras")

    def get_average_strategy(self, infoset, legal_actions):
        encoded = self.encode(infoset)
        prediction = self.strategy_net(encoded, training=False).numpy()[0]
        strategy = {a: prediction[i] for i, a in enumerate(self.output_decisions) if a in legal_actions}
        total = sum(strategy.values())
        if total > 0:
            return {a: strategy[a] / total for a in strategy}
        else:
            return {a: 1 / len(legal_actions) for a in legal_actions}