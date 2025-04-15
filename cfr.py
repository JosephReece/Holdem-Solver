# cfr.py
import random
import numpy as np
import tensorflow as tf
import pickle
from collections import defaultdict

# ---------------------------------------------------------------------
# Abstract GameState Interface
# ---------------------------------------------------------------------
class GameState:
    def is_terminal(self):
        raise NotImplementedError

    def current_player(self):
        raise NotImplementedError

    def legal_player_actions(self):
        raise NotImplementedError

    def legal_chance_outcomes(self):
        raise NotImplementedError

    def next_state(self, action):
        raise NotImplementedError

    def utility(self):
        raise NotImplementedError

    def information_set(self):
        raise NotImplementedError

# ---------------------------------------------------------------------
# Tabular CFR Solver
# ---------------------------------------------------------------------
def float_dict():
            return defaultdict(float)

class CFRSolver:
    def __init__(self, root_state):
        self.root_state = root_state
        
        self.regret_sums = defaultdict(float_dict)
        self.strategy_sums = defaultdict(float_dict)

    def get_strategy(self, info_set, legal_actions):
        regrets = np.array([self.regret_sums[info_set][a] for a in legal_actions])
        regrets = np.maximum(regrets, 0)
        normalizing_sum = regrets.sum()
        if normalizing_sum > 0:
            strategy = regrets / normalizing_sum
        else:
            strategy = np.ones(len(legal_actions)) / len(legal_actions)

        for i, a in enumerate(legal_actions):
            self.strategy_sums[info_set][a] += strategy[i]

        return {a: strategy[i] for i, a in enumerate(legal_actions)}

    def cfr(self, state, p0, p1):
        if state.is_terminal():
            return state.utility()

        current_player = state.current_player()
        if current_player is None:
            util = np.zeros(2)
            for outcome in state.legal_chance_outcomes():
                next_state = state.next_state(outcome)
                outcome_util = self.cfr(next_state, p0, p1)
                util += np.array(outcome_util) / len(state.legal_chance_outcomes())
            return tuple(util)

        info_set = state.information_set()
        legal_actions = state.legal_player_actions()
        strategy = self.get_strategy(info_set, legal_actions)

        action_utils = {}
        node_util = 0.0

        for a in legal_actions:
            next_state = state.next_state(a)
            if current_player == 0:
                util = self.cfr(next_state, p0 * strategy[a], p1)
                action_utils[a] = util[0]
            else:
                util = self.cfr(next_state, p0, p1 * strategy[a])
                action_utils[a] = util[1]
            node_util += strategy[a] * action_utils[a]

        for a in legal_actions:
            regret = action_utils[a] - node_util
            weight = p1 if current_player == 0 else p0
            self.regret_sums[info_set][a] += weight * regret

        return (node_util, -node_util) if current_player == 0 else (-node_util, node_util)

    def train(self, iterations):
        for i in range(iterations):
            self.cfr(self.root_state, 1.0, 1.0)
            if (i + 1) % 10 == 0:
                print(f"CFR Iteration {i+1}/{iterations} completed.")

    def get_average_strategy(self, info_set, legal_actions):
        strategy_sum = self.strategy_sums[info_set]
        total = sum(strategy_sum[a] for a in legal_actions)
        if total == 0:
            return {a: 1 / len(legal_actions) for a in legal_actions}
        return {a: strategy_sum[a] / total for a in legal_actions}

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump((self.regret_sums, self.strategy_sums), f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.regret_sums, self.strategy_sums = pickle.load(f)

# ---------------------------------------------------------------------
# Monte Carlo Deep CFR Solver
# ---------------------------------------------------------------------
import random
import numpy as np
import tensorflow as tf
from collections import defaultdict, deque

@tf.keras.utils.register_keras_serializable()
class RegretNet(tf.keras.Model):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(output_dim)
        ])

    def call(self, x):
        return self.model(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable()
class StrategyNet(tf.keras.Model):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(output_dim, activation='softmax')
        ])

    def call(self, x):
        return self.model(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
class MonteCarloDeepCFRSolver:
    def __init__(self, root_state, input_vocabulary, output_decisions, max_input_length):
        self.root_state = root_state
        self.input_vocabulary = {token: i + 1 for i, token in enumerate(input_vocabulary)}
        self.output_decisions = output_decisions
        self.max_input_length = max_input_length
        self.input_dim = max_input_length
        self.output_dim = len(output_decisions)

        self.regret_nets = [RegretNet(self.input_dim, self.output_dim) for _ in range(2)]
        self.strategy_net = StrategyNet(self.input_dim, self.output_dim)

        # Compile the networks
        for net in self.regret_nets:
            net.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                   loss=tf.keras.losses.MeanSquaredError())
        
        self.strategy_net.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                loss=tf.keras.losses.CategoricalCrossentropy())

        self.regret_memory = [deque(maxlen=100000) for _ in range(2)]
        self.strategy_memory = deque(maxlen=100000)

    def encode(self, info_set):
        tokens = info_set.split()
        indices = [self.input_vocabulary.get(tok, 0) for tok in tokens]
        indices = [0] * (self.max_input_length - len(indices)) + indices[-self.max_input_length:]
        return tf.convert_to_tensor([indices], dtype=tf.float32)

    def traverse(self, state, player, traversing_player, pi=1.0):
        if state.is_terminal():
            return state.utility()[player]

        if state.player_cards is None:
            outcome = random.choice(state.legal_chance_outcomes())
            return self.traverse(state.next_state(outcome), player, traversing_player, pi)

        current_player = state.current_player()
        infoset = state.information_set()
        legal_actions = state.legal_player_actions()
        encoded = self.encode(infoset)

        if current_player == traversing_player:
            strategy = self.get_strategy(current_player, encoded, legal_actions)
            action_utilities = {}
            node_util = 0.0

            for action in legal_actions:
                next_state = state.next_state(action)
                util = self.traverse(next_state, player, traversing_player, pi * strategy[action])
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
            next_state = state.next_state(action)
            return self.traverse(next_state, player, traversing_player, pi)

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

        if state.player_cards is None:
            outcome = random.choice(state.legal_chance_outcomes())
            return self.collect_strategy(state.next_state(outcome))

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
        next_state = state.next_state(action)
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