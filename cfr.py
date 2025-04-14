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
class MonteCarloDeepCFRSolver:
    def __init__(self, root_state, model, input_vocabulary, output_decisions, max_input_length):
        self.root_state = root_state
        self.model = model
        self.input_vocabulary = {token: i + 1 for i, token in enumerate(input_vocabulary)}
        self.output_decisions = output_decisions
        self.max_input_length = max_input_length
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.replay_buffer = []

    def encode(self, info_set):
        tokens = info_set.split()
        indices = [self.input_vocabulary.get(tok, 0) for tok in tokens]
        indices = [0] * (self.max_input_length - len(indices)) + indices[-self.max_input_length:]
        return np.array(indices)

    def traverse(self, state, player, iteration):
        if state.is_terminal():
            return state.utility()[player]

        if state.player_cards is None:
            outcomes = state.legal_chance_outcomes()
            outcome = random.choice(outcomes)
            return self.traverse(state.next_state(outcome), player, iteration)

        current_player = state.current_player()
        infoset = state.information_set()
        legal_actions = state.legal_player_actions()

        strategy = self.get_strategy(infoset, legal_actions)
        action = random.choices(legal_actions, weights=[strategy[a] for a in legal_actions])[0]
        next_state = state.next_state(action)
        util = self.traverse(next_state, player, iteration)

        if current_player == player:
            self.replay_buffer.append((infoset, legal_actions, util))

        return util

    def train(self, iterations=50, simulations_per_iteration=5, batch_size=128, epochs=10):
        print(f"Starting training with {iterations} iterations, {simulations_per_iteration} simulations per iteration")
        for i in range(iterations):
            print(f"Iteration {i+1}/{iterations}")
            
            for _ in range(simulations_per_iteration):
                self.traverse(self.root_state, 0, i)
                self.traverse(self.root_state, 1, i)

            self._train_model(batch_size, epochs)

    def _train_model(self, batch_size, epochs):
        X, y = [], []
        for info_set, legal_actions, target in self.replay_buffer:
            encoded = self.encode(info_set)
            X.append(encoded)
            label = [0.0] * len(self.output_decisions)
            for i, action in enumerate(self.output_decisions):
                if action in legal_actions:
                    label[i] = target
            y.append(label)
        self.replay_buffer = []

        if X:
            X = np.array(X)
            y = np.array(y)
            self.model.compile(optimizer=self.optimizer, loss=self.loss_fn)
            self.model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=0)

    def get_strategy(self, info_set, legal_actions):
        encoded = self.encode(info_set)
        prediction = self.model(np.array([encoded]), training=False).numpy()[0]
        strategy = {a: prediction[i] for i, a in enumerate(self.output_decisions) if a in legal_actions}

        total = sum(strategy.values())
        if total > 0:
            return {a: strategy[a] / total for a in strategy}
        else:
            return {a: 1 / len(legal_actions) for a in legal_actions}

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)