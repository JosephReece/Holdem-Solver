#!/usr/bin/env python3
"""
A very simplified CFR implementation for no-limit Texas Hold’em preflop betting,
using snake_case naming and the following action set:

  "fold", "check/call", "33", "50", "66", "90", "120", "150", "ai"

Game flow (simplified):
  1. Both players post a 1-chip ante. They are each dealt 2 hole cards.
  2. A single betting round occurs with a maximum of three decision nodes:
       • Node 0: Player 0 (acting first) may either check or bet (taking one of the bet/raise actions).
         (Note: folding is not allowed when there is no bet to call.)
       • Node 1: Player 1 then acts. If faced with a bet this node permits fold, call, or a re-raise.
         If no bet exists (Player 0 checked) then player 1 may either check or bet.
       • Node 2 (final): If player 1 elects to bet when no bet existed before or to re-raise when
         facing a bet, then player 0 gets a final decision—call or fold.
  3. If no one folds, the betting round ends. Then five community cards (the board) are dealt,
     and a showdown occurs using the compare_hands function.
     
Chip tracking:
  – Each player starts with a stack of 100 chips.
  – The ante reduces the stack by 1 (so initial stack becomes 99); the pot begins at 2.
  – For bet/raise actions the bet size is computed as a percentage of the pot at the time of the action.
  – “all-in” commits the rest of a player’s stack.
  
NOTE: This abstraction is meant for research and may not capture all subtleties of full-limit NLHE.
"""

import random
import copy
from card import Card, Deck, compare_hands, get_hand_type, hand_ranks

# Global node map for storing CFR nodes (information sets)
node_map = {}

# --- CFR Node class -----------------------------------------------------------
class Node:
    def __init__(self, info_set):
        self.info_set = info_set
        # We assume the current decision has 9 available actions.
        self.regret_sum = [0.0] * 9
        self.strategy = [0.0] * 9
        self.strategy_sum = [0.0] * 9

    def get_strategy(self, realization_weight):
        # Regret-matching: only positive regrets count.
        self.strategy = [r if r > 0 else 0 for r in self.regret_sum]
        normalizing_sum = sum(self.strategy)
        if normalizing_sum > 0:
            self.strategy = [s / normalizing_sum for s in self.strategy]
        else:
            self.strategy = [1.0 / len(self.strategy)] * len(self.strategy)
        self.strategy_sum = [
            self.strategy_sum[i] + realization_weight * self.strategy[i]
            for i in range(len(self.strategy))
        ]
        return self.strategy

    def get_average_strategy(self):
        normalizing_sum = sum(self.strategy_sum)
        if normalizing_sum > 0:
            return [s / normalizing_sum for s in self.strategy_sum]
        else:
            return [1.0 / len(self.strategy_sum)] * len(self.strategy_sum)

# --- Action labels and helper mappings ----------------------------------------
# Action ordering is fixed so that indices correspond to the order used in the Node:
# 0: "fold"
# 1: "check/call"
# 2: "33"   (bet/raise 33% of pot)
# 3: "50"   (bet/raise 50% of pot)
# 4: "66"   (bet/raise 66% of pot)
# 5: "90"   (bet/raise 90% of pot)
# 6: "120"  (bet/raise 120% of pot)
# 7: "150"  (bet/raise 150% of pot)
# 8: "ai"   (all-in)
ACTION_LABELS = ["fold", "check/call", "33", "50", "66", "90", "120", "150", "ai"]

# Mapping for bet/raise percentages (for actions 2 through 7)
BET_PERCENTAGE = {
    "33": 0.33,
    "50": 0.50,
    "66": 0.66,
    "90": 0.90,
    "120": 1.20,
    "150": 1.50,
}

# --- Game State Representation ------------------------------------------------
# We represent the state as a dictionary with keys:
#   "history": list of action codes (strings) taken so far.
#   "pot": current pot size (float)
#   "to_call": current amount that the acting player must put in to call.
#   "player_stacks": list of two floats representing the current chip stacks.
#   "hole_cards": [player0_hole, player1_hole] where each is a list of 2 Card objects.
#   "deck": a list of Card objects remaining (used for board draws)
#
# The initial state (prebetting) is created in run_training().

# --- Available Actions --------------------------------------------------------
def available_actions(state, current_player):
    """
    Returns a list of available actions (strings) for the current state.
    When no bet is outstanding (to_call == 0), folding is disallowed.
    """
    if state["to_call"] > 0:
        return ACTION_LABELS  # All actions available.
    else:
        # Remove "fold" when there is nothing to call.
        return ACTION_LABELS[1:]  # Skip index 0 ("fold").

# --- State Update Functions ---------------------------------------------------
def apply_action(state, action, player):
    """
    Given a state, an action string, and the acting player,
    return a new state (a deepcopy with updates) and a flag indicating terminal node.
    
    This function updates:
      - the pot size,
      - the player's stack,
      - the "to_call" (i.e. the bet that the opponent must match), and 
      - appends the chosen action to the history.
      
    In this simplified betting round:
      • If the action is "fold", we mark the state as terminal (the acting player folds).
      • If the action is "check/call", then the acting player contributes the required amount 
        (if any). In the final decision node, this ends betting.
      • If the action is a bet/raise (one of "33", "50", "66", "90", "120", "150") then a bet
        is constructed: bet_amount = percentage * current pot (rounded to, say, two decimals).
      • If the action is "ai", then the player commits all of their remaining chips.
    """
    new_state = copy.deepcopy(state)
    new_state["history"].append(action)
    acting_stack = new_state["player_stacks"][player]
    
    # Determine if we are in a calling situation (i.e. to_call > 0)
    if action == "fold":
        # No chip contributions: the player folds immediately.
        return new_state, True, "fold"
    elif action == "check/call":
        amt = new_state["to_call"]
        # If the player lacks enough to call, they go all-in.
        call_amt = min(amt, acting_stack)
        new_state["player_stacks"][player] -= call_amt
        new_state["pot"] += call_amt
        new_state["to_call"] = 0.0  # Calling clears the outstanding bet.
        return new_state, True, "call"  # Terminal betting round.
    elif action in BET_PERCENTAGE:
        # A bet or raise.
        bet_factor = BET_PERCENTAGE[action]
        bet_amount = round(new_state["pot"] * bet_factor, 2)
        # If the player cannot fully bet this amount, treat it as an all-in.
        if bet_amount >= acting_stack:
            bet_amount = acting_stack
            action = "ai"
        new_state["player_stacks"][player] -= bet_amount
        new_state["pot"] += bet_amount
        new_state["to_call"] = bet_amount  # Opponent will need to call this bet.
        return new_state, False, "bet"
    elif action == "ai":
        # All-in: commit entire acting stack.
        bet_amount = acting_stack
        new_state["player_stacks"][player] = 0.0
        new_state["pot"] += bet_amount
        new_state["to_call"] = bet_amount
        return new_state, False, "bet"
    else:
        raise ValueError("Unrecognized action: " + action)

# --- Terminal Utility Calculations -------------------------------------------
def showdown_utility(state):
    """
    Terminal node reached by showdown: draw a board of 5 cards from state["deck"],
    then determine the winner using compare_hands.
    
    Returns the utility for player 0, defined as:
      final_stack (player0) - starting_stack (which was 99, after the ante).
    The opponent wins the pot if they win showdown.
    
    In our simplified model, the winner collects the entire pot.
    """
    # Sample 5 community cards (a chance node)
    deck_copy = copy.deepcopy(state["deck"])
    random.shuffle(deck_copy)
    board = deck_copy[:5]
    # Build each player's full hand (hole cards + board)
    player0_hand = state["hole_cards"][0] + board
    player1_hand = state["hole_cards"][1] + board
    winner = compare_hands(player0_hand, player1_hand)
    # Distribute pot accordingly.
    final_stacks = state["player_stacks"].copy()
    if winner == "Player 1":
        final_stacks[1] += state["pot"]
    else:
        final_stacks[0] += state["pot"]
    # Utility for player 0 is the change from baseline (99 after ante)
    return final_stacks[0] - 99

def fold_utility(state, folding_player):
    """
    Terminal node reached when a player folds.
    The opponent wins the pot. Returns the utility for player 0.
    If player 0 folds, utility is negative; if player 1 folds, utility is positive.
    """
    final_stacks = state["player_stacks"].copy()
    # The opponent receives the pot.
    if folding_player == 0:
        final_stacks[1] += state["pot"]
    else:
        final_stacks[0] += state["pot"]
    return final_stacks[0] - 99

# --- CFR Recursive Function ---------------------------------------------------
def cfr(state, current_player, p0, p1):
    """
    The main recursive counterfactual regret minimization (CFR) function.
    
    In our simplified betting structure, there are at most 3 decision nodes:
      - Node 0: Player 0's initial action.
      - Node 1: Player 1's action.
      - Node 2: (if needed) Player 0’s final decision when faced with an outstanding bet.
      
    Args:
        state: a dictionary representing the current game state.
        current_player: integer 0 or 1, indicating who is to act.
        p0, p1: reach probabilities for players 0 and 1.
    
    Returns:
        The counterfactual utility for player 0.
    """
    history = state["history"]
    # Determine stage by history length.
    stage = len(history)
    
    # Terminal nodes: reached if
    #   (a) an action caused a fold (the state was marked terminal by apply_action),
    #   (b) a "check/call" action ends betting (we assume this is terminal), or
    #   (c) if stage == 2 (final decision) finishing by call.
    if stage >= 1 and state.get("terminal", False):
        # Terminal state reached by call or fold.
        last_action = history[-1]
        if last_action == "fold":
            # Identify who folded. The acting player who chose "fold" is current_player.
            return fold_utility(state, current_player)
        else:
            # Terminal by call; proceed to showdown.
            return showdown_utility(state)
    
    # If no further action is possible in our simplified tree, go to showdown.
    if stage == 0 and current_player == 0:
        # Should always have an action.
        pass
    if stage > 2:
        return showdown_utility(state)
    
    # Determine information set key.
    # We include the player's hole cards (as a sorted string of ranks, for example) and the history.
    player_hole = "".join(sorted([card.rank for card in state["hole_cards"][current_player]]))
    info_set = f"{player_hole}|{'-'.join(history)}|pot:{state['pot']:.2f}|to_call:{state['to_call']:.2f}"
    
    if info_set not in node_map:
        # Determine the number of available actions for this node.
        avail_actions = available_actions(state, current_player)
        node_map[info_set] = Node(info_set)
        # Adjust node’s arrays if the number of available actions is less than 9.
        num_actions = len(avail_actions)
        if num_actions < 9:
            node = node_map[info_set]
            node.regret_sum = node.regret_sum[:num_actions]
            node.strategy = node.strategy[:num_actions]
            node.strategy_sum = node.strategy_sum[:num_actions]
    node = node_map[info_set]
    avail_actions = available_actions(state, current_player)
    
    # Get the mixed strategy for this node.
    strategy = node.get_strategy(p0 if current_player == 0 else p1)
    
    util = [0.0] * len(avail_actions)
    node_util = 0.0
    
    # Iterate over available actions.
    for i, action in enumerate(avail_actions):
        # Apply the action.
        next_state, is_terminal, act_type = apply_action(state, action, current_player)
        # If the action ends the betting round (call or fold), mark the state as terminal.
        if is_terminal:
            next_state["terminal"] = True
        # Switch current player if betting continues.
        next_player = 1 - current_player
        if is_terminal:
            util[i] = -cfr(next_state, next_player, p0 * strategy[i], p1)  # sign flip for opponent
        else:
            util[i] = -cfr(next_state, next_player, p0 * strategy[i], p1)
        node_util += strategy[i] * util[i]
    
    # Regret update.
    for i, action in enumerate(avail_actions):
        regret = util[i] - node_util
        if current_player == 0:
            node.regret_sum[i] += p1 * regret
        else:
            node.regret_sum[i] += p0 * regret
    
    return node_util

# --- Training Function --------------------------------------------------------
def train(iterations):
    """
    Train the CFR algorithm over a specified number of iterations.
    Each iteration deals new hole cards and constructs an initial state for the betting round.
    After training the average game value and the average strategies for each information set are printed.
    """
    util_sum = 0.0
    for i in range(iterations):
        # Create a fresh deck and shuffle.
        deck = Deck()
        deck.shuffle()
        # Deal hole cards: 2 for each player.
        player0_hole = deck.draw_cards(2)
        player1_hole = deck.draw_cards(2)
        # The remaining deck is used for the board later.
        remaining_deck = deck.cards.copy()

        # Set up initial state.
        state = {
            "history": [],
            "pot": 2.0,  # 1 chip ante from each player.
            "to_call": 0.0,
            "player_stacks": [99.0, 99.0],  # After posting the ante.
            "hole_cards": [player0_hole, player1_hole],
            "deck": remaining_deck,
        }
        util_sum += cfr(state, 0, 1.0, 1.0)
    
    print("Average game value (from player 0 perspective): {:.3f}".format(util_sum / iterations))
    print("\nInformation sets and average strategies:")
    for info_set in sorted(node_map.keys()):
        node = node_map[info_set]
        avg_strategy = node.get_average_strategy()
        print("  {}: {}".format(info_set, [round(s, 3) for s in avg_strategy]))

# --- Main Execution -----------------------------------------------------------
if __name__ == "__main__":
    # You can adjust the number of iterations as needed.
    train(100)