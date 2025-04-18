import random
import time
import numpy as np
import matplotlib.pyplot as plt

from cfr import GameState, MonteCarloDeepCFRSolver
from card import (
    Card, Deck, rank_order, rank_value, suits, hand_ranks,
    compare_hands, calculate_equity, canonicalize_hand
)

import sys
sys.setrecursionlimit(1000)

# ---------------------------------------------------------------------
# Game Parameters
# ---------------------------------------------------------------------
INITIAL_STACK = 200
SB_AMOUNT = 1
BB_AMOUNT = 2

# ---------------------------------------------------------------------
# NLHEState Class
# ---------------------------------------------------------------------
class NLHEState(GameState):
    def __init__(self, players=None, board=None, current_street="PREFLOP", history=None, is_chance=False):
        if players is None:
            deck = Deck()
            hole_cards = [deck.draw_cards(2), deck.draw_cards(2)]
            self.players = {
                0: {"stack": INITIAL_STACK - SB_AMOUNT, "contribution": SB_AMOUNT, "hole_cards": hole_cards[0], "all_in": False},
                1: {"stack": INITIAL_STACK - BB_AMOUNT, "contribution": BB_AMOUNT, "hole_cards": hole_cards[1], "all_in": False}
            }
            self.is_chance_node = False
        else:
            self.players = players
            self.is_chance_node = is_chance

        self.board = board if board is not None else []
        self.current_street = current_street
        self.history = history if history is not None else {s: [] for s in ["PREFLOP", "FLOP", "TURN", "RIVER"]}

        self.pot = self.players[0]["contribution"] + self.players[1]["contribution"]
        self.current_bet = self._extract_current_bet()
        self.raise_count = self._extract_raise_count()

    def _extract_current_bet(self):
        actions = self.history[self.current_street]
        for action in reversed(actions):
            if action[1].startswith("BET") or action[1] == "ALL_IN":
                return action[2]
        return 0

    def _extract_raise_count(self):
        return sum(1 for a in self.history[self.current_street] if a[1].startswith("BET") or a[1] == "ALL_IN")

    def is_chance(self):
        return self.is_chance_node

    def is_terminal(self):
        if any("FOLD" == a[1] for s in self.history.values() for a in s):
            return True
        if self.players[0]["all_in"] and self.players[1]["all_in"]:
            return True
        if self.current_street == "RIVER" and self.history["RIVER"] and self.history["RIVER"][-1][1] in ["CALL", "CHECK", "FOLD", "ALL_IN"]:
            return True
        return False

    def current_player(self):
        if self.is_chance() or self.is_terminal():
            return None
        starting_player = 0 if self.current_street == "PREFLOP" else 1
        num_actions = len(self.history[self.current_street])
        return (starting_player + num_actions) % 2

    def legal_player_actions(self):
        if self.is_chance() or self.is_terminal():
            return []

        if self.current_street == "PREFLOP":
            # No actions yet - BTN's first action
            if not self.history["PREFLOP"]:
                return ["FOLD", "CALL", "BET_100"]  # Lock no rips to start
            
            # Handle raises based on raise count
            if self.raise_count < 4: # Allow another raise
                return ["FOLD", "CALL", "BET_100", "ALL_IN"]
            else: # After 4-bet, only all-in, call or fold
                return ["FOLD", "CALL", "ALL_IN"]

        else: # FLOP, TURN, RIVER
            # First action in street
            defaults = ["CHECK", "BET_20", "BET_100", "ALL_IN"]
            if not self.history[self.current_street]:
                return defaults
            
            last_action = self.history[self.current_street][-1][1]
            if last_action == "CHECK":
                return defaults
            
            if last_action in ["BET_20", "BET_100"]:
                return ["FOLD", "CALL", "ALL_IN"] # Can only all-in as raise
            elif last_action == "ALL_IN":
                return ["FOLD", "CALL"]
            
        print(self.current_street)
        print(self.history)

    def next_state_from_chance(self):
        used_cards = [card for p in self.players.values() for card in p["hole_cards"]] + self.board
        deck = Deck(without=used_cards)
        new_board = self.board.copy()

        if self.current_street == "PREFLOP":
            new_board += deck.draw_cards(3)
            new_street = "FLOP"
        elif self.current_street == "FLOP":
            new_board += deck.draw_cards(1)
            new_street = "TURN"
        elif self.current_street == "TURN":
            new_board += deck.draw_cards(1)
            new_street = "RIVER"
        else:
            raise ValueError("No chance move available from RIVER.")

        return NLHEState(
            players=self._copy_players(),
            board=new_board,
            current_street=new_street,
            history={k: v.copy() for k, v in self.history.items()},  # Preserve history
            is_chance=False
        )

    def next_state_from_action(self, action):
        if self.is_chance():
            raise ValueError("Cannot call next_state_from_action on a chance node.")

        acting_player = self.current_player()
        new_history = {k: v.copy() for k, v in self.history.items()}
        new_players = self._copy_players()
        pot = self.pot

        if action == "FOLD":
            new_history[self.current_street].append((acting_player, "FOLD", 0))
        elif action == "CHECK":
            new_history[self.current_street].append((acting_player, "CHECK", 0))
        elif action == "CALL":
            call_amount = int(self.current_bet)
            new_players[acting_player]["stack"] -= call_amount
            new_players[acting_player]["contribution"] += call_amount
            pot += call_amount
            new_history[self.current_street].append((acting_player, "CALL", call_amount))
        elif action in ["BET_20", "BET_100"]:
            bet_amount = int(0.2 * pot) if action == "BET_20" else pot
            new_players[acting_player]["stack"] -= bet_amount
            new_players[acting_player]["contribution"] += bet_amount
            pot += bet_amount
            new_history[self.current_street].append((acting_player, action, bet_amount))
        elif action == "ALL_IN":
            all_in_amount = new_players[acting_player]["stack"]
            new_players[acting_player]["stack"] = 0
            new_players[acting_player]["contribution"] += all_in_amount
            pot += all_in_amount
            new_players[acting_player]["all_in"] = True
            new_history[self.current_street].append((acting_player, "ALL_IN", all_in_amount))
        else:
            raise ValueError("Invalid action")

        # Check for end-of-street conditions
        street_actions = new_history[self.current_street]
        if self.current_street != "RIVER" and len(street_actions) >= 2:
            last_action = street_actions[-1][1]
            second_last_action = street_actions[-2][1]

            both_all_in = new_players[0]["all_in"] and new_players[1]["all_in"]
            if (last_action == "CALL" or (last_action == "CHECK" and second_last_action == "CHECK")) or both_all_in:
                return NLHEState(
                    players=new_players,
                    board=self.board.copy(),
                    current_street=self.current_street,
                    history=new_history,
                    is_chance=True  # Proceed to chance node
                )

        return NLHEState(
            players=new_players,
            board=self.board.copy(),
            current_street=self.current_street,
            history=new_history,
            is_chance=False
        )

    def utility(self):
        all_actions = [a for actions in self.history.values() for a in actions]
        pot = self.pot
        if any(a[1] == "FOLD" for a in all_actions):
            last_action = all_actions[-1]
            folded_player = last_action[0]
            winner = 1 - folded_player
            return (pot, -pot) if winner == 0 else (-pot, pot)

        if self.players[0]["all_in"] and self.players[1]["all_in"] and self.current_street != "RIVER":
            equity = calculate_equity(self.players[0]["hole_cards"], self.players[1]["hole_cards"], self.board)
            return (pot * (2 * equity - 1), -pot * (2 * equity - 1))

        result = compare_hands(self.players[0]["hole_cards"] + self.board, self.players[1]["hole_cards"] + self.board)
        if result == "Player 1":
            return (pot, -pot)
        elif result == "Player 2":
            return (-pot, pot)
        else:
            return (0, 0)

    def information_set(self):
        current_player_index = self.current_player()
        if current_player_index is None:
            return "EOS"

        canonical = canonicalize_hand(self.players[current_player_index]["hole_cards"], self.board)
        hole = canonical["hole"]

        tokens = [
            "BOS",
            "BTN" if current_player_index == 0 else "BB",
            hole,
            "STACK_SIZE", *list(str(self.players[current_player_index]["stack"])),
            "SB_SIZE", *list(str(SB_AMOUNT)),
            "BB_SIZE", *list(str(BB_AMOUNT))
        ]

        streets = ["PREFLOP", "FLOP", "TURN", "RIVER"]
        current_street_index = streets.index(self.current_street)
        for street in streets[:current_street_index + 1]:
            if not self.history[street] and street != self.current_street:
                continue
            tokens.append(street)

            if street == "PREFLOP":
                tokens += ["BTN", "POST", *list(str(SB_AMOUNT)), "BB", "POST", *list(str(BB_AMOUNT))]
            else:
                pot = sum(p["contribution"] for p in self.players.values())
                pot_tokens = list(str(int(pot)))
                tokens += ["POT_SIZE", *pot_tokens, *canonical[street.lower()].split(" ")]

            for action in self.history[street]:
                position = "BTN" if action[0] == 0 else "BB"
                act = action[1]
                amt = list(str(int(action[2]))) if action[2] > 0 else []
                tokens += [position, act] + amt

        tokens.append("EOS")
        return " ".join(tokens)

    def _copy_players(self):
        return {
            i: {
                "stack": p["stack"],
                "contribution": p["contribution"],
                "hole_cards": p["hole_cards"].copy(),
                "all_in": p["all_in"]
            } for i, p in self.players.items()
        }

# ---------------------------------------------------------------------
# Training and Evaluation
# ---------------------------------------------------------------------

input_vocabulary =  [
    # Special tokens
    "PAD", "BOS", "EOS",
    # Ranks
    "RANK_2", "RANK_3", "RANK_4", "RANK_5", "RANK_6", 
    "RANK_7", "RANK_8", "RANK_9", "RANK_T", "RANK_J", 
    "RANK_K", "RANK_Q", "RANK_A",
    # Suits
    "SUIT_1", "SUIT_2", "SUIT_3", "SUIT_4",
    # Positions
    "BTN", "BB",
    # Actions
    "FOLD", "CHECK", "POST", "CALL", "BET_20", "BET_100", "ALL_IN",
    # Streets and sizes
    "PREFLOP", "FLOP", "TURN", "RIVER",
    "STACK_SIZE", "POT_SIZE", "SB_SIZE", "BB_SIZE",
    # Numeric tokens
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
]

output_decisions = ['FOLD', 'CHECK', 'CALL', 'BET_20', 'BET_100', 'ALL_IN']

def create_info_set_for_hand(hand):
    """Create information set string for a given hand."""
    if hand[0] == hand[1]:  # Pair
        info_set = f"BOS BTN RANK_{hand[0]} SUIT_1 RANK_{hand[0]} SUIT_2"
    elif hand.endswith('s'):  # Suited
        info_set = f"BOS BTN RANK_{hand[0]} SUIT_1 RANK_{hand[1]} SUIT_1"
    else:  # Offsuit
        info_set = f"BOS BTN RANK_{hand[0]} SUIT_1 RANK_{hand[1]} SUIT_2"
    
    return info_set + f" STACK_SIZE 200 SB_SIZE 1 BB_SIZE 2 PREFLOP BTN POST 1 BB POST 2 EOS"

def print_hand_state(state):
    """Print the current state of the hand."""
    def print_cards(cards):
        return ', '.join(str(card.rank + card.suit[0].lower()) for card in cards)
    
    print(f"\nInitial stacks: BTN: {state.players[0]['stack']}, BB: {state.players[1]['stack']}")
    print(f"BTN cards: {print_cards(sorted(state.players[0]['hole_cards'], key=lambda card: -rank_value[card.rank]))}")
    print(f"BB cards: {print_cards(sorted(state.players[1]['hole_cards'], key=lambda card: -rank_value[card.rank]))}")
    print("\nPREFLOP")

def play_example_hand(deep_solver):
    """Play and print an example hand."""
    state = NLHEState()
    print_hand_state(state)

    while not state.is_terminal():
        if state.is_chance():
            state = state.next_state_from_chance()
            if state.current_street != "PREFLOP":
                print(f"\n{state.current_street}: {' '.join(str(card) for card in state.board)}")
                print(f"Pot: {state.pot}")
        else:
            current_player = "BTN" if state.current_player() == 0 else "BB"
            strategy = deep_solver.get_average_strategy(state.information_set(), state.legal_player_actions())
            action = max(strategy.items(), key=lambda x: x[1])[0]
            
            # Display amount based on action type
            if action == "CALL":
                print(f"{current_player} {action} {state.current_bet}")
            elif action.startswith("BET"):
                bet_amount = int(0.2 * state.pot) if action == "BET_20" else state.pot
                print(f"{current_player} {action} ({bet_amount})")
            elif action == "ALL_IN":
                all_in_amount = state.players[state.current_player()]["stack"]
                print(f"{current_player} {action} ({all_in_amount})")
            else:
                print(f"{current_player} {action}")
                
            state = state.next_state_from_action(action)

    print("\nFinal outcome:")
    utility = state.utility()
    print(f"BTN profit: {utility[0]}")
    print(f"BB profit: {utility[1]}")

def generate_preflop_grid(deep_solver):
    """Generate and plot the preflop raising frequency grid."""
    grid = np.zeros((13, 13))
    ranks = [f"RANK_{r}" for r in rank_order]
    
    for i, r1 in enumerate(ranks):
        for j, r2 in enumerate(ranks[::-1]):
            suit2 = "SUIT_1" if i < j else "SUIT_2"
            info_set = f"BOS BTN {r1} SUIT_1 {r2} {suit2} STACK_SIZE 200 SB_SIZE {SB_AMOUNT} BB_SIZE {BB_AMOUNT} PREFLOP BTN POST {SB_AMOUNT} BB POST {BB_AMOUNT} BTN RAISE_100 EOS"
            strategy = deep_solver.get_average_strategy(info_set, ["BET_100", "CALL", "FOLD"])
            grid[i, j] = strategy.get("BET_100", 0)

    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap='viridis', origin='lower', vmin=0, vmax=1)
    plt.colorbar(label='Raise Frequency')
    plt.xticks(ticks=range(13), labels=rank_order[::-1])
    plt.yticks(ticks=range(13), labels=rank_order)
    plt.title("BTN Raising Frequency Grid (Preflop)")
    plt.xlabel("Card 2")
    plt.ylabel("Card 1")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Initialize model
    deep_solver = MonteCarloDeepCFRSolver(
        root_state=NLHEState(),
        input_vocabulary=input_vocabulary,
        output_decisions=output_decisions,
        max_input_length=50
    )

    # # Load existing model or start fresh
    # try:
    #     deep_solver.load("models/nlhe")
    # except Exception:
    #     print("No existing model found, starting fresh training")

    # # Training loop
    # sessions, iterations = 1, 5
    # traversals_per_iter, batch_size, epochs = 1000, 256, 5
    
    # start_time = time.time()
    # for session in range(sessions):
    #     print(f"\nTraining Session {session + 1}/{sessions}")
    #     session_start = time.time()
        
    #     deep_solver.train(iterations, traversals_per_iter, batch_size, epochs)
    #     deep_solver.save("models/nlhe")
        
    #     session_time = time.time() - session_start
    #     print(f"Session completed in {session_time:.1f} seconds")

    #     # Evaluate test hands
    #     test_hands = ["AA", "KK", "JTs", "55", "72o"]
    #     print("\nEvaluating preflop strategies:")
    #     for hand in test_hands:
    #         info_set = create_info_set_for_hand(hand)
    #         strategy = deep_solver.get_average_strategy(info_set, ["CALL", "FOLD", "BET_100"])
    #         print(f"\nHand: {hand}")
    #         print("Strategy:", {k: f"{v:.3f}" for k, v in strategy.items()})

    # total_time = time.time() - start_time
    # print(f"\nTotal training time: {total_time:.1f} seconds")
    
    # Run examples
    print("\nPlaying example hand:")
    play_example_hand(deep_solver)
    generate_preflop_grid(deep_solver)
