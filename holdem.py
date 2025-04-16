"""
holdem.py

NLHE Heads-up (100BBs deep, 1/2) deep MCCFR solver that maintains player information
in a list of dictionaries. Each player dictionary contains:
  - "stack": the current chip count (in BBs)
  - "cards": a list of Card objects representing the hole cards (empty if not yet dealt)
  - "all_in": a boolean indicating if the player is all in

This version also tokenizes the game history (stored as a dictionary for each betting street)
into a string via the information_set() method following these tokenization rules:

  • General Information:
      BOS <HERO_POSITION> <HERO_CARD1> <HERO_CARD2> STACK_SIZE <stack digits> SB_SIZE <sb digits> BB_SIZE <bb digits>
  • PREFLOP:
      PREFLOP <action tokens>  (e.g., BTN POST 1, BB POST 2, BTN RAISE 5, BB CALL 5)
  • FLOP:
      FLOP POT_SIZE <pot digits> <3 board cards> <flop action tokens>
  • TURN:
      TURN POT_SIZE <pot digits> <turn card> <turn action tokens>
  • RIVER:
      RIVER POT_SIZE <pot digits> <river card> <river action tokens> EOS

The deep solver (MonteCarloDeepCFRSolver) is used exclusively.
Dependencies from card.py (Card, Deck, rank_order, suits, hand_ranks, compare_hands, calculate_equity, canonicalize_hand)
and the CFR module (GameState, MonteCarloDeepCFRSolver) are assumed to be available.
"""

import numpy as np
import matplotlib.pyplot as plt
from cfr import GameState, MonteCarloDeepCFRSolver
from card import Card, Deck, rank_order, suits, hand_ranks, compare_hands, calculate_equity, canonicalize_hand

# ---------------------------------------------------------------------
# Token definitions
# ---------------------------------------------------------------------

# Input vocabulary tokens
vocabulary = [
    # Special tokens
    "PAD",  # Padding Token
    "BOS",  # Beginning of sequence
    "EOS",  # End of sequence
    
    # Rank tokens (13)
    "RANK_2", "RANK_3", "RANK_4", "RANK_5", "RANK_6", 
    "RANK_7", "RANK_8", "RANK_9", "RANK_T", "RANK_J", 
    "RANK_K", "RANK_Q", "RANK_A",
    
    # Suit tokens (4)
    "SUIT_1", "SUIT_2", "SUIT_3", "SUIT_4",
    
    # Position tokens
    "BTN",  # Button
    "BB",   # Big Blind
    
    # Action tokens
    "FOLD",
    "CHECK",
    "POST",
    "CALL",
    "BET",
    "RAISE",
    "ALL_IN",  # Use this instead of call, bet and raise when a player goes all in
    
    # Game state tokens
    "PREFLOP",
    "FLOP",
    "TURN",
    "RIVER",
    
    # Sizing tokens
    "STACK_SIZE",
    "POT_SIZE",
    "SB_SIZE",
    "BB_SIZE",
    
    # Numeric tokens
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
]

output_decisions = [
    'FOLD',
    'PASSIVE_ACTION',  # Represents CALL or CHECK (decided by context)
    'BET_20',          # In preflop: a pot–sized raise; later: a 20% pot bet
    'BET_100',         # 100% pot bet (only later when no bet yet on street)
    'ALL_IN'
]

# ---------------------------------------------------------------------
# NLHEState Class (using players as a list of dictionaries)
# ---------------------------------------------------------------------
class NLHEState(GameState):
    def __init__(self, phase="CHANCE_HOLE", players=None, board_cards=None,
                 pot=0, history=None, bet_count=0, current_street=None,
                 next_player=0, sb=1, bb=2):
        """
        phase: current phase (e.g., "PREFLOP", "FLOP", "TURN", "RIVER", "TERMINAL")
        players: list of dictionaries, one per player; each dict has keys "stack", "cards", and "all_in"
                 If not provided, initializes two players with 100BB each, no cards, not all in.
        board_cards: list of Card objects on the board
        pot: current pot size
        history: dictionary with keys "PREFLOP", "FLOP", "TURN", "RIVER" (each a list of tokenized action strings)
        bet_count: number of raises on the current street
        current_street: betting round indicator (e.g., "PREFLOP", "FLOP", etc.)
        next_player: index (0 for BTN, 1 for BB) of the player to act
        sb, bb: blind sizes
        """
        self.phase = phase
        self.players = players if players is not None else [
            {"stack": 100.0, "cards": [], "all_in": False},
            {"stack": 100.0, "cards": [], "all_in": False}
        ]
        self.board_cards = board_cards if board_cards is not None else []
        self.pot = pot
        self.history = history if history is not None else {"PREFLOP": [], "FLOP": [], "TURN": [], "RIVER": []}
        self.bet_count = bet_count
        self.current_street = current_street  # e.g., "PREFLOP"
        self.next_player = next_player
        self.sb = sb
        self.bb = bb

    def is_terminal(self):
        return self.phase == "TERMINAL"

    def current_player(self):
        if self.is_terminal() or self.phase.startswith("CHANCE"):
            return None
        return self.next_player

    def legal_chance_outcomes(self):
        if self.phase == "CHANCE_HOLE":
            return []  # Sampling used in practice
        elif self.phase in ["CHANCE_FLOP", "CHANCE_TURN", "CHANCE_RIVER"]:
            return []
        return []

    def legal_player_actions(self):
        actions = ["FOLD", "PASSIVE_ACTION"]
        if self.current_street == "PREFLOP":
            if self.bet_count == 0:
                actions.append("BET_20")
            else:
                if self.bet_count < 4:
                    actions.append("BET_20")
                else:
                    actions.append("ALL_IN")
        else:
            if not self.history[self.current_street]:
                actions.append("BET_20")
                actions.append("BET_100")
            else:
                actions.append("ALL_IN")
        return actions

    def record_action(self, action):
        """
        Converts the internal action to a token string and records it in the history for the current street.
        For illustration (with dummy bet amounts):
          - "FOLD" -> "FOLD"
          - "PASSIVE_ACTION" -> "CHECK" if no bet exists, else "CALL 5"
          - "BET_20" -> "RAISE 5" in preflop; "BET 4" later
          - "BET_100" -> "BET 8"
          - "ALL_IN" -> "ALL_IN"
        The acting player's position (BTN/BB) is prepended.
        """
        pos = "BTN" if self.next_player == 0 else "BB"
        if action == "FOLD":
            token_str = "FOLD"
        elif action == "PASSIVE_ACTION":
            token_str = "CALL 5" if self.history[self.current_street] else "CHECK"
        elif action == "BET_20":
            token_str = "RAISE 5" if self.current_street == "PREFLOP" else "BET 4"
        elif action == "BET_100":
            token_str = "BET 8"
        elif action == "ALL_IN":
            token_str = "ALL_IN"
        else:
            token_str = action

        tokenized_action = pos + " " + token_str
        if self.current_street in self.history:
            self.history[self.current_street].append(tokenized_action)

    def next_state(self, action):
        """
        Returns a new state after applying the given action.
          - Records the tokenized action in the appropriate street.
          - Updates players' stacks, the pot, and bet counts.
          - Advances the phase/round if appropriate.
        """
        # Create a new state with a deep copy of players and history.
        new_state = NLHEState(
            phase=self.phase,
            players=[p.copy() for p in self.players],
            board_cards=self.board_cards.copy(),
            pot=self.pot,
            history={k: v.copy() for k, v in self.history.items()},
            bet_count=self.bet_count,
            current_street=self.current_street,
            next_player=self.next_player,
            sb=self.sb,
            bb=self.bb
        )
        cp = self.current_player()
        if new_state.current_street is not None:
            new_state.record_action(action)

        if action == "FOLD":
            new_state.phase = "TERMINAL"
            return new_state

        elif action == "PASSIVE_ACTION":
            call_amount = 5 if new_state.history[new_state.current_street] and "CALL" in new_state.history[new_state.current_street][-1] else 0
            new_state.players[cp]['stack'] -= call_amount
            new_state.pot += call_amount

        elif action in ["BET_20", "BET_100"]:
            bet_size = new_state.pot if self.current_street == "PREFLOP" else (0.2 * new_state.pot if action == "BET_20" else new_state.pot)
            new_state.players[cp]['stack'] -= bet_size
            new_state.pot += bet_size
            new_state.bet_count += 1

        elif action == "ALL_IN":
            all_in_amount = new_state.players[cp]['stack']
            new_state.pot += all_in_amount
            new_state.players[cp]['stack'] = 0
            new_state.players[cp]['all_in'] = True

        # If both players are all in, end the game.
        if all(player['all_in'] for player in new_state.players):
            new_state.phase = "TERMINAL"
            return new_state

        # Advance phase if enough actions have been taken.
        total_actions = sum(len(v) for v in new_state.history.values())
        if total_actions >= 2:
            if self.current_street == "PREFLOP":
                new_state.phase = "CHANCE_FLOP"
                new_state.current_street = "FLOP"
                deck = Deck()
                new_state.board_cards = deck.draw_cards(3)
                new_state.bet_count = 0
            elif self.current_street == "FLOP":
                new_state.phase = "CHANCE_TURN"
                new_state.current_street = "TURN"
                deck = Deck()
                new_state.board_cards += deck.draw_cards(1)
                new_state.bet_count = 0
            elif self.current_street == "TURN":
                new_state.phase = "CHANCE_RIVER"
                new_state.current_street = "RIVER"
                deck = Deck()
                new_state.board_cards += deck.draw_cards(1)
                new_state.bet_count = 0
            elif self.current_street == "RIVER":
                new_state.phase = "TERMINAL"

        if new_state.current_player() is not None:
            new_state.next_player = 1 - cp

        return new_state

    def information_set(self):
        """
        Returns the tokenized string representing all information available to the acting player.
        The tokenization includes:
          • General Info: BOS <HERO_POSITION> <HERO_CARD1> <HERO_CARD2> STACK_SIZE <stack digits> SB_SIZE <sb digits> BB_SIZE <bb digits>
          • PREFLOP: the preflop action tokens (if any)
          • FLOP: "FLOP POT_SIZE <pot> <3 board cards> <flop action tokens>"
          • TURN: "TURN POT_SIZE <pot> <turn card> <turn action tokens>"
          • RIVER: "RIVER POT_SIZE <pot> <river card> <river action tokens> EOS"
        """
        cp = self.current_player()
        if cp is None:
            return ""
        pos = "BTN" if cp == 0 else "BB"
        tokens = []
        tokens.append("BOS")
        tokens.append(pos)
        # Get hero's cards from the players list.
        hero_cards = self.players[cp]['cards']
        for card in hero_cards:
            tokens.append(f"RANK_{card.rank}")
            tokens.append(card.suit)
        eff_stack = str(int(self.players[cp]['stack']))
        tokens.append("STACK_SIZE")
        tokens.extend(list(eff_stack))
        tokens.append("SB_SIZE")
        tokens.extend(list(str(self.sb)))
        tokens.append("BB_SIZE")
        tokens.extend(list(str(self.bb)))

        # PREFLOP section.
        if self.history["PREFLOP"]:
            tokens.append("PREFLOP")
            tokens.extend(" ".join(self.history["PREFLOP"]).split())

        # FLOP section.
        if (self.current_street in ["FLOP", "TURN", "RIVER"]) or self.history["FLOP"]:
            tokens.append("FLOP")
            tokens.append("POT_SIZE")
            tokens.extend(list(str(int(self.pot))))
            if len(self.board_cards) >= 3:
                for card in self.board_cards[:3]:
                    tokens.append(f"RANK_{card.rank}")
                    tokens.append(card.suit)
            tokens.extend(" ".join(self.history["FLOP"]).split())

        # TURN section.
        if (self.current_street in ["TURN", "RIVER"]) or self.history["TURN"]:
            if len(self.board_cards) >= 4:
                tokens.append("TURN")
                tokens.append("POT_SIZE")
                tokens.extend(list(str(int(self.pot))))
                card = self.board_cards[3]
                tokens.append(f"RANK_{card.rank}")
                tokens.append(card.suit)
            tokens.extend(" ".join(self.history["TURN"]).split())

        # RIVER section.
        if self.current_street == "RIVER" or self.history["RIVER"]:
            if len(self.board_cards) >= 5:
                tokens.append("RIVER")
                tokens.append("POT_SIZE")
                tokens.extend(list(str(int(self.pot))))
                card = self.board_cards[4]
                tokens.append(f"RANK_{card.rank}")
                tokens.append(card.suit)
            tokens.extend(" ".join(self.history["RIVER"]).split())
            tokens.append("EOS")
        else:
            tokens.append("EOS")

        return " ".join(tokens)

    def utility(self):
        """
        Computes terminal utilities based on:
          - A fold action (the folder loses the pot)
          - Both players all-in (using calculate_equity)
          - Otherwise, showdown outcome via compare_hands.
        """
        for street in self.history:
            for act in self.history[street]:
                if "FOLD" in act:
                    # Assume the player who folded loses.
                    folded_player = 1 - self.next_player
                    util = [0, 0]
                    util[1 - folded_player] = self.pot
                    util[folded_player] = -self.pot
                    return tuple(util)

        if all(player['all_in'] for player in self.players):
            equity = calculate_equity(self.players[0]['cards'], self.players[1]['cards'], self.board_cards)
            util0 = equity * self.pot
            util1 = (1 - equity) * self.pot
            return (util0, util1)

        if len(self.board_cards) >= 3:
            result = compare_hands(self.players[0]['cards'] + self.board_cards,
                                   self.players[1]['cards'] + self.board_cards)
            if result == "Player 0":
                return (self.pot, -self.pot)
            elif result == "Player 1":
                return (-self.pot, self.pot)
            else:
                return (0, 0)
        return (0, 0)


# ---------------------------------------------------------------------
# Deep MCCFR Training and Evaluation (Deep-Only)
# ---------------------------------------------------------------------
if __name__ == '__main__':
    input_vocabulary = vocabulary
    max_input_length = 40  # Adjust based on tokenized sequence length

    # Create a deep solver root state, starting in the PREFLOP phase.
    # The players list now holds each player's stack, cards, and all_in status.
    deep_root = NLHEState(
        phase="PREFLOP",
        current_street="PREFLOP",
        players=[{"stack": 100.0, "cards": [], "all_in": False}, {"stack": 100.0, "cards": [], "all_in": False}]
    )
    deep_solver = MonteCarloDeepCFRSolver(
        root_state=deep_root,
        input_vocabulary=input_vocabulary,
        output_decisions=output_decisions,
        max_input_length=max_input_length
    )
    try:
        deep_solver.load("models")
    except Exception as e:
        print("No existing deep model found, starting fresh training")

    # Set training parameters.
    sessions = 1
    iterations = 1
    traversals_per_iter = 10
    batch_size = 256
    epochs = 10

    for session in range(sessions):
        print(f"\nTraining Session {session + 1}/{sessions}")
        deep_solver.train(iterations, traversals_per_iter, batch_size, epochs)
        deep_solver.save("models")
        
        # Evaluate strategy on sample information sets.
        infosets = [
            "RANK_A SUIT_1 RANK_K SUIT_2 PREFLOP",  # e.g., strong starting hands
            "RANK_7 SUIT_1 RANK_2 SUIT_2 PREFLOP",  # e.g., weak starting hands
            "RANK_Q SUIT_1 RANK_J SUIT_2 PREFLOP"
        ]
        print("\nStrategy Evaluation after Session {}:".format(session + 1))
        for info_set in infosets:
            legal = ["FOLD", "PASSIVE_ACTION", "BET_20"]
            strategy = deep_solver.get_average_strategy(info_set, legal)
            print(f"InfoSet: {info_set:<40} | Strategy: " +
                  ", ".join(f"{a}={strategy.get(a, 0):.2f}" for a in legal))
    
    # --- Example: BTN Preflop Decision Grid ---
    hand_grid = np.zeros((13, 13), dtype=int)
    # For each hand, assign: 0 = Fold, 1 = Call/Check, 2 = Raise.
    for i, r1 in enumerate(rank_order):
        for j, r2 in enumerate(rank_order):
            if i == j:
                # For pairs, use two different suits.
                hole_cards = [Card(f"RANK_{r1}", "SUIT_1"), Card(f"RANK_{r1}", "SUIT_2")]
            elif i > j:
                # Lower triangle: suited.
                hole_cards = [Card(f"RANK_{r1}", "SUIT_1"), Card(f"RANK_{r2}", "SUIT_1")]
            else:
                # Upper triangle: offsuit.
                hole_cards = [Card(f"RANK_{r1}", "SUIT_1"), Card(f"RANK_{r2}", "SUIT_2")]

            state = NLHEState(
                phase="PREFLOP",
                players=[
                    {"stack": 100.0, "cards": hole_cards, "all_in": False},
                    {"stack": 100.0, "cards": [], "all_in": False}
                ],
                board_cards=[],
                pot=1.5,
                history={"PREFLOP": [], "FLOP": [], "TURN": [], "RIVER": []},
                bet_count=0,
                current_street="PREFLOP",
                next_player=0,
                sb=1,
                bb=2
            )
            info_set = state.information_set()
            legal = ["FOLD", "PASSIVE_ACTION", "BET_20"]
            strat = deep_solver.get_average_strategy(info_set, legal)
            best_action = max(legal, key=lambda a: strat.get(a, 0))
            if best_action == "FOLD":
                hand_grid[i, j] = 0
            elif best_action == "PASSIVE_ACTION":
                hand_grid[i, j] = 1
            else:
                hand_grid[i, j] = 2

    plt.figure(figsize=(8, 8))
    cmap = plt.cm.get_cmap("RdYlGn", 3)
    im = plt.imshow(hand_grid, cmap=cmap, origin='lower')
    plt.xticks(np.arange(13), rank_order)
    plt.yticks(np.arange(13), rank_order)
    plt.xlabel("Second Card")
    plt.ylabel("First Card")
    plt.title("BTN Preflop Decision Grid\n0=Fold (Red), 1=Call/Check (Yellow), 2=Raise (Green)")
    for i in range(13):
        for j in range(13):
            action_letter = {0: "F", 1: "C", 2: "R"}.get(hand_grid[i, j], "")
            plt.text(j, i, action_letter, ha="center", va="center", color="black", fontsize=12)
    plt.colorbar(im, ticks=[0, 1, 2])
    plt.savefig("btn_decision_grid.png")
    plt.show()

    print("\nTraining Statistics:")
    print("Sessions:", sessions)
    print("Iterations per session:", iterations)
    print("Traversals per iteration:", traversals_per_iter)
    print("Batch size:", batch_size)
    print("Epochs per iteration:", epochs)