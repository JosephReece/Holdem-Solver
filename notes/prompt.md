Using my class MonteCarloDeepCFRSolver, create a solver for NLHE heads up 100BBs deep (1/2). Use these tokens and use these decisions at each node

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
    "SUIT_1",
    "SUIT_2",
    "SUIT_3",
    "SUIT_4",
    
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
    "ALL_IN", # Use this instead of call, bet and raise when a player goes all in
    
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

# Outputs for the network 
output_decisions = [
    'FOLD',
    'PASSIVE_ACTION', #Check or Call
    'BET_20', #Bet 20% of pot, or raise 20% of the pot
    'BET_100',
    'ALL_IN'
]

Include preflop, flop, turn and river decisions.  When players are both all in, use calculate_equity for regrets.

To reduce complexity, the following abstractions are in place.
Preflop: Players may only 5-bet as an all in or fold. Each raise must always be pot-sized.

So the longest line would be: BTN Call, BB Raise, BTN 3-Bet,  BB 4-Bet, BTN All-in.
But another line could be: BTN Raise, BB Call

On the flop, turn and river, players may only 3-Bet as the an all-in or fold to a riase. So on the flop, turn or river a line could be: BB Bet, BTN Bet, BB Raise, BTN All-In. But it could also just be something like: BB Check, BTN Bet, BB Call.

Save the strategies at regular intervals. Provide some general statistics at the end of the training and also provide a classic coloured grid of raising first in, calling or folding for the button.

I have the following code you should use from card.py:
from card import (
    Card,   # Represents a single playing card with .rank and .suit
    Deck,   # shuffle(), draw_card(), draw_cards(count), shuffles on init, also has the parameter without=[] for constructor (option to exclude some cards)
    rank_order,  # List of card ranks ['2', ..., '10', 'J', 'Q', 'K', 'A']
    suits,       # List of card suits ['Hearts', 'Spades', 'Clubs', 'Diamonds']
    hand_ranks,  # Dict mapping poker hands to strength values, 1 is a high card, 9 is a straight flush eg 'High Card' -> 1
    compare_hands,     # Compares two poker hands to determine winner, takes hand1 and hand2, len(hand) >= 5, returns "Player 1", "Player 2" or "Tie"
    calculate_equity,  # Hand equity calculation, takes hole1, hole2 and board, returns hole1's equity
    canonicalize_hand(hole_cards, board_cards), # Converts a poker hand into a dictionary each with canonical token string representation for eg {'turn': 'RANK_A SUIT_1'}
)