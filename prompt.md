Create a basic CFR solver for NLHE heads up 100BBs deep (1/2). Use these decisions at each node

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

Save the generated strategy in a json file and load it if it exists before training. To evaluate it, provide some general statistics at the end of the training and also provide a classic coloured grid of raising first in, calling or folding for the button. (Red is raise, green is call, grey is fold).

I have the following code you should use from card.py:
from card import (
    Card,   # Represents a single playing card with rank and suit
    Deck,   # shuffle(), draw_card(), draw_cards(count), shuffles on init, also has the parameter without=[] for constructor (option to exclude some cards)
    rank_order,  # List of card ranks ['2'-'10', 'J', 'Q', 'K', 'A']
    suits,       # List of card suits ['Hearts', 'Spades', 'Clubs', 'Diamonds']
    hand_ranks,  # Dict mapping poker hands to strength values, 1 is a high card, 9 is a straight flush
    compare_hands,     # Compares two poker hands to determine winner, takes hand1 and hand2, len(hand) >= 5
    calculate_equity,  # Hand equity calculation, takes hole1, hole2 and board
)