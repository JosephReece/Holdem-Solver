import random
from itertools import combinations

rank_order = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
rank_value = {rank: i for i, rank in enumerate(rank_order)}

suits = ['Hearts', 'Spades', 'Clubs', 'Diamonds']

hand_ranks = {
    'High Card': 1,
    'One Pair': 2,
    'Two Pair': 3,
    'Three of a Kind': 4,
    'Straight': 5,
    'Flush': 6,
    'Full House': 7,
    'Four of a Kind': 8,
    'Straight Flush': 9,
}

class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit

    def __str__(self):
        return f"{self.rank} of {self.suit}"

    def __eq__(self, other):
        return self.rank == other.rank and self.suit == other.suit

    def __hash__(self):
        return hash((self.rank, self.suit))

class Deck:
    def __init__(self, without=[]):
        self.initialize_deck(without)
        self.shuffle()

    def initialize_deck(self, without):
        self.cards = [Card(rank, suit) for suit in suits for rank in rank_order if Card(rank, suit) not in without]

    def shuffle(self):
        random.shuffle(self.cards)

    def draw_card(self):
        if len(self.cards) == 0:
            raise Exception("No cards left in the deck.")
        return self.cards.pop()

    def draw_cards(self, count):
        return [self.draw_card() for _ in range(count)]

# Hand evaluation functions
def is_flush(hand):
    return all(card.suit == hand[0].suit for card in hand)

def is_straight(hand):
    rank_indices = sorted([rank_order.index(card.rank) for card in hand])
    if rank_indices == [0, 1, 2, 3, 12]:  # Wheel straight
        return True
    return all(rank_indices[i] == rank_indices[i - 1] + 1 for i in range(1, len(rank_indices)))

def get_rank_counts(hand):
    counts = {}
    for card in hand:
        counts[card.rank] = counts.get(card.rank, 0) + 1
    return counts

def get_hand_type(hand):
    is_hand_flush = is_flush(hand)
    is_hand_straight = is_straight(hand)

    if is_hand_straight and is_hand_flush:
        return 'Straight Flush'

    rank_counts = get_rank_counts(hand)
    counts = sorted(rank_counts.values(), reverse=True)

    if counts[0] == 4:
        return 'Four of a Kind'
    if counts[0] == 3 and counts[1] == 2:
        return 'Full House'
    if is_hand_flush:
        return 'Flush'
    if is_hand_straight:
        return 'Straight'
    if counts[0] == 3:
        return 'Three of a Kind'
    if counts[0] == 2 and counts[1] == 2:
        return 'Two Pair'
    if counts[0] == 2:
        return 'One Pair'

    return 'High Card'

def compare_high_card(hand1, hand2):
    ranks1 = sorted([rank_order.index(card.rank) for card in hand1], reverse=True)
    ranks2 = sorted([rank_order.index(card.rank) for card in hand2], reverse=True)
    for i in range(len(ranks1)):
        if ranks1[i] != ranks2[i]:
            return "Player 1" if ranks1[i] > ranks2[i] else "Player 2"
    return "Tie"

def get_kickers(hand, filter_func):
    return [card for card in hand if filter_func(card)]

def is_wheel_straight(hand):
    rank_indices = sorted([rank_order.index(card.rank) for card in hand])
    return rank_indices == [0, 1, 2, 3, 12]

def compare_straights(hand1, hand2):
    is_wheel1 = is_wheel_straight(hand1)
    is_wheel2 = is_wheel_straight(hand2)
    if is_wheel1 and not is_wheel2:
        return "Player 2"
    if not is_wheel1 and is_wheel2:
        return "Player 1"
    return compare_high_card(hand1, hand2)

def compare_straight_flush(hand1, hand2):
    return compare_straights(hand1, hand2)

def compare_four_of_a_kind(hand1, hand2):
    counts1 = get_rank_counts(hand1)
    counts2 = get_rank_counts(hand2)
    r1 = next(rank_order.index(rank) for rank, count in counts1.items() if count == 4)
    r2 = next(rank_order.index(rank) for rank, count in counts2.items() if count == 4)
    if r1 != r2:
        return "Player 1" if r1 > r2 else "Player 2"
    return compare_high_card(
        get_kickers(hand1, lambda c: counts1[c.rank] != 4),
        get_kickers(hand2, lambda c: counts2[c.rank] != 4)
    )

def compare_full_house(hand1, hand2):
    counts1 = get_rank_counts(hand1)
    counts2 = get_rank_counts(hand2)
    t1 = next(rank_order.index(rank) for rank, count in counts1.items() if count == 3)
    t2 = next(rank_order.index(rank) for rank, count in counts2.items() if count == 3)
    if t1 != t2:
        return "Player 1" if t1 > t2 else "Player 2"
    p1 = next(rank_order.index(rank) for rank, count in counts1.items() if count == 2)
    p2 = next(rank_order.index(rank) for rank, count in counts2.items() if count == 2)
    return "Player 1" if p1 > p2 else "Player 2" if p1 < p2 else "Tie"

def compare_flush(hand1, hand2): return compare_high_card(hand1, hand2)
def compare_straight(hand1, hand2): return compare_straights(hand1, hand2)
def compare_three_of_a_kind(hand1, hand2):
    counts1, counts2 = get_rank_counts(hand1), get_rank_counts(hand2)
    t1 = next(rank_order.index(rank) for rank, count in counts1.items() if count == 3)
    t2 = next(rank_order.index(rank) for rank, count in counts2.items() if count == 3)
    if t1 != t2:
        return "Player 1" if t1 > t2 else "Player 2"
    return compare_high_card(
        get_kickers(hand1, lambda c: counts1[c.rank] != 3),
        get_kickers(hand2, lambda c: counts2[c.rank] != 3)
    )

def compare_two_pair(hand1, hand2):
    counts1, counts2 = get_rank_counts(hand1), get_rank_counts(hand2)
    p1 = sorted([rank_order.index(rank) for rank, count in counts1.items() if count == 2], reverse=True)
    p2 = sorted([rank_order.index(rank) for rank, count in counts2.items() if count == 2], reverse=True)
    if p1[0] != p2[0]: return "Player 1" if p1[0] > p2[0] else "Player 2"
    if p1[1] != p2[1]: return "Player 1" if p1[1] > p2[1] else "Player 2"
    return compare_high_card(
        get_kickers(hand1, lambda c: rank_order.index(c.rank) not in p1),
        get_kickers(hand2, lambda c: rank_order.index(c.rank) not in p2)
    )

def compare_one_pair(hand1, hand2):
    counts1, counts2 = get_rank_counts(hand1), get_rank_counts(hand2)
    p1 = next(rank_order.index(rank) for rank, count in counts1.items() if count == 2)
    p2 = next(rank_order.index(rank) for rank, count in counts2.items() if count == 2)
    if p1 != p2: return "Player 1" if p1 > p2 else "Player 2"
    return compare_high_card(
        get_kickers(hand1, lambda c: rank_order.index(c.rank) != p1),
        get_kickers(hand2, lambda c: rank_order.index(c.rank) != p2)
    )

def compare_high_card_hand(hand1, hand2): return compare_high_card(hand1, hand2)

hand_type_comparisons = {
    'Straight Flush': compare_straight_flush,
    'Four of a Kind': compare_four_of_a_kind,
    'Full House': compare_full_house,
    'Flush': compare_flush,
    'Straight': compare_straight,
    'Three of a Kind': compare_three_of_a_kind,
    'Two Pair': compare_two_pair,
    'One Pair': compare_one_pair,
    'High Card': compare_high_card_hand
}

def compare_5_card_hands(hand1, hand2):
    type1, type2 = get_hand_type(hand1), get_hand_type(hand2)
    r1, r2 = hand_ranks[type1], hand_ranks[type2]
    if r1 != r2:
        return "Player 1" if r1 > r2 else "Player 2"
    return hand_type_comparisons[type1](hand1, hand2)

def compare_hands(hand1, hand2):
    best1 = max(list(combinations(hand1, 5)), key=lambda h: (hand_ranks[get_hand_type(h)], sorted([rank_order.index(c.rank) for c in h], reverse=True)))
    best2 = max(list(combinations(hand2, 5)), key=lambda h: (hand_ranks[get_hand_type(h)], sorted([rank_order.index(c.rank) for c in h], reverse=True)))
    return compare_5_card_hands(best1, best2)

# Equity for hole1
def calculate_equity(hole1, hole2, board):
    wins = 0
    ties = 0
    remaining_cards = [Card(rank, suit) for suit in suits for rank in rank_order if Card(rank, suit) not in hole1 + hole2 + board]
    total_runs = 1000

    for _ in range(total_runs):
        sim_board = board.copy()
        sim_remaining = remaining_cards.copy()
        random.shuffle(sim_remaining)
        
        # Complete the board if needed
        cards_needed = 5 - len(board)
        if cards_needed > 0:
            sim_board.extend(sim_remaining[:cards_needed])
        
        result = compare_hands(hole1 + sim_board, hole2 + sim_board)
        
        if result == "Player 1":
            wins += 1
        elif result == "Tie":
            ties += 1

    return (wins + ties/2) / total_runs

def canonicalize_hand(hole_cards, board_cards):
    # Split board safely into flop, turn, river
    flop = board_cards[:3]
    turn = board_cards[3:4] if len(board_cards) > 3 else []
    river = board_cards[4:5] if len(board_cards) > 4 else []

    # Sort hole and flop cards by rank
    def sort_by_rank(cards):
        return sorted(cards, key=lambda c: rank_value[c.rank], reverse=True)

    sorted_hole = sort_by_rank(hole_cards)
    sorted_flop = sort_by_rank(flop)

    all_cards = sorted_hole + sorted_flop + turn + river

    # Suit mapping
    suit_map = {}
    next_suit_id = 1

    def map_suit(suit):
        nonlocal next_suit_id
        if suit not in suit_map:
            suit_map[suit] = next_suit_id
            next_suit_id += 1
        return suit_map[suit]

    canonical_strings = [
        f"{card.rank}{map_suit(card.suit)}" for card in all_cards
    ]

    return ",".join(canonical_strings)