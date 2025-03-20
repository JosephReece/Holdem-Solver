import random

rank_order = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

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

class Deck:
    def __init__(self, cards = None):
        self.suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        self.ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        self.initialize_deck(cards)
        self.shuffle()
    
    def initialize_deck(self, cards):
        if cards:
            self.cards = cards
        else:
            self.cards = []
            for suit in self.suits:
                for rank in self.ranks:
                    self.cards.append(Card(rank, suit))
    
    def shuffle(self):
        random.shuffle(self.cards)
    
    def draw_card(self):
        if len(self.cards) == 0:
            raise Exception("No cards left in the deck.")
        return self.cards.pop()
    
    def draw_cards(self, count):
        drawn_cards = []
        for _ in range(count):
            drawn_cards.append(self.draw_card())
        return drawn_cards

# Functions to evaluate hand types
def is_flush(hand):
    return all(card.suit == hand[0].suit for card in hand)

def is_straight(hand):
    rank_indices = sorted([rank_order.index(card.rank) for card in hand])
    
    # Check for wheel straight (A-2-3-4-5)
    if rank_indices == [0, 1, 2, 3, 12]:
        return True
    
    # Check for regular straight
    for i in range(1, len(rank_indices)):
        if rank_indices[i] != rank_indices[i - 1] + 1:
            return False
    return True

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

# Helper function to compare high cards
def compare_high_card(hand1, hand2):
    ranks1 = sorted([rank_order.index(card.rank) for card in hand1], reverse=True)
    ranks2 = sorted([rank_order.index(card.rank) for card in hand2], reverse=True)
    
    for i in range(len(ranks1)):
        if ranks1[i] != ranks2[i]:
            return "Player 1" if ranks1[i] > ranks2[i] else "Player 2"
    
    return "Tie"

# Extract kickers from a hand based on a filter function
def get_kickers(hand, filter_func):
    return [card for card in hand if filter_func(card)]

# Check if a hand is a wheel straight (A-2-3-4-5)
def is_wheel_straight(hand):
    rank_indices = sorted([rank_order.index(card.rank) for card in hand])
    return rank_indices == [0, 1, 2, 3, 12]

# Special comparison for straights, handling the wheel as the lowest straight
def compare_straights(hand1, hand2):
    is_wheel1 = is_wheel_straight(hand1)
    is_wheel2 = is_wheel_straight(hand2)
    
    # If one is a wheel and the other isn't, the wheel loses
    if is_wheel1 and not is_wheel2:
        return "Player 2"
    if not is_wheel1 and is_wheel2:
        return "Player 1"
    
    # If both are wheels or neither are wheels, compare normally
    return compare_high_card(hand1, hand2)

# Define comparison functions for each hand type
def compare_straight_flush(hand1, hand2):
    return compare_straights(hand1, hand2)

def compare_four_of_a_kind(hand1, hand2):
    counts1 = get_rank_counts(hand1)
    counts2 = get_rank_counts(hand2)
    
    # Get the rank with count 4
    four_of_a_kind_rank1 = next(rank for rank, count in counts1.items() if count == 4)
    four_of_a_kind_rank2 = next(rank for rank, count in counts2.items() if count == 4)
    
    rank1_index = rank_order.index(four_of_a_kind_rank1)
    rank2_index = rank_order.index(four_of_a_kind_rank2)
    
    if rank1_index != rank2_index:
        return "Player 1" if rank1_index > rank2_index else "Player 2"
    
    # If four of a kind ranks are the same, compare the kicker
    kickers1 = get_kickers(hand1, lambda card: card.rank != four_of_a_kind_rank1)
    kickers2 = get_kickers(hand2, lambda card: card.rank != four_of_a_kind_rank2)
    
    return compare_high_card(kickers1, kickers2)

def compare_full_house(hand1, hand2):
    counts1 = get_rank_counts(hand1)
    counts2 = get_rank_counts(hand2)
    
    # Get the rank with count 3
    three_of_a_kind_rank1 = next(rank for rank, count in counts1.items() if count == 3)
    three_of_a_kind_rank2 = next(rank for rank, count in counts2.items() if count == 3)
    
    rank1_index = rank_order.index(three_of_a_kind_rank1)
    rank2_index = rank_order.index(three_of_a_kind_rank2)
    
    if rank1_index != rank2_index:
        return "Player 1" if rank1_index > rank2_index else "Player 2"
    
    # If three of a kind ranks are the same, compare the pair
    pair_rank1 = next(rank for rank, count in counts1.items() if count == 2)
    pair_rank2 = next(rank for rank, count in counts2.items() if count == 2)
    
    pair_rank1_index = rank_order.index(pair_rank1)
    pair_rank2_index = rank_order.index(pair_rank2)
    
    if pair_rank1_index != pair_rank2_index:
        return "Player 1" if pair_rank1_index > pair_rank2_index else "Player 2"
    
    return "Tie"

def compare_flush(hand1, hand2):
    return compare_high_card(hand1, hand2)

def compare_straight(hand1, hand2):
    return compare_straights(hand1, hand2)

def compare_three_of_a_kind(hand1, hand2):
    counts1 = get_rank_counts(hand1)
    counts2 = get_rank_counts(hand2)
    
    # Get the rank with count 3
    three_of_a_kind_rank1 = next(rank for rank, count in counts1.items() if count == 3)
    three_of_a_kind_rank2 = next(rank for rank, count in counts2.items() if count == 3)
    
    rank1_index = rank_order.index(three_of_a_kind_rank1)
    rank2_index = rank_order.index(three_of_a_kind_rank2)
    
    if rank1_index != rank2_index:
        return "Player 1" if rank1_index > rank2_index else "Player 2"
    
    # If three of a kind ranks are the same, compare the kickers
    kickers1 = get_kickers(hand1, lambda card: card.rank != three_of_a_kind_rank1)
    kickers2 = get_kickers(hand2, lambda card: card.rank != three_of_a_kind_rank2)
    
    return compare_high_card(kickers1, kickers2)

def compare_two_pair(hand1, hand2):
    counts1 = get_rank_counts(hand1)
    counts2 = get_rank_counts(hand2)
    
    # Get the ranks with count 2
    pair_ranks1 = sorted([rank_order.index(rank) for rank, count in counts1.items() if count == 2], reverse=True)
    pair_ranks2 = sorted([rank_order.index(rank) for rank, count in counts2.items() if count == 2], reverse=True)
    
    # Compare the higher pair
    if pair_ranks1[0] != pair_ranks2[0]:
        return "Player 1" if pair_ranks1[0] > pair_ranks2[0] else "Player 2"
    
    # Compare the lower pair
    if pair_ranks1[1] != pair_ranks2[1]:
        return "Player 1" if pair_ranks1[1] > pair_ranks2[1] else "Player 2"
    
    # If both pairs are the same, compare the kicker
    high_pair_rank1 = rank_order[pair_ranks1[0]]
    low_pair_rank1 = rank_order[pair_ranks1[1]]
    high_pair_rank2 = rank_order[pair_ranks2[0]]
    low_pair_rank2 = rank_order[pair_ranks2[1]]
    
    kickers1 = get_kickers(hand1, lambda card: card.rank != high_pair_rank1 and card.rank != low_pair_rank1)
    kickers2 = get_kickers(hand2, lambda card: card.rank != high_pair_rank2 and card.rank != low_pair_rank2)
    
    return compare_high_card(kickers1, kickers2)

def compare_one_pair(hand1, hand2):
    counts1 = get_rank_counts(hand1)
    counts2 = get_rank_counts(hand2)
    
    # Get the rank with count 2
    pair_rank1 = next(rank for rank, count in counts1.items() if count == 2)
    pair_rank2 = next(rank for rank, count in counts2.items() if count == 2)
    
    rank1_index = rank_order.index(pair_rank1)
    rank2_index = rank_order.index(pair_rank2)
    
    if rank1_index != rank2_index:
        return "Player 1" if rank1_index > rank2_index else "Player 2"
    
    # If pair ranks are the same, compare the kickers
    kickers1 = get_kickers(hand1, lambda card: card.rank != pair_rank1)
    kickers2 = get_kickers(hand2, lambda card: card.rank != pair_rank2)
    
    return compare_high_card(kickers1, kickers2)

def compare_high_card_hand(hand1, hand2):
    return compare_high_card(hand1, hand2)

# Map hand types to their comparison functions
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

def get_all_combinations(arr, k):
    results = []
    
    def combination(start, chosen):
        if len(chosen) == k:
            results.append(chosen.copy())
            return
        for i in range(start, len(arr)):
            chosen.append(arr[i])
            combination(i + 1, chosen)
            chosen.pop()
    
    combination(0, [])
    return results

# Main function to compare two 5 card hands
def compare_5_card_hands(hand1, hand2):
    hand_type1 = get_hand_type(hand1)
    hand_type2 = get_hand_type(hand2)
    
    rank1 = hand_ranks[hand_type1]
    rank2 = hand_ranks[hand_type2]
    
    if rank1 != rank2:
        return "Player 1" if rank1 > rank2 else "Player 2"
    
    # If the hand types are the same, use the appropriate comparison function
    return hand_type_comparisons[hand_type1](hand1, hand2)

def compare_hands(hand1, hand2):
    combinations1 = get_all_combinations(hand1, 5)
    combinations2 = get_all_combinations(hand2, 5)
    
    def find_best_hand(combinations):
        best_hand = combinations[0]
        for combo in combinations:
            if compare_5_card_hands(combo, best_hand) == "Player 1":
                best_hand = combo
        return best_hand

    best_hand1 = find_best_hand(combinations1)
    best_hand2 = find_best_hand(combinations2)
    
    return compare_5_card_hands(best_hand1, best_hand2)

