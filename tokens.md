# Streamlined GTO No-Limit Hold'em Transformer Token Set

## Card Representation Tokens
### Rank Tokens (13)
- `RANK_2`, `RANK_3`, `RANK_4`, `RANK_5`, `RANK_6`, `RANK_7`, `RANK_8`, `RANK_9`, `RANK_T`, `RANK_J`, `RANK_Q`, `RANK_K`, `RANK_A`

### Suit Tokens (4)
- `SUIT_C` (clubs), `SUIT_D` (diamonds), `SUIT_H` (hearts), `SUIT_S` (spades)

## Player Position Tokens
- `BTN` (Button)
- `BB` (Big Blind)

## Action Tokens
- `FOLD`
- `CHECK`
- `CALL`
- `BET`
- `RAISE`
- `ALL_IN`

## Stack Size Tokens (in octal)
- `STACK_SIZE`
- Individual octal digits: `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`
- Examples:
  - Stack of 100 (decimal) = `144` (octal)
  - Stack of 1000 (decimal) = `1750` (octal)

## Pot Size Tokens
- `POT_SIZE`
- Uses same octal digits as stack sizes

## Bet Size Tokens (in octal)
- Uses same octal digits as stack sizes

## Game State Tokens
- `PREFLOP`
- `FLOP`
- `TURN`
- `RIVER`

## Special Tokens
- `EOS` (End of sequence)
- `BOS` (Beginning of sequence)
- `SDM` (Start decision making - when betting or raising a size is needed so multiple tokens are needed)

## Outputs
- `FOLD`
- `PASSIVE_ACTION` (Check or Call)
- `START_AGGRESSIVE_ACTION` (Bet or Raise)
- `END_AGGRESSIVE_ACTION`
- `ALL_IN`
- `OCTAL_0`
- `OCTAL_1`
- `OCTAL_2`
- `OCTAL_3`
- `OCTAL_4`
- `OCTAL_5`
- `OCTAL_6`
- `OCTAL_7`

<br />
<br />
<br />

# SDM usage

The hero (BTN) has started raising over the previous bet from the BB. The raise is starting with (or simply is) the octal digit 6 (eg 6, 64 or 671)
BOS BTN ... BB BET 2 SDM 6

# Example Hand

## Implied Information (No tokens)
Blinds are 1/2

## General Information
Start the sequence, describe hero position, hero cards and effective stack size

`BOS BTN RANK_A SUIT_H RANK_A SUIT_C STACK_SIZE 1 4 4`

## Preflop
Show alternating action with position and then the action (and an octal value if the action requires)

`PREFLOP BB CHECK BTN BET 5 BB CALL`

## Flop
Describe pot size and community cards, then action

`FLOP POT_SIZE 1 2 RANK_3 SUIT_C RANK_9 SUIT_C RANK_J SUIT_S BB CHECK BTN BET 4 BB CALL`

## Turn

`TURN POT_SIZE 2 2 RANK_4 SUIT_S BB CHECK BTN BET 7 BB RAISE 3 1 BTN CALL`

## River
End the sequence at the end

`RIVER POT_SIZE 1 0 4 RANK_T SUIT_S BB CHECK BTN CHECK EOS`