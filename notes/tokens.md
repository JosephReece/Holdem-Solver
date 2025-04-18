# No-Limit Hold'em Transformer Token Set

## Card Representation Tokens
### Rank Tokens (13)
- `RANK_2`, `RANK_3`, `RANK_4`, `RANK_5`, `RANK_6`, `RANK_7`, `RANK_8`, `RANK_9`, `RANK_T`, `RANK_J`, `RANK_Q`, `RANK_K`, `RANK_A`

### Suit Tokens (4)
- `SUIT_1`, `SUIT_2`, `SUIT_3`, `SUIT_4`

## Player Position Tokens
- `BTN` (Button)
- `BB` (Big Blind)

## Action Tokens
- `FOLD`
- `CHECK`
- `POST`
- `CALL`
- `BET`
- `RAISE`
- `ALL_IN`

## Blind Size Tokens
- `SB_SIZE`
- `BB_SIZE`
- Individual digits: `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`

## Stack Size Tokens
- `STACK_SIZE`
- Uses same digits as stack sizes

## Pot Size Tokens
- `POT_SIZE`
- Uses same digits as stack sizes

## Bet Size Tokens
- Uses same digits as stack sizes

## Game State Tokens
- `PREFLOP`
- `FLOP`
- `TURN`
- `RIVER`

## Special Tokens
- `EOS` (End of sequence)
- `BOS` (Beginning of sequence)

## Outputs
- `FOLD`
- `CHECK`
- `CALL`
- `BET_20` (Bet or Raise 20% of pot)
- `BET_100`
- `ALL_IN`

# Example Hand
It's worth noting that when the sequence doesn't end on the river (eg halfway through the flop), it will still end with `EOS`

## General Information
Start the sequence, describe hero position (eg BTN), hero cards (eg Ah, As), effective stack size (eg 200) and blinds (eg 1/2).

`BOS BTN RANK_A SUIT_1 RANK_A SUIT_2 STACK_SIZE 2 0 0 SB_SIZE 1 BB_SIZE 2`

## Preflop
Show alternating action with position and then the action and a value if the action requires (eg BTN RAISE 5)

`PREFLOP BTN POST 1 BB POST 2 BTN RAISE 5 BB CALL 5`

## Flop
Describe pot size (eg 10) and community cards (eg 3h 9h Js), then action

`FLOP POT_SIZE 1 0 RANK_3 SUIT_1 RANK_9 SUIT_1 RANK_J SUIT_2 BB CHECK BTN BET 4 BB CALL 4`

## Turn
Describe pot size (eg 18) and community cards (eg 4c), then action

`TURN POT_SIZE 1 8 RANK_4 SUIT_3 BB CHECK BTN BET 7 BB RAISE 2 2 BTN CALL 2 2`

## River
Describe pot size (eg 62) and community cards (eg Tc), then action, then end sequence (EOS)

`RIVER POT_SIZE 6 2 RANK_T SUIT_3 BB CHECK BTN CHECK EOS`