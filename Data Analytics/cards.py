import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Card and Deck Classes
class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit

    def __repr__(self):
        return f"{self.rank} of {self.suit}"

class Deck:
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    suits = ['Clubs', 'Diamonds', 'Hearts', 'Spades']

    def __init__(self, num_decks=1):
        self.num_decks = num_decks
        print("Generating deck")
        self.cards = self.create_deck()
        self.shuffle()

    def create_deck(self):
        return [Card(rank, suit) for rank in self.ranks for suit in self.suits] * self.num_decks

    def shuffle(self):
        print("Shuffling deck")
        random.shuffle(self.cards)

    def deal(self, num_cards):
        print(f"Dealing {num_cards} cards")
        dealt_cards = self.cards[:num_cards]
        self.cards = self.cards[num_cards:]
        return dealt_cards

    def reshuffle(self, r_min, r_max):
        remaining_cards = len(self.cards)
        reshuffle_chance = 100 * (r_max - remaining_cards) / (r_max - r_min)
        if random.randint(0, 100) < reshuffle_chance:
            print("Reshuffling deck")
            self.cards = self.create_deck()
            self.shuffle()

# Player Class
class Player:
    def __init__(self, name):
        self.name = name
        self.hand = []
        self.current_bet = 0
        self.total_bet = 0

    def receive_cards(self, cards):
        print(f"{self.name} receives cards: {cards}")
        self.hand.extend(cards)

    def bet(self, amount):
        print(f"{self.name} bets {amount}")
        self.current_bet += amount
        self.total_bet += amount

    def reset_bet(self):
        self.current_bet = 0

    def discard_and_draw(self, deck, discard_indices):
        for index in sorted(discard_indices, reverse=True):
            self.hand.pop(index)
        print(f"{self.name} discards {len(discard_indices)} cards and draws new ones")
        self.hand.extend(deck.deal(len(discard_indices)))

# Game Class
class Game:
    def __init__(self, num_players, ante, small_blind, big_blind, num_decks, reshuffle_min, reshuffle_max, model):
        self.num_players = num_players
        self.ante = ante
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.players = [Player(f"Player {i+1}") for i in range(num_players)]
        self.deck = Deck(num_decks)
        self.pot = 0
        self.reshuffle_min = reshuffle_min
        self.reshuffle_max = reshuffle_max
        self.model = model
        self.states = []
        self.actions = []
        self.rewards = []

    def initial_deal(self):
        print("Initial deal")
        for player in self.players:
            player.receive_cards(self.deck.deal(5))
            player.bet(self.ante)
            self.pot += self.ante

    def betting_round(self, starting_player_index):
        print("Starting betting round")
        max_bet = 0
        for i in range(starting_player_index, starting_player_index + self.num_players):
            player = self.players[i % self.num_players]
            action = self.get_player_action(player, max_bet)
            if action == 0:  # Fold
                print(f"{player.name} folds")
                self.players.remove(player)
                reward = -player.current_bet
                self.store_transition(player, action, reward)
            elif action == 1:  # Call
                call_amount = max_bet - player.current_bet
                player.bet(call_amount)
                self.pot += call_amount
                reward = 0
                self.store_transition(player, action, reward)
            elif action == 2:  # Raise
                raise_amount = self.get_raise_amount(player, max_bet)
                max_bet += raise_amount
                player.bet(max_bet - player.current_bet)
                self.pot += (max_bet - player.current_bet)
                reward = raise_amount
                self.store_transition(player, action, reward)

    def get_player_action(self, player, max_bet):
        state = encode_game_state(player, max_bet, self.pot, self.deck.cards)
        action_probs = self.model.predict(state.reshape(1, -1))
        action = np.argmax(action_probs)
        print(f"{player.name} takes action {['Fold', 'Call', 'Raise'][action]}")
        self.states.append(state)
        self.actions.append(action)
        return action

    def get_raise_amount(self, player, max_bet):
        state = encode_game_state(player, max_bet, self.pot, self.deck.cards)
        raise_amount = self.model.predict(state.reshape(1, -1))[0][3]  # Assuming the raise amount is the 4th output
        print(f"{player.name} raises by {int(raise_amount * 10)}")
        return int(raise_amount * 10)  # Scale appropriately

    def draw_phase(self):
        print("Starting draw phase")
        for player in self.players:
            discard_indices = self.get_discard_indices(player)
            player.discard_and_draw(self.deck, discard_indices)
        self.deck.reshuffle(self.reshuffle_min, self.reshuffle_max)

    def get_discard_indices(self, player):
        state = encode_game_state(player, 0, self.pot, self.deck.cards)
        discard_probs = self.model.predict(state.reshape(1, -1))
        num_discards = np.argmax(discard_probs)
        print(f"{player.name} decides to discard {num_discards} cards")
        self.states.append(state)
        self.actions.append(num_discards)
        return random.sample(range(5), num_discards)

    def determine_winner(self):
        print("Determining winner")
        ranked_hands = sorted(self.players, key=lambda p: self.rank_hand(p.hand), reverse=True)
        winner = ranked_hands[0]
        reward = self.pot
        self.store_transition(winner, -1, reward)  # -1 indicates the winning action
        return winner

    def rank_hand(self, hand):
        values = '23456789TJQKA'
        value_dict = {v: i for i, v in enumerate(values)}
        suits = {'Clubs': 0, 'Diamonds': 1, 'Hearts': 2, 'Spades': 3}

        def hand_rank(hand):
            ranks = sorted([value_dict[card.rank] for card in hand], reverse=True)
            suits = [card.suit for card in hand]
            if len(set(suits)) == 1:  # Flush
                if ranks == [12, 11, 10, 9, 8]:  # Royal Flush
                    return (10,)
                elif ranks == list(range(ranks[0], ranks[0] - 5, -1)):  # Straight Flush
                    return (9, ranks[0])
                return (6, ranks)
            if ranks.count(ranks[0]) == 4 or ranks.count(ranks[1]) == 4:  # Four of a Kind
                return (8, ranks[0], ranks[-1])
            if ranks.count(ranks[0]) == 3 and ranks.count(ranks[3]) == 2:  # Full House
                return (7, ranks[0], ranks[3])
            if ranks == list(range(ranks[0], ranks[0] - 5, -1)):  # Straight
                return (5, ranks[0])
            if ranks.count(ranks[0]) == 3 or ranks.count(ranks[2]) == 3:  # Three of a Kind
                return (4, ranks[0], ranks[3:])
            if len(set(ranks[:2])) == 1 and len(set(ranks[2:4])) == 1:  # Two Pair
                return (3, ranks[0], ranks[2], ranks[-1])
            if ranks.count(ranks[0]) == 2 or ranks.count(ranks[1]) == 2 or ranks.count(ranks[2]) == 2:  # One Pair
                return (2, ranks[0], ranks[2:])
            return (1, ranks)  # High Card

        return hand_rank(hand)

    def store_transition(self, player, action, reward):
        index = len(self.states) - 1
        self.rewards.append(reward)
        self.actions[index] = action

    def play_game(self):
        print("Starting a new game")
        self.initial_deal()
        self.betting_round(0)
        self.draw_phase()
        self.betting_round(0)
        winner = self.determine_winner()
        print(f"The winner is {winner.name} with hand {winner.hand} and pot {self.pot}")

# Neural Network for Decision Making
def create_model(input_shape):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))  # Actions: Fold, Call, Raise, Discard Count
    return model

def encode_game_state(player, max_bet, pot, remaining_deck):
    state = np.zeros(60)  # Example state size
    for i, card in enumerate(player.hand):
        state[i] = card.rank
    state[50] = player.current_bet
    state[51] = player.total_bet
    state[52] = max_bet
    state[53] = pot
    state[54] = len(remaining_deck)
    return state

# Custom callback to print loss and accuracy during training
class TrainingCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch+1}: Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}")

# Training the Model
num_games = 10000
input_shape = (60,)
model = create_model(input_shape)

states = []
actions = []
rewards = []

# Simulate games for training
for _ in range(num_games):
    game = Game(num_players=5, ante=10, small_blind=5, big_blind=10, num_decks=1, reshuffle_min=10, reshuffle_max=20, model=model)
    game.play_game()
    states.extend(game.states)
    actions.extend(game.actions)
    rewards.extend(game.rewards)

# Prepare training data
X_train = np.array(states)
y_train = np.zeros((len(actions), 4))

for i in range(len(actions)):
    y_train[i][actions[i]] = rewards[i]

# Train the model
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, callbacks=[TrainingCallback()])

# Save the model
model.save('poker_neural_network.h5')

# Example game simulation with trained model
game = Game(num_players=5, ante=10, small_blind=5, big_blind=10, num_decks=1, reshuffle_min=10, reshuffle_max=20, model=model)
game.play_game()
