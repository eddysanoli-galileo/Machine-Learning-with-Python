"""Tabular QL agent"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple
from . import framework
from . import utils

DEBUG = False

# ============================================== #
# SETTINGS                                       #
# ============================================== #

GAMMA = 0.5                 # discounted factor

# Epsilon
TRAINING_EP = 0.5           # epsilon-greedy parameter for training
TESTING_EP = 0.05           # epsilon-greedy parameter for testing

# Runs, epochs and episodes
NUM_RUNS = 10
NUM_EPOCHS = 200
NUM_EPIS_TRAIN = 25         # number of episodes for training at each epoch
NUM_EPIS_TEST = 50          # number of episodes for testing

# Learning rate
ALPHA = 0.1                 # learning rate for training

# Actions and object
ACTIONS = framework.get_actions()
OBJECTS = framework.get_objects()
NUM_ACTIONS = len(ACTIONS)
NUM_OBJECTS = len(OBJECTS)


# ============================================== #
# FUNCTION: EPSILON GREEDY                       #
# ============================================== #

def epsilon_greedy(
    state_1 : int,
    state_2 : int,
    q_func : np.ndarray, 
    epsilon : float
) -> Tuple[int, int]:
    """
    Returns an action selected by an epsilon-Greedy exploration policy

    Args:
        state_1 (int): Index describing the current room (Sr)
        state_2 (int): Index describing the current quest (Sq)
        q_func (np.ndarray): current Q-function (4D array)
        epsilon (float): the probability of choosing a random command

    Returns:
        (int, int): the indices describing the action/object to take (a, b) where 
        "a" is the action and "b" is the object on which to act upon.
    """

    # Variable aliases
    sr : int = state_1
    sq : int = state_2

    # We generate a random number between 0 and 1. If that number is smaller than (1- epsilon)
    # we exploit. In any other case, we explore.
    random_number : float = np.random.random_sample()
    greedy : bool = (random_number < (1-epsilon))

    # Exploit (Greedy action):
    # Take the best possible action according to the current policy Pi(s)
    if greedy:

        # All possible actions for a given state S (2D array)
        q_s : np.ndarray = q_func[sr, sq, :, :]

        # Greedy action (Action C (a,b) that maximizes the Q value)
        # NOTE: "argmax" flattens the 2D array and returns the linear index that corresponds
        # to the highest value in that 2D array. We use "unravel_index" to convert that linear
        # index back to a 2D index
        pi_s = np.unravel_index(np.argmax(q_s), q_s.shape)

        # We extract the action and object from the policy for the current state (Pi(s))
        a, b = pi_s

    # Explore (Non-greedy action):
    # Take a random action from all the possible ones
    else:

        # Random action
        _, _, NUM_ACTIONS, NUM_OBJECTS = q_func.shape

        # Select a random action and object
        a : int = np.random.randint(NUM_ACTIONS)
        b : int = np.random.randint(NUM_OBJECTS)
     
    # Return a tuple "C" of an action and an object
    return (a, b)


# ============================================== #
# FUNCTION: TABULAR Q LEARNING                   #
# ============================================== #

def tabular_q_learning(
    q_func : np.ndarray, 
    current_state_1 : int, current_state_2 : int, 
    action_index : int,
    object_index : int, 
    reward : float, 
    next_state_1 : int, next_state_2 : int,
    terminal : bool
) -> None:
    """
    Update q_func for a given transition

    Args:
        q_func (np.ndarray): current Q-function
        current_state_1 (int): index of the current room (S_r)
        current_state_2 (int): index of the current quest (S_q)
        action_index (int): index of the current action
        object_index (int): index of the current object
        reward (float): the immediate reward the agent receives from playing current command
        next_state_1 (int): index describing the next room (S'_r) 
        next_state_2 (int,): index describing the next quest (S'_q)
        terminal (bool): True if this episode is over

    Returns:
        None
    """

    # Shorter names for states and actions
    sr : int = current_state_1
    sq : int = current_state_2
    a : int = action_index
    b : int = object_index
    sr_prime : int = next_state_1
    sq_prime : int = next_state_2

    # This is the last step in the episode there are no "future events"
    if terminal:

        # We set the max value for all future actions to zero, since there are no more
        # actions after this
        max_Q_next = 0
    
    else: 

        # Get the maximum Q value that we can get from taking one of the C' possible
        # actions (a, b)
        max_Q_next = np.max(q_func[sr_prime, sq_prime, :, :])
    
    # Update according to the Q-learning algorithm
    # - q_func[sr, sq, a, b] : scalar
    # - reward : scalar
    # - max_Q : scalar
    q_update = (1 - ALPHA) * q_func[sr, sq, a, b] + (ALPHA) * (reward + GAMMA * max_Q_next)

    # Q(s, c) = updated Q(s, c)
    # - s = (sr, sq)
    #   sr: (current_state_1) Current room
    #   sq: (current_state_2) Current quest
    # - c = (a, b)
    #   a: (action_index) Action to execute
    #   b: (object_index) Object that will "receive" the action to execute (ie. eat APPLE)
    q_func[current_state_1, current_state_2, action_index, object_index] = q_update

    # This function shouldn't return anything
    return None  



# ============================================== #
# FUNCTION: RUN EPISODE                          #
# ============================================== #


# pragma: coderesponse template
def run_episode(for_training, dict_room_desc, dict_quest_desc):
    """ Runs one episode
    If for training, update Q function
    If for testing, computes and return cumulative discounted reward

    Args:
        for_training (bool): True if for training

    Returns:
        None
    """
    # Use a different epsilon depending on whether the current run is for training or testing
    epsilon = TRAINING_EP if for_training else TESTING_EP

    # The reward for the current episode starts at 0
    episode_reward = 0

    # Discount
    # NOTE: Remember that in each episode the power of gamma increases by one.
    # This means that it starts with a power of zero (gamma^0 = 1) and then in each
    # step it increases by one (gamma^1 = gamma, gamma^2 = gamma^2, ...)
    episode_gamma = 1

    # Initialize a new game, we get back:
    # - Description of the initial room
    # - A description of the quest for this episode
    # - A variable indicating if the game is done (Initially false to indicate that its starting)
    (current_room_desc, current_quest_desc, terminal) = framework.newGame()


    while not terminal:

        # Map from a description (Sr, Sq) to an index (ir, iq) like the ones used previously
        ir : int = dict_room_desc[current_room_desc]
        iq : int = dict_quest_desc[current_quest_desc]

        # Choose an action using the epsilon-greedy method
        action_index, object_index = epsilon_greedy(ir, iq, q_func, epsilon)

        # Aliases for the action and object indexes
        a = action_index
        b = object_index

        # Run a step of the game. This returns:
        # - Future room description (S'r)
        # - Future quest description (S'q)
        # - Reward (R)
        # - Terminal: Whether the game has finished or not
        next_room_desc, next_quest_desc, reward, terminal = framework.step_game(current_room_desc, current_quest_desc, a, b)

        # Map from next state descriptions to indices
        ir_prime : int = dict_room_desc[next_room_desc]
        iq_prime : int = dict_quest_desc[next_quest_desc]

        # Aliases for states
        sr = ir
        sq = iq
        sr_prime = ir_prime
        sq_prime = iq_prime

        if for_training:

            # Update the Q function
            # NOTE: The update gets done "in place" meaning that we dont have to return anything
            # the Q function get updated inside the function
            tabular_q_learning(q_func, sr, sq, a, b, reward, sr_prime, sq_prime, terminal)

        if not for_training:
            
            # Calculate the discounted reward
            episode_reward += reward * episode_gamma

            # Increase the power of the gamma used for next episode
            episode_gamma *= GAMMA

        # The current "next_step" will consist of the "current_step" in the
        # next pass of the while loop
        current_room_desc = next_room_desc
        current_quest_desc = next_quest_desc

    if not for_training:
        return episode_reward


def run_epoch(dict_room_desc, dict_quest_desc):
    """Runs one epoch and returns reward averaged over test episodes"""
    rewards = []

    for _ in range(NUM_EPIS_TRAIN):
        run_episode(True, dict_room_desc, dict_quest_desc)

    for _ in range(NUM_EPIS_TEST):
        rewards.append(run_episode(False, dict_room_desc, dict_quest_desc))

    return np.mean(np.array(rewards))


def run(NUM_ROOM_DESC, NUM_QUESTS, dict_room_desc, dict_quest_desc):
    """Returns array of test reward per epoch for one run"""
    global q_func
    q_func = np.zeros((NUM_ROOM_DESC, NUM_QUESTS, NUM_ACTIONS, NUM_OBJECTS))

    single_run_epoch_rewards_test = []
    pbar = tqdm(range(NUM_EPOCHS), ncols=80)
    for _ in pbar:
        single_run_epoch_rewards_test.append(run_epoch(dict_room_desc, dict_quest_desc))
        pbar.set_description(
            "Avg reward: {:0.6f} | Ewma reward: {:0.6f}".format(
                np.mean(single_run_epoch_rewards_test),
                utils.ewma(single_run_epoch_rewards_test)))
    return single_run_epoch_rewards_test


if __name__ == '__main__':
    # Data loading and build the dictionaries that use unique index for each state
    (dict_room_desc, dict_quest_desc) = framework.make_all_states_index()
    NUM_ROOM_DESC = len(dict_room_desc)
    NUM_QUESTS = len(dict_quest_desc)

    # set up the game
    framework.load_game_data()

    epoch_rewards_test = []  # shape NUM_RUNS * NUM_EPOCHS

    for _ in range(NUM_RUNS):
        epoch_rewards_test.append(run())

    epoch_rewards_test = np.array(epoch_rewards_test)

    x = np.arange(NUM_EPOCHS)
    fig, axis = plt.subplots()
    axis.plot(x, np.mean(epoch_rewards_test,
                         axis=0))  # plot reward per epoch averaged per run
    axis.set_xlabel('Epochs')
    axis.set_ylabel('reward')
    axis.set_title(('Tablular: nRuns=%d, Epilon=%.2f, Epi=%d, alpha=%.4f' %
                    (NUM_RUNS, TRAINING_EP, NUM_EPIS_TRAIN, ALPHA)))
    plt.show()
