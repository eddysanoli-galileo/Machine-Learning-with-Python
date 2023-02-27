"""Linear QL agent"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from . import framework
from . import utils
from typing import Tuple

DEBUG = False


GAMMA = 0.5             # discounted factor
TRAINING_EP = 0.5       # epsilon-greedy parameter for training
TESTING_EP = 0.05       # epsilon-greedy parameter for testing
NUM_RUNS = 5
NUM_EPOCHS = 600
NUM_EPIS_TRAIN = 25     # number of episodes for training at each epoch
NUM_EPIS_TEST = 50      # number of episodes for testing
ALPHA = 0.001           # learning rate for training

ACTIONS = framework.get_actions()
OBJECTS = framework.get_objects()
NUM_ACTIONS = len(ACTIONS)
NUM_OBJECTS = len(OBJECTS)


def tuple2index(action_index, object_index):
    """Converts a tuple (a,b) to an index c"""
    return action_index * NUM_OBJECTS + object_index


def index2tuple(index):
    """Converts an index c to a tuple (a,b)"""
    return index // NUM_OBJECTS, index % NUM_OBJECTS


# ============================================== #
# EPSILON GREEDY                                 #
# ============================================== #

def epsilon_greedy(
    state_vector: np.ndarray, 
    theta: np.ndarray, 
    epsilon: float
) -> Tuple[int, int]:
    """
    Returns an action selected by an epsilon-greedy exploration policy

    Args:
        state_vector (np.ndarray): Vector representation of the state Psi_r(s)
        theta (np.ndarray): Matrix weights that help compute the weighted sum of
        the "state_vector"
        epsilon (float): The probability of choosing a random command

    Returns:
        (int, int): Indices describing the action/object (a, b) to take
    """
    
    # We generate a random number between 0 and 1. If that number is smaller than (1- epsilon)
    # we exploit. In any other case, we explore.
    random_number : float = np.random.random_sample()
    greedy : bool = (random_number < (1-epsilon))
    
    # Exploit (Greedy action):
    # Take the best possible action according to the current policy Pi(s)
    if greedy:

        # We get the values for the Q function approximation
        # Shape: (C, S) x (S, 1) = (S, 1)
        #   S: Number of states in new Psi(s) representation
        #   C: Number of available actions (a * b = actions * objects)
        q_value : np.ndarray = (theta @ state_vector)

        # Greedy action (Action C (a,b) that maximizes the Q value)
        # NOTE: "argmax" flattens the 2D array and returns the linear index that corresponds
        # to the highest value in that 2D array. However, since the "q_value" is basically a
        # column vector, this flattening does not matter
        pi_s = int(np.argmax(q_value))

        # We need to convert the index of the column array that contains all
        # possible permutations of "a" and "b" (C), into a tuple of indexes that
        # will index a new square matrix of size (a, b). This will help us extract
        # the index of "a" and "b", getting the action and object in the process
        a, b = index2tuple(pi_s)

    # Explore (Non-greedy action):
    # Take a random action from all the possible ones
    else:

        # Select a random action and object
        a : int = np.random.randint(NUM_ACTIONS)
        b : int = np.random.randint(NUM_OBJECTS)
     
    # Return a tuple "C" composed of both an action and an object
    return (a, b)

# ============================================== #
# LINEAR Q LEARNING                              #
# ============================================== #

def linear_q_learning(
    theta : np.ndarray, 
    current_state_vector : np.ndarray, 
    action_index : int, 
    object_index : int,
    reward : float, 
    next_state_vector : np.ndarray, 
    terminal : bool
) -> None:
    """
    Update theta for a given transition

    Args:
        theta (np.ndarray): current weight matrix
        current_state_vector (np.ndarray): vector representation of current state
        action_index (int): index of the current action
        object_index (int): index of the current object
        reward (float): the immediate reward the agent receives from playing current command
        next_state_vector (np.ndarray): vector representation of next state
        terminal (bool): True if this episode is over

    Returns:
        None
    """

    # This is the last step in the episode there are no "future events"
    if terminal:

        # We set the max value for all future actions to zero, since there are no more
        # actions after this
        max_Q_next = 0
    
    else:

        # We build the Q function by using the next state vector
        q_values_next : np.ndarray = theta @ next_state_vector

        # Get the maximum Q value that we can get from taking one of the C' possible
        # actions (a, b)
        max_Q_next = np.max(q_values_next)


    # Current Q-function
    q_values_current : np.ndarray = theta @ current_state_vector

    # Get the linear index that corresponds to the current action (a, b)
    c = tuple2index(action_index, object_index)

    # Select the Q function values that correspond to the current action C (a,b)
    q_value_c : float = q_values_current[c]
    
    # NOTE:
    # Now that the Q values are generated using a vector representation
    # of state in conjunction with a value matrix, we dont actually need to
    # update the Q values directly, but the parameters theta. For this we simply
    # minimize a simple quadratic loss function:
    # 
    #   L(theta)  = 0.5 * (y - Q(s, c theta))^2
    #   L'(theta) = (y - Q(s, c, theta)) * phi(s,c)
    # 
    # By taking its gradient, we can use said gradient to update the theta parameter
    # by using a simple gradient descent update
    # 
    #  Theta = Theta - alpha * (gradient)

    # Target value "y" or the sampled version of the bellman operator
    y : float = reward + (GAMMA * max_Q_next)

    # Calculate the gradient of the loss function
    loss_gradient = (y - q_value_c) * current_state_vector

    # Theta parameter update using gradient descent
    theta[c] = theta[c] + ALPHA * loss_gradient

    # This function shouldn't return anything
    return None


# ============================================== #
# RUN EPISODE                                    #
# ============================================== #

def run_episode(for_training):
    """ 
    Runs one episode
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
    episode_reward : float = 0.0

    # Discount
    # NOTE: Remember that in each episode the power of gamma increases by one.
    # This means that it starts with a power of zero (gamma^0 = 1) and then in each
    # step it increases by one (gamma^1 = gamma, gamma^2 = gamma^2, ...)
    episode_gamma : float = 1

    # Initialize a new game, we get back:
    # - Description of the initial room
    # - A description of the quest for this episode
    # - A variable indicating if the game is done (Initially false to indicate that its starting)
    (current_room_desc, current_quest_desc, terminal) = framework.newGame()

    # If this is not the final episode
    while not terminal:

        # Get the state "S" by concatenating Sr (Room description) and Sq (Quest description)
        current_state : str = current_room_desc + current_quest_desc

        # Get a smaller dimensional representation of the current state
        # by using the bag of words encoding strategy
        current_state_vector : np.ndarray = utils.extract_bow_feature_vector(current_state, dictionary)

        # Choose an action using the epsilon-greedy method
        action_index, object_index = epsilon_greedy(current_state_vector, theta, epsilon)

         # Run a step of the game. This returns:
        # - Future room description (S'r)
        # - Future quest description (S'q)
        # - Reward (R)
        # - Terminal: Whether the game has finished or not
        next_room_desc, next_quest_desc, reward, terminal = framework.step_game(
            current_room_desc, 
            current_quest_desc, 
            action_index, 
            object_index
        )

        # Get the state "S'" by concatenating the future room and quest descriptions
        next_state : str = next_room_desc + next_quest_desc

        # Create the "next_state_vector" using the bag of words encoding strategy
        next_state_vector : np.ndarray = utils.extract_bow_feature_vector(next_state, dictionary)

        if for_training:
            linear_q_learning(
                theta, 
                current_state_vector, 
                action_index, object_index, 
                reward, 
                next_state_vector, 
                terminal
            )

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

# ============================================== #
# RUN EPOCH                                      #
# ============================================== #

def run_epoch():
    """Runs one epoch and returns reward averaged over test episodes"""
    rewards = []

    for _ in range(NUM_EPIS_TRAIN):
        run_episode(for_training=True)

    for _ in range(NUM_EPIS_TEST):
        rewards.append(run_episode(for_training=False))

    return np.mean(np.array(rewards))


def run(action_dim : int, state_dim : int, dict):
    """Returns array of test reward per epoch for one run"""
    global theta
    global dictionary

    # Assign the global dictionary variable the value of dict
    dictionary = dict

    # Create an initially empty theta parameter
    theta = np.zeros([action_dim, state_dim])

    single_run_epoch_rewards_test = []
    pbar = tqdm(range(NUM_EPOCHS), ncols=80)
    for _ in pbar:
        single_run_epoch_rewards_test.append(run_epoch())
        pbar.set_description(
            "Avg reward: {:0.6f} | Ewma reward: {:0.6f}".format(
                np.mean(single_run_epoch_rewards_test),
                utils.ewma(single_run_epoch_rewards_test)))
    return single_run_epoch_rewards_test

