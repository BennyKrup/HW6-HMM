import pytest
from hmm import HiddenMarkovModel
import numpy as np




def test_mini_weather():
    """
    TODO: 
    Create an instance of your HMM class using the "small_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "small_weather_input_output.npz" file.

    Ensure that the output of your Forward algorithm is correct. 

    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    In addition, check for at least 2 edge cases using this toy model. 
    """

    mini_hmm=np.load('./data/mini_weather_hmm.npz')
    mini_input=np.load('./data/mini_weather_sequences.npz')



    observation_states = mini_hmm['observation_states']
    hidden_states = mini_hmm['hidden_states']
    prior_p = mini_hmm['prior_p']
    transition_p = mini_hmm['transition_p']
    emission_p = mini_hmm['emission_p']

    observation_state_sequence = mini_input['observation_state_sequence']
    best_hidden_state_sequence = mini_input['best_hidden_state_sequence']

    # Instantiate the HMM
    hmm = HiddenMarkovModel(observation_states, hidden_states, prior_p, transition_p, emission_p)

    # Forward algorithm test
    forward_probability = hmm.forward(observation_state_sequence)
    assert isinstance(forward_probability, float), "Forward probability should be a float."

    # Viterbi algorithm test
    predicted_sequence = hmm.viterbi(observation_state_sequence)
    assert predicted_sequence == list(best_hidden_state_sequence), "Viterbi algorithm returned an incorrect sequence."
    
        #empty observation sequence
    try:
        hmm.forward(np.array([]))
        assert False, "Expected ValueError for empty observation sequence not raised."
    except ValueError:
        pass

    if len(observation_states) > 0:  # Ensure there's at least one observation state to test
        invalid_observation = 'invalid_state'  # Use a string value that is not in observation_states
        try:
            hmm.forward(np.array([invalid_observation]))
            assert False, "Expected ValueError for invalid observation not raised."
        except ValueError:
            pass


def test_full_weather():

    """
    TODO: 
    Create an instance of your HMM class using the "full_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "full_weather_input_output.npz" file
        
    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    """

    mini_hmm=np.load('./data/full_weather_hmm.npz')
    mini_input=np.load('./data/full_weather_sequences.npz')



    observation_states = mini_hmm['observation_states']
    hidden_states = mini_hmm['hidden_states']
    prior_p = mini_hmm['prior_p']
    transition_p = mini_hmm['transition_p']
    emission_p = mini_hmm['emission_p']

    observation_state_sequence = mini_input['observation_state_sequence']
    best_hidden_state_sequence = mini_input['best_hidden_state_sequence']

    # Instantiate the HMM
    hmm = HiddenMarkovModel(observation_states, hidden_states, prior_p, transition_p, emission_p)

    # Forward algorithm test
    forward_probability = hmm.forward(observation_state_sequence)
    assert isinstance(forward_probability, float), "Forward probability should be a float."

    # Viterbi algorithm test
    predicted_sequence = hmm.viterbi(observation_state_sequence)
    assert predicted_sequence == list(best_hidden_state_sequence), "Viterbi algorithm returned an incorrect sequence."
    
        #empty observation sequence
    try:
        hmm.forward(np.array([]))
        assert False, "Expected ValueError for empty observation sequence not raised."
    except ValueError:
        pass

    if len(observation_states) > 0:  # Ensure there's at least one observation state to test
        invalid_observation = 'invalid_state'  # Use a string value that is not in observation_states
        try:
            hmm.forward(np.array([invalid_observation]))
            assert False, "Expected ValueError for invalid observation not raised."
        except ValueError:
            pass














