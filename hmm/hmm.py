import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """             
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {state: index for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p= prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p
        

        #check data
        if transition_p.shape != (len(hidden_states), len(hidden_states)):
            raise ValueError("transition_p must be a square matrix of shape (len(hidden_states), len(hidden_states)).")
        if emission_p.shape != (len(hidden_states), len(observation_states)):
            raise ValueError("emission_p must have a shape of (len(hidden_states), len(observation_states)).")
        if prior_p.shape[0] != len(hidden_states):
            raise ValueError("prior_p must have a length equal to the number of hidden states.")

        # Probability value checks
        if not ((0 <= transition_p).all() and (transition_p <= 1).all()):
            raise ValueError("All values in transition_p must be between 0 and 1.")
        if not ((0 <= emission_p).all() and (emission_p <= 1).all()):
            raise ValueError("All values in emission_p must be between 0 and 1.")
        if not ((0 <= prior_p).all() and (prior_p <= 1).all()):
            raise ValueError("All values in prior_p must be between 0 and 1.")
        #check if probabilites sum to 1, if not normalize and raise warning
        if not np.isclose(self.prior_p.sum(), 1):
            print("Warning: prior probabilities do not sum to 1. Normalizing prior probabilities.")
            self.prior_p = self.prior_p / self.prior_p.sum()
        if not np.isclose(self.transition_p.sum(axis=1), 1).all():
            print("Warning: transition probabilities do not sum to 1. Normalizing transition probabilities.")
            self.transition_p = self.transition_p / self.transition_p.sum(axis=1)[:, None]
        if not np.isclose(self.emission_p.sum(axis=1), 1).all():
            print("Warning: emission probabilities do not sum to 1. Normalizing emission probabilities.")
            self.emission_p = self.emission_p / self.emission_p.sum(axis=1)[:, None]

        

    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        TODO 

        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """        
        
        # Check for empty sequence
        if len(input_observation_states) == 0:
            raise ValueError("input_observation_states cannot be empty.")

        # Check if all observations are valid
        for obs in input_observation_states:
            if obs not in self.observation_states_dict:
                raise ValueError(f"Observation {obs} not found in observation_states.")


        # Step 1. Initialize variables

        forward_table = np.zeros((len(input_observation_states), len(self.hidden_states)))

        for state, state_index in self.hidden_states_dict.items(): #initialize forward table

            obs_index = self.observation_states_dict[input_observation_states[0]]
            #print("State index:", state_index, "Type:", type(state_index))
            #print("Obs index:", obs_index, "Type:", type(obs_index))
            forward_table[0, state_index] = self.prior_p[state_index] * self.emission_p[state_index, obs_index]
 
       
        # Step 2. Calculate probabilities

        for t in range(1, len(input_observation_states)): #calculate forward probability for observation states
            for state, state_index in self.hidden_states_dict.items():
                obs_index = self.observation_states_dict[input_observation_states[t]]
                forward_table[t, state_index] = sum(forward_table[t-1, prev_state]
                                                     * self.transition_p[prev_state, state_index]
                                                     * self.emission_p[state_index, obs_index]
                                                       for prev_state in range(len(self.hidden_states)))

        # Step 3. Return final probability
        forward_probability = sum(forward_table[len(input_observation_states)-1, :])
        return forward_probability
    
        


    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        TODO

        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """        
        
        # Step 1. Initialize variables
        #switch to log probabilities to avoid underflow, numpy handles log(0) as -inf
        log_prior_p = np.log(self.prior_p)
        log_transition_p = np.log(self.transition_p)
        log_emmision_p = np.log(self.emission_p)


        n_states = len(self.hidden_states)
        n_observations = len(decode_observation_states)
        viterbi_log_prob = np.full((n_states, n_observations), -np.inf)  # Initialize with -inf
        path = np.zeros((n_states, n_observations), dtype=int)

        #initialize viterbi table
        first_obs_index = self.observation_states_dict[decode_observation_states[0]]
        viterbi_log_prob[:, 0] = log_prior_p + log_emmision_p[:, first_obs_index]


       
       # Step 2. Calculate Probabilities
        for t in range(1, n_observations):
            for j in range(n_states):
                current_obs_index = self.observation_states_dict[decode_observation_states[t]]
                for i in range(n_states):
                    prob = viterbi_log_prob[i, t-1] + log_transition_p[i, j] + log_emmision_p[j, current_obs_index]
                    if prob > viterbi_log_prob[j, t]:
                        viterbi_log_prob[j, t] = prob
                        path[j, t] = i

            
        # Step 3. Traceback
        best_path = []
        last_state = np.argmax(viterbi_log_prob[:, -1])
        best_path.append(last_state)

        for t in range(n_observations - 1, 0, -1):
            best_path.insert(0, path[last_state, t])
            last_state = path[last_state, t]


        # Step 4. Return best hidden state sequence 
        best_hidden_state_sequence = [self.hidden_states[i] for i in best_path]
        return best_hidden_state_sequence      