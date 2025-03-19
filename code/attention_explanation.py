import numpy as np

# Function to calculate attention scores using dot product
def calculate_attention_scores(decoder_hidden_state, encoder_hidden_states):
    # Calculate dot product between decoder hidden state and each encoder hidden state
    dot_products = np.dot(encoder_hidden_states, decoder_hidden_state)
    
    return dot_products

# Function to normalize attention scores using softmax
def normalize_attention_scores(attention_scores):
    # Apply softmax to obtain normalized weights
    attention_weights = np.exp(attention_scores) / np.sum(np.exp(attention_scores), axis=0)
    
    return attention_weights

# Function to calculate weighted sum for attention output
def calculate_attention_output(encoder_hidden_states, attention_weights):
    # Perform weighted sum to get attention output
    attention_output = np.dot(encoder_hidden_states.T, attention_weights)
    
    return attention_output

# Example data
encoder_hidden_states = np.array([[0.1, 0.2, 0.3],
                                  [0.4, 0.5, 0.6],
                                  [0.7, 0.8, 0.9]])

decoder_hidden_state = np.array([0.2, 0.4, 0.6])

# Calculate attention scores
# decoder's hidden states are the queries that is uses to query the encoder's keys and 
# determine how much "attention" it should pay to the encoder's values
attention_scores = calculate_attention_scores(decoder_hidden_state, encoder_hidden_states)

# Normalize attention scores
attention_weights = normalize_attention_scores(attention_scores)

# Calculate attention output
# the values are then weighted by the calculated attention scores to generate the output
attention_output = calculate_attention_output(encoder_hidden_states, attention_weights)

#Encoder : keys, values
#Decoder : queries
# Print results
print("Attention Scores:", attention_scores)
print("Attention Weights:", attention_weights)
print("Attention Output:", attention_output)
