import numpy as np

# Sample input: a sentence represented as word embeddings
sentence = np.array([[0.1, 0.2, 0.3, 0.4],  # "it"
                     [0.5, 0.6, 0.7, 0.8],  # "refers"
                     [0.9, 1.0, 1.1, 1.2],  # "to"
                     [1.3, 1.4, 1.5, 1.6],  # "robot"
                     [1.7, 1.8, 1.9, 2.0]])  # "."

# Self-Attention Calculation
def self_attention(query, keys, values):
    # Expand dimensions to handle 1D arrays
    query = query[np.newaxis, :]
    keys = keys.T  # Transpose keys to make dot product work

    # Calculate the dot product between query and keys
    dot_product = np.dot(query, keys)

    # Apply softmax to obtain attention weights
    exp_dot_product = np.exp(dot_product)
    attention_weights = exp_dot_product / exp_dot_product.sum(axis=1, keepdims=True)

    # Calculate the weighted sum of values
    output = np.dot(attention_weights, values)

    return output, attention_weights

# Assuming "it" is the query, and other words are keys and values
query = sentence[0]  # "it" (query)
keys = sentence[1:-1]  # Other words in the sentence except for "."
values = sentence[1:-1]  # Values are the same as keys in this example

output, attention_weights = self_attention(query, keys, values)

# The output is the result of self-attention, and attention_weights represent how 
# much each word "it" attends to other words in the sentence.

# This will be a vector with weighted information from the input values
print("Output of self-attention:", output)

# This will be a vector of attention weights indicating how much the query attends
# to each key
print("Attention Weights:", attention_weights)
