import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

class Encoder :
    def __init__(self, d_model=512, num_heads = 8):
        self.d_model = d_model
        self.num_heads = num_heads
        self.WQ = np.random.randn(d_model,d_model)
        self.WK = np.random.randn(d_model,d_model)
        self.WV = np.random.randn(d_model,d_model)
        self.W1 = np.random.randn(d_model,d_model*4) #FFN Layer 1
        self.W2 = np.random.randn(d_model*4,d_model) #FFN Layer 1

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        d_k = Q.shape[-1]
        scores = np.matmul(Q,K.T) / np.sqrt(d_k)
        if mask is not None :
            scores = np.where(mask,float('-inf'),scores)
        attention_weights = softmax(scores)
        output = np.matmul(attention_weights, V)
        return output

    def multi_head_attention(self, Q, K, V):
        head_size = self.d_model // self.num_heads
        Q_split = np.array(np.split(np.dot(Q,self.WQ),self.num_heads,axis=-1))
        K_split = np.array(np.split(np.dot(K,self.WK),self.num_heads,axis=-1))
        V_split = np.array(np.split(np.dot(V,self.WV),self.num_heads,axis=-1))
        outputs = [self.scaled_dot_product_attention(Q_split[i],K_split[i],V_split[i]) for i in range(self.num_heads)]
        result = np.concatenate(outputs,axis=-1)
        return result

    def feed_forward(self,x):
        return np.maximum(0,np.dot(x,self.W1)) @ self.W2 #ReLU + Linear

    def layer_norm(self,x,epsilon=1e-6):
        return (x - np.mean(x,axis=-1,keepdims=True)) / (np.std(x,axis=-1,keepdims=True) + epsilon)
    
    def add_norm(self,x,sublayer):
        return self.layer_norm(x + sublayer)

    def forward(self,x):
        attention_output = self.add_norm(x,self.multi_head_attention(x,x,x))
        ff_output = self.add_norm(attention_output,self.feed_forward(attention_output))
        return ff_output


class Decoder :
    def __init__(self,d_model=512,num_heads=8):
        self.d_model = d_model
        self.num_heads = num_heads
        self.W1 = np.random.randn(d_model,d_model * 4) #FFN 1
        self.W2 = np.random.randn(d_model * 4, d_model) #FFN 2

    def masked_decoder_self_attention(self, query, key, value):
        num_tokens = query.shape[0]
        attention_scores = np.matmul(query, key.T) / np.sqrt(query.shape[1])
        mask = np.triu(np.ones(attention_scores.shape), k=1)
        attention_scores = np.where(mask, float('-inf'), attention_scores)

        attention_weights = softmax(attention_scores)
        attention_output = np.matmul(attention_weights, value)
        return attention_output

    def encoder_decoder_attention(self, decoder_query, encoder_keys, encoder_values):
        attention_scores = np.matmul(decoder_query, encoder_keys.T) / np.sqrt(decoder_query.shape[1])
        attention_scores = np.exp(attention_scores) / np.sum(np.exp(attention_scores), axis = 1, keepdims=True)
        attention_output = np.matmul(attention_scores, encoder_values)
        return attention_output

    def feed_forward(self,x):
        return np.maximum(0,np.dot(x,self.W1)) @ self.W2 #ReLU + Linear

    def add_norm(self,x,sublayer):
        return self.layer_norm(x + sublayer)

    def layer_norm(self, x, epsilon=1e-6):
        return (x - np.mean(x, axis=-1, keepdims=True)) / (np.std(x,axis=-1,keepdims=True) + epsilon)

    def forward(self,x,encoder_output):
        masked_attention_output = self.add_norm(x, self.masked_decoder_self_attention(x,x,x))
        encoder_decoder_attention_output = self.add_norm(masked_attention_output, self.encoder_decoder_attention(masked_attention_output,encoder_output,encoder_output))
        ff_output = self.add_norm(encoder_decoder_attention_output,self.feed_forward(encoder_decoder_attention_output))
        return ff_output

class Transformer:
    def __init__(self,d_model=512,num_heads=8,max_length=100):
        self.encoder = Encoder(d_model,num_heads)
        self.decoder = Decoder(d_model,num_heads)
        self.positional_encoding = self.create_positional_encoding(max_length, d_model)

    def create_positional_encoding(self,max_length,d_model):
        pos = np.arange(max_length)[:,np.newaxis]
        i = np.arange(d_model)[np.newaxis,:]
        angle_rates = 1. / np.power(10000,(2*(i//2))/np.float32(d_model))
        pos_encoding = pos * angle_rates
        pos_encoding[:,0::2] = np.sin(pos_encoding[:,0::2])
        pos_encoding[:,1::2] = np.cos(pos_encoding[:,1::2])
        return pos_encoding

    def forward(self, encoder_input, decoder_input):
        # add positional encoding
        encoder_seq_length = encoder_input.shape[0]
        encoder_input += self.positional_encoding[:encoder_seq_length]
        encoder_output = self.encoder.forward(encoder_input)
        
        decoder_seq_length = decoder_input.shape[0]
        decoder_input += self.positional_encoding[:decoder_seq_length]
        decoder_output = self.decoder.forward(decoder_input, encoder_output)
        return decoder_output

encoder_input = np.random.randn(10,512)
decoder_input = np.random.randn(10,512)

transformer = Transformer()

decoder_output = transformer.forward(encoder_input,decoder_input)

print("Final Transformer Decoder Output Shape:", decoder_output.shape)

