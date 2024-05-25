
# Positional Encoding
import numpy as np
import torch

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Initialize a positional encoding matrix of zeros
        pe = np.zeros((max_len, d_model))  
        
        # Get positions (0, 1, ..., max_len-1) and reshape to column
        position = np.arange(0, max_len).reshape(-1, 1)
        
        # Compute the division term for sine and cosine functions
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        

        pe[:, 0::2] = np.sin(position * div_term)  # Apply sine to even indices in the array
        pe[:, 1::2] = np.cos(position * div_term)  # Apply cosine to odd indices in the array
        pe = pe[np.newaxis, ...]  # Add a new axis to match the batch dimension
        
        # Register 'pe' as a buffer to avoid it being considered a model parameter
        self.register_buffer('pe', torch.tensor(pe, dtype=torch.float32))
        

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]  # Add positional encoding to the input tensor (already embedded) 
        return x

# Example usage
d_model = 512  # Dimension of the model
pos_encoding = PositionalEncoding(d_model)  # Initialize positional encoding
x = torch.rand(2, 10, d_model)  # Random tensor for input
output = pos_encoding(x)  # Apply positional encoding
print(output)  # Print the output with positional encoding added








# Scaled Dot-Product Attention
import torch  
import torch.nn.functional as F  # Importing the functional module for applying functions like softmax

def scaled_dot_product_attention(Q, K, V):
    
    # Dimension of the keys
    d_k = Q.size(-1)
    print("d_k =", d_k)
    
    # Calculate the dot products of the query and key matrices, and scale by the square root of the key dimension
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    print("Scores shape is", scores.shape)
    
    # Apply softmax to get the attention weights
    attention_weights = F.softmax(scores, dim=-1)
    print("\nAttention weights is of shape :", attention_weights.shape,"\n")
    
    # Multiply the attention weights with the value matrix to get the weighted sum
    return torch.matmul(attention_weights, V), attention_weights
    

# Example usage
Q = torch.rand(2, 3, 4)  # Create a random tensor for queries (batch_size, seq_length, d_k)
K = torch.rand(2, 3, 4)  # Create a random tensor for keys (batch_size, seq_length, d_k)
V = torch.rand(2, 3, 4)  # Create a random tensor for values (batch_size, seq_length, d_k)
output, weights = scaled_dot_product_attention(Q, K, V)
print(output)  # Print the output tensor
print(weights)  # Print the attention weights








# Multi-Head Attention
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model  # Total dimension of the model (embedding dimension)
        self.num_heads = num_heads  # Number of attention heads
        self.d_k = d_model // num_heads  # Dimension of each head

        # Define linear transformations for query, key, and value
        self.W_q = torch.nn.Linear(d_model, d_model)
        self.W_k = torch.nn.Linear(d_model, d_model)
        self.W_v = torch.nn.Linear(d_model, d_model)
        self.fc = torch.nn.Linear(d_model, d_model)  # Linear layer to combine the outputs (W_O)
        

    def forward(self, Q, K, V):
        batch_size = Q.size(0)  # Get the batch size

        # Apply linear transformations and split into multiple heads : input X = (Q,K,V) --> (QW_Q, KW_K, VW_V)
        print("The shapes of Q,K,V before linear projection are : ", Q.shape, K.shape, V.shape)
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  # Q --> QW_Q
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  # K --> KW_K
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  # V --> VW_V
        print("\nThe shapes of Q,K,V after linear projection and transposition between 2nd and 3rd dimensions are : ", Q.shape, K.shape, V.shape)

        # Scaled dot-product attention for each head
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        attention_weights = F.softmax(scores, dim=-1)  # Softmax to get attention weights
        print("\nThe shape of attention weights is : ", attention_weights.shape)
        context = torch.matmul(attention_weights, V)  # Weighted sum of values
        print("\nThe shape of context is : ", context.shape)

        # Concatenate heads and apply the final linear transformation
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        print("\nThe shape of context after transposition is : ", context.shape)
        output = self.fc(context)  # Final linear layer (W_O)
        print("\nThe output is : ", output.shape, "\n")
        return output

# Example usage
d_model = 512  # Dimension of the model
num_heads = 8  # Number of heads
multihead_attn = MultiHeadAttention(d_model, num_heads)  # Initialize multi-head attention
Q = torch.rand(2, 10, d_model)  # Random tensor for queries (batch_size, seq_length, embedding_dim)
K = torch.rand(2, 10, d_model)  # Random tensor for keys (batch_size, seq_length, embedding_dim)
V = torch.rand(2, 10, d_model)  # Random tensor for values (batch_size, seq_length, embedding_dim)
output = multihead_attn(Q, K, V)  # Apply multi-head attention
print(output)  # Print the output







# # Masked Scaled Dot-Product Attention
import torch
import torch.nn.functional as F

def masked_scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    # Apply the mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output, attention_weights

# Example usage with a mask
Q = torch.rand(2, 3, 4)  # Create a random tensor for queries
K = torch.rand(2, 3, 4)  # Create a random tensor for keys
V = torch.rand(2, 3, 4)  # Create a random tensor for values

# Mask matrix : Create a lower triangular matrix of shape (3, 3) and add a batch dimension
mask = torch.tril(torch.ones(3, 3)).unsqueeze(0) 

# Repeat the mask tensor along the batch dimension to match the desired shape
mask = mask.repeat(2, 1, 1)
print("The shape of mask is ", mask.shape,"\n")

output, weights = masked_scaled_dot_product_attention(Q, K, V, mask=mask)
print(output)
print(weights)


 







# Masked Multi-Head Attention

class MaskedMultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super(MaskedMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = torch.nn.Linear(d_model, d_model)
        self.W_k = torch.nn.Linear(d_model, d_model)
        self.W_v = torch.nn.Linear(d_model, d_model)
        self.fc = torch.nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.fc(context)
        return output

# Example usage with a mask
d_model = 512
num_heads = 8
masked_multihead_attn = MaskedMultiHeadAttention(d_model, num_heads)
Q = torch.rand(2, 10, d_model)
K = torch.rand(2, 10, d_model)
V = torch.rand(2, 10, d_model)

# Mask matrix : Create a lower triangular matrix of shape (10, 10) and add two batch dimensions
mask = torch.tril(torch.ones(10, 10)).unsqueeze(0).unsqueeze(0)  

# Repeat the mask tensor along the batch dimension to match the desired shape
mask = mask.repeat(2, 8, 1, 1)
print("The shape of mask is ", mask.shape, "\n")

output = masked_multihead_attn(Q, K, V, mask=mask)
print(output)





