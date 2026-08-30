Packages

```python
import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Embedding, MultiHeadAttention, Dense, Input, Dropout, LayerNormalization
```

Positional Encoding

- in sequence to sequence tasks, the relative order of the data is extremely important to its meaning
- when training sequential NN such as RNNs, inputs are fed into the network in order
- information about the order of the data was automatically fed into the model
- however, when training a transformer network using multi-head attention, data is fed into the model all at once
- while this dramatically reduces training time, there is no information about the order of the data
- this is where positional encoding is useful - specifically encode the positions of the inputs and pass them into the network using the sine and cosine formulas

    - d is the dimension of the word embedding and positional encoding
    - pas is the position of the word
    - k refers to each of the different dimensions in the positional encodings, with i equal to k // 2
    - to develop some intuition about positional encodings, think of them broadly as a feature that contains the information about the relative positions of words
    - the sum of the positional encoding and word embedding is ultimately what is fed into the model
    - hard coding the positions in, say by adding a matrix of 1s or whole number to the word embedding, the semantic meaning is distorted
    - conversely, the values of the sine and cosine equations are small enough that adding the positional encoding to a word embedding does not significantly distort the values, but instead enrich with positional information
    - using a combination of these equations helps the transformer network attend to the relative positions of the input data
    - Sine and Cosine Angles
        - even though the sine and cosine positional encoding equations take in different arguments, the inner terms for both equations are the same

        - consider the inner term when calculating the positional encoding for a word in a sequence

        - the angle is the same for both
        - the angles for PE(pos,2) and PE(pos,3) are the same as well, since for both, i = 1 and therefore the inner term is

        - the relationship holds true for all paired sine and cosine curves:

    - Exercise 1 - get_angles
        - if k = [0, 1, 2, 3, 4, 5], then i must be i = [0, 0, 1, 1, 2, 2]
        - i = k // 2

        ```python
        # UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
        # GRADED FUNCTION get_angles
        def get_angles(pos, k, d):
            """
            Get the angles for the positional encoding

            Arguments:
                pos -- Column vector containing the positions [[0], [1], ...,[N-1]]
                k --   Row vector containing the dimension span [[0, 1, 2, ..., d-1]]
                d(integer) -- Encoding size

            Returns:
                angles -- (pos, d) numpy array 
            """

            # START CODE HERE
            # Get i from dimension span k
            i = k // 2
            # Calculate the angles using pos, i and d
            angles = pos / (10000 ** ((2 * i) / d))
            # END CODE HERE

            return angles
        ```

- Sine and Cosine Positional Encodings
    - use the angles to calculate the sine and cosine positional encodings

    - Exercise 2 - positional_encoding
        - use the sine equation when i is an even number and the sine and the cosine equations when i is an odd number
        - np.newaxis can be used

        ```python
        # UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
        # GRADED FUNCTION positional_encoding
        def positional_encoding(positions, d):
            """
            Precomputes a matrix with all the positional encodings 

            Arguments:
                positions (int) -- Maximum number of positions to be encoded 
                d (int) -- Encoding size 

            Returns:
                pos_encoding -- (1, position, d_model) A matrix with the positional encodings
            """
            # START CODE HERE
            # initialize a matrix angle_rads of all the angles 
            angle_rads = get_angles(np.arange(positions)[:,np.newaxis],
                                    np.arange(d)[np.newaxis,:],
                                    d)

            # apply sin to even indices in the array; 2i
            angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

            # apply cos to odd indices in the array; 2i+1
            angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
            # END CODE HERE

            pos_encoding = angle_rads[np.newaxis, ...]

            return tf.cast(pos_encoding, dtype=tf.float32)
        ```

        ```python
        pos_encoding = positional_encoding(50, 512)

        print (pos_encoding.shape)

        plt.pcolormesh(pos_encoding[0], cmap='RdBu')
        plt.xlabel('d')
        plt.xlim((0, 512))
        plt.ylabel('Position')
        plt.colorbar()
        plt.show()
        ```

        - each row represents a positional encoding
        - none of the rows are identical

Masking

- there are two types of masks that are useful when building the transformer network: the padding mask and the look-ahead mask
- both help the softmax computation give the appropriate weights to the words in the input sequence
- Padding Mask
    - oftentimes, the input sequence will exceed the max length of a sequence the network can process
    - if the max length is five, for example, and the model is fed

    - which might get vectorized as

    - when passing sequences into a transformer model, it is important that they are of uniform length
    - this can be achieved by padding the sequence with zeros, and truncating sentences that exceed the max length of the model

    - sequences longer than the max length of five will be truncated, and zeros will be added to the truncated sequence to achieve uniform length
    - similarity, for sequences shorter than the max length, zeros will also be added for padding
    - however, the zeros will affect the softmax calculation - this is when a padding mask comes in handy
    - define a boolean mask that specifies which elements must attend (1) and which elemnets must be ignored (0)
    - later use the mask to set all the zeros in the sequence to a value close to negative infinity
    - after masking, the input should go from

    - so that when a softmax is taken, the zeros don’t affect the score
    - the MultiheadAttention layer implemented in Keras uses this masking logic
    - the following function only creates the mask of an already padded sequence
    - the TensorFlow Tokenizer and Hugging Face Tokenizer internally handle padding (and truncating) the input sequence

        ```python
        def create_padding_mask(decoder_token_ids):
            """
            Creates a matrix mask for the padding cells

            Arguments:
                decoder_token_ids -- (n, m) matrix

            Returns:
                mask -- (n, 1, m) binary tensor
            """    
            seq = 1 - tf.cast(tf.math.equal(decoder_token_ids, 0), tf.float32)

            # add extra dimensions to add the padding
            # to the attention logits. 
            # this will allow for broadcasting later when comparing sequences
            return seq[:, tf.newaxis, :] 
        ```

        ```python
        x = tf.constant([[7., 6., 0., 0., 1.], [1., 2., 3., 0., 0.], [0., 0., 0., 4., 5.]])
        print(create_padding_mask(x))

        '''
        tf.Tensor(
        [[[1. 1. 0. 0. 1.]]

         [[1. 1. 1. 0. 0.]]

         [[0. 0. 0. 1. 1.]]], shape=(3, 1, 5), dtype=float32)
        '''
        ```

    - multiplying (1 - mask) by -1e9 and add it to the sample input sequences, the zeros are essentially set to negative infinity
    - notice the difference when taking the softmax of the original sequence and the masked sequence

        ```python
        print(tf.keras.activations.softmax(x))
        print(tf.keras.activations.softmax(x + (1 - create_padding_mask(x)) * -1.0e9))

        '''
        tf.Tensor(
        [[7.2876644e-01 2.6809821e-01 6.6454901e-04 6.6454901e-04 1.8064314e-03]
         [8.4437378e-02 2.2952460e-01 6.2391251e-01 3.1062774e-02 3.1062774e-02]
         [4.8541026e-03 4.8541026e-03 4.8541026e-03 2.6502505e-01 7.2041273e-01]], shape=(3, 5), dtype=float32)
        tf.Tensor(
        [[0. 0. 1. 1. 0.]
         [0. 0. 0. 1. 1.]
         [1. 1. 1. 0. 0.]], shape=(3, 5), dtype=float32)
        tf.Tensor(
        [[[7.2973627e-01 2.6845497e-01 0.0000000e+00 0.0000000e+00 1.8088354e-03]
          [2.4472848e-01 6.6524094e-01 0.0000000e+00 0.0000000e+00 9.0030573e-02]
          [6.6483547e-03 6.6483547e-03 0.0000000e+00 0.0000000e+00 9.8670328e-01]]

         [[7.3057163e-01 2.6876229e-01 6.6619506e-04 0.0000000e+00 0.0000000e+00]
          [9.0030573e-02 2.4472848e-01 6.6524094e-01 0.0000000e+00 0.0000000e+00]
          [3.3333334e-01 3.3333334e-01 3.3333334e-01 0.0000000e+00 0.0000000e+00]]

         [[0.0000000e+00 0.0000000e+00 0.0000000e+00 2.6894143e-01 7.3105860e-01]
          [0.0000000e+00 0.0000000e+00 0.0000000e+00 5.0000000e-01 5.0000000e-01]
          [0.0000000e+00 0.0000000e+00 0.0000000e+00 2.6894143e-01 7.3105860e-01]]], shape=(3, 3, 5), dtype=float32)
        '''
        ```

- Look-ahead Mask
    - the look-ahead mask follows similar intuition
    - in training, access the complete correct output of your training example
    - the look-ahead mask helps the model pretend that it correctly predicted a part of the output and se if, without looking ahead, it can correctly predict the next output
    - for example, if the expected correct output is [1, 2, 3] and you wanted to see if given that the model correctly predicted the first value it could predict the second, value, you would mask out the second and third values
    - so input the masked sequence [1, -1e9, -1e9] and see if it could generate [1, 2, -1e9]

        ```python
        def create_look_ahead_mask(sequence_length):
            """
            Returns a lower triangular matrix filled with ones

            Arguments:
                sequence_length -- matrix size

            Returns:
                mask -- (size, size) tensor
            """
            mask = tf.linalg.band_part(tf.ones((1, sequence_length, sequence_length)), -1, 0)
            return mask 
        ```

        ```python
        x = tf.random.uniform((1, 3))
        temp = create_look_ahead_mask(x.shape[1])
        temp

        '''
        <tf.Tensor: shape=(1, 3, 3), dtype=float32, numpy=
        array([[[1., 0., 0.],
                [1., 1., 0.],
                [1., 1., 1.]]], dtype=float32)>
        '''
        ```

Self-Attention

- the use of self-attention paired with traditional conv networks allows for parallelization which speeds up training
- implement scaled dot product attention which takes in a query, key, value, and a mask as inputs to return rich, attention-based vector representations of the words in the sequence

- Q is the matrix of queries
- K is the matrix of keys
- V is the matrix of values
- M is the optional mask to apply
- d_k is the dimension of the keys, which is used to scale everything down so the softmax doesn’t explode
- Exercise 3 - scaled_dot_product_attention
    - the boolean mask param can be passed in as none or as either padding or look-ahead
        - multiply (1. - mask) by -1e9 before applying the softmax

    ```python
    # UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
    # GRADED FUNCTION scaled_dot_product_attention
    def scaled_dot_product_attention(q, k, v, mask):
        """
        Calculate the attention weights.
          q, k, v must have matching leading dimensions.
          k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
          The mask has different shapes depending on its type(padding or look ahead) 
          but it must be broadcastable for addition.

        Arguments:
            q -- query shape == (..., seq_len_q, depth)
            k -- key shape == (..., seq_len_k, depth)
            v -- value shape == (..., seq_len_v, depth_v)
            mask: Float tensor with shape broadcastable 
                  to (..., seq_len_q, seq_len_k). Defaults to None.

        Returns:
            output -- attention_weights
        """
        # START CODE HERE

        matmul_qk = tf.matmul(q, k.T)  # (..., seq_len_q, seq_len_k)

        # scale matmul_qk
        dk = np.sqrt(k.shape[-1])
        scaled_attention_logits = matmul_qk / dk

        # add the mask to the scaled tensor.
        if mask is not None: # Don't replace this None
            scaled_attention_logits += (1. - mask) * -1e9

        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = tf.keras.activations.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        # END CODE HERE

        return output, attention_weights
    ```

Encoder

- the transformer encoder layer pairs self-attention and conv NN style of processing to improve the speed of training and passes K and V matrices to the decoder
- the encoder pairs multi-head attention and a feed forward NN

    - MultiHeadAttention computes the self-attention several times to detect different features
    - feed forward NN contains two Dense layers
- the input sentence first passes through a multi-head attention layer, where the encoder looks at other words in the input sentence as it encodes a specific word
- the outputs of the multi-head attention layer are then fed to a feed forward NN
- the exact same feed forward network is independently applied to each position
    - for the MultiHeadAttention layer, use the Keras implementation
    - also use the Sequential API with two dense layers to build the feed forward NN layer
- in python, the __call__ method can call an object like a function
- tensorflow leverages this by providing a call function within Keras layers
- can use a layer like a_Layer(arg) and tensorflow will internally execute a_Layer.call(arg) to process the data

    ```python
    def FullyConnected(embedding_dim, fully_connected_dim):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(fully_connected_dim, activation='relu'),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(embedding_dim)  # (batch_size, seq_len, embedding_dim)
        ])
    ```

- Encoder Layer
    - pair multi-head attention and feed forward NN together in an encoder layer
    - use residual connections and layer normalization to help speed up training
    - Exercise 4 - EncoderLayer
        - pass the Q, V, K matrices and a boolean mask to a multi-head attention layer
        - to compute self-attention Q, V, and K should be the same
        - set the default values for return_attention_scores and training
        - also perform Dropout in this multi-head attention layer during training
        - add a skip connection by adding the original input x and the output of the multi-head attention layer
        - after adding the skip connection, pass the output through the first normalization layer
        - repeat steps 1-3, but with the feed forward NN with a dropout layer instead of the multi-head attention layer
        - the __init__ method creates all the layers that will be accessed by the call method
        - to use a layer defined inside the __init__ method, use the syntax self.[layer name]
        - for MultiHeadAttention, if query, key and value are the same, the function performs self-attention
        - the call arguments for self.mha are
            - query: query tensor of shape (B, T, dim)
            - value: value tensor of shape (B, S, dim)
            - key: optional key Tensor of shape (B, S, dim), if not given , will use value for both key and value, which is the most common case
            - attention_mask: a boolean mask of shape (B, T, S) that prevents attention to certain positions
                - the boolean mask specifies which query elements can attend to which key elements, 1 indicates attention and 0 indicates no attention
                - broadcasting can happen for the missing batch dimensions and the head dimension
            - return_attention_scores: a boolean to indicate whether the output should be (attention_output, attention_scores) if True, or attention_output if False, false by default
            - training: python boolean indicating whether the layer should behave in training mode (adding dropout) or in inference mode (no dropout)
                - defaults to either using the training mode of the parent layer/model, or False (inference) if there is no parent layer

        ```python
        # UNQ_C4 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
        # GRADED FUNCTION EncoderLayer
        class EncoderLayer(tf.keras.layers.Layer):
            """
            The encoder layer is composed by a multi-head self-attention mechanism,
            followed by a simple, positionwise fully connected feed-forward network. 
            This architecture includes a residual connection around each of the two 
            sub-layers, followed by layer normalization.
            """
            def __init__(self, embedding_dim, num_heads, fully_connected_dim,
                         dropout_rate=0.1, layernorm_eps=1e-6):
                super(EncoderLayer, self).__init__()

                self.mha = MultiHeadAttention(num_heads=num_heads,
                                              key_dim=embedding_dim,
                                              dropout=dropout_rate)

                self.ffn = FullyConnected(embedding_dim=embedding_dim,
                                          fully_connected_dim=fully_connected_dim)

                self.layernorm1 = LayerNormalization(epsilon=layernorm_eps)
                self.layernorm2 = LayerNormalization(epsilon=layernorm_eps)

                self.dropout_ffn = Dropout(dropout_rate)

            def call(self, x, training, mask):
                """
                Forward pass for the Encoder Layer

                Arguments:
                    x -- Tensor of shape (batch_size, input_seq_len, embedding_dim)
                    training -- Boolean, set to true to activate
                                the training mode for dropout layers
                    mask -- Boolean mask to ensure that the padding is not 
                            treated as part of the input
                Returns:
                    encoder_layer_out -- Tensor of shape (batch_size, input_seq_len, embedding_dim)
                """
                # START CODE HERE
                # calculate self-attention using mha(~1 line).
                # Dropout is added by Keras automatically if the dropout parameter is non-zero during training
                self_mha_output = self.mha(x, x, x, mask)  # Self attention (batch_size, input_seq_len, embedding_dim)

                # skip connection
                # apply layer normalization on sum of the input and the attention output to get the  
                # output of the multi-head attention layer (~1 line)
                skip_x_attention = self.layernorm1(x + self_mha_output)  # (batch_size, input_seq_len, embedding_dim)

                # pass the output of the multi-head attention layer through a ffn (~1 line)
                ffn_output = self.ffn(skip_x_attention)  # (batch_size, input_seq_len, embedding_dim)

                # apply dropout layer to ffn output during training (~1 line)
                # use `training=training` 
                ffn_output = self.dropout_ffn(ffn_output)

                # apply layer normalization on sum of the output from multi-head attention (skip connection) and ffn output to get the
                # output of the encoder layer (~1 line)
                encoder_layer_out = self.layernorm2(skip_x_attention + ffn_output)  # (batch_size, input_seq_len, embedding_dim)
                # END CODE HERE

                return encoder_layer_out

        ```

- Full Encoder
    - embed the input and add the positional encodings
    - then feed the encoded embeddings to a stack of Encoder layers

    - Exercise 5 - Encoder
        - use the call() method to embed the input, add positional encoding, and implement multiple encoder layers
        - pass the input through the Embedding layer
        - scale the embedding by multiplying it by the square root of the embedding dimension
        - remember to cast the embedding dimension to data type tf.float32 before computing the square root
        - add the position encoding: self.pos_encoding[:, :seq_len, :] to the embedding
        - pass the encoded embedding through a dropout layer, remembering to use the training param to set the model training mode
        - pass the output of the dropout layer through the stack of encoding layers using a for loop

            ```python
             # UNQ_C5 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
            # GRADED FUNCTION
            class Encoder(tf.keras.layers.Layer):
                """
                The entire Encoder starts by passing the input to an embedding layer 
                and using positional encoding to then pass the output through a stack of
                encoder Layers

                """  
                def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, input_vocab_size,
                           maximum_position_encoding, dropout_rate=0.1, layernorm_eps=1e-6):
                    super(Encoder, self).__init__()

                    self.embedding_dim = embedding_dim
                    self.num_layers = num_layers

                    self.embedding = Embedding(input_vocab_size, self.embedding_dim)
                    self.pos_encoding = positional_encoding(maximum_position_encoding, 
                                                            self.embedding_dim)

                    self.enc_layers = [EncoderLayer(embedding_dim=self.embedding_dim,
                                                    num_heads=num_heads,
                                                    fully_connected_dim=fully_connected_dim,
                                                    dropout_rate=dropout_rate,
                                                    layernorm_eps=layernorm_eps) 
                                       for _ in range(self.num_layers)]

                    self.dropout = Dropout(dropout_rate)

                def call(self, x, training, mask):
                    """
                    Forward pass for the Encoder

                    Arguments:
                        x -- Tensor of shape (batch_size, input_seq_len)
                        training -- Boolean, set to true to activate
                                    the training mode for dropout layers
                        mask -- Boolean mask to ensure that the padding is not 
                                treated as part of the input
                    Returns:
                        x -- Tensor of shape (batch_size, input_seq_len, embedding_dim)
                    """
                    seq_len = tf.shape(x)[1]

                    # START CODE HERE
                    # Pass input through the Embedding layer
                    x = self.embedding(x)  # (batch_size, input_seq_len, embedding_dim)
                    # Scale embedding by multiplying it by the square root of the embedding dimension
                    x *= np.sqrt(tf.cast(self.embedding_dim, dtype=tf.float32))
                    # Add the position encoding to embedding
                    x += self.pos_encoding[:, :seq_len, :]
                    # Pass the encoded embedding through a dropout layer
                    # use `training=training`
                    x = self.dropout(x, training)
                    # Pass the output through the stack of encoding layers 
                    for i in range(self.num_layers):
                        x = self.enc_layers[i](x, training, mask)
                    # END CODE HERE

                    return x  # (batch_size, input_seq_len, embedding_dim)
            ```

Decoder

- the decoder layer takes the K and V matrices generated by the Encoder and computes the second multi-head attention layer with the Q matrix from the output

- Decoder Layer
    - pair multi-head attention with a feed forward NN, but his time implement two multi-head attention layers
    - use residual connections and layer norm to speed up training
    - Exercise 6 - DecoderLayer
        - block 1 is a multi-head attention layer with a residual connection, and look-ahead mask
        - like in the encoderlayer, dropout is defined within the multi-head attention layer
        - block 2 will take into account the outptu of the encoder, so the multi-head attention layer will receive K and V from the encoder, and Q from the block 1
        - then apply a norm layer and a residual connection, just like before with the encoderlayer
        - finally, block 3 is a feed forward NN with dropout and norm layers and a residual connection
        - the first two blocks are fairly similar to the EncoderLayer except return attention_scores when computing self-attention

        ```python
        # UNQ_C6 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
        # GRADED FUNCTION DecoderLayer
        class DecoderLayer(tf.keras.layers.Layer):
            """
            The decoder layer is composed by two multi-head attention blocks, 
            one that takes the new input and uses self-attention, and the other 
            one that combines it with the output of the encoder, followed by a
            fully connected block. 
            """
            def __init__(self, embedding_dim, num_heads, fully_connected_dim, dropout_rate=0.1, layernorm_eps=1e-6):
                super(DecoderLayer, self).__init__()

                self.mha1 = MultiHeadAttention(num_heads=num_heads,
                                              key_dim=embedding_dim,
                                              dropout=dropout_rate)

                self.mha2 = MultiHeadAttention(num_heads=num_heads,
                                              key_dim=embedding_dim,
                                              dropout=dropout_rate)

                self.ffn = FullyConnected(embedding_dim=embedding_dim,
                                          fully_connected_dim=fully_connected_dim)

                self.layernorm1 = LayerNormalization(epsilon=layernorm_eps)
                self.layernorm2 = LayerNormalization(epsilon=layernorm_eps)
                self.layernorm3 = LayerNormalization(epsilon=layernorm_eps)

                self.dropout_ffn = Dropout(dropout_rate)

            def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
                """
                Forward pass for the Decoder Layer

                Arguments:
                    x -- Tensor of shape (batch_size, target_seq_len, embedding_dim)
                    enc_output --  Tensor of shape(batch_size, input_seq_len, embedding_dim)
                    training -- Boolean, set to true to activate
                                the training mode for dropout layers
                    look_ahead_mask -- Boolean mask for the target_input
                    padding_mask -- Boolean mask for the second multihead attention layer
                Returns:
                    out3 -- Tensor of shape (batch_size, target_seq_len, embedding_dim)
                    attn_weights_block1 -- Tensor of shape(batch_size, num_heads, target_seq_len, input_seq_len)
                    attn_weights_block2 -- Tensor of shape(batch_size, num_heads, target_seq_len, input_seq_len)
                """

                # START CODE HERE
                # enc_output.shape == (batch_size, input_seq_len, embedding_dim)

                # BLOCK 1
                # calculate self-attention and return attention scores as attn_weights_block1.
                # Dropout will be applied during training (~1 line).
                mult_attn_out1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask, return_attention_scores=True)  # (batch_size, target_seq_len, embedding_dim)

                # apply layer normalization (layernorm1) to the sum of the attention output and the input (~1 line)
                Q1 = self.layernorm1(mult_attn_out1 + x)

                # BLOCK 2
                # calculate self-attention using the Q from the first block and K and V from the encoder output. 
                # Dropout will be applied during training
                # Return attention scores as attn_weights_block2 (~1 line) 
                mult_attn_out2, attn_weights_block2 = self.mha2(Q1, enc_output, enc_output, padding_mask, return_attention_scores=True)  # (batch_size, target_seq_len, embedding_dim)

                # apply layer normalization (layernorm2) to the sum of the attention output and the output of the first block (~1 line)
                mult_attn_out2 = self.layernorm2(mult_attn_out2 + Q1)  # (batch_size, target_seq_len, embedding_dim)

                #BLOCK 3
                # pass the output of the second block through a ffn
                ffn_output = self.ffn(mult_attn_out2)  # (batch_size, target_seq_len, embedding_dim)

                # apply a dropout layer to the ffn output
                # use `training=training`
                ffn_output = self.dropout_ffn(ffn_output)

                # apply layer normalization (layernorm3) to the sum of the ffn output and the output of the second block
                out3 = self.layernorm3(ffn_output + mult_attn_out2)  # (batch_size, target_seq_len, embedding_dim)
                # END CODE HERE

                return out3, attn_weights_block1, attn_weights_block2

        ```

- Full Decoder
    - build a full transformer decoder
    - embed the output and add positional encodings

    - Exercise 7 - Decoder
        - use the call() method to embed the output, add positional encoding, and implement multiple decoder layers
        - initialize the decoder with an embedding layer, positional encoding, and multiple decoderlayers
            - pass the generated output through the embedding layer
            - scale the embedding by multiplying it by the square root of the embedding dimension, cast the embedding dimension to data type tf.float32 before compute the square root
            - add the position encoding: self.pos_encoding[:, :seq_len, :] to the embedding
            - pass the encoded embedding through a dropout layer, remembering to use the training param to set the model training mode
            - pass the output of the dropout layer through the stack of decoding layers using a for loop

        ```python
        # UNQ_C7 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
        # GRADED FUNCTION Decoder
        class Decoder(tf.keras.layers.Layer):
            """
            The entire Encoder starts by passing the target input to an embedding layer 
            and using positional encoding to then pass the output through a stack of
            decoder Layers

            """ 
            def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, target_vocab_size,
                       maximum_position_encoding, dropout_rate=0.1, layernorm_eps=1e-6):
                super(Decoder, self).__init__()

                self.embedding_dim = embedding_dim
                self.num_layers = num_layers

                self.embedding = Embedding(target_vocab_size, self.embedding_dim)
                self.pos_encoding = positional_encoding(maximum_position_encoding, self.embedding_dim)

                self.dec_layers = [DecoderLayer(embedding_dim=self.embedding_dim,
                                                num_heads=num_heads,
                                                fully_connected_dim=fully_connected_dim,
                                                dropout_rate=dropout_rate,
                                                layernorm_eps=layernorm_eps) 
                                   for _ in range(self.num_layers)]
                self.dropout = Dropout(dropout_rate)

            def call(self, x, enc_output, training, 
                   look_ahead_mask, padding_mask):
                """
                Forward  pass for the Decoder

                Arguments:
                    x -- Tensor of shape (batch_size, target_seq_len, embedding_dim)
                    enc_output --  Tensor of shape(batch_size, input_seq_len, embedding_dim)
                    training -- Boolean, set to true to activate
                                the training mode for dropout layers
                    look_ahead_mask -- Boolean mask for the target_input
                    padding_mask -- Boolean mask for the second multihead attention layer
                Returns:
                    x -- Tensor of shape (batch_size, target_seq_len, embedding_dim)
                    attention_weights - Dictionary of tensors containing all the attention weights
                                        each of shape Tensor of shape (batch_size, num_heads, target_seq_len, input_seq_len)
                """

                seq_len = tf.shape(x)[1]
                attention_weights = {}

                # START CODE HERE
                # create word embeddings 
                x = self.embedding(x)  # (batch_size, target_seq_len, embedding_dim)

                # scale embeddings by multiplying by the square root of their dimension
                x *= np.sqrt(tf.cast(self.embedding_dim, tf.float32))

                # calculate positional encodings and add to word embedding
                x += self.pos_encoding[:, :seq_len, :]

                # apply a dropout layer to x
                # use `training=training`
                x = self.dropout(x)

                # use a for loop to pass x through a stack of decoder layers and update attention_weights (~4 lines total)
                for i in range(self.num_layers):
                    # pass x and the encoder output through a stack of decoder layers and save the attention weights
                    # of block 1 and 2 (~1 line)
                    x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                         look_ahead_mask, padding_mask)

                    #update attention_weights dictionary with the attention weights of block 1 and block 2
                    attention_weights['decoder_layer{}_block1_self_att'.format(i+1)] = block1
                    attention_weights['decoder_layer{}_block2_decenc_att'.format(i+1)] = block2
                # END CODE HERE

                # x.shape == (batch_size, target_seq_len, embedding_dim)
                return x, attention_weights
        ```

Transformer

- the input first passes through an encoder, which is just repeated encoder layers
    - embedding and positional encoding of the input
    - multi-head attention on the input
    - feed forward NN to help detect features
- then the predicted output passes through a decoder, consisting of the decoder layers
    - embedding and positional encoding of the output
    - multi-head attention on the generated output
    - multi-head attention with the Q from the first multi-head attention layer and the K and V from the encoder
    - a feed forward NN to help detect features
- finally, after the Nth Decoder layer, one dense layer and a softmax are applied to generate prediction to the next output in the sequence
- Exercise 8 - Transformer
    - pass the input through the encoder with the appropriate mask
    - pass the encoder output and the target through the decoder with the appropriate mask
    - apply a linear transformation and a softmax to get a prediction

    ```python
    # UNQ_C8 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
    # GRADED FUNCTION Transformer
    class Transformer(tf.keras.Model):
        """
        Complete transformer with an Encoder and a Decoder
        """
        def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, input_vocab_size, 
                   target_vocab_size, max_positional_encoding_input,
                   max_positional_encoding_target, dropout_rate=0.1, layernorm_eps=1e-6):
            super(Transformer, self).__init__()

            self.encoder = Encoder(num_layers=num_layers,
                                   embedding_dim=embedding_dim,
                                   num_heads=num_heads,
                                   fully_connected_dim=fully_connected_dim,
                                   input_vocab_size=input_vocab_size,
                                   maximum_position_encoding=max_positional_encoding_input,
                                   dropout_rate=dropout_rate,
                                   layernorm_eps=layernorm_eps)

            self.decoder = Decoder(num_layers=num_layers, 
                                   embedding_dim=embedding_dim,
                                   num_heads=num_heads,
                                   fully_connected_dim=fully_connected_dim,
                                   target_vocab_size=target_vocab_size, 
                                   maximum_position_encoding=max_positional_encoding_target,
                                   dropout_rate=dropout_rate,
                                   layernorm_eps=layernorm_eps)

            self.final_layer = Dense(target_vocab_size, activation='softmax')

        def call(self, input_sentence, output_sentence, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
            """
            Forward pass for the entire Transformer
            Arguments:
                input_sentence -- Tensor of shape (batch_size, input_seq_len)
                                  An array of the indexes of the words in the input sentence
                output_sentence -- Tensor of shape (batch_size, target_seq_len)
                                  An array of the indexes of the words in the output sentence
                training -- Boolean, set to true to activate
                            the training mode for dropout layers
                enc_padding_mask -- Boolean mask to ensure that the padding is not 
                        treated as part of the input
                look_ahead_mask -- Boolean mask for the target_input
                dec_padding_mask -- Boolean mask for the second multihead attention layer
            Returns:
                final_output -- Describe me
                attention_weights - Dictionary of tensors containing all the attention weights for the decoder
                                    each of shape Tensor of shape (batch_size, num_heads, target_seq_len, input_seq_len)

            """
            # START CODE HERE
            # call self.encoder with the appropriate arguments to get the encoder output
            enc_output = self.encoder(input_sentence, training, enc_padding_mask)  # (batch_size, inp_seq_len, embedding_dim)

            # call self.decoder with the appropriate arguments to get the decoder output
            # dec_output.shape == (batch_size, tar_seq_len, embedding_dim)
            dec_output, attention_weights = self.decoder(output_sentence, enc_output, training, look_ahead_mask, dec_padding_mask)

            # pass decoder output through a linear layer and softmax (~2 lines)
            final_output = self.final_layer(dec_output) # (batch_size, tar_seq_len, target_vocab_size)
            # END CODE HERE

            return final_output, attention_weights
    ```