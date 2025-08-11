"""Particle transformer in TensorFlow."""

import tensorflow as tf
import numpy as np


def prepare_interaction(x: tf.Tensor) -> tf.Tensor:
    """Prepare the features for interaction matrix U.

    Args:
        x : tf.Tensor
            Input tensor of shape (N, L, 3), where N is the batch size,
            L is the number of particles, and 3 is the feature dimension
            corresponding to (pt_rel, delta_eta, delta_phi).
    Returns:
        tf.Tensor
            Output tensor of shape (N, 3, L, L), where N is the batch size,
            L is the number of particles, and 3 is the feature dimension
            corresponding to (delta, kt, z).
    """
    
    # Expand dimensions for broadcasting
    x_i = tf.expand_dims(x, axis=-2)  # (N, L, 3) -> (N, L, 1, 3)
    x_j = tf.expand_dims(x, axis=-3)  # (N, L, 3) -> (N, 1, L, 3)

    # Split features
    pt_rel_i, delta_eta_i, delta_phi_i = tf.unstack(x_i, axis=-1)  # (N, L, 1)
    pt_rel_j, delta_eta_j, delta_phi_j = tf.unstack(x_j, axis=-1)  # (N, 1, L)

    # Calculate delta and mod delta_phi to [-pi, pi]
    delta_eta_diff = delta_eta_i - delta_eta_j
    delta_phi_diff = (delta_phi_i - delta_phi_j + np.pi) % (2 * np.pi) - np.pi
    delta = tf.sqrt(delta_eta_diff ** 2 + delta_phi_diff ** 2)  # (N, L, L)

    # Calculate kt and z
    pt_rel_min = tf.minimum(pt_rel_i, pt_rel_j)  # (N, L, L)
    kt = pt_rel_min * delta  # (N, L, L)
    z = pt_rel_min / (pt_rel_i + pt_rel_j)  # (N, L, L)

    # Stack and clamp values to avoid numerical issues
    features = tf.stack([delta, kt, z], axis=-3)  # (N, 3, L, L)
    features = tf.clip_by_value(features, 1e-9, tf.float32.max)
    return tf.math.log(features)


class ParticleFeatureEmbedding(tf.keras.layers.Layer):
    
    def __init__(self, input_dim: int, embedding_dims: list, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.embedding_dims = embedding_dims
        
        dims = [input_dim] + embedding_dims
        dims = [(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        
        self.layers = []
        for _input_dim, _embed_dim in dims:
            self.layers.append(tf.keras.layers.LayerNormalization())
            self.layers.append(tf.keras.layers.Dense(_embed_dim))
            self.layers.append(tf.keras.layers.Activation('gelu'))
    
    def call(self, x):
        for layer in self.layers:
            x = layer(x)
        return x  # (N, L, D) -> (N, L, E)


class InteractionMatrixEmbedding(tf.keras.layers.Layer):
    
    def __init__(self, input_dim: int, embedding_dims: list, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.embedding_dims = embedding_dims
        
        dims = [input_dim] + embedding_dims
        dims = [(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        
        self.layers = []
        for _input_dim, _embed_dim in dims:
            self.layers.append(tf.keras.layers.Conv2D(_embed_dim, kernel_size=1))
            self.layers.append(tf.keras.layers.Activation('gelu'))
    
    def call(self, x):
        for layer in self.layers:
            x = layer(x)
        return x  # (N, C, L, L) -> (N, H, L, L)


class MultiheadAttention(tf.keras.layers.Layer):
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, 
                 use_bias: bool = False, use_head_scale: bool = True, **kwargs):
        super().__init__(**kwargs)
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear projections for queries, keys, and values
        self.q_linear = tf.keras.layers.Dense(embed_dim, use_bias=use_bias)
        self.k_linear = tf.keras.layers.Dense(embed_dim, use_bias=use_bias)
        self.v_linear = tf.keras.layers.Dense(embed_dim, use_bias=use_bias)
        
        # Linear projection for the output
        self.out_linear = tf.keras.layers.Dense(embed_dim, use_bias=use_bias)
        
        self.dropout_layer = tf.keras.layers.Dropout(dropout)
        
        # Head scale
        self.use_head_scale = use_head_scale
        if use_head_scale:
            self.gamma = self.add_weight(
                name='gamma',
                shape=(num_heads,),
                initializer='ones',
                trainable=True
            )
    
    def call(self, query, key, value, attn_mask=None, key_padding_mask=None, training=None):
        batch_size = tf.shape(query)[0]
        
        # Linear projections
        q = self.q_linear(query)  # (N, L, E) or (N, 1, E)
        k = self.k_linear(key)    # (N, L, E) or (N, 1+L, E)
        v = self.v_linear(value)  # (N, L, E) or (N, 1+L, E)
        
        # Reshape for multi-head attention
        q = tf.reshape(q, [batch_size, -1, self.num_heads, self.head_dim])
        q = tf.transpose(q, [0, 2, 1, 3])  # (N, H, L, D)
        
        k = tf.reshape(k, [batch_size, -1, self.num_heads, self.head_dim])
        k = tf.transpose(k, [0, 2, 1, 3])  # (N, H, L, D)
        
        v = tf.reshape(v, [batch_size, -1, self.num_heads, self.head_dim])
        v = tf.transpose(v, [0, 2, 1, 3])  # (N, H, L, D)
        
        # Scaled dot-product attention
        scores = tf.matmul(q, k, transpose_b=True) / tf.sqrt(float(self.head_dim))
        
        # Apply masks
        if key_padding_mask is not None:
            # Expand dimensions for broadcasting
            scores_mask = tf.expand_dims(tf.expand_dims(key_padding_mask, axis=1), axis=1)
            scores = tf.where(scores_mask, -1e9, scores)
            
            v_mask = tf.expand_dims(tf.expand_dims(key_padding_mask, axis=1), axis=-1)
            v = tf.where(v_mask, 0.0, v)
        
        if attn_mask is not None:
            scores = scores + attn_mask
        
        # Softmax
        attn_weights = tf.nn.softmax(scores, axis=-1)
        attn_weights = self.dropout_layer(attn_weights, training=training)
        
        # Apply attention weights
        weighted_sum = tf.matmul(attn_weights, v)  # (N, H, L, D)
        
        # Apply head scaling
        if self.use_head_scale:
            weighted_sum = tf.einsum('bhtd,h->bhtd', weighted_sum, self.gamma)
        
        # Concatenate heads
        weighted_sum = tf.transpose(weighted_sum, [0, 2, 1, 3])  # (N, L, H, D)
        weighted_sum = tf.reshape(weighted_sum, [batch_size, -1, self.embed_dim])
        
        # Final linear projection
        output = self.out_linear(weighted_sum)
        
        return output


class AttentionBlock(tf.keras.layers.Layer):
    
    def __init__(self, embed_dim: int, num_heads: int, fc_dim: int, 
                 dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        
        self.embed_dim = embed_dim
        
        self.attention = MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            dropout=dropout
        )
        
        self.pre_attn_norm = tf.keras.layers.LayerNormalization()
        self.post_attn_norm = tf.keras.layers.LayerNormalization()
        
        self.pre_fc_norm = tf.keras.layers.LayerNormalization()
        self.post_fc_norm = tf.keras.layers.LayerNormalization()
        
        self.fc1 = tf.keras.layers.Dense(fc_dim)
        self.fc2 = tf.keras.layers.Dense(embed_dim)
        self.activation = tf.keras.layers.Activation('gelu')
        
        self.act_dropout = tf.keras.layers.Dropout(dropout)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        
        # Residual scaling parameter
        self.lambda_residual = self.add_weight(
            name='lambda_residual',
            shape=(embed_dim,),
            initializer='ones',
            trainable=True
        )
    
    def call(self, x, x_clt=None, attn_mask=None, key_padding_mask=None, training=None):
        
        if x_clt is not None:  # Class Attention Block
            # Add mask for class token
            if key_padding_mask is not None:
                clt_padding_mask = tf.zeros_like(key_padding_mask[:, 0:1])  # (N, 1)
                key_padding_mask = tf.concat([clt_padding_mask, key_padding_mask], axis=1)
            
            # First residual
            residual = x_clt
            
            # Multi-head attention
            combined_x = tf.concat([x_clt, x], axis=1)  # (N, 1+L, E)
            combined_x = self.pre_attn_norm(combined_x)
            x = self.attention(
                query=x_clt, 
                key=combined_x, 
                value=combined_x, 
                key_padding_mask=key_padding_mask,
                training=training
            )
            
        else:  # Particle Attention Block
            # First residual
            residual = x
            
            # Particle Multi-head attention
            x = self.pre_attn_norm(x)
            x = self.attention(
                query=x, 
                key=x, 
                value=x, 
                attn_mask=attn_mask, 
                key_padding_mask=key_padding_mask,
                training=training
            )
        
        x = self.post_attn_norm(x)
        x = self.dropout1(x, training=training)
        
        # First residual connection
        x = x + residual
        
        # Second residual
        residual = x
        
        # Feed forward network
        x = self.pre_fc_norm(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.act_dropout(x, training=training)
        x = self.post_fc_norm(x)
        x = self.fc2(x)
        x = self.dropout2(x, training=training)
        
        # Residual scaling
        residual = residual * self.lambda_residual
        
        # Second residual connection
        x = x + residual
        
        return x


class ParticleTransformer(tf.keras.Model):
    
    def __init__(self, score_dim: int, parameters: dict, **kwargs):
        """Particle Transformer.

        Args:
            score_dim : int
                Dimension of final output.
            parameters : dict
                Hyperparameters for the model.
        """
        super().__init__(**kwargs)
        
        self.score_dim = score_dim
        self.parameters = parameters
        
        # Particle Embedding
        self.par_embedding = ParticleFeatureEmbedding(
            input_dim=parameters['ParEmbed']['input_dim'],
            embedding_dims=parameters['ParEmbed']['embed_dim']
        )
        
        # Embedding dimension
        atte_embed_dim = parameters['ParEmbed']['embed_dim'][-1]
        
        # Particle Attention Blocks
        self.par_atte_blocks = [
            AttentionBlock(
                embed_dim=atte_embed_dim,
                **parameters['ParAtteBlock']
            ) for _ in range(parameters['num_ParAtteBlock'])
        ]
        
        # Class Attention Blocks
        self.class_atte_blocks = [
            AttentionBlock(
                embed_dim=atte_embed_dim,
                **parameters['ClassAtteBlock']
            ) for _ in range(parameters['num_ClassAtteBlock'])
        ]
        
        # Class token
        self.class_token = self.add_weight(
            name='class_token',
            shape=(1, 1, atte_embed_dim),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True
        )
        
        self.layer_norm = tf.keras.layers.LayerNormalization()
        
        self.fc = tf.keras.Sequential([
            tf.keras.layers.Dense(atte_embed_dim),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(0.1)
        ])
        
        self.final_layer = tf.keras.layers.Dense(score_dim)
    
    def call(self, x, training=None):
        """
        Args:
            x : tf.Tensor
                Input tensor of shape (N, L, D), where N is the batch size,
                L is the sequence length, and D is the feature dimension.
        """
        batch_size = tf.shape(x)[0]
        
        # Create padding mask
        key_padding_mask = tf.math.is_nan(x[..., 0])  # (N, L)
        
        # Fill NaN values with zeros
        x = tf.where(tf.expand_dims(key_padding_mask, axis=-1), 0.0, x)
        
        # Particle embedding
        x = self.par_embedding(x)  # (N, L, E)
        
        # Particle Attention Blocks
        for block in self.par_atte_blocks:
            x = block(x, x_clt=None, attn_mask=None, 
                     key_padding_mask=key_padding_mask, training=training)
        
        # Class Attention Blocks
        class_token = tf.tile(self.class_token, [batch_size, 1, 1])  # (N, 1, E)
        for block in self.class_atte_blocks:
            class_token = block(x, x_clt=class_token, 
                              key_padding_mask=key_padding_mask, training=training)
        
        # Final processing
        class_token = tf.squeeze(self.layer_norm(class_token), axis=1)  # (N, E)
        class_token = self.fc(class_token, training=training)
        class_token = self.final_layer(class_token)
        
        return class_token


class ParT_Baseline(ParticleTransformer):
    def __init__(self, num_channels: int = 3, **kwargs):
        hyperparameters = {
            "ParEmbed": {
                "input_dim": 3 + num_channels,  # (pt, eta, phi) + one-hot_encoding
                "embed_dim": [64, 512, 64]
            },
            "ParAtteBlock": {
                "num_heads": 8,
                "fc_dim": 512,
                "dropout": 0.1
            },
            "ClassAtteBlock": {
                "num_heads": 8,
                "fc_dim": 512,
                "dropout": 0.0
            },
            "num_ParAtteBlock": 6,
            "num_ClassAtteBlock": 2
        }
        super().__init__(score_dim=1, parameters=hyperparameters, **kwargs)


class ParT_Light(ParticleTransformer):
    def __init__(self, num_channels: int = 3, **kwargs):
        hyperparameters = {
            "ParEmbed": {
                "input_dim": 3 + num_channels,  # (pt, eta, phi) + one-hot_encoding
                "embed_dim": [64, 256, 64]
            },
            "ParAtteBlock": {
                "num_heads": 4,
                "fc_dim": 256,
                "dropout": 0.1
            },
            "ClassAtteBlock": {
                "num_heads": 4,
                "fc_dim": 256,
                "dropout": 0.0
            },
            "num_ParAtteBlock": 3,
            "num_ClassAtteBlock": 1
        }
        super().__init__(score_dim=1, parameters=hyperparameters, **kwargs)


# Example usage:
if __name__ == "__main__":
    # Create model
    model = ParT_Light(num_channels=3)
    
    # Example input (batch_size=2, sequence_length=10, features=6)
    x = tf.random.normal((2, 10, 6))
    
    # Add some NaN values to simulate padding
    x = tf.concat([x[:, :5], tf.fill([2, 5, 6], float('nan'))], axis=1)
    
    # Forward pass
    output = model(x)
    print(f"Output shape: {output.shape}")
    print(f"Model summary:")
    model.build(input_shape=(None, None, 6))
    model.summary()