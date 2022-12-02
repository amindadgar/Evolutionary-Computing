import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dropout, Layer
from tensorflow.keras.layers import Embedding, Input, GlobalAveragePooling1D, Dense
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential, Model
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

class TransformerBlock(Layer):
    def __init__(self, d_model, num_heads, feed_forward_layer1=None, feed_forward_layer2=None):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.layernorm = LayerNormalization(epsilon=1e-6)
        
        ## the created_model parameter will be None at first, because we've not created a model yet before first feed_forward layer
        self.feedforward1, self.layernorm1 = self.__create_feed_forward(d_model, feed_forward_layer1)
        self.feedforward2, self.layernorm2 = self.__create_feed_forward(d_model, feed_forward_layer2)
        

    def __create_feed_forward(self, d_model, feed_forward_configs):

        ## if the layer was available
        if feed_forward_configs is not None:

            ## extracting the information
            dense_layer_neuron_count = feed_forward_configs[0]
            if feed_forward_configs[1] == 'R':
                activation_function = 'relu'
            else:
                activation_function = 'sigmoid'

            dropout_rate = feed_forward_configs[2]

            created_model = Sequential()
            created_model.add(Dense(dense_layer_neuron_count, activation=activation_function))
            created_model.add(Dropout(dropout_rate))
            created_model.add(Dense(d_model))

            ## if layer normalization parameter was true
            layernorm = feed_forward_configs[3]
        else:
            created_model = None
        
        
        return created_model, layernorm


    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        out1 = self.layernorm(inputs + attn_output)
        
        if (self.feedforward1 is not None) and (self.feedforward2 is not None):
            out2 = self.feedforward1(out1)
            feedforward_output = self.feedforward2(out2)
            if self.layernorm1:
                feedforward_output = self.layernorm(out1 + feedforward_output)
        
        elif (self.feedforward1 is None) and (self.feedforward2 is not None):
            feedforward_output = self.feedforward2(out1)
            if self.layernorm2:
                feedforward_output = self.layernorm(out1 + feedforward_output)

        elif (self.feedforward1 is not None) and (self.feedforward2 is None):
            feedforward_output = self.feedforward1(out1)
            if self.layernorm1:
                feedforward_output = self.layernorm(out1 + feedforward_output)
        else:
            feedforward_output = out1

        return feedforward_output


class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, d_model):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=d_model)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=d_model)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


def load_data():
    vocab_size = 20000  # Only consider the top 20k words
    maxlen = 200  # Only consider the first 200 words of each movie review

    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
    print(len(x_train), "Training sequences")
    print(len(x_test), "Test sequences")

    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

    return x_train, x_test, y_train, y_test


def create_model(phenotype_chromosome, maxlen=200, vocab_size=2000):
    d_model = phenotype_chromosome[0]

    attention_layer_configs1 = phenotype_chromosome[1]
    attention_layer_configs2 = phenotype_chromosome[2]
    attention_layer_configs3 = phenotype_chromosome[3]

    ## the last feed forward layer of the network
    FFN_layer_configs = phenotype_chromosome[4]


    inputs = Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, d_model)
    x = embedding_layer(inputs)
    
    ## the the first attention layer was available to use
    ## if it was not available then all the tuple will be None, so no need to check all tuples
    ## that's why we checked the index=2
    if attention_layer_configs1[2] is not None:
        transformer_block1 = TransformerBlock(d_model, attention_layer_configs1[2], attention_layer_configs1[0], attention_layer_configs1[1])
        x = transformer_block1(x)
    if attention_layer_configs2[2] is not None:
        transformer_block2 = TransformerBlock(d_model, attention_layer_configs2[2], attention_layer_configs2[0], attention_layer_configs2[1])
        x = transformer_block2(x)
    if attention_layer_configs3[2] is not None:
        transformer_block3 = TransformerBlock(d_model, attention_layer_configs3[2], attention_layer_configs3[0], attention_layer_configs3[1])
        x = transformer_block3(x)

    x = GlobalAveragePooling1D()(x)   # dont edit this line

    if FFN_layer_configs[1] == 'R':
        FFN_activation_function = 'relu'
    else:
        FFN_activation_function = 'sigmoid'
    
    FFN_neuron_count = FFN_layer_configs[0]
    FFN_dropout_probability = FFN_layer_configs[2]
    
    x = Dense(FFN_neuron_count, activation=FFN_activation_function)(x)
    x = Dropout(FFN_dropout_probability)(x)
    outputs = Dense(2, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])


    return model

def start_training(phenotype_chromosome):

    model = create_model(phenotype_chromosome)
    x_train, x_test, y_train, y_test = load_data()

    model.fit(x_train, y_train, batch_size=64, epochs=5)