def convert_genotype_to_phenotype_values(chromsome):
    """
    Converts the chromsome from genotypic space into phenotypic values

    Parameters:
    ------------
    chromsome : string
        A chromsome in genotypic space
    
    Returns:
    ---------
    d_model : int
        the dimensionality of the model
    transformer_level1 : tuple
        a tuple with length of 4
        the first 3 indices are the feed forward parameters
        And the last one is the dropout probability value
    transformer_level2 : tuple
        a tuple with length of 4
        the first 3 indices are the feed forward parameters
        And the last one is the dropout probability value
    transformer_level3 : tuple
        a tuple with length of 4
        the first 3 indices are the feed forward parameters
        And the last one is the attention head count
    ffn_layer_architecture : tuple
        a tuple with length of 3, representing
            - Neuron Count : int
                5, 10, 20, or 30
            - Attention layer function type : character 
                Either 'R' for ReLU or `S` for Sigmoid
            - dropout probability : float 
                a value between 0 to 0.9
    """

    d_model_char, transformer_level1_str, transformer_level2_str, transformer_level3_str, ffn_layer_architecture_str = map_hyperparameters(chromsome)

    ## convert to actual values
    d_model = get_model_dimensionality(d_model_char)
    transformer_level1 = get_transformer_hyperparameters(transformer_level1_str)
    transformer_level2 = get_transformer_hyperparameters(transformer_level2_str)
    transformer_level3 = get_transformer_hyperparameters(transformer_level3_str)
    ffn_layer_architecture = get_feed_forward_hyperparameters(ffn_layer_architecture_str)


    return d_model, transformer_level1, transformer_level2, transformer_level3, ffn_layer_architecture


def map_hyperparameters(chromsome):
    """
    map the chromsome to hyperparameters of the problem
    exactly it divide the chromsome into different parts

    Parameters:
    ------------
    chromsome : string
        the chromsome in integer string
    
    Returns:
    ---------
    d_model : char
        the dimensionality of the model
    transformer_level1 : string
        4 characters
        the first 3 are the feed forward parameters
        And the last character is the dropout probability value
    transformer_level2 : string
        4 characters
        the first 3 are the feed forward parameters
        And the last character is the dropout probability value
    transformer_level3 : string
        4 characters
        the first 3 are the feed forward parameters
        And the last character is the attention head count
    ffn_layer_architecture : string
        3 characters, representing
            - Neuron Count
            - Attention layer function type
            - dropout probability
    
    """
    ## model dimensionality
    d_model = chromsome[0]

    ## first transformer layer configuration
    transformer_level1 = chromsome[1:5]

    ## second transformer layers configuration
    transformer_level2 = chromsome[5:9]

    ## third transformer layers configuration
    transformer_level3 = chromsome[9:13]

    ## the FFN, last layer configuration
    ffn_layer_architecture = chromsome[13:16]
    

    return d_model, transformer_level1, transformer_level2, transformer_level3, ffn_layer_architecture
        
    
def map_with_values(gene, possible_value_count):
    """
    map the gene with via possible values count
    if possible_value_count is 2, then 0-4 interval represents 0 and 5-9 represents 1

    Parameters:
    ------------
    gene : character
        the gene of the chromsome
    possible_value_count : int
        an integer value between 0 to 9
    """

    ## initialize the value variable
    mapped_value = None

    if possible_value_count == 2:
        if gene in ['0', '1', '2', '3', '4']:
            mapped_value = 0
        else:
            mapped_value = 1
    elif possible_value_count == 3:
        if gene in ['0', '1', '2', '3']:
            mapped_value = 0
        elif gene in ['4', '5', '6']:
            mapped_value = 1
        else:
            mapped_value = 2
    elif possible_value_count == 4:
        if gene in ['0', '1']:
            mapped_value = 0
        elif gene in ['2', '3', '4']:
            mapped_value = 1
        elif gene in ['5', '6', '7']:
            mapped_value = 2
        else:
            mapped_value = 3
    elif possible_value_count == 10:
        mapped_value = int(gene)
    else:
        raise ValueError(f'Wrong possible_value_count: {possible_value_count}!\nShould be 2, 3, 4 or 10!')

    return mapped_value

def get_feed_forward_hyperparameters(genes):
    """
    Get the feed forward hyperparameters from a 3 character string

    Parameters:
    ------------
    genes : 3 character string
        3 character representing
            - Neuron Count
            - Attention Function type
            - Dropout probability

    Returns:
    ---------
    neuron_count : int
        the actual neuron count, either 5, 10, 20, or 30
    attention_function_type : character
        the actual function type with a character showed
            - `R` representing `ReLU`
            - And `S` representing `Sigmoid`
    dropout_probability : float
        a float value representing 0, 0.1, 0.2, ..., 0.9
    """
    ## possible values for each hyperparameter
    Possible_neuron_values = [5, 10, 20, 30]
    Possible_attention_functions = ['R', 'S']
    Possible_dropouts = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    ## get the index of the value
    neuron_idx = map_with_values(genes[0], 4)
    attention_function_idx = map_with_values(genes[1], 2)
    dropout_idx = map_with_values(genes[2], 10)

    ## get the exact value
    neuron_count = Possible_neuron_values[neuron_idx]
    attention_function_type = Possible_attention_functions[attention_function_idx]
    dropout_probability = Possible_dropouts[dropout_idx]

    return neuron_count, attention_function_type, dropout_probability  

def get_transformer_hyperparameters(genes):
    """
    Get the transformer layer hyperparameters from a 4 character string

    Parameters:
    ------------
    genes : 4 character string
        the first 3 character representing
            - Neuron Count
            - Attention Function type
            - Dropout probability
        And the last one representing
            - Attention head count 

    Returns:
    ---------
    neuron_count : int
        the actual neuron count, either 5, 10, 20, or 30
    attention_function_type : character
        the actual function type with a character showed
            - `R` representing `ReLU`
            - And `S` representing `Sigmoid`
    dropout_probability : float
        a float value representing 0, 0.1, 0.2, ..., or 0.9
    attention_head_count : int
        an integer value representing values 1, 2, 4, or 8
    """

    neuron_count, attention_function_type, dropout_probability = get_feed_forward_hyperparameters(genes[0:3])
    

    Possible_attention_head_values = [1, 2, 4, 8]
    attention_head_idx = map_with_values(genes[3], 4)
    attention_head_count = Possible_attention_head_values[attention_head_idx]

    return neuron_count, attention_function_type, dropout_probability, attention_head_count

def get_model_dimensionality(gene):
    """
    get the model dimensionality from a one bit character

    Parameters:
    ------------
    gene : character
        one bit character representing the model dimensionality extracted from a chromsome
    
    Returns:
    ----------
    d_model_value : int
        either 16, 32, 64, 128 representing the model dimensionality
    """

    Possible_d_model = [16, 32, 64, 128]
    d_model_idx = map_with_values(gene, 4)

    d_model_value = Possible_d_model[d_model_idx]

    return d_model_value