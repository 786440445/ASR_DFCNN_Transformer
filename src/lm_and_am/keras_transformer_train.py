import numpy as np
from keras_transformer import get_model

# Build a small toy token dictionary
tokens = 'all work and no play makes jack a dull boy'.split(' ')
token_dict = {
    '<PAD>': 0,
    '<START>': 1,
    '<END>': 2,
}
for token in tokens:
    if token not in token_dict:
        token_dict[token] = len(token_dict)

# Generate toy data
encoder_inputs_no_padding = []
encoder_inputs, decoder_inputs, decoder_outputs = [], [], []
for i in range(1, len(tokens) - 1):
    encode_tokens, decode_tokens = tokens[:i], tokens[i:]
    encode_tokens = ['<START>'] + encode_tokens + ['<END>'] + ['<PAD>'] * (len(tokens) - len(encode_tokens))
    output_tokens = decode_tokens + ['<END>', '<PAD>'] + ['<PAD>'] * (len(tokens) - len(decode_tokens))
    decode_tokens = ['<START>'] + decode_tokens + ['<END>'] + ['<PAD>'] * (len(tokens) - len(decode_tokens))
    encode_tokens = list(map(lambda x: token_dict[x], encode_tokens))
    decode_tokens = list(map(lambda x: token_dict[x], decode_tokens))
    output_tokens = list(map(lambda x: [token_dict[x]], output_tokens))
    encoder_inputs_no_padding.append(encode_tokens[:i + 2])
    encoder_inputs.append(encode_tokens)
    decoder_inputs.append(decode_tokens)
    decoder_outputs.append(output_tokens)

# Build the model
model = get_model(
    token_num=len(token_dict),
    embed_dim=512,
    encoder_num=6,
    decoder_num=6,
    head_num=8,
    hidden_dim=512,
    attention_activation='relu',
    feed_forward_activation='relu',
    dropout_rate=0.05,
    embed_weights=np.random.random((13, 512)),
)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
)
model.summary()

# Train the model
model.fit(
    x=[np.asarray(encoder_inputs * 1000), np.asarray(decoder_inputs * 1000)],
    y=np.asarray(decoder_outputs * 1000),
    epochs=5,
)