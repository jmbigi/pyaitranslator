import tensorflow as tf
from typing import Any, Tuple
import typing
from BahdanauAttentionMod import BahdanauAttention
from ShapeCheckerMod import ShapeChecker
from CrossAttentionMod import CrossAttention


class Decoder(tf.keras.layers.Layer):
    @classmethod
    def add_method(cls, fun):
        setattr(cls, fun.__name__, fun)
        return fun

    def __init__(self, text_processor, units):
        super(Decoder, self).__init__()
        self.text_processor = text_processor
        self.vocab_size = text_processor.vocabulary_size()
        self.word_to_id = tf.keras.layers.StringLookup(
            vocabulary=text_processor.get_vocabulary(), mask_token="", oov_token="[UNK]"
        )
        self.id_to_word = tf.keras.layers.StringLookup(
            vocabulary=text_processor.get_vocabulary(),
            mask_token="",
            oov_token="[UNK]",
            invert=True,
        )
        self.start_token = self.word_to_id("[START]")
        self.end_token = self.word_to_id("[END]")

        self.units = units

        # 1. The embedding layer converts token IDs to vectors
        self.embedding = tf.keras.layers.Embedding(
            self.vocab_size, units, mask_zero=True
        )

        # 2. The RNN keeps track of what's been generated so far.
        self.rnn = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )

        # 3. The RNN output will be the query for the attention layer.
        self.attention = CrossAttention(units)

        # 4. This fully connected layer produces the logits for each
        # output token.
        self.output_layer = tf.keras.layers.Dense(self.vocab_size)


@Decoder.add_method
def call(self, context, x, state=None, return_state=False):
    shape_checker = ShapeChecker()
    shape_checker(x, "batch t")
    shape_checker(context, "batch s units")

    # 1. Lookup the embeddings
    x = self.embedding(x)
    shape_checker(x, "batch t units")

    # 2. Process the target sequence.
    x, state = self.rnn(x, initial_state=state)
    shape_checker(x, "batch t units")

    # 3. Use the RNN output as the query for the attention over the context.
    x = self.attention(x, context)
    self.last_attention_weights = self.attention.last_attention_weights
    shape_checker(x, "batch t units")
    shape_checker(self.last_attention_weights, "batch t s")

    # Step 4. Generate logit predictions for the next token.
    logits = self.output_layer(x)
    shape_checker(logits, "batch t target_vocab_size")

    if return_state:
        return logits, state
    else:
        return logits


@Decoder.add_method
def get_initial_state(self, context):
    batch_size = tf.shape(context)[0]
    start_tokens = tf.fill([batch_size, 1], self.start_token)
    done = tf.zeros([batch_size, 1], dtype=tf.bool)
    embedded = self.embedding(start_tokens)
    return start_tokens, done, self.rnn.get_initial_state(embedded)[0]


@Decoder.add_method
def tokens_to_text(self, tokens):
    words = self.id_to_word(tokens)
    result = tf.strings.reduce_join(words, axis=-1, separator=" ")
    result = tf.strings.regex_replace(result, "^ *\[START\] *", "")
    result = tf.strings.regex_replace(result, " *\[END\] *$", "")
    return result


@Decoder.add_method
def get_next_token(self, context, next_token, done, state, temperature=0.0):
    logits, state = self(context, next_token, state=state, return_state=True)

    if temperature == 0.0:
        next_token = tf.argmax(logits, axis=-1)
    else:
        logits = logits[:, -1, :] / temperature
        next_token = tf.random.categorical(logits, num_samples=1)

    # If a sequence produces an `end_token`, set it `done`
    done = done | (next_token == self.end_token)
    # Once a sequence is done it only produces 0-padding.
    next_token = tf.where(done, tf.constant(0, dtype=tf.int64), next_token)

    return next_token, done, state
