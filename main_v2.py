import sys

sys.path.insert(0, "./src")

import numpy as np
import einops
import tensorflow as tf
import tensorflow_text as tf_text
import matplotlib.pyplot as plt
import pathlib
import textwrap
from src.load_data_mod import load_data, tf_lower_and_split_punct, process_text
from src.EncoderMod import Encoder
from src.BahdanauAttentionMod import BahdanauAttention
from src.DecoderMod import Decoder, DecoderInput, call
from src.TrainTranslatorMod import (
    TrainTranslator,
    _loop_step,
    _preprocess,
    _tf_train_step,
    _train_step,
)
from src.MakeLossMod import MaskedLoss
from src.BatchLogsMod import BatchLogs
from src.TranslatorMod import (
    Translator,
    tokens_to_text,
    sample,
    translate_unrolled,
    translate_symbolic,
    tf_translate,
)
from src.plot_attention_mod import plot_attention

use_builtins = True

print("*** Descarga y prepara el conjunto de datos")

# Download the file
path_to_zip = tf.keras.utils.get_file(
    "spa-eng.zip",
    origin="http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip",
    extract=True,
)

path_to_file = pathlib.Path(path_to_zip).parent / "spa-eng/spa.txt"

target_raw, context_raw = load_data(path_to_file)
print(context_raw[-1])

print(target_raw[-1])

print("*** Crea un conjunto de datos tf.data")

BUFFER_SIZE = len(context_raw)
BATCH_SIZE = 64

is_train = np.random.uniform(size=(len(target_raw),)) < 0.8

train_raw = (
    tf.data.Dataset.from_tensor_slices((context_raw[is_train], target_raw[is_train]))
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
)
val_raw = (
    tf.data.Dataset.from_tensor_slices((context_raw[~is_train], target_raw[~is_train]))
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
)

for example_context_strings, example_target_strings in train_raw.take(1):
    print(example_context_strings[:5])
    print()
    print(example_target_strings[:5])
    break

print("*** Preprocesamiento de texto")

example_text = tf.constant("¿Todavía está en casa?")

print(example_text.numpy())
print(tf_text.normalize_utf8(example_text, "NFKD").numpy())

print(example_text.numpy().decode())
print(tf_lower_and_split_punct(example_text).numpy().decode())

max_vocab_size = 5000

context_text_processor = tf.keras.layers.TextVectorization(
    standardize=tf_lower_and_split_punct, max_tokens=max_vocab_size, ragged=True
)

context_text_processor.adapt(train_raw.map(lambda context, target: context))

# Here are the first 10 words from the vocabulary:
context_text_processor.get_vocabulary()[:10]

target_text_processor = tf.keras.layers.TextVectorization(
    standardize=tf_lower_and_split_punct, max_tokens=max_vocab_size, ragged=True
)

target_text_processor.adapt(train_raw.map(lambda context, target: target))
target_text_processor.get_vocabulary()[:10]

example_tokens = context_text_processor(example_context_strings)
example_tokens[:3, :]

context_vocab = np.array(context_text_processor.get_vocabulary())
tokens = context_vocab[example_tokens[0].numpy()]
" ".join(tokens)

plt.subplot(1, 2, 1)
plt.pcolormesh(example_tokens.to_tensor())
plt.title("Token IDs")

plt.subplot(1, 2, 2)
plt.pcolormesh(example_tokens.to_tensor() != 0)
plt.title("Mask")


### Process the dataset

train_ds = train_raw.map(process_text, tf.data.AUTOTUNE)
val_ds = val_raw.map(process_text, tf.data.AUTOTUNE)

for (ex_context_tok, ex_tar_in), ex_tar_out in train_ds.take(1):
    print(ex_context_tok[0, :10].numpy())
    print()
    print(ex_tar_in[0, :10].numpy())
    print(ex_tar_out[0, :10].numpy())


print("*** El modelo de codificador / decodificador")

embedding_dim = 256
units = 1024

print("*** El codificador")

# Convert the input text to tokens.
example_tokens = input_text_processor(example_input_batch)

# Encode the input sequence.
encoder = Encoder(input_text_processor.vocabulary_size(), embedding_dim, units)
example_enc_output, example_enc_state = encoder(example_tokens)

print(f"Input batch, shape (batch): {example_input_batch.shape}")
print(f"Input batch tokens, shape (batch, s): {example_tokens.shape}")
print(f"Encoder output, shape (batch, s, units): {example_enc_output.shape}")
print(f"Encoder state, shape (batch, units): {example_enc_state.shape}")

# Prueba la capa de atencion
print("*** La cabeza de atención")

attention_layer = BahdanauAttention(units)

(example_tokens != 0).shape

# Later, the decoder will generate this attention query
example_attention_query = tf.random.normal(shape=[len(example_tokens), 2, 10])

# Attend to the encoded tokens
context_vector, attention_weights = attention_layer(
    query=example_attention_query, value=example_enc_output, mask=(example_tokens != 0)
)

print(
    f"Attention result shape: (batch_size, query_seq_length, units):           {context_vector.shape}"
)
print(
    f"Attention weights shape: (batch_size, query_seq_length, value_seq_length): {attention_weights.shape}"
)

plt.subplot(1, 2, 1)
plt.pcolormesh(attention_weights[:, 0, :])
plt.title("Attention weights")

plt.subplot(1, 2, 2)
plt.pcolormesh(example_tokens != 0)
plt.title("Mask")

attention_weights.shape

attention_slice = attention_weights[0, 0].numpy()
attention_slice = attention_slice[attention_slice != 0]

plt.suptitle("Attention weights for one sequence")

plt.figure(figsize=(12, 6))
a1 = plt.subplot(1, 2, 1)
plt.bar(range(len(attention_slice)), attention_slice)
# freeze the xlim
plt.xlim(plt.xlim())
plt.xlabel("Attention weights")

a2 = plt.subplot(1, 2, 2)
plt.bar(range(len(attention_slice)), attention_slice)
plt.xlabel("Attention weights, zoomed")

# zoom in
top = max(a1.get_ylim())
zoom = 0.85 * top
a2.set_ylim([0.90 * top, top])
a1.plot(a1.get_xlim(), [zoom, zoom], color="k")

print("*** El decodificador")

Decoder.call = call

decoder = Decoder(output_text_processor.vocabulary_size(), embedding_dim, units)

# Convert the target sequence, and collect the "[START]" tokens
example_output_tokens = output_text_processor(example_target_batch)

start_index = output_text_processor.get_vocabulary().index("[START]")
first_token = tf.constant([[start_index]] * example_output_tokens.shape[0])

# Run the decoder
dec_result, dec_state = decoder(
    inputs=DecoderInput(
        new_tokens=first_token,
        enc_output=example_enc_output,
        mask=(example_tokens != 0),
    ),
    state=example_enc_state,
)

print(f"logits shape: (batch_size, t, output_vocab_size) {dec_result.logits.shape}")
print(f"state shape: (batch_size, dec_units) {dec_state.shape}")

sampled_token = tf.random.categorical(dec_result.logits[:, 0, :], num_samples=1)

vocab = np.array(output_text_processor.get_vocabulary())
first_word = vocab[sampled_token.numpy()]
first_word[:5]

dec_result, dec_state = decoder(
    DecoderInput(sampled_token, example_enc_output, mask=(example_tokens != 0)),
    state=dec_state,
)

print("*** Implementar el paso de formación")

TrainTranslator._preprocess = _preprocess

TrainTranslator._train_step = _train_step

TrainTranslator._loop_step = _loop_step

print("*** Prueba el paso de entrenamiento")

translator = TrainTranslator(
    embedding_dim,
    units,
    input_text_processor=input_text_processor,
    output_text_processor=output_text_processor,
    use_tf_function=False,
)

# Configure the loss and optimizer
translator.compile(
    optimizer=tf.optimizers.Adam(),
    loss=MaskedLoss(),
)

np.log(output_text_processor.vocabulary_size())

for n in range(10):
    print(translator.train_step([example_input_batch, example_target_batch]))
print()

TrainTranslator._tf_train_step = _tf_train_step

translator.use_tf_function = True

translator.train_step([example_input_batch, example_target_batch])

for n in range(10):
    print(translator.train_step([example_input_batch, example_target_batch]))
print()

losses = []
for n in range(100):
    print(".", end="")
    logs = translator.train_step([example_input_batch, example_target_batch])
    losses.append(logs["batch_loss"].numpy())

print()
plt.plot(losses)

train_translator = TrainTranslator(
    embedding_dim,
    units,
    input_text_processor=input_text_processor,
    output_text_processor=output_text_processor,
)

# Configure the loss and optimizer
train_translator.compile(
    optimizer=tf.optimizers.Adam(),
    loss=MaskedLoss(),
)

print("*** Entrena el modelo")

batch_loss = BatchLogs("batch_loss")

train_translator.fit(dataset, epochs=3, callbacks=[batch_loss])

plt.plot(batch_loss.logs)
plt.ylim([0, 3])
plt.xlabel("Batch #")
plt.ylabel("CE/token")

print("*** Traducir")

translator = Translator(
    encoder=train_translator.encoder,
    decoder=train_translator.decoder,
    input_text_processor=input_text_processor,
    output_text_processor=output_text_processor,
)

print("*** Convertir ID de token en texto")

Translator.tokens_to_text = tokens_to_text

example_output_tokens = tf.random.uniform(
    shape=[5, 2],
    minval=0,
    dtype=tf.int64,
    maxval=output_text_processor.vocabulary_size(),
)
translator.tokens_to_text(example_output_tokens).numpy()

print("*** Muestra de las predicciones del decodificador")

Translator.sample = sample

example_logits = tf.random.normal([5, 1, output_text_processor.vocabulary_size()])
example_output_tokens = translator.sample(example_logits, temperature=1.0)
example_output_tokens

print("*** Implementar el ciclo de traducción")

Translator.translate = translate_unrolled

input_text = tf.constant(
    [
        "hace mucho frio aqui.",  # "It's really cold here."
        "Esta es mi vida.",  # "This is my life.""
    ]
)

result = translator.translate(input_text=input_text)

print(result["text"][0].numpy().decode())
print(result["text"][1].numpy().decode())
print()

Translator.tf_translate = tf_translate

result = translator.tf_translate(input_text=input_text)

result = translator.tf_translate(input_text=input_text)

print(result["text"][0].numpy().decode())
print(result["text"][1].numpy().decode())
print()

print("*** [Opcional] Utilice un bucle simbólico")

Translator.translate = translate_symbolic

result = translator.translate(input_text=input_text)

print(result["text"][0].numpy().decode())
print(result["text"][1].numpy().decode())
print()

Translator.tf_translate = tf_translate

result = translator.tf_translate(input_text=input_text)

result = translator.tf_translate(input_text=input_text)

print(result["text"][0].numpy().decode())
print(result["text"][1].numpy().decode())
print()

a = result["attention"][0]

print(np.sum(a, axis=-1))

_ = plt.bar(range(len(a[0, :])), a[0, :])

plt.imshow(np.array(a), vmin=0.0)

print("*** Parcelas de atención etiquetadas")

i = 0
plot_attention(result["attention"][i], input_text[i], result["text"][i])

three_input_text = tf.constant(
    [
        # This is my life.
        "Esta es mi vida.",
        # Are they still home?
        "¿Todavía están en casa?",
        # Try to find out.'
        "Tratar de descubrir.",
    ]
)

result = translator.tf_translate(three_input_text)

for tr in result["text"]:
    print(tr.numpy().decode())

print()

three_input_text = tf.constant(
    [
        # This is my life.
        "Esta es mi vida.",
        # Are they still home?
        "¿Todavía están en casa?",
        # Try to find out.'
        "Tratar de descubrir.",
    ]
)

result = translator.tf_translate(three_input_text)

for tr in result["text"]:
    print(tr.numpy().decode())

print()

result["text"]

i = 0
plot_attention(result["attention"][i], three_input_text[i], result["text"][i])

i = 1
plot_attention(result["attention"][i], three_input_text[i], result["text"][i])

i = 2
plot_attention(result["attention"][i], three_input_text[i], result["text"][i])

long_input_text = tf.constant([inp[-1]])

print("Expected output:\n", "\n".join(textwrap.wrap(targ[-1])))

result = translator.tf_translate(long_input_text)

i = 0
plot_attention(result["attention"][i], long_input_text[i], result["text"][i])
_ = plt.suptitle("This never works")

print("*** Exportar")

tf.saved_model.save(
    translator, "translator", signatures={"serving_default": translator.tf_translate}
)

reloaded = tf.saved_model.load("translator")
result = reloaded.tf_translate(three_input_text)

result = reloaded.tf_translate(three_input_text)

for tr in result["text"]:
    print(tr.numpy().decode())

print()
