import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from load_data_mod import tf_lower_and_split_punct


def plot_attention(attention, sentence, predicted_sentence):
    sentence = tf_lower_and_split_punct(sentence).numpy().decode().split()
    predicted_sentence = predicted_sentence.numpy().decode().split() + ["[END]"]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)

    attention = attention[: len(predicted_sentence), : len(sentence)]

    ax.matshow(attention, cmap="viridis", vmin=0.0)

    fontdict = {"fontsize": 14}

    ax.set_xticklabels([""] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([""] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    ax.set_xlabel("Input text")
    ax.set_ylabel("Output text")
    plt.suptitle("Attention weights")
