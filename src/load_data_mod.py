import tensorflow as tf
import tensorflow_text as tf_text


def load_data(path):
    text = path.read_text(encoding="utf-8")

    lines = text.splitlines()
    pairs = [line.split("\t") for line in lines]

    inp = [inp for targ, inp in pairs]
    targ = [targ for targ, inp in pairs]

    return targ, inp


def tf_lower_and_split_punct(text):
    # Split accecented characters.
    text = tf_text.normalize_utf8(text, "NFKD")
    text = tf.strings.lower(text)
    # Keep space, a to z, and select punctuation.
    text = tf.strings.regex_replace(text, "[^ a-z.?!,¿]", "")
    # Add spaces around punctuation.
    text = tf.strings.regex_replace(text, "[.?!,¿]", r" \0 ")
    # Strip whitespace.
    text = tf.strings.strip(text)

    text = tf.strings.join(["[START]", text, "[END]"], separator=" ")
    return text
