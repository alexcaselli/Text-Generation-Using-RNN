# RNN Text Generation using TensorFlow
Give a large book from Project Gutenberg (e.g., The Count of Monte Cristo). Train a network to **predic the next textual character given a sequence of characters**. Such network can be used to generate text by sampling a character given the first one.

This project uses **TensorFlow and LSTM cells to train a recurrent mode**l able to generate sentences in plain English.

The same model can be used to **learn from your Json exported Telegram conversations** given a preprocessing function available in this repository which cast the Json formatted chat to the plain txt version giving some funny results.
