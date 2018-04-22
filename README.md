This repository contains a demo written with TensorFlow.js that lets users generate their own Nietschze quote. An LSTM [written in Keras](https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py) is trained on a database of Nietschze's writings and attempts to generate Nietschze-like quotes. The Keras model is then ported to TensorFlow.js.

You can run it immediately in your browser by going to https://reiinakano.github.io/tfjs-lstm-text-generation/.

To run it locally, you must install Yarn and run the following command to get all the dependencies.

```bash
yarn prep
```

Then, you can run

```bash
yarn start
```

You can then browse to `localhost:9966` to view the application.

This demo was written for the book [Deep Learning in the Browser](https://github.com/backstopmedia/deep-learning-browser). You can check out the book's main repository [here](https://github.com/backstopmedia/deep-learning-browser).
