import * as tf from '@tensorflow/tfjs';
import indices_char from './indices_char';
import char_indices from './char_indices';

const INPUT_LENGTH = 40;
const CHARS_TO_GENERATE = 200;

/**
 * Main application to start on window load
 */
class Main {
  /**
   * Constructor creates and initializes the variables needed for
   * the application
   */
  constructor() {
    // Initiate variables
    this.generatedSentence = document.getElementById("generated-sentence");
    this.inputSeed = document.getElementById("seed");
    this.generateButton = document.getElementById("generate-button");
    this.generateButton.onclick = () => {
      this.charsGenerated = 0;
      this.generatedSentence.innerText = this.inputSeed.value;
      this.generateButton.disabled = true;
      this.generateButton.innerText = "Pay attention to Nietzsche's words"
      this.generateText(this.inputSeed.value);
    }
    tf.loadModel('lstm/model.json').then((model) => {
      this.model = model;
      this.start();
    });
  }

  /**
   * Called after model has finished loading. 
   * Sets up UI elements for generating text.
   */
  start() {
    this.generateButton.innerText = "Generate new text";
    this.generateButton.disabled = false;
  }

  /**
   * Predicts next character from given text and updates UI accordingly.
   * This is the main tfjs loop.
   */
  generateText(text) {
    if (this.charsGenerated > CHARS_TO_GENERATE) {
      this.generateButton.disabled = false;
      this.generateButton.innerText = "Generate new text";
      return;
    }
    const index = tf.tidy(() => {
      const input = this.convert(text);
      const prediction = this.model.predict(input).squeeze();
      return this.sample(prediction);
    })
    index.data().then((indexData) => {
      this.charsGenerated += 1;
      index.dispose();
      this.generatedSentence.innerText += indices_char[indexData[0]];
      tf.nextFrame().then(() => this.generateText(this.generatedSentence.innerText))
    });
  }

  /**
   * Randomly samples next character weighted by model prediction.
   */
  sample(prediction) {
    return tf.tidy(() => {
      prediction = prediction.log();
      const diversity = tf.scalar(1.0);
      prediction = prediction.div(diversity);
      prediction = prediction.exp();
      prediction = prediction.div(prediction.sum());
      prediction = prediction.mul(tf.randomNormal(prediction.shape));
      return prediction.argMax();
    });
  }

  /**
   * Converts sentence to Tensor for feeding into model.
   */
  convert(sentence) {
    sentence = sentence.toLowerCase();
    sentence = sentence.split('').filter(x => x in char_indices).join('');
    if (sentence.length < INPUT_LENGTH) {
      sentence = sentence.padStart(INPUT_LENGTH);
    } else if (sentence.length > INPUT_LENGTH) {
      sentence = sentence.substring(sentence.length - INPUT_LENGTH);
    }
    const buffer = tf.buffer([1, INPUT_LENGTH, Object.keys(indices_char).length]);
    for (let i = 0; i < INPUT_LENGTH; i++) {
      let char = sentence.charAt(i)
      buffer.set(1, 0, i, char_indices[char]);
    }
    const input = buffer.toTensor();
    return input;
  }
}

window.addEventListener('load', () => new Main());
