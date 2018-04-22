import * as tf from '@tensorflow/tfjs';
import indices_char from './indices_char';
import char_indices from './char_indices';

const seed = "The wisdom of man";

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
    this.generateButton = document.getElementById("generate-button");
    this.generateButton.onclick = () => this.generateText();
    this.generatedSentence = document.getElementById("generated-sentence");
    tf.loadModel('lstm/model.json').then((model) => {
      this.model = model;
      this.start();
    });
  }

  start() {
    console.log('starting...');
    this.generateButton.innerText = "Generate new text";
    this.generateButton.disabled = false;
  }

  generateText() {
    const prediction = tf.tidy(() => {
      const input = this.convert(seed);
      return this.model.predict(input).squeeze();
      const index = this.sample(prediction);
      console.log(index);
    })
    prediction.print();
    this.sample(prediction).then((index) => console.log(index[0]));
    prediction.dispose();
  }

  sample(prediction) {
    const index = tf.tidy(() => {
      // TODO: Actually make this matter
      prediction = prediction.log();
      let diversity = tf.scalar(1.0);
      prediction = prediction.div(diversity);
      prediction = prediction.exp();
      prediction = prediction.div(prediction.sum());
      return prediction.argMax();
    })
    index.print();
    return index.data();
  }

  convert(sentence) {
    console.log(`converting ${sentence}`);
    // TODO: Handle OOV characters
    if (sentence.length < 40) {
      sentence = sentence.padStart(40);
    } else if (sentence.length > 40) {
      sentence = sentence.substring(sentence.length - 40);
    }
    console.log(`converted to ${sentence}`);
    const buffer = tf.buffer([1, 40, Object.keys(indices_char).length]);
    for (let i = 0; i < 40; i++) {
      let char = sentence.charAt(i)
      buffer.set(1, 0, i, char_indices[char]);
    }
    const input = buffer.toTensor();
    input.print();
    return input;
  }
}

window.addEventListener('load', () => new Main());
