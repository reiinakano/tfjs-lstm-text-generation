import * as tf from '@tensorflow/tfjs';
import indices_char from './indices_char';
import char_indices from './char_indices';

const seed = "The stupidity of man knows no bounds";

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
    this.generateButton = document.getElementById("generate-button");
    this.generateButton.onclick = () => {
      this.charsGenerated = 0;
      this.generatedSentence.innerText = seed;
      this.generateButton.disabled = true;
      this.generateButton.innerText = "Generating text.."
      this.generateText(seed);
    }
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

  generateText(text) {
    if (this.charsGenerated > 200) {
      this.generateButton.disabled = false;
      this.generateButton.innerText = "Generate new text";
      return;
    }
    const index = tf.tidy(() => {
      const input = this.convert(text);
      const prediction = this.model.predict(input).squeeze();
      return prediction.argMax();
    })
    index.data().then((indexData) => {
      this.charsGenerated += 1;
      index.dispose();
      console.log(indexData[0]);
      this.generatedSentence.innerText += indices_char[indexData[0]];
      tf.nextFrame().then(() => this.generateText(this.generatedSentence.innerText))
    });
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
    return input;
  }
}

window.addEventListener('load', () => new Main());
