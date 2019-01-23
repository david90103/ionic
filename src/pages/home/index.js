/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs';
import {
  generateData
} from './data';
import {
  plotData,
  plotDataAndPredictions,
  renderCoefficients
} from './ui';

// mnist
import {
  MnistData
} from './data';
import * as model from './model';
import * as ui from './ui';

/**
 * We want to learn the coefficients that give correct solutions to the
 * following cubic equation:
 *      y = a * x^3 + b * x^2 + c * x + d
 * In other words we want to learn values for:
 *      a
 *      b
 *      c
 *      d
 * Such that this function produces 'desired outputs' for y when provided
 * with x. We will provide some examples of 'xs' and 'ys' to allow this model
 * to learn what we mean by desired outputs and then use it to produce new
 * values of y that fit the curve implied by our example.
 */

// Step 1. Set up variables, these are the things we want the model
// to learn in order to do prediction accurately. We will initialize
// them with random values.
const a = tf.variable(tf.scalar(Math.random()));
const b = tf.variable(tf.scalar(Math.random()));
const c = tf.variable(tf.scalar(Math.random()));
const d = tf.variable(tf.scalar(Math.random()));


// Step 2. Create an optimizer, we will use this later. You can play
// with some of these values to see how the model performs.
let numIterations = 75;
let learningRate = 0.5;

const optimizer = tf.train.sgd(learningRate);

// add start button listener
let startbtn = document.getElementById('start');
startbtn.addEventListener('click', () => {
  numIterations = document.getElementById('iteration').value;
  learningRate = document.getElementById('learnrate').value;
  console.log(numIterations);
  console.log(learningRate);
  learnCoefficients();
});

// Step 3. Write our training process functions.

/*
 * This function represents our 'model'. Given an input 'x' it will try and
 * predict the appropriate output 'y'.
 *
 * It is also sometimes referred to as the 'forward' step of our training
 * process. Though we will use the same function for predictions later.
 *
 * @return number predicted y value
 */
function predict(x) {
  // y = a * x ^ 3 + b * x ^ 2 + c * x + d
  return tf.tidy(() => {
    return a.mul(x.pow(tf.scalar(3, 'int32')))
      .add(b.mul(x.square()))
      .add(c.mul(x))
      .add(d);
  });
}

/*
 * This will tell us how good the 'prediction' is given what we actually
 * expected.
 *
 * prediction is a tensor with our predicted y values.
 * labels is a tensor with the y values the model should have predicted.
 */
function loss(prediction, labels) {
  // Having a good error function is key for training a machine learning model
  const error = prediction.sub(labels).square().mean();
  return error;
}

/*
 * This will iteratively train our model.
 *
 * xs - training data x values
 * ys — training data y values
 */
async function train(xs, ys, numIterations) {
  for (let iter = 0; iter < numIterations; iter++) {
    // optimizer.minimize is where the training happens.

    // The function it takes must return a numerical estimate (i.e. loss)
    // of how well we are doing using the current state of
    // the variables we created at the start.

    // This optimizer does the 'backward' step of our training process
    // updating variables defined previously in order to minimize the
    // loss.
    optimizer.minimize(() => {
      // Feed the examples into the model
      const pred = predict(xs);
      return loss(pred, ys);
    });

    // Use tf.nextFrame to not block the browser.
    await tf.nextFrame();
  }
}

async function learnCoefficients() {
  const trueCoefficients = {
    a: -.8,
    b: -.2,
    c: .9,
    d: .5
  };
  const trainingData = generateData(100, trueCoefficients);

  // Plot original data
  renderCoefficients('#data .coeff', trueCoefficients);
  await plotData('#data .plot', trainingData.xs, trainingData.ys)

  // See what the predictions look like with random coefficients
  renderCoefficients('#random .coeff', {
    a: a.dataSync()[0],
    b: b.dataSync()[0],
    c: c.dataSync()[0],
    d: d.dataSync()[0],
  });
  const predictionsBefore = predict(trainingData.xs);
  await plotDataAndPredictions(
    '#random .plot', trainingData.xs, trainingData.ys, predictionsBefore);

  // Train the model!
  await train(trainingData.xs, trainingData.ys, numIterations);

  // See what the final results predictions are after training.

  renderCoefficients('#trained .coeff', {
    a: a.dataSync()[0],
    b: b.dataSync()[0],
    c: c.dataSync()[0],
    d: d.dataSync()[0],
  });
  const predictionsAfter = predict(trainingData.xs);
  await plotDataAndPredictions(
    '#trained .plot', trainingData.xs, trainingData.ys, predictionsAfter);

  predictionsBefore.dispose();
  predictionsAfter.dispose();
}

//----------------------------------------mnist-----------------------------------


let data;
async function load() {
  data = new MnistData();
  await data.load();
}

async function train2() {
  ui.isTraining();
  await model.train(data, ui.trainingLog);
}

async function test() {
  const testExamples = 50;
  const batch = data.nextTestBatch(testExamples);
  const predictions = model.predict(batch.xs);
  const labels = model.classesFromLabel(batch.labels);

  ui.showTestResults(batch, predictions, labels);
}

async function mnist() {
  await load();
  await train2();
  test();
}


//---------------------------------------- my model -----------------------------------
// Build and compile model.
const mymodel = tf.sequential();
mymodel.add(tf.layers.dense({
  units: 1,
  inputShape: [1]
}));
mymodel.compile({
  optimizer: 'sgd',
  loss: 'meanSquaredError'
});

// Generate some synthetic data for training.
const xs = tf.tensor2d([
  [1],
  [2],
  [3],
  [4]
], [4, 1]);
const ys = tf.tensor2d([
  [1],
  [3],
  [5],
  [7]
], [4, 1]);

// Train model with fit().
mymodel.fit(xs, ys, {
  epochs: 100
});

let mymodelbtn = document.getElementById('myModelStart');
mymodelbtn.addEventListener('click', () => {
  let mydata = document.getElementById('inputdata').value;
  // Run inference with predict().
  let result = mymodel.predict(tf.tensor2d([
    [parseInt(mydata)]
  ], [1, 1]));
  let answer = document.getElementById('mymodelanswer');
  answer.innerHTML = "Result is : " + result;

});


// learnCoefficients();
mnist();