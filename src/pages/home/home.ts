import { Component } from '@angular/core';
import { NavController } from 'ionic-angular';
import * as model from './model';
import * as ui from './ui';
import * as data from './data';
import * as tf from '@tensorflow/tfjs';

@Component({
  selector: 'page-home',
  templateUrl: 'index.html'
})

export class HomePage {

  constructor(public navCtrl: NavController) {

  }

  ngAfterViewInit() {

    const a = tf.variable(tf.scalar(Math.random()));
    const b = tf.variable(tf.scalar(Math.random()));
    const c = tf.variable(tf.scalar(Math.random()));
    const d = tf.variable(tf.scalar(Math.random()));

    let numIterations = 75;
    let learningRate = 0.5;

    const optimizer = tf.train.sgd(learningRate);

    // add start button listener
    let startbtn = document.getElementById('start');
    startbtn.addEventListener('click', () => {
      numIterations = Number((<HTMLInputElement>document.getElementById('iteration')).value);
      learningRate = Number((<HTMLInputElement>document.getElementById('learnrate')).value);
      console.log(numIterations);
      console.log(learningRate);
      learnCoefficients();
    });

    function predict(x) {
      // y = a * x ^ 3 + b * x ^ 2 + c * x + d
      return tf.tidy(() => {
        return a.mul(x.pow(tf.scalar(3, 'int32')))
          .add(b.mul(x.square()))
          .add(c.mul(x))
          .add(d);
      });
    }


    function loss(prediction, labels) {
      // Having a good error function is key for training a machine learning model
      const error = prediction.sub(labels).square().mean();
      return error;
    }


    async function train(xs, ys, numIterations) {
      for (let iter = 0; iter < numIterations; iter++) {

        optimizer.minimize(() => {
          const pred = predict(xs);
          return loss(pred, ys);
        });
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
      const trainingData = data.generateData(100, trueCoefficients);

      // Plot original data
      ui.renderCoefficients('#data .coeff', trueCoefficients);
      await ui.plotData('#data .plot', trainingData.xs, trainingData.ys)

      // See what the predictions look like with random coefficients
      ui.renderCoefficients('#random .coeff', {
        a: a.dataSync()[0],
        b: b.dataSync()[0],
        c: c.dataSync()[0],
        d: d.dataSync()[0],
      });
      const predictionsBefore = predict(trainingData.xs);
      await ui.plotDataAndPredictions(
        '#random .plot', trainingData.xs, trainingData.ys, predictionsBefore);

      // Train the model!
      await train(trainingData.xs, trainingData.ys, numIterations);

      // See what the final results predictions are after training.

      ui.renderCoefficients('#trained .coeff', {
        a: a.dataSync()[0],
        b: b.dataSync()[0],
        c: c.dataSync()[0],
        d: d.dataSync()[0],
      });
      const predictionsAfter = predict(trainingData.xs);
      await ui.plotDataAndPredictions(
        '#trained .plot', trainingData.xs, trainingData.ys, predictionsAfter);

      predictionsBefore.dispose();
      predictionsAfter.dispose();
    }

    //----------------------------------------mnist-----------------------------------


    let mnistdata;
    async function load() {
      mnistdata = new data.MnistData();
      await mnistdata.load();
    }

    async function train2() {
      // ui.isTraining();
      await model.train(mnistdata, ui.trainingLog);
    }

    async function test() {
      const testExamples = 50;
      const batch = mnistdata.nextTestBatch(testExamples);
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
      let mydata = (<HTMLInputElement>document.getElementById('inputdata')).value;
      // Run inference with predict().
      let result = mymodel.predict(tf.tensor2d([
        [parseInt(mydata)]
      ], [1, 1]));
      let answer = document.getElementById('mymodelanswer');
      answer.innerHTML = "Result is : " + result;

    });

    // learnCoefficients();
    mnist();

  }
}
