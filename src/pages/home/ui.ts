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

import renderChart from 'vega-embed';

export async function plotData(container, xs, ys) {
  const xvals = await xs.data();
  const yvals = await ys.data();

  const values = Array.from(yvals).map((y, i) => {
    return {
      'x': xvals[i],
      'y': yvals[i]
    };
  });

  const spec = {
    '$schema': 'https://vega.github.io/schema/vega-lite/v2.json',
    'width': 300,
    'height': 300,
    'data': {
      'values': values
    },
    'mark': 'point',
    'encoding': {
      'x': {
        'field': 'x',
        'type': 'quantitative'
      },
      'y': {
        'field': 'y',
        'type': 'quantitative'
      }
    }
  };

  return renderChart(container, String(spec), {
    actions: false
  });
}

export async function plotDataAndPredictions(container, xs, ys, preds) {
  const xvals = await xs.data();
  const yvals = await ys.data();
  const predVals = await preds.data();

  const values = Array.from(yvals).map((y, i) => {
    return {
      'x': xvals[i],
      'y': yvals[i],
      pred: predVals[i]
    };
  });

  const spec = {
    '$schema': 'https://vega.github.io/schema/vega-lite/v2.json',
    'width': 300,
    'height': 300,
    'data': {
      'values': values
    },
    'layer': [{
      'mark': 'point',
      'encoding': {
        'x': {
          'field': 'x',
          'type': 'quantitative'
        },
        'y': {
          'field': 'y',
          'type': 'quantitative'
        }
      }
    },
    {
      'mark': 'line',
      'encoding': {
        'x': {
          'field': 'x',
          'type': 'quantitative'
        },
        'y': {
          'field': 'pred',
          'type': 'quantitative'
        },
        'color': {
          'value': 'tomato'
        }
      },
    }
    ]
  };

  return renderChart(container, String(spec), {
    actions: false
  });
}

export function renderCoefficients(container, coeff) {
  document.querySelector(container).innerHTML =
    `<span>a=${coeff.a.toFixed(3)}, b=${coeff.b.toFixed(3)}, c=${
    coeff.c.toFixed(3)},  d=${coeff.d.toFixed(3)}</span>`;
}

//////////////////////////////////////////////    mnist    /////////////////////////////////////

import * as tfvis from '@tensorflow/tfjs-vis';

let statusElement;
let messageElement;
let imagesElement;

export function isTraining() {
  statusElement = document.getElementById('status');
  statusElement.innerText = 'Training...';
}

const lossArr = [];
export function trainingLog(loss, iteration) {
  messageElement = document.getElementById('message');
  messageElement.innerText = `loss[${iteration}]: ${loss}`;
  lossArr.push({
    x: iteration,
    y: loss
  });
  const container = {
    name: 'Loss',
    tab: 'Training'
  };
  const options = {
    xLabel: 'Training Step',
    yLavel: 'Loss',
  };
  const data = {
    values: lossArr,
    series: ['loss']
  };
  tfvis.render.linechart(data, container, options);
}

export function showTestResults(batch, predictions, labels) {
  statusElement = document.getElementById('status');
  statusElement.innerText = 'Testing...';

  const testExamples = batch.xs.shape[0];
  let totalCorrect = 0;
  for (let i = 0; i < testExamples; i++) {
    const image = batch.xs.slice([i, 0], [1, batch.xs.shape[1]]);

    const div = document.createElement('div');
    div.className = 'pred-container';

    const canvas = document.createElement('canvas');
    draw(image.flatten(), canvas);

    const pred = document.createElement('div');

    const prediction = predictions[i];
    const label = labels[i];
    const correct = prediction === label;
    if (correct) {
      totalCorrect++;
    }

    pred.className = `pred ${(correct ? 'pred-correct' : 'pred-incorrect')}`;
    pred.innerText = `pred: ${prediction}`;

    div.appendChild(pred);
    div.appendChild(canvas);

    imagesElement = document.getElementById('images');
    imagesElement.appendChild(div);
  }

  const accuracy = 100 * totalCorrect / testExamples;
  const displayStr =
    `Accuracy: ${accuracy.toFixed(2)}% (${totalCorrect} / ${testExamples})`;
  messageElement = document.getElementById('message');
  messageElement.innerText = `${displayStr}\n`;
  console.log(displayStr);
}

export function draw(image, canvas) {
  const [width, height] = [28, 28];
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  const imageData = new ImageData(width, height);
  const data = image.dataSync();
  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;
    imageData.data[j + 0] = data[i] * 255;
    imageData.data[j + 1] = data[i] * 255;
    imageData.data[j + 2] = data[i] * 255;
    imageData.data[j + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}
