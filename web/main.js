import * as ort from 'onnxruntime-web';
import cv from '@techstark/opencv-js';
import './style.css';

navigator.getUserMedia = (navigator.getUserMedia ||
    navigator.webkitGetUserMedia ||
    navigator.mozGetUserMedia ||
    navigator.msGetUserMedia);

let solveBtn = document.querySelector('button#solve');
let sudokuOutline = document.querySelector('div.outline');

// Initalize localizer model
let session = null;
ort.InferenceSession.create('/localizer.ort', {})
    .then((sess) => { session = sess; })
    .catch((e) => {
        alert('Could not load localizer model.');
        console.error(e);
    });

// Request permission for camera, and set it as video source
let video = document.querySelector('.camera video');
navigator.mediaDevices.getUserMedia({
    video: { aspectRatio: { exact: 1.0 }, facingMode: { ideal: 'environment' } },
    audio: false
}).then((stream) => {
    video.srcObject = stream;
    video.play();
}).catch((e) => {
    alert('You need to enable camera permissions to use this webpage.')
    console.error(e);
});


// The width and height of the input image required by the model
const WIDTH = 400, HEIGHT = 400;
let canvas = document.createElement('canvas');
canvas.width = WIDTH;
canvas.height = HEIGHT;

function main() {
    if (session === null || video.paused) {
        setTimeout(main, 200);
        return;
    }

    // Start processing frames
    processFrame();
}


async function processFrame() {
    let context = canvas.getContext('2d', { willReadFrequently: true });
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    let mat = cv.imread(canvas);
    cv.cvtColor(mat, mat, cv.COLOR_RGB2GRAY);
    mat.convertTo(mat, cv.CV_32F, 1. / 255);
    
    const input = new ort.Tensor('float32', mat.data32F, [1, HEIGHT, WIDTH]);
    const { classification, localization } = await session.run({ input });

    const sudokuPresent = 0.9 < classification.data[0];
    if (sudokuPresent) {
        let [x1, y1, x2, y2] = localization.data;
        let scaleX = video.clientWidth / WIDTH, scaleY = video.clientHeight / HEIGHT;

        let padding = 20;
        sudokuOutline.style.top = y1 * scaleY - padding + 'px';
        sudokuOutline.style.left = x1 * scaleX - padding + 'px';
        sudokuOutline.style.width = (x2 - x1) * scaleX + padding * 2 + 'px';
        sudokuOutline.style.height = (y2 - y1) * scaleY + padding * 2 + 'px';

        // // sudokuOutline.style.top = 
        let pt1 = new cv.Point(localization.data[0], localization.data[1]);
        let pt2 = new cv.Point(localization.data[2], localization.data[3]);
        let color = new cv.Scalar(255, 0, 0, 255);
        cv.rectangle(mat, pt1, pt2, color, 2);
        sudokuOutline.style.opacity = "100%";
        solveBtn.disabled = false;
    } else {
        sudokuOutline.style.opacity = "0%";
        solveBtn.disabled = true;
    }

    cv.imshow(canvas, mat);
    mat.delete();

    setTimeout(processFrame, 20);
}

// Show canvas
document.body.appendChild(canvas);

if (cv.getBuildInformation) {
    main();
} else {
    // WASM
    cv['onRuntimeInitialized'] = main;
}
