const canvas = document.getElementById('drawCanvas');
const ctx = canvas.getContext('2d');

const pixelSize = 10;
canvas.width = 280;
canvas.height = 280;

ctx.fillStyle = 'black';
let isDrawing = false;

function getMousePos(canvas, evt) {
    const rect = canvas.getBoundingClientRect();
    return {
        x: Math.floor((evt.clientX - rect.left) / (rect.width / canvas.width) / pixelSize),
        y: Math.floor((evt.clientY - rect.top) / (rect.height / canvas.height) / pixelSize)
    };
}

canvas.addEventListener('mousedown', (e) => {
    isDrawing = true;
    const pos = getMousePos(canvas, e);
    draw(pos.x, pos.y);
});

canvas.addEventListener('mousemove', (e) => {
    if (isDrawing) {
        const pos = getMousePos(canvas, e);
        draw(pos.x, pos.y);
    }
});

canvas.addEventListener('mouseup', () => isDrawing = false);
canvas.addEventListener('mouseout', () => isDrawing = false);

document.getElementById('clearButton').addEventListener('click', clearCanvas);
document.getElementById('predictButton').addEventListener('click', predictDigit);

function draw(x, y) {
    ctx.fillRect(x * pixelSize, y * pixelSize, pixelSize, pixelSize);
}

function drawGrid() {
    for (let x = 0; x <= canvas.width; x += pixelSize) {
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
    }

    for (let y = 0; y <= canvas.height; y += pixelSize) {
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
    }

    ctx.strokeStyle = '#EEE';
    ctx.stroke();
}

function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawGrid();
    document.getElementById('predictionOutput').innerText = 'None';
}

function predictDigit() {
    const dataURL = canvas.toDataURL('image/png');
    fetch('/predict-mnist', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image: dataURL })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('predictionOutput').innerText = data.prediction;
    })
    .catch(err => {
        console.error('Error:', err);
    });
}

drawGrid();
