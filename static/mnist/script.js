const canvas = document.getElementById('drawCanvas');
const ctx = canvas.getContext('2d');

const pixelSize = 10;
canvas.width = 280;
canvas.height = 280;

ctx.fillStyle = 'black';
let isDrawing = false;

let chartInstance = null;

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
    throttledPredictDigit();
});

canvas.addEventListener('mousemove', (e) => {
    if (isDrawing) {
        const pos = getMousePos(canvas, e);
        draw(pos.x, pos.y);
        throttledPredictDigit();
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

    if (chartInstance) {
        updatePredictionDistribution(Array(10).fill(0));
    }
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
        updatePredictionDistribution(data.distribution);
    })
    .catch(err => {
        console.error('Error:', err);
    });
}

function throttle(func, limit) {
    let lastFunc;
    let lastRan;

    return function() {
        const context = this;
        const args = arguments;
        if (!lastRan) {
            func.apply(context, args);
            lastRan = Date.now();
        } else {
            clearTimeout(lastFunc);
            lastFunc = setTimeout(function() {
                if ((Date.now() - lastRan) >= limit) {
                    func.apply(context, args);
                    lastRan = Date.now();
                }
            }, limit - (Date.now() - lastRan));
        }
    };
}

function initializeChart() {
    const labels = Array.from({ length: 10 }, (_, i) => i);

    const data = {
        labels: labels,
        datasets: [{
            label: 'Prediction Probability',
            data: Array(10).fill(0),
            backgroundColor: '#a4b6ef'
        }]
    };

    const config = {
        type: 'bar',
        data: data,
        options: {
            responsive: false,
            maintainAspectRatio: false,
            drawBorder: false,
            scales: {
                y: {
                    beginAtZero: true,
                    min: 0,
                    max: 1,
                    ticks: {
                        callback: function(value) {
                            if (value === 0 || value === 1) {
                                return value;
                            }
                            return '';
                        },
                        color: '#fff',
                        font: {
                            size: 14,
                            weight: 'bold'
                        }
                    },
                    grid: {
                        display: false,
                        borderDash: [3, 3],
                        color: 'rgba(200, 200, 200, 0.8)',
                        drawBorder: false
                    },
                    title: {
                        display: true,
                        text: 'Probability',
                        color: '#fff',
                        font: {
                            size: 16,
                            weight: 'bold'
                        }
                    }
                },
                x: {
                    grid: {
                        display: false,
                        drawBorder: false
                    },
                    ticks: {
                        color: '#fff',
                        font: {
                            size: 14,
                            weight: 'bold'
                        },
                    },
                    title: {
                        display: true,
                        text: 'Digit',
                        color: '#fff',
                        font: {
                            size: 16,
                            weight: 'bold'
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    labels: {
                        font: {
                            size: 18,
                            weight: 'bold'
                        },
                        usePointStyle: true,
                        boxWidth: 0,
                        filter: function(item, chart) {
                            return item.text !== 'Prediction Probability';
                        }
                    }
                }
            },
            onClick: (e) => {
                const points = chartInstance.getElementsAtEventForMode(e, 'nearest', { intersect: true }, false);
                if (points.length) {
                    const firstPoint = points[0];
                    const label = chartInstance.data.labels[firstPoint.index];
                    if (label === 'Prediction Probability') {
                        e.stopPropagation();
                    }
                }
            },
            layout: {
                padding: {
                    left: 20,
                    right: 20,
                    top: 20,
                    bottom: 20
                },
            },
        }
    };

    const ctx = document.getElementById('distributionChart').getContext('2d');
    chartInstance = new Chart(ctx, config);
}

function updatePredictionDistribution(distribution) {
    if (chartInstance) {
        chartInstance.data.datasets[0].data = distribution;
        chartInstance.update();
    }
}

drawGrid();
initializeChart();
const throttledPredictDigit = throttle(predictDigit, 1000);
