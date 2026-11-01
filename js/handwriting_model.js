/**
 * Handwritten Digit Recognition (Task 3) - Browser Neural Network Inference Engine
 * Trained on MNIST Dataset (70,000 images) with MLP Neural Network
 * Architecture: 784 -> 256 (ReLU) -> 128 (ReLU) -> 10 (Softmax)
 * Test Accuracy: 94.61%
 */

class HandwritingRecognitionEngine {
  constructor() {
    this.weightsLoaded = false;
    this.scalerMean = null;
    this.scalerScale = null;
    this.W0 = null; // 784 * 256
    this.b0 = null; // 256
    this.W1 = null; // 256 * 128
    this.b1 = null; // 128
    this.W2 = null; // 128 * 10
    this.b2 = null; // 10

    // Drawing Canvas
    this.canvas = null;
    this.ctx = null;
    this.isDrawing = false;
    this.brushSize = 22;
    this.isEraser = false;
    this.undoStack = [];
    this.maxUndos = 20;

    // 28x28 Preview Canvas
    this.previewCanvas = null;
    this.previewCtx = null;

    this.init();
  }

  async init() {
    try {
      await this.loadWeights();
      this.initCanvases();
      this.bindEvents();
      console.log('Handwriting Recognition Engine initialized successfully.');

      // Load sample digit '3' by default
      this.loadSampleDigit(3);
    } catch (err) {
      console.error('Failed to initialize handwriting engine:', err);
    }
  }

  async loadWeights() {
    const res = await fetch('data/handwriting_weights.bin');
    if (!res.ok) throw new Error(`HTTP ${res.status} loading handwriting weights`);
    const buffer = await res.arrayBuffer();
    const f32 = new Float32Array(buffer);

    let offset = 0;
    // 1. Scaler Mean (784)
    this.scalerMean = f32.subarray(offset, offset + 784);
    offset += 784;

    // 2. Scaler Scale (784)
    this.scalerScale = f32.subarray(offset, offset + 784);
    offset += 784;

    // 3. W0 (784 * 256)
    this.W0 = f32.subarray(offset, offset + (784 * 256));
    offset += 784 * 256;

    // 4. b0 (256)
    this.b0 = f32.subarray(offset, offset + 256);
    offset += 256;

    // 5. W1 (256 * 128)
    this.W1 = f32.subarray(offset, offset + (256 * 128));
    offset += 256 * 128;

    // 6. b1 (128)
    this.b1 = f32.subarray(offset, offset + 128);
    offset += 128;

    // 7. W2 (128 * 10)
    this.W2 = f32.subarray(offset, offset + (128 * 10));
    offset += 128 * 10;

    // 8. b2 (10)
    this.b2 = f32.subarray(offset, offset + 10);
    offset += 10;

    this.weightsLoaded = true;
  }

  initCanvases() {
    this.canvas = document.getElementById('digit-drawing-canvas');
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });

    this.previewCanvas = document.getElementById('digit-preview-28');
    if (this.previewCanvas) {
      this.previewCtx = this.previewCanvas.getContext('2d');
    }

    this.clearCanvas(false);
  }

  clearCanvas(triggerPredict = true) {
    if (!this.ctx) return;
    this.ctx.fillStyle = '#000000';
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    this.saveState();
    if (triggerPredict) {
      this.resetPredictionDisplay();
    }
  }

  saveState() {
    if (!this.ctx) return;
    const imgData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
    this.undoStack.push(imgData);
    if (this.undoStack.length > this.maxUndos) {
      this.undoStack.shift();
    }
  }

  undo() {
    if (this.undoStack.length > 1) {
      this.undoStack.pop(); // discard current
      const prevState = this.undoStack[this.undoStack.length - 1];
      this.ctx.putImageData(prevState, 0, 0);
      this.predictCurrentCanvas();
    } else {
      this.clearCanvas(true);
    }
  }

  bindEvents() {
    if (!this.canvas) return;

    // Mouse events
    const startDraw = (e) => {
      this.isDrawing = true;
      this.draw(e);
    };

    const stopDraw = () => {
      if (this.isDrawing) {
        this.isDrawing = false;
        this.ctx.beginPath();
        this.saveState();
        this.predictCurrentCanvas();
      }
    };

    const moveDraw = (e) => {
      if (!this.isDrawing) return;
      this.draw(e);
    };

    this.canvas.addEventListener('mousedown', startDraw);
    window.addEventListener('mouseup', stopDraw);
    this.canvas.addEventListener('mousemove', moveDraw);

    // Touch events for mobile/tablet
    const getTouchPos = (touchEvent) => {
      const rect = this.canvas.getBoundingClientRect();
      const touch = touchEvent.touches[0];
      return {
        clientX: touch.clientX,
        clientY: touch.clientY
      };
    };

    this.canvas.addEventListener('touchstart', (e) => {
      e.preventDefault();
      startDraw(getTouchPos(e));
    }, { passive: false });

    window.addEventListener('touchend', stopDraw);

    this.canvas.addEventListener('touchmove', (e) => {
      e.preventDefault();
      moveDraw(getTouchPos(e));
    }, { passive: false });

    // Canvas tools
    const clearBtn = document.getElementById('digit-clear-btn');
    if (clearBtn) clearBtn.addEventListener('click', () => {
      if (window.app && window.app.playSound) window.app.playSound('click');
      this.clearCanvas(true);
    });

    const undoBtn = document.getElementById('digit-undo-btn');
    if (undoBtn) undoBtn.addEventListener('click', () => {
      if (window.app && window.app.playSound) window.app.playSound('click');
      this.undo();
    });

    const eraserBtn = document.getElementById('digit-eraser-btn');
    if (eraserBtn) eraserBtn.addEventListener('click', () => {
      this.isEraser = !this.isEraser;
      eraserBtn.classList.toggle('active', this.isEraser);
    });

    const brushSlider = document.getElementById('digit-brush-size');
    if (brushSlider) {
      brushSlider.addEventListener('input', (e) => {
        this.brushSize = parseInt(e.target.value);
        const label = document.getElementById('digit-brush-val');
        if (label) label.textContent = `${this.brushSize}px`;
      });
    }

    const centerBtn = document.getElementById('digit-center-btn');
    if (centerBtn) {
      centerBtn.addEventListener('click', () => {
        this.centerBoundingBox();
        this.predictCurrentCanvas();
      });
    }

    // Sample Digits Buttons (0-9)
    document.querySelectorAll('[data-sample-digit]').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const digit = parseInt(e.currentTarget.dataset.sampleDigit);
        this.loadSampleDigit(digit);
      });
    });

    // Custom Image Upload
    const fileInput = document.getElementById('digit-file-input');
    const uploadBtn = document.getElementById('digit-upload-btn');
    if (uploadBtn && fileInput) {
      uploadBtn.addEventListener('click', () => fileInput.click());
      fileInput.addEventListener('change', (e) => this.handleImageUpload(e));
    }
  }

  draw(e) {
    const rect = this.canvas.getBoundingClientRect();
    const scaleX = this.canvas.width / rect.width;
    const scaleY = this.canvas.height / rect.height;

    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    this.ctx.lineWidth = this.brushSize;
    this.ctx.lineCap = 'round';
    this.ctx.lineJoin = 'round';

    if (this.isEraser) {
      this.ctx.strokeStyle = '#000000';
    } else {
      this.ctx.strokeStyle = '#ffffff';
    }

    this.ctx.lineTo(x, y);
    this.ctx.stroke();
    this.ctx.beginPath();
    this.ctx.moveTo(x, y);
  }

  loadSampleDigit(digit) {
    if (window.app && window.app.playSound) window.app.playSound('click');

    // Update active button state
    document.querySelectorAll('[data-sample-digit]').forEach(b => {
      b.classList.toggle('active', parseInt(b.dataset.sampleDigit) === digit);
    });

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = `assets/digits/digit_${digit}.png`;
    img.onload = () => {
      this.ctx.fillStyle = '#000000';
      this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

      // Draw centered 28x28 scaled up to 280x280
      this.ctx.imageSmoothingEnabled = false;
      this.ctx.drawImage(img, 28, 28, 224, 224);

      this.saveState();
      this.predictCurrentCanvas();
    };
  }

  handleImageUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    if (window.app && window.app.playSound) window.app.playSound('click');
    const reader = new FileReader();
    reader.onload = (evt) => {
      const img = new Image();
      img.onload = () => {
        this.ctx.fillStyle = '#000000';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Calculate aspect fit
        const maxDim = 224;
        let w = img.width;
        let h = img.height;
        if (w > h) {
          h = Math.round((h * maxDim) / w);
          w = maxDim;
        } else {
          w = Math.round((w * maxDim) / h);
          h = maxDim;
        }
        const dx = (this.canvas.width - w) / 2;
        const dy = (this.canvas.height - h) / 2;

        this.ctx.drawImage(img, dx, dy, w, h);

        // If background is bright, invert it to match MNIST (white digit on black bg)
        this.autoInvertIfBright();

        this.saveState();
        this.predictCurrentCanvas();
      };
      img.src = evt.target.result;
    };
    reader.readAsDataURL(file);
  }

  autoInvertIfBright() {
    const imgData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
    const data = imgData.data;
    let totalBrightness = 0;
    const numPixels = data.length / 4;

    for (let i = 0; i < data.length; i += 4) {
      const gray = (data[i] * 0.299) + (data[i + 1] * 0.587) + (data[i + 2] * 0.114);
      totalBrightness += gray;
    }

    const meanBrightness = totalBrightness / numPixels;
    // Invert if white paper/background (> 127)
    if (meanBrightness > 127) {
      for (let i = 0; i < data.length; i += 4) {
        data[i] = 255 - data[i];
        data[i + 1] = 255 - data[i + 1];
        data[i + 2] = 255 - data[i + 2];
      }
      this.ctx.putImageData(imgData, 0, 0);
    }
  }

  centerBoundingBox() {
    const w = this.canvas.width;
    const h = this.canvas.height;
    const imgData = this.ctx.getImageData(0, 0, w, h);
    const data = imgData.data;

    let minX = w, maxX = 0, minY = h, maxY = 0;
    let found = false;

    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = (y * w + x) * 4;
        const brightness = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        if (brightness > 30) {
          found = true;
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }

    if (!found) return;

    const boxW = maxX - minX + 1;
    const boxH = maxY - minY + 1;

    // Create offscreen canvas for cropped digit
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = boxW;
    tempCanvas.height = boxH;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(this.canvas, minX, minY, boxW, boxH, 0, 0, boxW, boxH);

    // Clear main canvas
    this.ctx.fillStyle = '#000000';
    this.ctx.fillRect(0, 0, w, h);

    // Draw centered with margin
    const targetSize = 210;
    let drawW, drawH;
    if (boxW > boxH) {
      drawW = targetSize;
      drawH = (boxH * targetSize) / boxW;
    } else {
      drawH = targetSize;
      drawW = (boxW * targetSize) / boxH;
    }

    const posX = (w - drawW) / 2;
    const posY = (h - drawH) / 2;
    this.ctx.drawImage(tempCanvas, posX, posY, drawW, drawH);
    this.saveState();
  }

  // --- Preprocessing & 28x28 Rasterization ---
  getNormalized28x28() {
    // Render 28x28 offscreen
    const offscreen = document.createElement('canvas');
    offscreen.width = 28;
    offscreen.height = 28;
    const offCtx = offscreen.getContext('2d');

    offCtx.fillStyle = '#000000';
    offCtx.fillRect(0, 0, 28, 28);
    offCtx.imageSmoothingEnabled = true;
    offCtx.imageSmoothingQuality = 'high';
    offCtx.drawImage(this.canvas, 0, 0, 28, 28);

    const imgData = offCtx.getImageData(0, 0, 28, 28);
    const data = imgData.data;

    // Render onto preview canvas in UI
    if (this.previewCtx) {
      this.previewCtx.imageSmoothingEnabled = false;
      this.previewCtx.clearRect(0, 0, 84, 84);
      this.previewCtx.drawImage(offscreen, 0, 0, 84, 84);
    }

    // Extract 784 normalized float values (0.0 to 1.0)
    const rawPixels = new Float32Array(784);
    let hasSignal = false;

    for (let i = 0; i < 784; i++) {
      const idx = i * 4;
      // Convert RGB to grayscale
      const gray = (data[idx] * 0.299 + data[idx + 1] * 0.587 + data[idx + 2] * 0.114) / 255.0;
      rawPixels[i] = gray;
      if (gray > 0.05) hasSignal = true;
    }

    return { rawPixels, hasSignal };
  }

  predictCurrentCanvas() {
    if (!this.weightsLoaded) return;

    const { rawPixels, hasSignal } = this.getNormalized28x28();
    if (!hasSignal) {
      this.resetPredictionDisplay();
      return;
    }

    const startTime = performance.now();

    // 1. Standardize using scaler mean and scale
    // x_scaled = (x - mean) / scale
    const x = new Float32Array(784);
    for (let i = 0; i < 784; i++) {
      x[i] = (rawPixels[i] - this.scalerMean[i]) / this.scalerScale[i];
    }

    // 2. Hidden Layer 1: 784 -> 256 (ReLU)
    // h1 = max(0, x @ W0 + b0)
    const h1 = new Float32Array(256);
    for (let j = 0; j < 256; j++) {
      let sum = this.b0[j];
      let wIdx = j; // column j in row-major W0
      for (let i = 0; i < 784; i++) {
        sum += x[i] * this.W0[wIdx];
        wIdx += 256;
      }
      h1[j] = sum > 0 ? sum : 0;
    }

    // 3. Hidden Layer 2: 256 -> 128 (ReLU)
    // h2 = max(0, h1 @ W1 + b1)
    const h2 = new Float32Array(128);
    for (let j = 0; j < 128; j++) {
      let sum = this.b1[j];
      let wIdx = j;
      for (let i = 0; i < 256; i++) {
        sum += h1[i] * this.W1[wIdx];
        wIdx += 128;
      }
      h2[j] = sum > 0 ? sum : 0;
    }

    // 4. Output Layer: 128 -> 10 (Softmax)
    // logits = h2 @ W2 + b2
    const logits = new Float32Array(10);
    let maxLogit = -Infinity;
    for (let j = 0; j < 10; j++) {
      let sum = this.b2[j];
      let wIdx = j;
      for (let i = 0; i < 128; i++) {
        sum += h2[i] * this.W2[wIdx];
        wIdx += 10;
      }
      logits[j] = sum;
      if (sum > maxLogit) maxLogit = sum;
    }

    // Stable Softmax
    let sumExp = 0;
    const probs = new Float32Array(10);
    for (let j = 0; j < 10; j++) {
      probs[j] = Math.exp(logits[j] - maxLogit);
      sumExp += probs[j];
    }

    let predictedDigit = 0;
    let maxProb = 0;
    for (let j = 0; j < 10; j++) {
      probs[j] /= sumExp;
      if (probs[j] > maxProb) {
        maxProb = probs[j];
        predictedDigit = j;
      }
    }

    const latency = (performance.now() - startTime).toFixed(1);
    this.displayResults(predictedDigit, maxProb, probs, latency);
  }

  displayResults(predDigit, confidence, allProbs, latency) {
    const digitHero = document.getElementById('digit-predicted-hero');
    const confHero = document.getElementById('digit-predicted-conf');
    const latencyElem = document.getElementById('digit-latency-val');

    if (digitHero) digitHero.textContent = predDigit;
    if (confHero) confHero.textContent = `${(confidence * 100).toFixed(1)}% Confidence`;
    if (latencyElem) latencyElem.textContent = `${latency} ms`;

    // Update 10 probability bars
    const container = document.getElementById('digit-prob-bars');
    if (!container) return;
    container.innerHTML = '';

    for (let d = 0; d < 10; d++) {
      const p = allProbs[d] * 100;
      const isTop = d === predDigit;
      const row = document.createElement('div');
      row.className = `digit-bar-row ${isTop ? 'top-prediction' : ''}`;
      row.innerHTML = `
        <div class="digit-label">
          <span class="digit-badge">${d}</span>
          <span class="digit-pct">${p.toFixed(1)}%</span>
        </div>
        <div class="digit-track">
          <div class="digit-fill" style="width: ${Math.max(2, p)}%;"></div>
        </div>
      `;
      container.appendChild(row);
    }
  }

  resetPredictionDisplay() {
    const digitHero = document.getElementById('digit-predicted-hero');
    const confHero = document.getElementById('digit-predicted-conf');
    if (digitHero) digitHero.textContent = '-';
    if (confHero) confHero.textContent = 'Draw a digit to test';

    const container = document.getElementById('digit-prob-bars');
    if (container) {
      container.innerHTML = '<div class="empty-state-prompt">Canvas empty — draw any digit (0-9)</div>';
    }
    if (this.previewCtx) {
      this.previewCtx.fillStyle = '#000000';
      this.previewCtx.fillRect(0, 0, 84, 84);
    }
  }
}

window.HandwritingRecognitionEngine = HandwritingRecognitionEngine;
