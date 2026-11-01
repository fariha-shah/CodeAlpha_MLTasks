/**
 * Handwritten Digit Recognition (Task 3)
 * High-Accuracy MNIST Center-of-Mass Pipeline & In-Browser Neural Network
 * Architecture: 784 -> 256 (ReLU) -> 128 (ReLU) -> 10 (Softmax)
 */

class HandwritingRecognitionEngine {
  constructor() {
    this.weightsLoaded = false;
    this.scalerMean = null;
    this.scalerScale = null;
    this.W0 = null;
    this.b0 = null;
    this.W1 = null;
    this.b1 = null;
    this.W2 = null;
    this.b2 = null;

    this.canvas = null;
    this.ctx = null;
    this.isDrawing = false;
    this.brushSize = 24;
    this.isEraser = false;
    this.undoStack = [];
    this.maxUndos = 20;

    this.previewCanvas = null;
    this.previewCtx = null;

    this.init();
  }

  async init() {
    try {
      await this.loadWeights();
      this.initCanvases();
      this.bindEvents();
      console.log('Handwriting Recognition Engine ready with MNIST CoM Normalization.');

      // Load initial authentic MNIST sample 3
      this.loadSampleDigit(3);
    } catch (err) {
      console.error('Error initializing handwriting engine:', err);
    }
  }

  async loadWeights() {
    const res = await fetch('data/handwriting_weights.bin');
    if (!res.ok) throw new Error(`HTTP ${res.status} loading handwriting weights`);
    const buffer = await res.arrayBuffer();
    const f32 = new Float32Array(buffer);

    let offset = 0;
    this.scalerMean = f32.subarray(offset, offset + 784);
    offset += 784;

    this.scalerScale = f32.subarray(offset, offset + 784);
    offset += 784;

    this.W0 = f32.subarray(offset, offset + (784 * 256));
    offset += 784 * 256;

    this.b0 = f32.subarray(offset, offset + 256);
    offset += 256;

    this.W1 = f32.subarray(offset, offset + (256 * 128));
    offset += 256 * 128;

    this.b1 = f32.subarray(offset, offset + 128);
    offset += 128;

    this.W2 = f32.subarray(offset, offset + (128 * 10));
    offset += 128 * 10;

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
      this.undoStack.pop();
      const prevState = this.undoStack[this.undoStack.length - 1];
      this.ctx.putImageData(prevState, 0, 0);
      this.predictCurrentCanvas();
    } else {
      this.clearCanvas(true);
    }
  }

  bindEvents() {
    if (!this.canvas) return;

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

    const getTouchPos = (touchEvent) => {
      const rect = this.canvas.getBoundingClientRect();
      const touch = touchEvent.touches[0];
      return { clientX: touch.clientX, clientY: touch.clientY };
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

    // Buttons
    document.getElementById('digit-clear-btn')?.addEventListener('click', () => {
      if (window.app?.playSound) window.app.playSound('click');
      this.clearCanvas(true);
    });

    document.getElementById('digit-undo-btn')?.addEventListener('click', () => {
      if (window.app?.playSound) window.app.playSound('click');
      this.undo();
    });

    const eraserBtn = document.getElementById('digit-eraser-btn');
    if (eraserBtn) {
      eraserBtn.addEventListener('click', () => {
        this.isEraser = !this.isEraser;
        eraserBtn.classList.toggle('active', this.isEraser);
      });
    }

    const brushSlider = document.getElementById('digit-brush-size');
    if (brushSlider) {
      brushSlider.addEventListener('input', (e) => {
        this.brushSize = parseInt(e.target.value);
        const label = document.getElementById('digit-brush-val');
        if (label) label.textContent = `${this.brushSize}px`;
      });
    }

    document.getElementById('digit-predict-now-btn')?.addEventListener('click', () => {
      this.predictCurrentCanvas();
    });

    // Preset digits
    document.querySelectorAll('[data-sample-digit]').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const digit = parseInt(e.currentTarget.dataset.sampleDigit);
        this.loadSampleDigit(digit);
      });
    });

    // Custom file upload
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
    this.ctx.strokeStyle = this.isEraser ? '#000000' : '#ffffff';

    this.ctx.lineTo(x, y);
    this.ctx.stroke();
    this.ctx.beginPath();
    this.ctx.moveTo(x, y);
  }

  loadSampleDigit(digit) {
    if (window.app?.playSound) window.app.playSound('click');

    document.querySelectorAll('[data-sample-digit]').forEach(b => {
      b.classList.toggle('active', parseInt(b.dataset.sampleDigit) === digit);
    });

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = `assets/digits/digit_${digit}.png`;
    img.onload = () => {
      this.ctx.fillStyle = '#000000';
      this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

      this.ctx.imageSmoothingEnabled = false;
      this.ctx.drawImage(img, 28, 28, 224, 224);

      this.saveState();
      this.predictCurrentCanvas();
    };
  }

  handleImageUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    if (window.app?.playSound) window.app.playSound('click');
    const reader = new FileReader();
    reader.onload = (evt) => {
      const img = new Image();
      img.onload = () => {
        this.ctx.fillStyle = '#000000';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        const maxDim = 220;
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
      totalBrightness += (data[i] * 0.299) + (data[i + 1] * 0.587) + (data[i + 2] * 0.114);
    }

    if (totalBrightness / numPixels > 127) {
      for (let i = 0; i < data.length; i += 4) {
        data[i] = 255 - data[i];
        data[i + 1] = 255 - data[i + 1];
        data[i + 2] = 255 - data[i + 2];
      }
      this.ctx.putImageData(imgData, 0, 0);
    }
  }

  /**
   * Authentic MNIST Center-of-Mass Preprocessing Pipeline
   * 1. Finds tight bounding box of drawn pixels.
   * 2. Resizes into 20x20 bounding box preserving aspect ratio.
   * 3. Places inside 28x28 grid and computes mass centroid (CoM).
   * 4. Centers CoM at exactly (13.5, 13.5) with strictly zero borders.
   */
  getNormalizedMNIST28x28() {
    const w = this.canvas.width;
    const h = this.canvas.height;
    const imgData = this.ctx.getImageData(0, 0, w, h);
    const data = imgData.data;

    let minX = w, maxX = 0, minY = h, maxY = 0;
    let totalIntensity = 0;

    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = (y * w + x) * 4;
        const gray = (data[idx] * 0.299 + data[idx + 1] * 0.587 + data[idx + 2] * 0.114);
        if (gray > 30) {
          totalIntensity += gray;
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }

    // Canvas empty
    if (totalIntensity < 500 || minX > maxX || minY > maxY) {
      return { rawPixels: null, hasSignal: false };
    }

    const boxW = Math.max(1, maxX - minX + 1);
    const boxH = Math.max(1, maxY - minY + 1);

    // 1. Crop bounding box
    const cropCanvas = document.createElement('canvas');
    cropCanvas.width = boxW;
    cropCanvas.height = boxH;
    const cropCtx = cropCanvas.getContext('2d');
    cropCtx.drawImage(this.canvas, minX, minY, boxW, boxH, 0, 0, boxW, boxH);

    // 2. Scale into 20x20 box preserving aspect ratio
    let targetW, targetH;
    if (boxW > boxH) {
      targetW = 20;
      targetH = Math.max(1, Math.round((boxH * 20) / boxW));
    } else {
      targetH = 20;
      targetW = Math.max(1, Math.round((boxW * 20) / boxH));
    }

    const scaledCanvas = document.createElement('canvas');
    scaledCanvas.width = targetW;
    scaledCanvas.height = targetH;
    const scaledCtx = scaledCanvas.getContext('2d');
    scaledCtx.imageSmoothingEnabled = true;
    scaledCtx.drawImage(cropCanvas, 0, 0, targetW, targetH);

    // 3. Place inside 28x28 grid
    const startX = 4 + Math.floor((20 - targetW) / 2);
    const startY = 4 + Math.floor((20 - targetH) / 2);

    const canvas28 = document.createElement('canvas');
    canvas28.width = 28;
    canvas28.height = 28;
    const ctx28 = canvas28.getContext('2d');
    ctx28.fillStyle = '#000000';
    ctx28.fillRect(0, 0, 28, 28);
    ctx28.drawImage(scaledCanvas, startX, startY);

    const d28 = ctx28.getImageData(0, 0, 28, 28);
    const p28 = d28.data;

    // 4. Calculate Center of Mass (CoM)
    let mass = 0;
    let weightedX = 0;
    let weightedY = 0;
    const grid28 = new Float32Array(28 * 28);

    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        const i = (y * 28 + x) * 4;
        const val = (p28[i] * 0.299 + p28[i + 1] * 0.587 + p28[i + 2] * 0.114) / 255.0;
        grid28[y * 28 + x] = val;
        if (val > 0.05) {
          mass += val;
          weightedX += x * val;
          weightedY += y * val;
        }
      }
    }

    const finalPixels = new Float32Array(784);
    if (mass > 0) {
      const comX = weightedX / mass;
      const comY = weightedY / mass;
      const shiftX = Math.round(13.5 - comX);
      const shiftY = Math.round(13.5 - comY);

      // Shift matrix to center CoM
      for (let y = 0; y < 28; y++) {
        const srcY = y - shiftY;
        if (srcY < 0 || srcY >= 28) continue;
        for (let x = 0; x < 28; x++) {
          const srcX = x - shiftX;
          if (srcX < 0 || srcX >= 28) continue;
          finalPixels[y * 28 + x] = grid28[srcY * 28 + srcX];
        }
      }

      // Zero out 2 border pixels so variance scaling never blows up
      for (let x = 0; x < 28; x++) {
        finalPixels[0 * 28 + x] = 0;
        finalPixels[1 * 28 + x] = 0;
        finalPixels[26 * 28 + x] = 0;
        finalPixels[27 * 28 + x] = 0;
      }
      for (let y = 0; y < 28; y++) {
        finalPixels[y * 28 + 0] = 0;
        finalPixels[y * 28 + 1] = 0;
        finalPixels[y * 28 + 26] = 0;
        finalPixels[y * 28 + 27] = 0;
      }
    } else {
      finalPixels.set(grid28);
    }

    // 5. Render onto Preview Box in UI
    if (this.previewCtx) {
      const previewImg = this.previewCtx.createImageData(28, 28);
      for (let i = 0; i < 784; i++) {
        const v = Math.min(255, Math.round(finalPixels[i] * 255));
        const idx = i * 4;
        previewImg.data[idx] = v;
        previewImg.data[idx + 1] = v;
        previewImg.data[idx + 2] = v;
        previewImg.data[idx + 3] = 255;
      }
      // Draw 28x28 scaled up to 84x84
      const temp28 = document.createElement('canvas');
      temp28.width = 28;
      temp28.height = 28;
      temp28.getContext('2d').putImageData(previewImg, 0, 0);

      this.previewCtx.imageSmoothingEnabled = false;
      this.previewCtx.clearRect(0, 0, 84, 84);
      this.previewCtx.drawImage(temp28, 0, 0, 84, 84);
    }

    return { rawPixels: finalPixels, hasSignal: true };
  }

  predictCurrentCanvas() {
    if (!this.weightsLoaded) return;

    const { rawPixels, hasSignal } = this.getNormalizedMNIST28x28();
    if (!hasSignal) {
      this.resetPredictionDisplay();
      return;
    }

    const t0 = performance.now();

    // 1. Standardize using clamped scale
    const x = new Float32Array(784);
    for (let i = 0; i < 784; i++) {
      const s = Math.max(this.scalerScale[i], 0.05);
      x[i] = (rawPixels[i] - this.scalerMean[i]) / s;
    }

    // 2. Hidden Layer 1: 784 -> 256 (ReLU)
    const h1 = new Float32Array(256);
    for (let j = 0; j < 256; j++) {
      let sum = this.b0[j];
      let wIdx = j;
      for (let i = 0; i < 784; i++) {
        sum += x[i] * this.W0[wIdx];
        wIdx += 256;
      }
      h1[j] = sum > 0 ? sum : 0;
    }

    // 3. Hidden Layer 2: 256 -> 128 (ReLU)
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

    const latency = (performance.now() - t0).toFixed(1);
    this.displayResults(predictedDigit, maxProb, probs, latency);
  }

  displayResults(predDigit, confidence, allProbs, latency) {
    const digitHero = document.getElementById('digit-predicted-hero');
    const confHero = document.getElementById('digit-predicted-conf');
    const latencyElem = document.getElementById('digit-latency-val');

    if (digitHero) digitHero.textContent = predDigit;
    if (confHero) confHero.textContent = `${(confidence * 100).toFixed(1)}% Confidence`;
    if (latencyElem) latencyElem.textContent = `Inference: ${latency} ms`;

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
          <span class="digit-badge ${isTop ? 'acid-text' : ''}">Digit ${d}</span>
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
      container.innerHTML = '<div class="empty-state-prompt">Draw any digit (0–9) on the canvas</div>';
    }
    if (this.previewCtx) {
      this.previewCtx.fillStyle = '#000000';
      this.previewCtx.fillRect(0, 0, 84, 84);
    }
  }
}

window.HandwritingRecognitionEngine = HandwritingRecognitionEngine;
