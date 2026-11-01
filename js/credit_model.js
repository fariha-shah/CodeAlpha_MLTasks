/**
 * Credit Scoring Model (Task 1)
 * High-Accuracy 100-Tree Random Forest In-Browser Evaluator
 * Calibrated against German Credit Dataset
 */

class CreditScoringEngine {
  constructor() {
    this.modelData = null;
    this.initialized = false;
    this.init();
  }

  async init() {
    try {
      const res = await fetch('data/credit_model.json');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      this.modelData = await res.json();
      this.initialized = true;
      console.log(`Credit Scoring Engine ready with ${this.modelData.trees.length} trees.`);
      this.bindEvents();
      this.evaluateCurrentInputs();
    } catch (err) {
      console.error('Failed to load credit model data:', err);
    }
  }

  bindEvents() {
    const inputIds = [
      'credit-age', 'credit-sex', 'credit-job', 'credit-housing',
      'credit-savings', 'credit-checking', 'credit-amount', 'credit-duration',
      'credit-purpose'
    ];

    inputIds.forEach(id => {
      const el = document.getElementById(id);
      if (el) {
        // Listen to BOTH input (sliders) and change (dropdowns)
        ['input', 'change'].forEach(evt => {
          el.addEventListener(evt, () => {
            this.syncSliders(id);
            this.evaluateCurrentInputs();
          });
        });
      }
    });

    // Evaluate button
    document.getElementById('credit-evaluate-btn')?.addEventListener('click', () => {
      if (window.app?.playSound) window.app.playSound('success');
      this.evaluateCurrentInputs();
      window.app?.toast('Credit evaluation calculated!', 'success');
    });

    // Presets
    document.querySelectorAll('[data-persona]').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const persona = e.currentTarget.getAttribute('data-persona');
        this.loadPersona(persona);
      });
    });

    // Export
    document.getElementById('credit-export-btn')?.addEventListener('click', () => {
      this.exportReport();
    });
  }

  syncSliders(changedId) {
    if (changedId === 'credit-age') {
      const val = document.getElementById('credit-age').value;
      const disp = document.getElementById('credit-age-val');
      if (disp) disp.textContent = `${val} yrs`;
    } else if (changedId === 'credit-amount') {
      const val = document.getElementById('credit-amount').value;
      const disp = document.getElementById('credit-amount-val');
      if (disp) disp.textContent = `€${Number(val).toLocaleString()}`;
    } else if (changedId === 'credit-duration') {
      const val = document.getElementById('credit-duration').value;
      const disp = document.getElementById('credit-duration-val');
      if (disp) disp.textContent = `${val} mo`;
    }
  }

  loadPersona(personaKey) {
    const presets = {
      low_risk: {
        age: 46, sex: 'male', job: 2, housing: 'own',
        savings: 'rich', checking: 'rich', amount: 2000, duration: 12, purpose: 'car'
      },
      high_risk: {
        age: 21, sex: 'female', job: 1, housing: 'rent',
        savings: 'little', checking: 'little', amount: 12000, duration: 48, purpose: 'vacation/others'
      },
      moderate_risk: {
        age: 33, sex: 'male', job: 2, housing: 'own',
        savings: 'moderate', checking: 'moderate', amount: 4500, duration: 24, purpose: 'furniture/equipment'
      }
    };

    const p = presets[personaKey];
    if (!p) return;

    if (window.app?.playSound) window.app.playSound('click');

    document.getElementById('credit-age').value = p.age;
    document.getElementById('credit-sex').value = p.sex;
    document.getElementById('credit-job').value = p.job;
    document.getElementById('credit-housing').value = p.housing;
    document.getElementById('credit-savings').value = p.savings;
    document.getElementById('credit-checking').value = p.checking;
    document.getElementById('credit-amount').value = p.amount;
    document.getElementById('credit-duration').value = p.duration;
    document.getElementById('credit-purpose').value = p.purpose;

    this.syncSliders('credit-age');
    this.syncSliders('credit-amount');
    this.syncSliders('credit-duration');

    document.querySelectorAll('[data-persona]').forEach(b => b.classList.remove('active-preset'));
    document.querySelector(`[data-persona="${personaKey}"]`)?.classList.add('active-preset');

    this.evaluateCurrentInputs();
  }

  getInputs() {
    return {
      age: parseFloat(document.getElementById('credit-age')?.value || 35),
      sex: document.getElementById('credit-sex')?.value || 'male',
      job: parseInt(document.getElementById('credit-job')?.value || 2),
      housing: document.getElementById('credit-housing')?.value || 'own',
      savings: document.getElementById('credit-savings')?.value || 'rich',
      checking: document.getElementById('credit-checking')?.value || 'rich',
      amount: parseFloat(document.getElementById('credit-amount')?.value || 2500),
      duration: parseFloat(document.getElementById('credit-duration')?.value || 12),
      purpose: document.getElementById('credit-purpose')?.value || 'car'
    };
  }

  evaluateTree(tree, x) {
    let node = 0;
    while (tree.children_left[node] !== -1) {
      const featIdx = tree.feature[node];
      const threshold = tree.threshold[node];
      if (x[featIdx] <= threshold) {
        node = tree.children_left[node];
      } else {
        node = tree.children_right[node];
      }
    }
    const val = tree.value[node]; // [countBad, countGood]
    const total = val[0] + val[1];
    if (total === 0) return [0.3, 0.7];
    return [val[0] / total, val[1] / total];
  }

  predict(featuresRaw) {
    if (!this.modelData) return { goodProb: 0.8, badProb: 0.2 };

    const enc = this.modelData.encoders;
    const sexCode = enc.Sex[featuresRaw.sex] ?? 1;
    const housingCode = enc.Housing[featuresRaw.housing] ?? 1;
    const savingsCode = enc['Saving accounts'][featuresRaw.savings] ?? 3;
    const checkingCode = enc['Checking account'][featuresRaw.checking] ?? 2;
    const purposeCode = enc.Purpose[featuresRaw.purpose] ?? 1;

    // Features order: ['Age', 'Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Credit amount', 'Duration', 'Purpose']
    const x = [
      featuresRaw.age,
      sexCode,
      featuresRaw.job,
      housingCode,
      savingsCode,
      checkingCode,
      featuresRaw.amount,
      featuresRaw.duration,
      purposeCode
    ];

    // Standardize: Age(0), Credit amount(6), Duration(7)
    const mean = this.modelData.scaler_mean;
    const scale = this.modelData.scaler_scale;

    x[0] = (x[0] - mean[0]) / scale[0];
    x[6] = (x[6] - mean[1]) / scale[1];
    x[7] = (x[7] - mean[2]) / scale[2];

    let sumBad = 0;
    let sumGood = 0;
    const trees = this.modelData.trees;

    for (let i = 0; i < trees.length; i++) {
      const [badP, goodP] = this.evaluateTree(trees[i], x);
      sumBad += badP;
      sumGood += goodP;
    }

    const badProb = sumBad / trees.length;
    const goodProb = sumGood / trees.length;

    return { badProb, goodProb };
  }

  evaluateCurrentInputs() {
    if (!this.initialized) return;

    const inputs = this.getInputs();
    const result = this.predict(inputs);

    // Map probability to FICO range: 300 to 850
    const score = Math.round(300 + (result.goodProb * 550));
    const goodPct = (result.goodProb * 100).toFixed(1);
    const badPct = (result.badProb * 100).toFixed(1);

    const scoreElem = document.getElementById('credit-score-number');
    if (scoreElem) scoreElem.textContent = score;

    const badgeElem = document.getElementById('credit-risk-badge');
    if (badgeElem) {
      if (score >= 680) {
        badgeElem.textContent = 'PRIME / LOW RISK (APPROVED)';
        badgeElem.className = 'risk-badge risk-good';
      } else if (score >= 570) {
        badgeElem.textContent = 'MODERATE RISK (CONDITIONAL)';
        badgeElem.className = 'risk-badge risk-moderate';
      } else {
        badgeElem.textContent = 'HIGH RISK / HIGH DEFAULT ODDS';
        badgeElem.className = 'risk-badge risk-bad';
      }
    }

    const probGoodElem = document.getElementById('credit-prob-good');
    const probBadElem = document.getElementById('credit-prob-bad');
    const probBarGood = document.getElementById('credit-prob-bar-good');
    const probBarBad = document.getElementById('credit-prob-bar-bad');

    if (probGoodElem) probGoodElem.textContent = `${goodPct}%`;
    if (probBadElem) probBadElem.textContent = `${badPct}%`;
    if (probBarGood) probBarGood.style.width = `${goodPct}%`;
    if (probBarBad) probBarBad.style.width = `${badPct}%`;

    this.updateGauge(score);

    // Loan calculations
    const apr = score >= 680 ? 0.059 : score >= 570 ? 0.105 : 0.169;
    const monthlyRate = apr / 12;
    const n = inputs.duration;
    const P = inputs.amount;

    let monthly = 0;
    if (monthlyRate > 0) {
      monthly = (P * (monthlyRate * Math.pow(1 + monthlyRate, n))) / (Math.pow(1 + monthlyRate, n) - 1);
    } else {
      monthly = P / n;
    }
    const totalRepay = monthly * n;
    const totalInterest = totalRepay - P;

    document.getElementById('loan-monthly-installment').textContent = `€${Math.round(monthly).toLocaleString()}/mo`;
    document.getElementById('loan-total-repayment').textContent = `€${Math.round(totalRepay).toLocaleString()}`;
    document.getElementById('loan-interest-est').textContent = `€${Math.round(totalInterest).toLocaleString()}`;
    document.getElementById('loan-est-apr').textContent = `${(apr * 100).toFixed(1)}%`;

    this.generateAdvice(inputs, result, score);
  }

  updateGauge(score) {
    const gaugePath = document.getElementById('credit-gauge-fill');
    if (!gaugePath) return;

    const minScore = 300;
    const maxScore = 850;
    const fraction = Math.max(0, Math.min(1, (score - minScore) / (maxScore - minScore)));
    const totalDash = 251.2;
    const dashOffset = totalDash * (1 - fraction);

    gaugePath.style.strokeDashoffset = dashOffset;
    if (score >= 680) {
      gaugePath.style.stroke = 'var(--accent-lime)'; // Acid Lime for excellent score
    } else if (score >= 570) {
      gaugePath.style.stroke = 'var(--accent-blue)'; // Electric Cobalt Blue
    } else {
      gaugePath.style.stroke = '#ff3344'; // Red alert
    }
  }

  generateAdvice(inputs, result, score) {
    const list = document.getElementById('credit-factors-list');
    const adviceElem = document.getElementById('credit-ai-advice');
    if (!list) return;

    list.innerHTML = '';
    const factors = [];

    if (inputs.amount > 6000) {
      factors.push({ text: `Requested amount (€${inputs.amount.toLocaleString()}) increases default exposure.`, risk: 'negative' });
    } else {
      factors.push({ text: `Loan principal (€${inputs.amount.toLocaleString()}) is comfortably within credit capacity.`, risk: 'positive' });
    }

    if (inputs.duration > 36) {
      factors.push({ text: `Extended loan term (${inputs.duration} months) raises structural credit uncertainty.`, risk: 'negative' });
    } else {
      factors.push({ text: `Manageable repayment term (${inputs.duration} mo) demonstrates lower risk.`, risk: 'positive' });
    }

    if (inputs.savings === 'rich' || inputs.savings === 'quite rich') {
      factors.push({ text: `Substantial savings reserves provide strong solvency cushion.`, risk: 'positive' });
    } else if (inputs.savings === 'little' || inputs.savings === 'unknown') {
      factors.push({ text: `Low or unverified savings reserves limit financial buffer.`, risk: 'negative' });
    }

    if (inputs.checking === 'little') {
      factors.push({ text: `Checking account balance indicates minimal liquidity buffer.`, risk: 'negative' });
    } else if (inputs.checking === 'rich') {
      factors.push({ text: `Healthy liquid checking account balance verifies positive cashflow.`, risk: 'positive' });
    }

    factors.forEach(f => {
      const li = document.createElement('li');
      li.className = `factor-item ${f.risk}`;
      const icon = f.risk === 'positive' ? '✓' : f.risk === 'negative' ? '⚠' : 'ℹ';
      li.innerHTML = `<span class="factor-icon">${icon}</span> <span>${f.text}</span>`;
      list.appendChild(li);
    });

    if (adviceElem) {
      if (score >= 680) {
        adviceElem.textContent = 'Excellent applicant profile. Low risk category with high probability of timely repayment. Recommended for expedited prime interest loan approval.';
      } else if (score >= 570) {
        adviceElem.textContent = 'Moderate risk profile. Recommend reducing loan duration to under 24 months or securing a co-signer to unlock preferred prime rates.';
      } else {
        adviceElem.textContent = 'High probability of repayment difficulties. Recommended steps: Lower loan amount by 25-40%, extend down payment, and provide verified checking reserves.';
      }
    }
  }

  exportReport() {
    if (window.app?.playSound) window.app.playSound('success');
    const inputs = this.getInputs();
    const result = this.predict(inputs);
    const score = Math.round(300 + (result.goodProb * 550));
    const isGood = score >= 570;

    const report = `===================================================
  CODEALPHA MACHINE LEARNING INTERNSHIP
  Task 1: Credit Risk Assessment Report
===================================================
Generated: ${new Date().toLocaleString()}
Model: Random Forest Classifier (100 Trees, German Credit Data)

APPLICANT PROFILE:
- Age: ${inputs.age} years
- Gender: ${inputs.sex.toUpperCase()}
- Job Skill Level: Level ${inputs.job}
- Housing: ${inputs.housing.toUpperCase()}
- Savings Account: ${inputs.savings.toUpperCase()}
- Checking Account: ${inputs.checking.toUpperCase()}
- Loan Amount: €${inputs.amount.toLocaleString()}
- Repayment Duration: ${inputs.duration} Months
- Loan Purpose: ${inputs.purpose.toUpperCase()}

AI EVALUATION OUTCOME:
- Credit Risk Category: ${score >= 680 ? 'PRIME (LOW RISK)' : score >= 570 ? 'FAIR / MODERATE RISK' : 'HIGH DEFAULT RISK'}
- Estimated Credit Score: ${score} / 850
- Good Credit Probability: ${(result.goodProb * 100).toFixed(1)}%
- Bad Credit Probability: ${(result.badProb * 100).toFixed(1)}%
===================================================`;

    const blob = new Blob([report], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `Credit_Risk_Report_${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
    window.app?.toast('Report downloaded successfully!', 'success');
  }
}

window.CreditScoringEngine = CreditScoringEngine;
