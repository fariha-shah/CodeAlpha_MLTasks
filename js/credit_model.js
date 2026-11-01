/**
 * Credit Scoring Model (Task 1) - Browser ML Inference Engine
 * Trained on German Credit Dataset with Random Forest Classifier
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
      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
      this.modelData = await res.json();
      this.initialized = true;
      console.log('Credit Scoring Engine initialized successfully.');
      this.bindEvents();
      this.evaluateCurrentInputs();
    } catch (err) {
      console.error('Failed to load credit model data:', err);
      const outputElem = document.getElementById('credit-risk-badge');
      if (outputElem) outputElem.textContent = 'Error loading model';
    }
  }

  bindEvents() {
    // Sliders & inputs
    const inputs = [
      'credit-age', 'credit-sex', 'credit-job', 'credit-housing',
      'credit-savings', 'credit-checking', 'credit-amount', 'credit-duration',
      'credit-purpose'
    ];

    inputs.forEach(id => {
      const el = document.getElementById(id);
      if (el) {
        el.addEventListener('input', () => {
          this.syncSliders(id);
          this.evaluateCurrentInputs();
        });
      }
    });

    // Preset personas
    document.querySelectorAll('[data-persona]').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const persona = e.currentTarget.getAttribute('data-persona');
        this.loadPersona(persona);
      });
    });

    // Export report
    const exportBtn = document.getElementById('credit-export-btn');
    if (exportBtn) {
      exportBtn.addEventListener('click', () => this.exportReport());
    }
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
        savings: 'rich', checking: 'rich', amount: 2500, duration: 12, purpose: 'car'
      },
      high_risk: {
        age: 21, sex: 'female', job: 1, housing: 'rent',
        savings: 'little', checking: 'little', amount: 9500, duration: 48, purpose: 'vacation/others'
      },
      moderate_risk: {
        age: 33, sex: 'male', job: 2, housing: 'own',
        savings: 'moderate', checking: 'moderate', amount: 4500, duration: 24, purpose: 'furniture/equipment'
      }
    };

    const p = presets[personaKey];
    if (!p) return;

    if (window.app && window.app.playSound) window.app.playSound('click');

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

    // Highlight active preset button
    document.querySelectorAll('[data-persona]').forEach(b => b.classList.remove('active-preset'));
    const activeBtn = document.querySelector(`[data-persona="${personaKey}"]`);
    if (activeBtn) activeBtn.classList.add('active-preset');

    this.evaluateCurrentInputs();
  }

  getInputs() {
    return {
      age: parseFloat(document.getElementById('credit-age')?.value || 35),
      sex: document.getElementById('credit-sex')?.value || 'male',
      job: parseInt(document.getElementById('credit-job')?.value || 2),
      housing: document.getElementById('credit-housing')?.value || 'own',
      savings: document.getElementById('credit-savings')?.value || 'unknown',
      checking: document.getElementById('credit-checking')?.value || 'unknown',
      amount: parseFloat(document.getElementById('credit-amount')?.value || 3000),
      duration: parseFloat(document.getElementById('credit-duration')?.value || 24),
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
    if (!this.modelData) return { goodProb: 0.7, badProb: 0.3 };

    const enc = this.modelData.encoders;
    const sexCode = enc.Sex[featuresRaw.sex] ?? 1;
    const housingCode = enc.Housing[featuresRaw.housing] ?? 1;
    const savingsCode = enc['Saving accounts'][featuresRaw.savings] ?? 4;
    const checkingCode = enc['Checking account'][featuresRaw.checking] ?? 3;
    const purposeCode = enc.Purpose[featuresRaw.purpose] ?? 1;

    // Feature vector: ['Age', 'Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Credit amount', 'Duration', 'Purpose']
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

    // Standardize numeric columns: Age (0), Amount (6), Duration (7)
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

    return {
      badProb,
      goodProb,
      prediction: goodProb >= 0.5 ? 1 : 0
    };
  }

  evaluateCurrentInputs() {
    if (!this.initialized) return;

    const inputs = this.getInputs();
    const result = this.predict(inputs);

    // Compute credit score: 300 to 850 range
    // Good probability mapped across 300 - 850
    const score = Math.round(300 + (result.goodProb * 550));
    const isGood = result.goodProb >= 0.5;

    // Update UI elements
    const scoreElem = document.getElementById('credit-score-number');
    if (scoreElem) scoreElem.textContent = score;

    const badgeElem = document.getElementById('credit-risk-badge');
    if (badgeElem) {
      badgeElem.textContent = isGood ? 'GOOD CREDIT (LOW RISK)' : 'HIGH CREDIT RISK';
      badgeElem.className = isGood ? 'risk-badge risk-good' : 'risk-badge risk-bad';
    }

    const probGoodElem = document.getElementById('credit-prob-good');
    const probBadElem = document.getElementById('credit-prob-bad');
    const probBarGood = document.getElementById('credit-prob-bar-good');
    const probBarBad = document.getElementById('credit-prob-bar-bad');

    const goodPct = (result.goodProb * 100).toFixed(1);
    const badPct = (result.badProb * 100).toFixed(1);

    if (probGoodElem) probGoodElem.textContent = `${goodPct}%`;
    if (probBadElem) probBadElem.textContent = `${badPct}%`;
    if (probBarGood) probBarGood.style.width = `${goodPct}%`;
    if (probBarBad) probBarBad.style.width = `${badPct}%`;

    // Radial Gauge Needle / Arc
    this.updateGauge(score, isGood);

    // Loan Simulator calculations
    const apr = isGood ? 0.065 : 0.145; // 6.5% for good risk, 14.5% for high risk
    const monthlyRate = apr / 12;
    const nMonths = inputs.duration;
    const principal = inputs.amount;
    
    // Monthly installment formula
    let monthlyPayment = 0;
    if (monthlyRate > 0) {
      monthlyPayment = (principal * (monthlyRate * Math.pow(1 + monthlyRate, nMonths))) / (Math.pow(1 + monthlyRate, nMonths) - 1);
    } else {
      monthlyPayment = principal / nMonths;
    }
    const totalRepaid = monthlyPayment * nMonths;
    const totalInterest = totalRepaid - principal;

    const monthlyElem = document.getElementById('loan-monthly-installment');
    const totalRepayElem = document.getElementById('loan-total-repayment');
    const totalInterestElem = document.getElementById('loan-interest-est');
    const estAprElem = document.getElementById('loan-est-apr');

    if (monthlyElem) monthlyElem.textContent = `€${Math.round(monthlyPayment).toLocaleString()}/mo`;
    if (totalRepayElem) totalRepayElem.textContent = `€${Math.round(totalRepaid).toLocaleString()}`;
    if (totalInterestElem) totalInterestElem.textContent = `€${Math.round(totalInterest).toLocaleString()}`;
    if (estAprElem) estAprElem.textContent = `${(apr * 100).toFixed(1)}%`;

    // Recommendations & Risk factors
    this.generateAdvice(inputs, result, score);
  }

  updateGauge(score, isGood) {
    const gaugePath = document.getElementById('credit-gauge-fill');
    if (!gaugePath) return;

    // Semicircle gauge: stroke-dasharray ~ 251.2
    const minScore = 300;
    const maxScore = 850;
    const fraction = Math.max(0, Math.min(1, (score - minScore) / (maxScore - minScore)));
    const totalDash = 251.2;
    const dashOffset = totalDash * (1 - fraction);

    gaugePath.style.strokeDashoffset = dashOffset;
    if (score >= 700) {
      gaugePath.style.stroke = 'var(--accent-emerald)';
    } else if (score >= 580) {
      gaugePath.style.stroke = 'var(--accent-amber)';
    } else {
      gaugePath.style.stroke = 'var(--accent-rose)';
    }
  }

  generateAdvice(inputs, result, score) {
    const list = document.getElementById('credit-factors-list');
    const adviceElem = document.getElementById('credit-ai-advice');
    if (!list) return;

    list.innerHTML = '';
    const factors = [];

    if (inputs.amount > 6000) {
      factors.push({ text: `High Loan Amount (€${inputs.amount.toLocaleString()}) elevates lender exposure`, risk: 'negative' });
    } else {
      factors.push({ text: `Requested amount (€${inputs.amount.toLocaleString()}) is within manageable band`, risk: 'positive' });
    }

    if (inputs.duration > 36) {
      factors.push({ text: `Extended loan term (${inputs.duration} months) raises default uncertainty`, risk: 'negative' });
    } else {
      factors.push({ text: `Compact repayment term (${inputs.duration} mo) demonstrates lower risk`, risk: 'positive' });
    }

    if (inputs.savings === 'rich' || inputs.savings === 'quite rich') {
      factors.push({ text: `Strong savings reserves cushion against repayment shocks`, risk: 'positive' });
    } else if (inputs.savings === 'little' || inputs.savings === 'unknown') {
      factors.push({ text: `Low or unverified savings reserves provide minimal buffer`, risk: 'negative' });
    }

    if (inputs.age < 25) {
      factors.push({ text: `Early credit age profile (< 25 years old)`, risk: 'neutral' });
    } else if (inputs.age > 40) {
      factors.push({ text: `Established age profile (${inputs.age} years) correlates with loan stability`, risk: 'positive' });
    }

    factors.forEach(f => {
      const li = document.createElement('li');
      li.className = `factor-item ${f.risk}`;
      const icon = f.risk === 'positive' ? '✓' : f.risk === 'negative' ? '⚠' : 'ℹ';
      li.innerHTML = `<span class="factor-icon">${icon}</span> <span>${f.text}</span>`;
      list.appendChild(li);
    });

    if (adviceElem) {
      if (result.goodProb >= 0.7) {
        adviceElem.textContent = 'Excellent applicant profile. Low risk category with high probability of timely repayment. Recommended for expedited prime interest loan approval.';
      } else if (result.goodProb >= 0.5) {
        adviceElem.textContent = 'Borderline acceptable profile. Consider reducing the loan duration to under 24 months or securing a co-signer / collateral to ensure lower interest rates.';
      } else {
        adviceElem.textContent = 'High probability of repayment difficulties. Recommended steps: Lower requested amount by 25-40%, extend down payment, and furnish verified checking account balance history.';
      }
    }
  }

  exportReport() {
    if (window.app && window.app.playSound) window.app.playSound('success');
    const inputs = this.getInputs();
    const result = this.predict(inputs);
    const score = Math.round(300 + (result.goodProb * 550));
    const isGood = result.goodProb >= 0.5;

    const report = `===================================================
  CODEALPHA MACHINE LEARNING INTERNSHIP
  Task 1: Credit Risk Assessment Report
===================================================
Generated: ${new Date().toLocaleString()}
Model: Random Forest Classifier (Accuracy: 74.50%, ROC-AUC: 0.753)
Dataset: German Credit Data (1000 records)

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
- Credit Risk Category: ${isGood ? 'GOOD CREDIT (LOW RISK)' : 'HIGH CREDIT RISK'}
- Estimated Credit Score: ${score} / 850
- Good Credit Probability: ${(result.goodProb * 100).toFixed(1)}%
- Bad Credit Probability: ${(result.badProb * 100).toFixed(1)}%
- Recommendation: ${isGood ? 'Approved for Prime Rate Financing' : 'Requires Collateral / Reduced Term'}
===================================================`;

    const blob = new Blob([report], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `Credit_Risk_Report_${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
    if (window.app && window.app.toast) {
      window.app.toast('Credit Assessment Report downloaded successfully!', 'success');
    }
  }
}

window.CreditScoringEngine = CreditScoringEngine;
