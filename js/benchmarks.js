/**
 * Model Benchmarks & Metrics Visualization (Studio Analytics)
 * Comparing models across all 3 CodeAlpha tasks
 */

class BenchmarksEngine {
  constructor() {
    this.currentView = 'comparison';
    this.init();
  }

  init() {
    this.bindEvents();
    this.renderFeatureImportance();
    this.renderMetricsTable();
  }

  bindEvents() {
    document.querySelectorAll('[data-bench-filter]').forEach(btn => {
      btn.addEventListener('click', (e) => {
        document.querySelectorAll('[data-bench-filter]').forEach(b => b.classList.remove('active'));
        e.currentTarget.classList.add('active');
        const filter = e.currentTarget.dataset.benchFilter;
        this.filterTable(filter);
      });
    });
  }

  renderFeatureImportance() {
    const container = document.getElementById('benchmark-feature-bars');
    if (!container) return;

    const features = [
      { name: 'Credit Amount', pct: 25.3, desc: 'Requested loan principal size' },
      { name: 'Applicant Age', pct: 17.6, desc: 'Customer age and financial maturity' },
      { name: 'Loan Duration', pct: 15.8, desc: 'Total repayment horizon (months)' },
      { name: 'Checking Account', pct: 13.3, desc: 'Liquid operational bank balance' },
      { name: 'Loan Purpose', pct: 9.3, desc: 'Use of funds (car, radio, business, etc.)' },
      { name: 'Saving Accounts', pct: 6.2, desc: 'Emergency cushion and wealth reserves' },
      { name: 'Job Skill Level', pct: 5.2, desc: 'Employment stability & earning level' },
      { name: 'Housing Status', pct: 4.1, desc: 'Residential ownership vs rental' },
      { name: 'Gender (Sex)', pct: 3.2, desc: 'Demographic baseline' }
    ];

    container.innerHTML = '';
    features.forEach(f => {
      const row = document.createElement('div');
      row.className = 'feature-row';
      row.innerHTML = `
        <div class="feat-meta">
          <span class="feat-name">${f.name}</span>
          <span class="feat-desc">${f.desc}</span>
          <span class="feat-pct">${f.pct}%</span>
        </div>
        <div class="feat-track">
          <div class="feat-fill" style="width: ${f.pct * 3.5}%;"></div>
        </div>
      `;
      container.appendChild(row);
    });
  }

  renderMetricsTable() {
    // Populate or filter table rows
  }

  filterTable(category) {
    const rows = document.querySelectorAll('#benchmark-table tbody tr');
    rows.forEach(row => {
      if (category === 'all' || row.dataset.category === category) {
        row.style.display = '';
      } else {
        row.style.display = 'none';
      }
    });
  }
}

window.BenchmarksEngine = BenchmarksEngine;
