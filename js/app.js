/**
 * Main Application Orchestrator - CodeAlpha ML Suite
 * Handles routing, themes, audio synthesis, toasts, and module lifecycles.
 */

class App {
  constructor() {
    this.currentSection = 'home';
    this.soundEnabled = true;
    this.audioCtx = null;
    this.theme = localStorage.getItem('codealpha_theme') || 'dark';

    this.init();
  }

  init() {
    this.applyTheme(this.theme);
    this.setupAudio();
    this.setupNavigation();
    this.setupThemeToggle();
    this.setupSoundToggle();
    this.setupMobileMenu();
    this.setupModal();

    // Initialize ML Engines
    window.creditEngine = new window.CreditScoringEngine();
    window.emotionEngine = new window.EmotionRecognitionEngine();
    window.handwritingEngine = new window.HandwritingRecognitionEngine();
    window.benchmarksEngine = new window.BenchmarksEngine();

    console.log('CodeAlpha ML Suite App Ready.');
  }

  // --- Theme Management ---
  applyTheme(theme) {
    this.theme = theme;
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('codealpha_theme', theme);

    const themeBtn = document.getElementById('theme-toggle-btn');
    if (themeBtn) {
      themeBtn.innerHTML = theme === 'dark'
        ? `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>`
        : `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>`;
    }
  }

  setupThemeToggle() {
    const btn = document.getElementById('theme-toggle-btn');
    if (btn) {
      btn.addEventListener('click', () => {
        const next = this.theme === 'dark' ? 'light' : 'dark';
        this.applyTheme(next);
        this.playSound('click');
        this.toast(`Switched to ${next} theme`, 'info');
      });
    }
  }

  // --- Sound Synthesizer via Web Audio API ---
  setupAudio() {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (AudioCtx) {
      this.audioCtx = new AudioCtx();
    }
  }

  playSound(type = 'click') {
    if (!this.soundEnabled || !this.audioCtx) return;

    if (this.audioCtx.state === 'suspended') {
      this.audioCtx.resume();
    }

    const t = this.audioCtx.currentTime;
    const osc = this.audioCtx.createOscillator();
    const gain = this.audioCtx.createGain();
    osc.connect(gain);
    gain.connect(this.audioCtx.destination);

    if (type === 'click') {
      osc.type = 'sine';
      osc.frequency.setValueAtTime(600, t);
      osc.frequency.exponentialRampToValueAtTime(120, t + 0.05);
      gain.gain.setValueAtTime(0.08, t);
      gain.gain.exponentialRampToValueAtTime(0.001, t + 0.05);
      osc.start(t);
      osc.stop(t + 0.05);
    } else if (type === 'success') {
      osc.type = 'triangle';
      osc.frequency.setValueAtTime(440, t);
      osc.frequency.setValueAtTime(659.25, t + 0.08); // E5
      gain.gain.setValueAtTime(0.12, t);
      gain.gain.exponentialRampToValueAtTime(0.001, t + 0.25);
      osc.start(t);
      osc.stop(t + 0.25);
    } else if (type === 'switch') {
      osc.type = 'sine';
      osc.frequency.setValueAtTime(320, t);
      osc.frequency.exponentialRampToValueAtTime(540, t + 0.08);
      gain.gain.setValueAtTime(0.06, t);
      gain.gain.exponentialRampToValueAtTime(0.001, t + 0.08);
      osc.start(t);
      osc.stop(t + 0.08);
    }
  }

  setupSoundToggle() {
    const btn = document.getElementById('sound-toggle-btn');
    if (btn) {
      btn.addEventListener('click', () => {
        this.soundEnabled = !this.soundEnabled;
        btn.classList.toggle('muted', !this.soundEnabled);
        btn.innerHTML = this.soundEnabled
          ? `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"/></svg>`
          : `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><line x1="23" y1="9" x2="17" y2="15"/><line x1="17" y1="9" x2="23" y2="15"/></svg>`;
        this.toast(this.soundEnabled ? 'Sound feedback on' : 'Sound muted', 'info');
      });
    }
  }

  // --- Navigation & Routing ---
  setupNavigation() {
    const navLinks = document.querySelectorAll('[data-target-section]');
    navLinks.forEach(link => {
      link.addEventListener('click', (e) => {
        e.preventDefault();
        const target = link.dataset.targetSection;
        this.navigateTo(target);
      });
    });

    // Handle hash links
    window.addEventListener('hashchange', () => {
      const hash = window.location.hash.replace('#', '');
      if (hash) this.navigateTo(hash, false);
    });

    if (window.location.hash) {
      const hash = window.location.hash.replace('#', '');
      this.navigateTo(hash, false);
    }
  }

  navigateTo(sectionId, updateHash = true) {
    const targetSection = document.getElementById(`section-${sectionId}`);
    if (!targetSection) return;

    this.playSound('switch');
    this.currentSection = sectionId;

    // Update section visibility
    document.querySelectorAll('.app-section').forEach(sec => {
      sec.classList.remove('active');
    });
    targetSection.classList.add('active');

    // Update nav links
    document.querySelectorAll('[data-target-section]').forEach(link => {
      link.classList.toggle('active', link.dataset.targetSection === sectionId);
    });

    // Close mobile menu if open
    const mobileMenu = document.getElementById('mobile-drawer');
    if (mobileMenu) mobileMenu.classList.remove('open');

    if (updateHash) {
      window.location.hash = sectionId;
    }

    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  setupMobileMenu() {
    const toggleBtn = document.getElementById('mobile-menu-btn');
    const drawer = document.getElementById('mobile-drawer');
    const closeBtn = document.getElementById('mobile-drawer-close');

    if (toggleBtn && drawer) {
      toggleBtn.addEventListener('click', () => {
        drawer.classList.add('open');
        this.playSound('click');
      });
    }

    if (closeBtn && drawer) {
      closeBtn.addEventListener('click', () => {
        drawer.classList.remove('open');
        this.playSound('click');
      });
    }
  }

  setupModal() {
    const guideBtn = document.getElementById('deploy-guide-btn');
    const modal = document.getElementById('deploy-modal');
    const closeBtn = document.getElementById('deploy-modal-close');

    if (guideBtn && modal) {
      guideBtn.addEventListener('click', () => {
        modal.classList.add('open');
        this.playSound('click');
      });
    }

    if (closeBtn && modal) {
      closeBtn.addEventListener('click', () => {
        modal.classList.remove('open');
        this.playSound('click');
      });
    }

    if (modal) {
      modal.addEventListener('click', (e) => {
        if (e.target === modal) modal.classList.remove('open');
      });
    }
  }

  // --- Floating Toast Notifications ---
  toast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    const icon = type === 'success' ? '✓' : type === 'warning' ? '⚠' : 'ℹ';
    toast.innerHTML = `<span class="toast-icon">${icon}</span> <span class="toast-msg">${message}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
      toast.classList.add('toast-show');
    }, 10);

    setTimeout(() => {
      toast.classList.remove('toast-show');
      setTimeout(() => toast.remove(), 300);
    }, 3200);
  }
}

// Global bootstrap
document.addEventListener('DOMContentLoaded', () => {
  window.app = new App();
});
