/**
 * Main Application Orchestrator - CodeAlpha ML Suite
 * Handles routing, themes, audio synthesis, toasts, and Lucide icons.
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

    // Render Lucide icons
    if (window.lucide) {
      window.lucide.createIcons();
    }

    console.log('CodeAlpha ML Suite App Ready (Powered by Lucide Icons).');
  }

  // --- Theme Management ---
  applyTheme(theme) {
    this.theme = theme;
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('codealpha_theme', theme);

    const themeBtn = document.getElementById('theme-toggle-btn');
    if (themeBtn) {
      themeBtn.innerHTML = theme === 'dark'
        ? `<i data-lucide="sun" style="width:18px; height:18px;"></i>`
        : `<i data-lucide="moon" style="width:18px; height:18px;"></i>`;
      if (window.lucide) window.lucide.createIcons();
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
      osc.frequency.setValueAtTime(659.25, t + 0.08);
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
          ? `<i data-lucide="volume-2" style="width:18px; height:18px;"></i>`
          : `<i data-lucide="volume-x" style="width:18px; height:18px;"></i>`;
        if (window.lucide) window.lucide.createIcons();
        this.toast(this.soundEnabled ? 'Audio feedback enabled' : 'Audio muted', 'info');
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

    document.querySelectorAll('.app-section').forEach(sec => {
      sec.classList.remove('active');
    });
    targetSection.classList.add('active');

    document.querySelectorAll('[data-target-section]').forEach(link => {
      link.classList.toggle('active', link.dataset.targetSection === sectionId);
    });

    const mobileMenu = document.getElementById('mobile-drawer');
    if (mobileMenu) mobileMenu.classList.remove('open');

    if (updateHash) {
      window.location.hash = sectionId;
    }

    if (window.lucide) window.lucide.createIcons();
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
        if (window.lucide) window.lucide.createIcons();
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
        if (window.lucide) window.lucide.createIcons();
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

  // --- Toast Notifications using Lucide Icons ---
  toast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    const iconName = type === 'success' ? 'check-circle-2' : type === 'warning' ? 'alert-triangle' : 'info';
    toast.innerHTML = `<span class="toast-icon"><i data-lucide="${iconName}" style="width:16px; height:16px;"></i></span> <span class="toast-msg">${message}</span>`;
    container.appendChild(toast);

    if (window.lucide) window.lucide.createIcons();

    setTimeout(() => {
      toast.classList.add('toast-show');
    }, 10);

    setTimeout(() => {
      toast.classList.remove('toast-show');
      setTimeout(() => toast.remove(), 300);
    }, 3200);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  window.app = new App();
});
