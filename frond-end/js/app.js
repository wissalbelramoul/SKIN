/**
 * Main Application Logic
 * Handles UI interactions, rendering, and tying into api.js
 */

const app = {
  // Application Data State
  state: {
    patients: [],
    analyses: []
  },

  // UI Utilities
  ui: {
    toggleModal(modalId) {
      const modal = document.getElementById(modalId);
      if (modal) {
        modal.classList.toggle('active');
      }
    },

    showError(elementId, message) {
      const el = document.getElementById(elementId);
      if (el) {
        el.textContent = message;
        // Auto hide after 5 seconds
        setTimeout(() => { if (el.textContent === message) el.textContent = ""; }, 5000);
      } else {
        alert(message);
      }
    },

    formatDate(dateStr) {
      if (!dateStr) return "-";
      return new Date(dateStr).toLocaleString('fr-FR', {
        day: '2-digit', month: '2-digit', year: 'numeric',
        hour: '2-digit', minute: '2-digit'
      });
    },
    
    updateAuthVisibility() {
      const isAuth = window.api.auth.isAuthenticated();
      const loginBtn = document.getElementById('nav-login-btn');
      const dashBtn = document.getElementById('nav-dash-btn');
      
      if (loginBtn && dashBtn) {
        if (isAuth) {
          loginBtn.classList.add('hidden');
          dashBtn.classList.remove('hidden');
        } else {
          loginBtn.classList.remove('hidden');
          dashBtn.classList.add('hidden');
        }
      }
    }
  },

  // Setup Drag & Drop zones
  setupDropZone(zoneId, inputId, previewId) {
    const zone = document.getElementById(zoneId);
    const input = document.getElementById(inputId);
    const preview = document.getElementById(previewId);

    if (!zone || !input || !preview) return;

    zone.addEventListener('click', () => input.click());
    
    zone.addEventListener('dragover', (e) => {
      e.preventDefault();
      zone.classList.add('dragover');
    });

    zone.addEventListener('dragleave', (e) => {
      e.preventDefault();
      zone.classList.remove('dragover');
    });

    zone.addEventListener('drop', (e) => {
      e.preventDefault();
      zone.classList.remove('dragover');
      
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        input.files = e.dataTransfer.files;
        app.updatePreview(input.files[0], preview, input);
      }
    });

    input.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        app.updatePreview(e.target.files[0], preview, input);
      }
    });
  },

  updatePreview(file, previewEl, inputEl) {
    if (!file.type.startsWith('image/')) {
      alert("Veuillez sélectionner une image valide.");
      inputEl.value = "";
      return;
    }
    previewEl.src = URL.createObjectURL(file);
    previewEl.classList.remove('hidden');

    // Trigger AI prediction if this is the Quick Try zone
    if (inputEl.id === 'ai-file-input') {
      const predictBtn = document.getElementById('ai-btn-predict');
      if (predictBtn) {
        predictBtn.disabled = false;
        // Optionally bind click explicitly if not already
        predictBtn.onclick = () => app.runQuickPrediction(file);
      }
    }
  },

  async runQuickPrediction(file) {
    const container = document.getElementById('ai-progress-container');
    const bar = document.getElementById('ai-progress-bar');
    const resultDiv = document.getElementById('ai-result');
    const predictBtn = document.getElementById('ai-btn-predict');

    predictBtn.disabled = true;
    container.classList.remove('hidden');
    resultDiv.classList.add('hidden');
    bar.style.width = '0%';

    try {
      const result = await window.api.predictImage(file, (percent) => {
        bar.style.width = percent + '%';
      });

      // Quick fake progress if fast local
      if (parseFloat(bar.style.width) < 100) {
         bar.style.width = '100%';
      }

      setTimeout(() => {
        container.classList.add('hidden');
        app.renderPredictionResult(result, resultDiv);
      }, 500);

    } catch (err) {
      container.classList.add('hidden');
      predictBtn.disabled = false;
      
      let msg = err.message;
      if (msg.includes("Network") || msg.includes("Failed to fetch")) {
        msg = "Serveur indisponible. Avez-vous démarré le backend `traefik` et `ml` ?";
      }
      
      resultDiv.innerHTML = `<p style="color: #F87171;">⚠️ ${msg}</p>`;
      resultDiv.classList.remove('hidden');
    }
  },

  renderPredictionResult(data, container) {
    if (!data || !data.top3) {
      container.innerHTML = `<p class="text-muted">Aucun résultat d'analyse disponible.</p>`;
      container.classList.remove('hidden');
      return;
    }

    let html = `<h3 class="mb-4" style="color: var(--emerald-400);">Résultats d'analyse</h3>`;
    data.top3.forEach((item, index) => {
      // 1st item gets special highlight
      const sizeStr = index === 0 ? "font-size: 1.1rem; font-weight: bold;" : "font-size: 0.9rem;";
      html += `
        <div class="flex justify-between items-center mb-2 p-2" style="background: rgba(0,0,0,0.2); border-radius: 0.5rem; ${sizeStr}">
          <span>${item.label}</span>
          <span style="color: var(--emerald-400);">${item.confidence}%</span>
        </div>
      `;
    });

    container.innerHTML = html;
    container.classList.remove('hidden');
    document.getElementById('ai-btn-predict').disabled = false; // allow retry
  },

  // ------------------------------------------------------------------
  // Auth Flows
  // ------------------------------------------------------------------

  setupAuthForms() {
    const loginForm = document.getElementById('login-form');
    if (loginForm) {
      loginForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const u = document.getElementById('login-email').value;
        const p = document.getElementById('login-password').value;
        const btn = loginForm.querySelector('button[type="submit"]');
        
        btn.disabled = true;
        btn.textContent = "Chargement...";
        
        try {
          await window.api.login(u, p);
          app.ui.toggleModal('login-modal');
          // Redirect to dashboard
          window.location.href = "dashboard.html";
        } catch (err) {
          app.ui.showError('login-error', "Échec de connexion : Vérifiez vos identifiants.");
        } finally {
          btn.disabled = false;
          btn.textContent = "Se connecter";
        }
      });
    }

    const regForm = document.getElementById('register-form');
    if (regForm) {
      regForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const u = document.getElementById('reg-email').value;
        const p = document.getElementById('reg-password').value;
        const btn = regForm.querySelector('button[type="submit"]');
        
        btn.disabled = true;
        btn.textContent = "Chargement...";

        try {
          await window.api.register(u, p);
          app.ui.toggleModal('register-modal');
          // Auto login after reg? Depends on backend, but let's just show login modal
          alert("Compte créé avec succès ! Veuillez vous connecter.");
          app.ui.toggleModal('login-modal');
        } catch (err) {
          app.ui.showError('reg-error', "Échec d'inscription : " + err.message);
        } finally {
          btn.disabled = false;
          btn.textContent = "S'inscrire";
        }
      });
    }
  },

  // ------------------------------------------------------------------
  // Dashboard Logics
  // ------------------------------------------------------------------

  setupDashboard() {
    // Tab switching
    const tabPatients = document.getElementById('tab-btn-patients');
    const tabAnalyses = document.getElementById('tab-btn-analyses');
    const secPatients = document.getElementById('section-patients');
    const secAnalyses = document.getElementById('section-analyses');

    if (!tabPatients) return; // Not on dashboard page

    tabPatients.onclick = (e) => {
      e.preventDefault();
      tabPatients.classList.add('active');
      tabAnalyses.classList.remove('active');
      secPatients.classList.remove('hidden');
      secAnalyses.classList.add('hidden');
      app.loadPatients();
    };

    tabAnalyses.onclick = (e) => {
      e.preventDefault();
      tabAnalyses.classList.add('active');
      tabPatients.classList.remove('active');
      secAnalyses.classList.remove('hidden');
      secPatients.classList.add('hidden');
      app.loadAnalyses();
    };

    // Forms
    document.getElementById('create-patient-form').onsubmit = async (e) => {
      e.preventDefault();
      const n = document.getElementById('patient-name').value;
      const notes = document.getElementById('patient-notes').value;
      try {
        await window.api.createPatient(n, notes);
        app.ui.toggleModal('create-patient-modal');
        document.getElementById('create-patient-form').reset();
        await app.loadPatients();
      } catch (err) {
        alert("Erreur lors de la création du patient.");
      }
    };

    document.getElementById('create-analysis-form').onsubmit = async (e) => {
      e.preventDefault();
      const pId = document.getElementById('analysis-patient-id').value;
      const fileInput = document.getElementById('analysis-file-input');
      
      if (!pId || fileInput.files.length === 0) return;

      const btn = e.target.querySelector('button[type="submit"]');
      btn.disabled = true;
      btn.textContent = "Envoi...";

      try {
        await window.api.createAnalysis(pId, fileInput.files[0]);
        app.ui.toggleModal('create-analysis-modal');
        document.getElementById('create-analysis-form').reset();
        document.getElementById('analysis-preview').classList.add('hidden');
        
        // Auto switch to analyses tab
        tabAnalyses.click();
      } catch (err) {
        alert("Erreur d'envoi. Avez-vous lancé RabbitMQ/Consul ?");
      } finally {
        btn.disabled = false;
        btn.textContent = "Envoyer (Worker Asynchrone)";
      }
    };

    // Initialize list
    app.loadPatients();

    // Setup dropzone for analysis
    app.setupDropZone('analysis-drop-zone', 'analysis-file-input', 'analysis-preview');
  },

  async loadPatients() {
    const list = document.getElementById('patients-list');
    const select = document.getElementById('analysis-patient-id');
    if (!list) return;

    try {
      const data = await window.api.getPatients();
      app.state.patients = data;
      
      if (data.length === 0) {
        list.innerHTML = `<li class="text-muted justify-center border-0 bg-transparent">Aucun patient trouvé.</li>`;
        if (select) select.innerHTML = `<option value="">Aucun patient...</option>`;
        return;
      }

      list.innerHTML = data.map(p => `
        <li>
          <div>
            <div style="font-weight: 600; color: var(--text-primary);">#${p.id} ${p.name}</div>
            <div style="font-size: 0.8rem;" class="text-muted">${p.notes || 'Aucune note'}</div>
          </div>
          <div class="text-secondary" style="font-size: 0.8rem;">Créé par ${p.created_by}</div>
        </li>
      `).join('');

      // Populate select
      if (select) {
        select.innerHTML = `<option value="">Sélectionner un patient...</option>` + 
          data.map(p => `<option value="${p.id}">#${p.id} ${p.name}</option>`).join('');
      }

    } catch (err) {
      list.innerHTML = `<li class="text-muted bg-transparent border-0" style="color:#F87171">Erreur de chargement des patients</li>`;
    }
  },

  async loadAnalyses() {
    const list = document.getElementById('analyses-list');
    if (!list) return;

    try {
      const data = await window.api.getAnalyses();
      app.state.analyses = data;
      
      if (data.length === 0) {
        list.innerHTML = `<li class="text-muted justify-center border-0 bg-transparent">Aucune analyse trouvée.</li>`;
        return;
      }

      list.innerHTML = data.map(a => {
        const isPending = a.status === 'pending';
        const badgeClass = isPending ? 'status-badge' : (a.status === 'error' ? 'status-badge error' : 'status-badge');
        
        let resultHtml = "";
        if (!isPending && a.result_json) {
          try {
            const parsed = JSON.parse(a.result_json);
            if (parsed.top3) {
              resultHtml = `<span style="color:var(--emerald-400)">${parsed.top3[0].label} (${parsed.top3[0].confidence}%)</span>`;
            }
          } catch(e) {}
        }
        
        const patientName = app.state.patients.find(p => p.id == a.patient_id)?.name || `Patient #${a.patient_id}`;

        return `
        <li>
          <div>
            <div style="font-weight: 600; color: var(--text-primary);">Analyse #${a.id} - ${patientName}</div>
            <div style="font-size: 0.8rem; margin-top:0.2rem;">
              <span class="${badgeClass}">${a.status}</span>
              ${resultHtml ? `<span style="margin-left: 0.5rem;">${resultHtml}</span>` : ''}
            </div>
          </div>
          <div class="text-secondary" style="font-size: 0.8rem;">Modèle: ${a.model_used || 'N/A'}</div>
        </li>
      `}).join('');

    } catch (err) {
      list.innerHTML = `<li class="text-muted bg-transparent border-0" style="color:#F87171">Erreur de chargement des analyses</li>`;
    }
  },

  // ------------------------------------------------------------------
  // Initializer
  // ------------------------------------------------------------------

  init() {
    // Setup listeners
    window.addEventListener('authStateChanged', app.ui.updateAuthVisibility);
    app.ui.updateAuthVisibility();
    
    // Smooth scroll for anchors
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
      anchor.addEventListener('click', function (e) {
        if (this.getAttribute('href') !== '#') {
          e.preventDefault();
          const target = document.querySelector(this.getAttribute('href'));
          if (target) {
            target.scrollIntoView({ behavior: 'smooth' });
          }
        }
      });
    });

    // Intersection observer for animations
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('show');
        }
      });
    }, { threshold: 0.1 });
    
    document.querySelectorAll('.fade-in').forEach(el => observer.observe(el));

    // Page Specific Setup
    app.setupAuthForms();
    app.setupDropZone('drop-zone', 'ai-file-input', 'ai-preview');
    
    // Check if on dashboard
    if (document.getElementById('section-patients')) {
      app.setupDashboard();
    }
  }
};

// Start
document.addEventListener("DOMContentLoaded", () => app.init());
window.app = app; // Expose for inline onclicks
