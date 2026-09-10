// Diktao PWA - Main Application Engine
import type { 
  DevisData, 
  CotisData, 
  ChantierData, 
  ModuleType, 
  ActiveScreen 
} from './types';
import { 
  getFreemiumState, 
  getRemainingQuota, 
  hasRemainingQuota, 
  consumeQuota, 
  resetQuotasForTesting, 
  saveDocument, 
  getSavedDocuments, 
  getSavedDocumentById, 
  deleteSavedDocument, 
  openDiktaoDB, 
  exportLocalDataBackup, 
  type SavedDoc 
} from './storage';
import { 
  initWasmEngine, 
  extractDocumentFromVoice, 
  computeDevisTotals, 
  computeCotisTotals 
} from './ai_engine';
import { 
  renderDevisToCanvas, 
  renderCotisToCanvas, 
  renderChantierToCanvas, 
  getCanvasBlob 
} from './document_renderer';
import { DiktaoSpeechRecognizer } from './use_speech';
import { PWAManager } from './pwa_manager';

// Application State
let activeScreen: ActiveScreen = 'splash';
let currentModule: ModuleType = 'devis';
let currentDevis: DevisData | null = null;
let currentCotis: CotisData | null = null;
let currentChantier: ChantierData | null = null;
let generatedCanvas: HTMLCanvasElement | null = null;
let speechRecognizer: DiktaoSpeechRecognizer | null = null;
let pwaManager: PWAManager | null = null;
let isModelReady = false;

// Sample Francophone African demonstration phrases
const SAMPLE_PHRASES: Record<ModuleType, string[]> = {
  devis: [
    "Devis pour M. Mensah, 3 sacs de ciment à 4500 le sac, main d'œuvre de maçonnerie 2 jours à 15000 par jour, transport 5000 francs",
    "Devis pour Madame Diallo, 10 pots de peinture blanche à 12000 le pot, 4 rouleaux à 2500, main d'œuvre de peinture 3 jours à 10000 francs",
    "Devis pour M. Kouassi, 15 paquets de fer de 12 à 18000 le paquet, transport 8000 francs, main d'oeuvre 1 jour à 20000"
  ],
  cotis: [
    "Koffi a payé sa cotisation mensuelle de 1000 francs et le droit de secours de 500 francs, Awa a payé 1000 francs seulement",
    "Amadou a versé 5000 francs pour la tontine et 1000 francs droit de secours, Fatou a payé 2500 francs partiel",
    "Mamadou a payé 2000 francs, Ibrahim a payé 1500 francs, Saliou doit sa cotisation de 1000 francs en attente"
  ],
  chantier: [
    "Aujourd'hui nous avons coulé la dalle du premier étage, consommé 40 sacs de ciment. Il faut commander 15 paquets de fer de 12 pour mardi",
    "Chantier Cocody : pose des tuyaux de plomberie et raccordement au rez-de-chaussée. Consommé 6 tubes PVC. Besoin urgent de 10 coudes de 50 d'ici demain",
    "Chantier Plateau : crépissage de la façade principale terminé. Consommé 25 sacs de ciment et 2 bennes de sable. Commander 30 sacs de ciment pour jeudi"
  ]
};

// Initialize Application
export async function initApp() {
  console.log('Initializing Diktao PWA...');

  // Setup PWA Manager
  pwaManager = new PWAManager((canInstall) => {
    updateInstallButtonVisibility(canInstall);
  });

  // Initialize IndexedDB Storage
  openDiktaoDB().then(() => {
    console.log('Diktao IndexedDB storage initialized');
  }).catch(err => {
    console.warn('IndexedDB initial setup note:', err);
  });

  // Pre-initialize WASM engine in background
  initWasmEngine((percent, msg) => {
    const statusEl = document.getElementById('engine-status-text');
    if (statusEl) statusEl.textContent = msg;
    if (percent === 100) isModelReady = true;
  }).then(ready => {
    isModelReady = ready;
    const statusEl = document.getElementById('engine-status-text');
    if (statusEl) statusEl.textContent = 'Moteur d’extraction locale prêt';
  });

  // Setup Speech Recognizer
  speechRecognizer = new DiktaoSpeechRecognizer(
    (transcript, isFinal) => {
      const transcriptInput = document.getElementById('voice-transcript-input') as HTMLTextAreaElement;
      if (transcriptInput) {
        transcriptInput.value = transcript;
        transcriptInput.scrollTop = transcriptInput.scrollHeight;
      }
    },
    (errorMsg) => {
      showToast(errorMsg, 'error');
    },
    (isListening) => {
      updateMicButtonUI(isListening);
    }
  );

  // Bind static DOM event listeners
  bindEventListeners();

  // Initial navigation
  renderScreen('splash');

  // Auto transition from splash to home after 1.8s
  setTimeout(() => {
    if (activeScreen === 'splash') {
      renderScreen('home');
    }
  }, 2200);
}

function updateInstallButtonVisibility(canInstall: boolean) {
  const installBanner = document.getElementById('pwa-install-banner');
  if (installBanner) {
    if (canInstall && !pwaManager?.getIsStandalone()) {
      installBanner.classList.remove('hidden');
    } else {
      installBanner.classList.add('hidden');
    }
  }
}

function updateMicButtonUI(isListening: boolean) {
  const micBtn = document.getElementById('mic-button');
  const micRipple = document.getElementById('mic-ripple');
  const micStatus = document.getElementById('mic-status-text');

  if (micBtn && micRipple && micStatus) {
    if (isListening) {
      micBtn.classList.add('scale-110', 'bg-red-500', 'shadow-red-500/50');
      micBtn.classList.remove('bg-[#FF6B4A]');
      micRipple.classList.remove('hidden');
      micStatus.textContent = "À l'écoute... parlez naturellement";
      micStatus.classList.add('text-red-600', 'font-bold');
      micStatus.classList.remove('text-slate-600');
    } else {
      micBtn.classList.remove('scale-110', 'bg-red-500', 'shadow-red-500/50');
      micBtn.classList.add('bg-[#FF6B4A]');
      micRipple.classList.add('hidden');
      micStatus.textContent = "Appuyez sur le micro pour parler";
      micStatus.classList.remove('text-red-600', 'font-bold');
      micStatus.classList.add('text-slate-600');
    }
  }
}

// Navigation Router
export function renderScreen(screen: ActiveScreen) {
  activeScreen = screen;

  // Hide all screens
  const screenIds = ['screen-splash', 'screen-home', 'screen-voice', 'screen-editor', 'screen-share', 'screen-limit', 'screen-about', 'screen-history'];
  screenIds.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.classList.add('hidden');
  });

  // Show target screen
  const target = document.getElementById(`screen-${screen}`);
  if (target) {
    target.classList.remove('hidden');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  // Update screen-specific state
  if (screen === 'home') {
    updateHomeQuotasUI();
  } else if (screen === 'voice') {
    setupVoiceScreenUI();
  } else if (screen === 'editor') {
    setupEditorScreenUI();
  } else if (screen === 'history') {
    setupHistoryScreenUI();
  }
}

function updateHomeQuotasUI() {
  const devisQuota = getRemainingQuota('devis');
  const cotisQuota = getRemainingQuota('cotis');
  const chantierQuota = getRemainingQuota('chantier');

  const devisBadge = document.getElementById('quota-devis');
  const cotisBadge = document.getElementById('quota-cotis');
  const chantierBadge = document.getElementById('quota-chantier');

  if (devisBadge) devisBadge.textContent = `${devisQuota}/5 gratuits ce mois`;
  if (cotisBadge) cotisBadge.textContent = `${cotisQuota}/5 gratuits ce mois`;
  if (chantierBadge) chantierBadge.textContent = `${chantierQuota}/5 gratuits ce mois`;
}

function selectModuleAndStart(module: ModuleType) {
  currentModule = module;

  // Freemium quota check
  if (!hasRemainingQuota(module)) {
    renderScreen('limit');
    const limitTitle = document.getElementById('limit-module-name');
    if (limitTitle) {
      if (module === 'devis') limitTitle.textContent = 'Devis-Éclair';
      else if (module === 'cotis') limitTitle.textContent = 'Cotis-Métier';
      else limitTitle.textContent = 'Rapport-Chantier-Pro';
    }
    return;
  }

  // Clear previous session if new document
  if (module === 'devis') currentDevis = null;
  else if (module === 'cotis') currentCotis = null;
  else currentChantier = null;

  renderScreen('voice');
}

function setupVoiceScreenUI() {
  speechRecognizer?.stop();
  speechRecognizer?.clear();

  const titleEl = document.getElementById('voice-module-title');
  const descEl = document.getElementById('voice-module-desc');
  const chipsContainer = document.getElementById('voice-chips-container');
  const transcriptInput = document.getElementById('voice-transcript-input') as HTMLTextAreaElement;

  if (transcriptInput) transcriptInput.value = '';

  let title = 'Devis-Éclair';
  let desc = 'Dictez les matériaux, quantités, prix unitaires et journées de travail.';
  if (currentModule === 'cotis') {
    title = 'Cotis-Métier';
    desc = 'Dictez les cotisations des membres (ex: Koffi 1000 FCFA, Awa 1000 FCFA).';
  } else if (currentModule === 'chantier') {
    title = 'Rapport-Chantier-Pro';
    desc = 'Dictez les travaux effectués ce jour, les matériaux sortis et les commandes.';
  }

  if (titleEl) titleEl.textContent = title;
  if (descEl) descEl.textContent = desc;

  // Render suggestion chips
  if (chipsContainer) {
    chipsContainer.innerHTML = '';
    const phrases = SAMPLE_PHRASES[currentModule];
    phrases.forEach((phrase, idx) => {
      const chip = document.createElement('button');
      chip.type = 'button';
      chip.className = 'text-left text-xs bg-slate-100 hover:bg-slate-200 text-slate-700 px-3 py-2 rounded-lg border border-slate-200 transition line-clamp-2';
      chip.innerHTML = `<span class="text-[#FF6B4A] font-semibold">Exemple ${idx + 1} :</span> "${phrase}"`;
      chip.onclick = () => {
        if (transcriptInput) {
          transcriptInput.value = phrase;
          speechRecognizer?.setText(phrase);
        }
      };
      chipsContainer.appendChild(chip);
    });
  }
}

async function handleAnalyzeVoice() {
  const transcriptInput = document.getElementById('voice-transcript-input') as HTMLTextAreaElement;
  const text = transcriptInput?.value?.trim();

  if (!text) {
    showToast("Veuillez dicter ou saisir un texte avant de valider.", "warning");
    return;
  }

  speechRecognizer?.stop();

  const analyzeBtn = document.getElementById('voice-analyze-btn') as HTMLButtonElement;
  if (analyzeBtn) {
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = `
      <svg class="animate-spin -ml-1 mr-2 h-4 w-4 text-white inline" fill="none" viewBox="0 0 24 24">
        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
      </svg>
      Extraction structurée en cours...
    `;
  }

  try {
    const extracted = await extractDocumentFromVoice(currentModule, text);

    if (currentModule === 'devis') {
      currentDevis = extracted as DevisData;
    } else if (currentModule === 'cotis') {
      const newCotis = extracted as CotisData;
      // If continuing an existing session, merge contributions!
      if (currentCotis && currentCotis.contributions) {
        currentCotis.contributions.push(...newCotis.contributions);
      } else {
        currentCotis = newCotis;
      }
    } else {
      currentChantier = extracted as ChantierData;
    }

    // Go to Editor
    renderScreen('editor');
  } catch (err) {
    console.error('Extraction error:', err);
    showToast("Erreur lors de l'extraction. Veuillez réessayer.", "error");
  } finally {
    if (analyzeBtn) {
      analyzeBtn.disabled = false;
      analyzeBtn.innerHTML = `Valider et générer le document →`;
    }
  }
}

function setupEditorScreenUI() {
  const editorDevis = document.getElementById('editor-devis-container');
  const editorCotis = document.getElementById('editor-cotis-container');
  const editorChantier = document.getElementById('editor-chantier-container');

  if (editorDevis) editorDevis.classList.add('hidden');
  if (editorCotis) editorCotis.classList.add('hidden');
  if (editorChantier) editorChantier.classList.add('hidden');

  if (currentModule === 'devis' && currentDevis) {
    if (editorDevis) editorDevis.classList.remove('hidden');
    renderDevisEditorForm();
  } else if (currentModule === 'cotis' && currentCotis) {
    if (editorCotis) editorCotis.classList.remove('hidden');
    renderCotisEditorForm();
  } else if (currentModule === 'chantier' && currentChantier) {
    if (editorChantier) editorChantier.classList.remove('hidden');
    renderChantierEditorForm();
  }
}

// -------------------------------------------------------------
// EDITOR FORMS IMPLEMENTATION
// -------------------------------------------------------------

function renderDevisEditorForm() {
  if (!currentDevis) return;

  const clientInput = document.getElementById('devis-client-name') as HTMLInputElement;
  const providerInput = document.getElementById('devis-provider-name') as HTMLInputElement;
  const currencySelect = document.getElementById('devis-currency') as HTMLSelectElement;
  const laborDaysInput = document.getElementById('devis-labor-days') as HTMLInputElement;
  const laborRateInput = document.getElementById('devis-labor-rate') as HTMLInputElement;

  if (clientInput) clientInput.value = currentDevis.client_name;
  if (providerInput) providerInput.value = currentDevis.provider_name;
  if (currencySelect) currencySelect.value = currentDevis.currency || 'FCFA';
  if (laborDaysInput) laborDaysInput.value = String(currentDevis.labor_days || 0);
  if (laborRateInput) laborRateInput.value = String(currentDevis.labor_price_per_day || 0);

  renderDevisItemsTable();
  updateDevisCalculationsDisplay();
}

function renderDevisItemsTable() {
  if (!currentDevis) return;
  const tableBody = document.getElementById('devis-items-tbody');
  if (!tableBody) return;

  tableBody.innerHTML = '';
  currentDevis.items.forEach((item, index) => {
    const tr = document.createElement('tr');
    tr.className = 'border-b border-slate-100 hover:bg-slate-50/60';
    tr.innerHTML = `
      <td class="py-2.5 px-2">
        <input type="text" value="${escapeHtml(item.description)}" class="w-full bg-transparent text-sm text-slate-800 focus:bg-white focus:ring-1 focus:ring-[#1A365D] rounded px-1.5 py-1 border border-transparent hover:border-slate-200" onchange="window.updateDevisItem(${index}, 'description', this.value)" />
      </td>
      <td class="py-2.5 px-2 w-20 text-center">
        <input type="number" min="1" value="${item.quantity}" class="w-full text-center bg-transparent text-sm text-slate-800 focus:bg-white focus:ring-1 focus:ring-[#1A365D] rounded px-1 py-1 border border-transparent hover:border-slate-200" onchange="window.updateDevisItem(${index}, 'quantity', this.value)" />
      </td>
      <td class="py-2.5 px-2 w-28 text-right">
        <input type="number" min="0" value="${item.unit_price}" class="w-full text-right bg-transparent text-sm text-slate-800 focus:bg-white focus:ring-1 focus:ring-[#1A365D] rounded px-1.5 py-1 border border-transparent hover:border-slate-200" onchange="window.updateDevisItem(${index}, 'unit_price', this.value)" />
      </td>
      <td class="py-2.5 px-2 w-28 text-right font-semibold text-slate-900 text-sm">
        ${formatMoney((item.quantity || 0) * (item.unit_price || 0))}
      </td>
      <td class="py-2.5 px-2 w-10 text-center">
        <button type="button" onclick="window.removeDevisItem(${index})" class="text-slate-400 hover:text-red-500 transition p-1">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path></svg>
        </button>
      </td>
    `;
    tableBody.appendChild(tr);
  });
}

function updateDevisCalculationsDisplay() {
  if (!currentDevis) return;
  const totals = computeDevisTotals(currentDevis);

  const subtotalItemsEl = document.getElementById('devis-subtotal-items');
  const subtotalLaborEl = document.getElementById('devis-subtotal-labor');
  const grandTotalEl = document.getElementById('devis-grand-total');

  if (subtotalItemsEl) subtotalItemsEl.textContent = `${formatMoney(totals.subtotal_items)} ${currentDevis.currency}`;
  if (subtotalLaborEl) subtotalLaborEl.textContent = `${formatMoney(totals.subtotal_labor)} ${currentDevis.currency}`;
  if (grandTotalEl) grandTotalEl.textContent = `${formatMoney(totals.total_general)} ${currentDevis.currency}`;
}

function renderCotisEditorForm() {
  if (!currentCotis) return;

  const assocInput = document.getElementById('cotis-assoc-name') as HTMLInputElement;
  if (assocInput) assocInput.value = currentCotis.association_name;

  renderCotisTable();
  updateCotisCalculationsDisplay();
}

function renderCotisTable() {
  if (!currentCotis) return;
  const tableBody = document.getElementById('cotis-items-tbody');
  if (!tableBody) return;

  tableBody.innerHTML = '';
  currentCotis.contributions.forEach((item, index) => {
    const tr = document.createElement('tr');
    tr.className = 'border-b border-slate-100 hover:bg-slate-50/60';
    tr.innerHTML = `
      <td class="py-2.5 px-2">
        <input type="text" value="${escapeHtml(item.member_name)}" class="w-full font-semibold text-sm text-slate-800 bg-transparent focus:bg-white rounded px-1.5 py-1 border border-transparent hover:border-slate-200" onchange="window.updateCotisItem(${index}, 'member_name', this.value)" />
      </td>
      <td class="py-2.5 px-2">
        <input type="text" value="${escapeHtml(item.purpose || 'Cotisation')}" class="w-full text-xs text-slate-600 bg-transparent focus:bg-white rounded px-1.5 py-1 border border-transparent hover:border-slate-200" onchange="window.updateCotisItem(${index}, 'purpose', this.value)" />
      </td>
      <td class="py-2.5 px-2 w-28 text-right">
        <input type="number" min="0" value="${item.amount}" class="w-full text-right font-bold text-sm text-slate-900 bg-transparent focus:bg-white rounded px-1.5 py-1 border border-transparent hover:border-slate-200" onchange="window.updateCotisItem(${index}, 'amount', this.value)" />
      </td>
      <td class="py-2.5 px-2 w-28 text-center">
        <select class="text-xs font-semibold rounded px-1.5 py-1 border border-slate-200 bg-white" onchange="window.updateCotisItem(${index}, 'payment_status', this.value)">
          <option value="payé" ${item.payment_status === 'payé' ? 'selected' : ''}>Payé</option>
          <option value="partiel" ${item.payment_status === 'partiel' ? 'selected' : ''}>Partiel</option>
          <option value="en attente" ${item.payment_status === 'en attente' ? 'selected' : ''}>En attente</option>
        </select>
      </td>
      <td class="py-2.5 px-2 w-10 text-center">
        <button type="button" onclick="window.removeCotisItem(${index})" class="text-slate-400 hover:text-red-500 transition p-1">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path></svg>
        </button>
      </td>
    `;
    tableBody.appendChild(tr);
  });
}

function updateCotisCalculationsDisplay() {
  if (!currentCotis) return;
  const totals = computeCotisTotals(currentCotis);

  const totalCollectedEl = document.getElementById('cotis-total-collected');
  const countMembersEl = document.getElementById('cotis-count-members');

  if (totalCollectedEl) totalCollectedEl.textContent = `${formatMoney(totals.total_collected)} FCFA`;
  if (countMembersEl) countMembersEl.textContent = `${totals.total_members_paid} membres`;
}

function renderChantierEditorForm() {
  if (!currentChantier) return;

  const siteInput = document.getElementById('chantier-site-name') as HTMLInputElement;
  if (siteInput) siteInput.value = currentChantier.site_name;

  // Render Work Done
  const workList = document.getElementById('chantier-work-list');
  if (workList) {
    workList.innerHTML = '';
    currentChantier.work_done.forEach((work, idx) => {
      const div = document.createElement('div');
      div.className = 'flex items-center gap-2 mb-2';
      div.innerHTML = `
        <span class="text-green-600 font-bold">✓</span>
        <input type="text" value="${escapeHtml(work)}" class="flex-1 text-sm bg-slate-50 border border-slate-200 rounded px-2.5 py-1.5 focus:bg-white" onchange="window.updateChantierWork(${idx}, this.value)" />
        <button type="button" onclick="window.removeChantierWork(${idx})" class="text-slate-400 hover:text-red-500 p-1">✕</button>
      `;
      workList.appendChild(div);
    });
  }

  // Render Materials Used
  const usedList = document.getElementById('chantier-used-list');
  if (usedList) {
    usedList.innerHTML = '';
    currentChantier.materials_used.forEach((item, idx) => {
      const div = document.createElement('div');
      div.className = 'flex items-center gap-2 mb-2';
      div.innerHTML = `
        <input type="text" placeholder="Matériau" value="${escapeHtml(item.material)}" class="flex-1 text-sm bg-slate-50 border border-slate-200 rounded px-2.5 py-1.5 focus:bg-white" onchange="window.updateChantierUsed(${idx}, 'material', this.value)" />
        <input type="text" placeholder="Quantité" value="${escapeHtml(item.quantity)}" class="w-28 text-sm bg-slate-50 border border-slate-200 rounded px-2.5 py-1.5 focus:bg-white" onchange="window.updateChantierUsed(${idx}, 'quantity', this.value)" />
        <button type="button" onclick="window.removeChantierUsed(${idx})" class="text-slate-400 hover:text-red-500 p-1">✕</button>
      `;
      usedList.appendChild(div);
    });
  }

  // Render Materials Needed
  const neededList = document.getElementById('chantier-needed-list');
  if (neededList) {
    neededList.innerHTML = '';
    currentChantier.materials_needed.forEach((item, idx) => {
      const div = document.createElement('div');
      div.className = 'flex items-center gap-2 mb-2';
      div.innerHTML = `
        <input type="text" placeholder="Matériau à commander" value="${escapeHtml(item.material)}" class="flex-1 text-sm bg-slate-50 border border-slate-200 rounded px-2.5 py-1.5 focus:bg-white" onchange="window.updateChantierNeeded(${idx}, 'material', this.value)" />
        <input type="text" placeholder="Quantité" value="${escapeHtml(item.quantity)}" class="w-24 text-sm bg-slate-50 border border-slate-200 rounded px-2.5 py-1.5 focus:bg-white" onchange="window.updateChantierNeeded(${idx}, 'quantity', this.value)" />
        <input type="text" placeholder="Délai (ex: Pour mardi)" value="${escapeHtml(item.deadline || '')}" class="w-32 text-xs bg-slate-50 border border-slate-200 rounded px-2 py-1.5 focus:bg-white" onchange="window.updateChantierNeeded(${idx}, 'deadline', this.value)" />
        <button type="button" onclick="window.removeChantierNeeded(${idx})" class="text-slate-400 hover:text-red-500 p-1">✕</button>
      `;
      neededList.appendChild(div);
    });
  }

  // Render Photos Gallery
  renderPhotosPreviewGrid();
}

function renderPhotosPreviewGrid() {
  if (!currentChantier) return;
  const grid = document.getElementById('chantier-photos-grid');
  if (!grid) return;

  grid.innerHTML = '';
  currentChantier.photos.forEach((photo, idx) => {
    const div = document.createElement('div');
    div.className = 'relative group rounded-xl overflow-hidden aspect-video border-2 border-slate-200 shadow-sm';
    div.innerHTML = `
      <img src="${photo}" alt="Photo ${idx + 1}" class="w-full h-full object-cover" />
      <button type="button" onclick="window.removeChantierPhoto(${idx})" class="absolute top-1.5 right-1.5 bg-black/60 hover:bg-red-600 text-white rounded-full p-1 transition shadow">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>
      </button>
      <div class="absolute bottom-1 left-1.5 bg-black/50 text-white text-[10px] font-semibold px-2 py-0.5 rounded">Cliché #${idx + 1}</div>
    `;
    grid.appendChild(div);
  });
}

// -------------------------------------------------------------
// DOCUMENT FINALIZATION & CANVAS RENDERING
// -------------------------------------------------------------

async function handleGenerateFinalDocument() {
  const canvas = document.createElement('canvas');
  generatedCanvas = canvas;

  const previewContainer = document.getElementById('share-canvas-preview');
  if (previewContainer) {
    previewContainer.innerHTML = '<div class="text-center py-10 text-slate-500 text-sm">Génération du document haute résolution en cours...</div>';
  }

  // Render Canvas depending on module
  if (currentModule === 'devis' && currentDevis) {
    renderDevisToCanvas(canvas, currentDevis);
    consumeQuota('devis');
    const thumb = canvas.toDataURL('image/jpeg', 0.5);
    await saveDocument({
      id: 'doc_' + Date.now(),
      type: 'devis',
      title: `Devis — ${currentDevis.client_name || 'Client'}`,
      date: currentDevis.date,
      createdAt: Date.now(),
      data: currentDevis,
      thumbnail: thumb,
    });
  } else if (currentModule === 'cotis' && currentCotis) {
    renderCotisToCanvas(canvas, currentCotis);
    consumeQuota('cotis');
    const thumb = canvas.toDataURL('image/jpeg', 0.5);
    await saveDocument({
      id: 'doc_' + Date.now(),
      type: 'cotis',
      title: `Registre — ${currentCotis.association_name}`,
      date: currentCotis.date,
      createdAt: Date.now(),
      data: currentCotis,
      thumbnail: thumb,
    });
  } else if (currentModule === 'chantier' && currentChantier) {
    await renderChantierToCanvas(canvas, currentChantier);
    consumeQuota('chantier');
    const thumb = canvas.toDataURL('image/jpeg', 0.5);
    await saveDocument({
      id: 'doc_' + Date.now(),
      type: 'chantier',
      title: `Rapport — ${currentChantier.site_name}`,
      date: currentChantier.date,
      createdAt: Date.now(),
      data: currentChantier,
      thumbnail: thumb,
    });
  }

  // Inject rendered canvas into preview container
  if (previewContainer) {
    previewContainer.innerHTML = '';
    canvas.className = 'w-full h-auto rounded-xl shadow-xl border border-slate-200';
    previewContainer.appendChild(canvas);
  }

  renderScreen('share');
}

// -------------------------------------------------------------
// SHARING & EXPORT ACTIONS
// -------------------------------------------------------------

async function handleDownloadPNG() {
  if (!generatedCanvas) return;
  try {
    const blob = await getCanvasBlob(generatedCanvas);
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `Diktao_${currentModule.toUpperCase()}_${new Date().toISOString().slice(0, 10)}.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showToast("Document PNG téléchargé avec succès !", "success");
  } catch (err) {
    console.error('Download error:', err);
    showToast("Erreur lors du téléchargement de l'image.", "error");
  }
}

async function handleShareWhatsApp() {
  let text = '';
  if (currentModule === 'devis' && currentDevis) {
    const totals = computeDevisTotals(currentDevis);
    text = `*DEVIS ESTIMATIF — ${currentDevis.client_name}*\n` +
      `Émis par : ${currentDevis.provider_name}\n` +
      `Date : ${currentDevis.date}\n\n` +
      `*Fournitures :*\n` +
      currentDevis.items.map(i => `• ${i.description} (x${i.quantity}) : ${formatMoney(i.quantity * i.unit_price)} ${currentDevis?.currency}`).join('\n') +
      (currentDevis.labor_days ? `\n• Main d'œuvre (${currentDevis.labor_days}j à ${formatMoney(currentDevis.labor_price_per_day || 0)}) : ${formatMoney(totals.subtotal_labor)} ${currentDevis.currency}` : '') +
      `\n\n*TOTAL GÉNÉRAL : ${formatMoney(totals.total_general)} ${currentDevis.currency}*\n\n_Généré via Diktao.app_`;
  } else if (currentModule === 'cotis' && currentCotis) {
    const totals = computeCotisTotals(currentCotis);
    text = `*BILAN DE COTISATIONS — ${currentCotis.association_name}*\n` +
      `Séance du : ${currentCotis.date}\n\n` +
      `*Versements enregistrés :*\n` +
      currentCotis.contributions.map(c => `• ${c.member_name} (${c.purpose || 'Cotisation'}) : ${formatMoney(c.amount)} FCFA [${c.payment_status.toUpperCase()}]`).join('\n') +
      `\n\n*TOTAL COLLECTÉ : ${formatMoney(totals.total_collected)} FCFA* (${totals.total_members_paid} membres)\n\n_Généré via Diktao.app_`;
  } else if (currentModule === 'chantier' && currentChantier) {
    text = `*RAPPORT JOURNALIER — ${currentChantier.site_name}*\n` +
      `Date : ${currentChantier.date}\n\n` +
      `*Travaux réalisés :*\n` +
      currentChantier.work_done.map(w => `✓ ${w}`).join('\n') +
      `\n\n*Matériaux consommés :*\n` +
      currentChantier.materials_used.map(m => `• ${m.material} : ${m.quantity}`).join('\n') +
      `\n\n*Besoins urgents pour la suite :*\n` +
      currentChantier.materials_needed.map(n => `⚠️ ${n.material} (${n.quantity}) — ${n.deadline || 'Dès que possible'}`).join('\n') +
      `\n\n_Généré via Diktao.app_`;
  }

  // Also trigger Web Share API with image file if supported
  if (generatedCanvas && navigator.canShare) {
    try {
      const blob = await getCanvasBlob(generatedCanvas);
      const file = new File([blob], `document-diktao.png`, { type: 'image/png' });
      if (navigator.canShare({ files: [file] })) {
        await navigator.share({
          title: 'Document Diktao',
          text,
          files: [file]
        });
        showToast("Partage effectué !", "success");
        return;
      }
    } catch (e) {
      // Fallback to wa.me URL
    }
  }

  // Standard WhatsApp URL scheme fallback
  const encoded = encodeURIComponent(text);
  window.open(`https://wa.me/?text=${encoded}`, '_blank');
}

async function handleNativeWebShare() {
  if (!generatedCanvas) return;
  try {
    const blob = await getCanvasBlob(generatedCanvas);
    const file = new File([blob], `document-diktao.png`, { type: 'image/png' });
    if (navigator.share) {
      await navigator.share({
        title: 'Document officiel Diktao',
        text: 'Voici votre document généré via Diktao.',
        files: [file]
      });
      showToast("Document partagé avec succès !", "success");
    } else {
      handleDownloadPNG();
    }
  } catch (e) {
    console.warn('Web Share cancelled or failed', e);
  }
}

// -------------------------------------------------------------
// HISTORY VIEW (IndexedDB Offline Storage)
// -------------------------------------------------------------

async function setupHistoryScreenUI() {
  const container = document.getElementById('history-docs-list');
  if (!container) return;

  container.innerHTML = `
    <div class="text-center py-10 text-slate-400 text-xs">
      <div class="w-6 h-6 border-2 border-[#1A365D] border-t-transparent rounded-full animate-spin mx-auto mb-2"></div>
      Chargement de la base locale IndexedDB...
    </div>
  `;

  const docs = await getSavedDocuments();
  if (docs.length === 0) {
    container.innerHTML = `
      <div class="text-center py-12 bg-white rounded-2xl border border-slate-200 p-6">
        <div class="w-16 h-16 mx-auto bg-slate-100 rounded-full flex items-center justify-center text-slate-400 mb-3">
          <svg class="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>
        </div>
        <h3 class="font-bold text-slate-800 text-lg">Aucun document archivé</h3>
        <p class="text-xs text-slate-500 mt-1">Vos devis, registres et rapports créés sont automatiquement conservés dans votre base de données locale IndexedDB.</p>
        <button type="button" onclick="window.renderScreen('home')" class="mt-4 px-4 py-2 bg-[#1A365D] text-white text-xs font-semibold rounded-xl">Créer un document</button>
      </div>
    `;
    return;
  }

  container.innerHTML = '';
  docs.forEach(doc => {
    const card = document.createElement('div');
    card.className = 'bg-white rounded-2xl p-3 border border-slate-200 shadow-sm flex items-center gap-3 mb-3 hover:border-slate-300 transition';
    
    let badgeColor = 'bg-blue-50 text-blue-700 border-blue-200';
    let typeLabel = 'Devis';
    if (doc.type === 'cotis') {
      badgeColor = 'bg-teal-50 text-teal-700 border-teal-200';
      typeLabel = 'Cotisations';
    } else if (doc.type === 'chantier') {
      badgeColor = 'bg-amber-50 text-amber-700 border-amber-200';
      typeLabel = 'Chantier';
    }

    const thumbHtml = doc.thumbnail 
      ? `<img src="${doc.thumbnail}" class="w-14 h-16 object-cover rounded-lg border border-slate-200 shrink-0 shadow-xs" alt="Aperçu" />`
      : `<div class="w-14 h-16 rounded-lg bg-slate-100 flex items-center justify-center shrink-0 text-slate-400">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>
        </div>`;

    card.innerHTML = `
      ${thumbHtml}
      <div class="flex-1 min-w-0 pr-1">
        <div class="flex items-center gap-1.5 mb-1">
          <span class="text-[10px] font-bold uppercase px-2 py-0.5 rounded border ${badgeColor}">${typeLabel}</span>
          <span class="text-[11px] text-slate-400 truncate">${doc.date}</span>
        </div>
        <h4 class="font-bold text-slate-800 text-sm truncate">${escapeHtml(doc.title)}</h4>
      </div>
      <div class="flex items-center gap-1 shrink-0">
        <button type="button" onclick="window.openArchivedDoc('${doc.id}')" class="px-3 py-1.5 bg-[#1A365D] hover:bg-[#0D204A] text-white text-xs font-semibold rounded-lg transition" title="Ouvrir dans l'éditeur">
          Ouvrir
        </button>
        <button type="button" onclick="window.deleteArchivedDoc('${doc.id}')" class="p-1.5 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded-lg transition" title="Supprimer de la base">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path></svg>
        </button>
      </div>
    `;
    container.appendChild(card);
  });
}

async function openArchivedDoc(docId: string) {
  const doc = await getSavedDocumentById(docId);
  if (!doc) {
    showToast("Document introuvable dans la base locale.", "error");
    return;
  }

  currentModule = doc.type;
  if (doc.type === 'devis') currentDevis = doc.data as DevisData;
  else if (doc.type === 'cotis') currentCotis = doc.data as CotisData;
  else currentChantier = doc.data as ChantierData;

  renderScreen('editor');
}

// -------------------------------------------------------------
// EVENT BINDINGS & GLOBAL HELPERS
// -------------------------------------------------------------

function bindEventListeners() {
  // Navigation & buttons
  document.getElementById('splash-start-btn')?.addEventListener('click', () => renderScreen('home'));
  document.getElementById('nav-home-btn')?.addEventListener('click', () => renderScreen('home'));
  document.getElementById('nav-history-btn')?.addEventListener('click', () => renderScreen('history'));
  document.getElementById('nav-about-btn')?.addEventListener('click', () => renderScreen('about'));

  // Module Selection
  document.getElementById('card-module-devis')?.addEventListener('click', () => selectModuleAndStart('devis'));
  document.getElementById('card-module-cotis')?.addEventListener('click', () => selectModuleAndStart('cotis'));
  document.getElementById('card-module-chantier')?.addEventListener('click', () => selectModuleAndStart('chantier'));

  // Voice Screen
  document.getElementById('mic-button')?.addEventListener('click', () => {
    speechRecognizer?.toggle();
  });
  document.getElementById('voice-clear-btn')?.addEventListener('click', () => {
    speechRecognizer?.clear();
    const input = document.getElementById('voice-transcript-input') as HTMLTextAreaElement;
    if (input) input.value = '';
  });
  document.getElementById('voice-analyze-btn')?.addEventListener('click', handleAnalyzeVoice);
  document.getElementById('voice-back-btn')?.addEventListener('click', () => renderScreen('home'));

  // Editor Actions
  document.getElementById('editor-revoice-btn')?.addEventListener('click', () => renderScreen('voice'));
  document.getElementById('editor-generate-btn')?.addEventListener('click', handleGenerateFinalDocument);
  document.getElementById('editor-back-btn')?.addEventListener('click', () => renderScreen('home'));

  // Devis specific add item
  document.getElementById('devis-add-item-btn')?.addEventListener('click', () => {
    if (currentDevis) {
      currentDevis.items.push({
        id: 'item_' + Date.now(),
        description: 'Nouvel article',
        quantity: 1,
        unit_price: 1000,
      });
      renderDevisItemsTable();
      updateDevisCalculationsDisplay();
    }
  });

  // Cotis specific add contribution & multi-session voice
  document.getElementById('cotis-add-item-btn')?.addEventListener('click', () => {
    if (currentCotis) {
      currentCotis.contributions.push({
        id: 'cotis_' + Date.now(),
        member_name: 'Nouveau membre',
        amount: 1000,
        purpose: 'Cotisation mensuelle',
        payment_status: 'payé',
      });
      renderCotisTable();
      updateCotisCalculationsDisplay();
    }
  });
  document.getElementById('cotis-voice-append-btn')?.addEventListener('click', () => {
    // Return to voice capture to append another voice session to the same meeting!
    renderScreen('voice');
  });

  // Chantier specific add work & materials
  document.getElementById('chantier-add-work-btn')?.addEventListener('click', () => {
    if (currentChantier) {
      currentChantier.work_done.push('Nouvelle tâche de chantier');
      renderChantierEditorForm();
    }
  });
  document.getElementById('chantier-add-used-btn')?.addEventListener('click', () => {
    if (currentChantier) {
      currentChantier.materials_used.push({
        id: 'mat_' + Date.now(),
        material: 'Matériau consommé',
        quantity: '1 unité',
      });
      renderChantierEditorForm();
    }
  });
  document.getElementById('chantier-add-needed-btn')?.addEventListener('click', () => {
    if (currentChantier) {
      currentChantier.materials_needed.push({
        id: 'mat_' + Date.now(),
        material: 'Matériau à commander',
        quantity: '1 unité',
        deadline: 'Bientôt',
      });
      renderChantierEditorForm();
    }
  });

  // Photo uploads
  const cameraInput = document.getElementById('chantier-camera-input') as HTMLInputElement;
  cameraInput?.addEventListener('change', (e: any) => {
    const files = e.target.files;
    if (!files || !currentChantier) return;

    for (let i = 0; i < files.length; i++) {
      if (currentChantier.photos.length >= 4) {
        showToast("Maximum 4 photos autorisées.", "warning");
        break;
      }
      const reader = new FileReader();
      reader.onload = (loadEvent) => {
        const dataUrl = loadEvent.target?.result as string;
        if (dataUrl && currentChantier) {
          currentChantier.photos.push(dataUrl);
          renderPhotosPreviewGrid();
        }
      };
      reader.readAsDataURL(files[i]);
    }
  });

  // Share Actions
  document.getElementById('share-whatsapp-btn')?.addEventListener('click', handleShareWhatsApp);
  document.getElementById('share-download-btn')?.addEventListener('click', handleDownloadPNG);
  document.getElementById('share-native-btn')?.addEventListener('click', handleNativeWebShare);
  document.getElementById('share-new-doc-btn')?.addEventListener('click', () => renderScreen('home'));

  // Freemium Limit Screen Actions
  document.getElementById('limit-home-btn')?.addEventListener('click', () => renderScreen('home'));
  document.getElementById('limit-reset-btn')?.addEventListener('click', () => {
    resetQuotasForTesting();
    showToast("Quotas de test réinitialisés à 5 documents !", "success");
    renderScreen('home');
  });

  // PWA Install Button
  document.getElementById('pwa-install-btn')?.addEventListener('click', () => {
    if (pwaManager?.getIsIOS()) {
      const iosModal = document.getElementById('ios-install-modal');
      if (iosModal) iosModal.classList.remove('hidden');
    } else {
      pwaManager?.promptInstall();
    }
  });
  document.getElementById('close-ios-modal')?.addEventListener('click', () => {
    const iosModal = document.getElementById('ios-install-modal');
    if (iosModal) iosModal.classList.add('hidden');
  });
}

// Window Globals for dynamic DOM inputs
(window as any).renderScreen = renderScreen;
(window as any).openArchivedDoc = openArchivedDoc;

(window as any).deleteArchivedDoc = async (docId: string) => {
  if (confirm('Voulez-vous supprimer ce document de votre base locale ?')) {
    await deleteSavedDocument(docId);
    showToast("Document supprimé des archives locales.", "info");
    await setupHistoryScreenUI();
  }
};

(window as any).exportLocalBackup = async () => {
  try {
    const backupJson = await exportLocalDataBackup();
    const blob = new Blob([backupJson], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `Diktao_Sauvegarde_IndexedDB_${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showToast("Sauvegarde JSON exportée avec succès !", "success");
  } catch (err) {
    console.error('Export error', err);
    showToast("Erreur lors de l'export de la sauvegarde.", "error");
  }
};

(window as any).updateDevisLabor = () => {
  if (!currentDevis) return;
  const daysEl = document.getElementById('devis-labor-days') as HTMLInputElement;
  const rateEl = document.getElementById('devis-labor-rate') as HTMLInputElement;
  if (daysEl) currentDevis.labor_days = parseFloat(daysEl.value) || 0;
  if (rateEl) currentDevis.labor_price_per_day = parseFloat(rateEl.value) || 0;
  updateDevisCalculationsDisplay();
};

(window as any).resetQuotasForTesting = () => {
  resetQuotasForTesting();
  updateHomeQuotasUI();
};

(window as any).updateDevisItem = (idx: number, field: string, val: string) => {
  if (!currentDevis || !currentDevis.items[idx]) return;
  if (field === 'description') currentDevis.items[idx].description = val;
  else if (field === 'quantity') currentDevis.items[idx].quantity = parseFloat(val) || 1;
  else if (field === 'unit_price') currentDevis.items[idx].unit_price = parseFloat(val) || 0;
  updateDevisCalculationsDisplay();
  renderDevisItemsTable();
};

(window as any).removeDevisItem = (idx: number) => {
  if (!currentDevis) return;
  currentDevis.items.splice(idx, 1);
  renderDevisItemsTable();
  updateDevisCalculationsDisplay();
};

(window as any).updateCotisItem = (idx: number, field: string, val: string) => {
  if (!currentCotis || !currentCotis.contributions[idx]) return;
  if (field === 'member_name') currentCotis.contributions[idx].member_name = val;
  else if (field === 'purpose') currentCotis.contributions[idx].purpose = val;
  else if (field === 'amount') currentCotis.contributions[idx].amount = parseFloat(val) || 0;
  else if (field === 'payment_status') currentCotis.contributions[idx].payment_status = val as any;
  updateCotisCalculationsDisplay();
};

(window as any).removeCotisItem = (idx: number) => {
  if (!currentCotis) return;
  currentCotis.contributions.splice(idx, 1);
  renderCotisTable();
  updateCotisCalculationsDisplay();
};

(window as any).updateChantierWork = (idx: number, val: string) => {
  if (currentChantier) currentChantier.work_done[idx] = val;
};
(window as any).removeChantierWork = (idx: number) => {
  if (currentChantier) {
    currentChantier.work_done.splice(idx, 1);
    renderChantierEditorForm();
  }
};

(window as any).updateChantierUsed = (idx: number, field: string, val: string) => {
  if (!currentChantier || !currentChantier.materials_used[idx]) return;
  if (field === 'material') currentChantier.materials_used[idx].material = val;
  else currentChantier.materials_used[idx].quantity = val;
};
(window as any).removeChantierUsed = (idx: number) => {
  if (currentChantier) {
    currentChantier.materials_used.splice(idx, 1);
    renderChantierEditorForm();
  }
};

(window as any).updateChantierNeeded = (idx: number, field: string, val: string) => {
  if (!currentChantier || !currentChantier.materials_needed[idx]) return;
  if (field === 'material') currentChantier.materials_needed[idx].material = val;
  else if (field === 'quantity') currentChantier.materials_needed[idx].quantity = val;
  else currentChantier.materials_needed[idx].deadline = val;
};
(window as any).removeChantierNeeded = (idx: number) => {
  if (currentChantier) {
    currentChantier.materials_needed.splice(idx, 1);
    renderChantierEditorForm();
  }
};

(window as any).removeChantierPhoto = (idx: number) => {
  if (currentChantier) {
    currentChantier.photos.splice(idx, 1);
    renderPhotosPreviewGrid();
  }
};

// Simple Toast Notification
function showToast(message: string, type: 'info' | 'success' | 'warning' | 'error' = 'info') {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const toast = document.createElement('div');
  let bg = 'bg-slate-800 text-white';
  if (type === 'success') bg = 'bg-emerald-600 text-white';
  else if (type === 'error') bg = 'bg-rose-600 text-white';
  else if (type === 'warning') bg = 'bg-amber-600 text-white';

  toast.className = `flex items-center gap-2 px-4 py-3 rounded-xl shadow-lg text-sm font-medium transition-all duration-300 transform translate-y-2 opacity-0 ${bg}`;
  toast.textContent = message;

  container.appendChild(toast);
  requestAnimationFrame(() => {
    toast.classList.remove('translate-y-2', 'opacity-0');
  });

  setTimeout(() => {
    toast.classList.add('opacity-0', 'translate-y-2');
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function formatMoney(amount: number): string {
  return (amount || 0).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ' ');
}

// Start application on DOM ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initApp);
} else {
  initApp();
}
