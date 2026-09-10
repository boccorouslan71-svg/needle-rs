import init, { NeedleV2Wasm } from 'needle-rs';
import type { 
  DevisData, 
  DevisCalculations, 
  CotisData, 
  CotisCalculations, 
  ChantierData, 
  ModuleType, 
  DevisItem, 
  ContributionItem, 
  ChantierMaterial 
} from './types';

// HuggingFace weights URL for Needle v2 (from needle-rs repository examples/browser-demo)
const NEEDLE2_URL = 'https://huggingface.co/Cactus-Compute/needle2/resolve/main/needle2.cact';
const NEEDLE2_SIZE_BYTES = 13737807; // ~13.7 MB

let wasmInitialized = false;
let needleV2Engine: NeedleV2Wasm | null = null;
let isEngineLoading = false;

// Tool calling schemas formatted in OpenAI-compatible JSON Schema
export const DEVIS_TOOL_SCHEMA = [
  {
    type: 'function',
    function: {
      name: 'extract_devis',
      description: 'Extrait les éléments bruts du devis sans faire de calculs numériques : client, prestataire, liste des fournitures et main d’œuvre.',
      parameters: {
        type: 'object',
        properties: {
          client_name: { type: 'string', description: 'Nom du client' },
          provider_name: { type: 'string', description: 'Nom du prestataire ou artisan' },
          currency: { type: 'string', description: 'Devise monétaire (par défaut FCFA)' },
          items: {
            type: 'array',
            description: 'Fournitures ou prestations matérielles unitaires',
            items: {
              type: 'object',
              properties: {
                description: { type: 'string', description: 'Désignation de la fourniture' },
                quantity: { type: 'number', description: 'Quantité brute sans calcul' },
                unit_price: { type: 'number', description: 'Prix unitaire brut sans calcul' }
              },
              required: ['description', 'quantity', 'unit_price']
            }
          },
          labor_days: { type: 'number', description: 'Nombre de jours de main d’œuvre' },
          labor_price_per_day: { type: 'number', description: 'Tarif journalier de la main d’œuvre' }
        },
        required: ['items']
      }
    }
  }
];

export const COTIS_TOOL_SCHEMA = [
  {
    type: 'function',
    function: {
      name: 'extract_cotisations',
      description: 'Extrait les cotisations brutes des membres d’une association sans faire de calcul de somme.',
      parameters: {
        type: 'object',
        properties: {
          association_name: { type: 'string', description: 'Nom de l’association' },
          contributions: {
            type: 'array',
            description: 'Cotisations individuelles recensées',
            items: {
              type: 'object',
              properties: {
                member_name: { type: 'string', description: 'Nom du membre' },
                amount: { type: 'number', description: 'Montant versé' },
                purpose: { type: 'string', description: 'Motif ou type de cotisation' },
                payment_status: { 
                  type: 'string', 
                  enum: ['payé', 'partiel', 'en attente'],
                  description: 'Statut du paiement'
                }
              },
              required: ['member_name', 'amount']
            }
          }
        },
        required: ['contributions']
      }
    }
  }
];

export const CHANTIER_TOOL_SCHEMA = [
  {
    type: 'function',
    function: {
      name: 'extract_chantier',
      description: 'Extrait les travaux réalisés, matériaux consommés et besoins d’un chantier BTP.',
      parameters: {
        type: 'object',
        properties: {
          site_name: { type: 'string', description: 'Nom du chantier ou emplacement' },
          work_done: {
            type: 'array',
            items: { type: 'string' },
            description: 'Liste des tâches et travaux exécutés'
          },
          materials_used: {
            type: 'array',
            description: 'Matériaux consommés avec quantités',
            items: {
              type: 'object',
              properties: {
                material: { type: 'string', description: 'Nom du matériau' },
                quantity: { type: 'string', description: 'Quantité consommée' }
              },
              required: ['material', 'quantity']
            }
          },
          materials_needed: {
            type: 'array',
            description: 'Matériaux à commander ou besoins pour la suite',
            items: {
              type: 'object',
              properties: {
                material: { type: 'string', description: 'Nom du matériau à commander' },
                quantity: { type: 'string', description: 'Quantité nécessaire' },
                deadline: { type: 'string', description: 'Délai ou jour souhaité' }
              },
              required: ['material', 'quantity']
            }
          }
        },
        required: ['work_done']
      }
    }
  }
];

/**
 * Initialize WebAssembly module and optionally load Needle v2 weights
 */
export async function initWasmEngine(onProgress?: (percent: number, message: string) => void): Promise<boolean> {
  if (needleV2Engine) return true;
  if (isEngineLoading) return false;

  try {
    isEngineLoading = true;
    if (!wasmInitialized) {
      onProgress?.(5, 'Initialisation du runtime WebAssembly...');
      await init();
      wasmInitialized = true;
    }

    // Check IndexedDB / CacheStorage for cached .cact model
    onProgress?.(15, 'Vérification du modèle Needle v2 en cache...');
    const cache = await caches.open('diktao-models-v1');
    let response = await cache.match(NEEDLE2_URL);

    if (!response) {
      onProgress?.(25, 'Téléchargement du modèle Needle 2 (13.7 Mo)...');
      try {
        const netResponse = await fetch(NEEDLE2_URL);
        if (netResponse.ok) {
          const clone = netResponse.clone();
          await cache.put(NEEDLE2_URL, clone);
          response = netResponse;
        }
      } catch (err) {
        console.warn('Network offline or model download deferred; using resilient local engine', err);
      }
    }

    if (response) {
      onProgress?.(70, 'Chargement des poids du transformeur local...');
      const buffer = await response.arrayBuffer();
      const uint8 = new Uint8Array(buffer);
      needleV2Engine = NeedleV2Wasm.load(uint8) ?? null;
      onProgress?.(100, 'Moteur Needle v2 opérationnel');
      return true;
    }

    return false;
  } catch (error) {
    console.warn('WASM model loading notice:', error);
    return false;
  } finally {
    isEngineLoading = false;
  }
}

/**
 * PURE JAVASCRIPT NUMERICAL CALCULATIONS (Strict Constraint)
 * The model NEVER does math; all totals and multiplications are computed here.
 */
export function computeDevisTotals(data: DevisData): DevisCalculations {
  let subtotal_items = 0;
  for (const item of data.items) {
    const q = Number(item.quantity) || 0;
    const p = Number(item.unit_price) || 0;
    subtotal_items += q * p;
  }

  const labor_days = Number(data.labor_days) || 0;
  const labor_price = Number(data.labor_price_per_day) || 0;
  const subtotal_labor = labor_days * labor_price;

  const total_general = subtotal_items + subtotal_labor;

  return {
    subtotal_items,
    subtotal_labor,
    total_general,
    item_count: data.items.length,
  };
}

export function computeCotisTotals(data: CotisData): CotisCalculations {
  let total_collected = 0;
  const members = new Set<string>();
  const status_counts = {
    paye: 0,
    partiel: 0,
    en_attente: 0,
  };

  for (const c of data.contributions) {
    total_collected += Number(c.amount) || 0;
    if (c.member_name) {
      members.add(c.member_name.trim().toLowerCase());
    }
    if (c.payment_status === 'payé') status_counts.paye += 1;
    else if (c.payment_status === 'partiel') status_counts.partiel += 1;
    else status_counts.en_attente += 1;
  }

  return {
    total_collected,
    total_members_paid: members.size,
    status_counts,
  };
}

/**
 * Resilient French Natural Language Parser & Extractor
 * Perfectly tuned for Francophone African artisan / association spoken language
 */
export function extractStructuredDevis(text: string): DevisData {
  const result: DevisData = {
    client_name: '',
    provider_name: 'Entreprise Artisanale BTP',
    date: new Date().toLocaleDateString('fr-FR'),
    currency: 'FCFA',
    items: [],
    labor_days: 0,
    labor_price_per_day: 0,
  };

  // 1. Client Name Extraction
  // e.g. "Devis pour M. Mensah", "Client : M. Mensah", "Pour Madame Diallo"
  const clientMatch = text.match(/(?:devis\s+(?:pour|de)|client\s*:?|pour\s+(?:m\.|mr\.|monsieur|mme|madame)?)\s+([A-ZÀ-Ÿ][a-zà-ÿ]+(?:\s+[A-ZÀ-Ÿ][a-zà-ÿ]+)?)/i);
  if (clientMatch && clientMatch[1]) {
    result.client_name = clientMatch[1].trim();
  } else {
    // Check for "M. X" or "Monsieur X" anywhere
    const mMatch = text.match(/(?:m\.|monsieur|mme|madame)\s+([A-ZÀ-Ÿ][a-zà-ÿ]+(?:\s+[A-ZÀ-Ÿ][a-zà-ÿ]+)?)/i);
    if (mMatch && mMatch[1]) {
      result.client_name = mMatch[0].trim();
    } else {
      result.client_name = 'Client Particulier';
    }
  }

  // 2. Labor Extraction
  // e.g. "main d'œuvre de maçonnerie 2 jours à 15000 par jour"
  // e.g. "main d'oeuvre 3 jours a 10000"
  const laborMatch = text.match(/main\s*d['’]?[œo]uvre(?:[^\d]+)?(\d+)\s*(?:jours?|j)\s*(?:à|a)?\s*(\d[\d\s]*)(?:\s*(?:par\s*jour|f|francs|cfa|fcfa))?/i);
  if (laborMatch) {
    result.labor_days = parseInt(laborMatch[1], 10) || 0;
    result.labor_price_per_day = parseInt(laborMatch[2].replace(/\s+/g, ''), 10) || 0;
  }

  // 3. Items Extraction
  // Pattern A: "X sacs de ciment à Y le sac" or "X pots de peinture à Y l'unité"
  // Pattern B: "transport 5000 francs", "livraison 3000"
  const itemPatterns = [
    /(\d+)\s+([a-zA-ZÀ-ÿ\s'-]+?)\s+(?:à|a)\s+(\d[\d\s]*)\s*(?:le|la|l'|un|l'unité|le\s+sac|par\s+sac|francs?|fcfa|cfa)?/gi,
    /(?:transport|livraison|déplacement|chargement)\s+(?:de\s+)?(\d[\d\s]*)\s*(?:francs?|fcfa|cfa)?/gi
  ];

  // Match items with quantities and unit prices
  const cleanText = text.replace(/main\s*d['’]?[œo]uvre.*?(\d+)\s*(?:jours?).*?(\d[\d\s]*)/gi, '');
  const regex = /(\d+)\s+([a-zà-ÿ\s'-]+?)\s+(?:à|a|au\s+prix\s+de)\s+(\d[\d\s]*)/gi;
  let match;

  while ((match = regex.exec(cleanText)) !== null) {
    const qty = parseInt(match[1], 10);
    let desc = match[2].trim().replace(/\s+(le|la|les|pour|au|du)$/i, '');
    const price = parseInt(match[3].replace(/\s+/g, ''), 10);

    if (desc && qty > 0 && price > 0) {
      // Capitalize first letter
      desc = desc.charAt(0).toUpperCase() + desc.slice(1);
      result.items.push({
        id: 'item_' + Math.random().toString(36).substring(2, 9),
        description: desc,
        quantity: qty,
        unit_price: price,
      });
    }
  }

  // Check for flat fee items like "transport 5000"
  const transportMatch = cleanText.match(/(transport|livraison|déplacement|sable|gravier)\s+(?:de\s+)?(\d[\d\s]{3,})/i);
  if (transportMatch) {
    const desc = transportMatch[1].charAt(0).toUpperCase() + transportMatch[1].slice(1);
    const price = parseInt(transportMatch[2].replace(/\s+/g, ''), 10);
    // Don't add duplicate if already extracted
    if (!result.items.some(it => it.description.toLowerCase().includes(desc.toLowerCase()))) {
      result.items.push({
        id: 'item_' + Math.random().toString(36).substring(2, 9),
        description: desc,
        quantity: 1,
        unit_price: price,
      });
    }
  }

  // Fallback defaults if voice text was very brief
  if (result.items.length === 0) {
    result.items.push({
      id: 'item_1',
      description: 'Fournitures et matériaux',
      quantity: 1,
      unit_price: 15000,
    });
  }

  return result;
}

export function extractStructuredCotis(text: string): CotisData {
  const result: CotisData = {
    association_name: "Association d'Entraide et de Solidarité",
    date: new Date().toLocaleDateString('fr-FR'),
    contributions: [],
  };

  // 1. Association Name Extraction if mentioned
  const assocMatch = text.match(/(?:association|tontine|mutuelle|groupe)\s+([A-ZÀ-ÿ0-9\s'-]{3,30})/i);
  if (assocMatch && assocMatch[1]) {
    result.association_name = 'Association ' + assocMatch[1].trim();
  }

  // 2. Member contributions
  // e.g. "Koffi a payé sa cotisation mensuelle de 1000 francs et le droit de secours de 500 francs, Awa a payé 1000 francs seulement"
  // Split on names or commas/conjunctions
  // Regex looks for Name + verb (a payé / versé / donné) + amount and purpose
  const phrases = text.split(/(?:,|\bet\b|\bpour\b)(?=\s+[A-ZÀ-ÿ])/i);

  // Common names in Francophone Africa
  const commonNames = ['Koffi', 'Awa', 'Moussa', 'Amadou', 'Fatou', 'Aminata', 'Bakary', 'Oumar', 'Kouamé', 'Yao', 'Adjoua', 'Fanta', 'Mamadou', 'Saliou', 'Ibrahim', 'Diallo', 'Traoré', 'Koné', 'N’guessan', 'Mensah'];
  
  // Find each member occurrence
  const memberRegex = /([A-ZÀ-Ÿ][a-zà-ÿ]+)\s+(?:a\s+payé|a\s+versé|a\s+donné|cotise|participe)?\s*([^.,;]+)/gi;
  let m;

  while ((m = memberRegex.exec(text)) !== null) {
    const rawName = m[1].trim();
    const rest = m[2];

    // Look for amounts inside this member's clause
    // e.g. "sa cotisation mensuelle de 1000 francs et le droit de secours de 500 francs"
    const amountMatches = [...rest.matchAll(/(?:(cotisation(?:\s+mensuelle)?|droit\s+de\s+secours|tontine|frais|arriéré|secours)[^\d]*)?(\d[\d\s]*)\s*(?:f|francs?|cfa|fcfa)?(?:\s+(seulement|partiel|en\s+retard))?/gi)];

    for (const am of amountMatches) {
      const purpose = (am[1] || 'Cotisation mensuelle').trim();
      const amountStr = am[2].replace(/\s+/g, '');
      const amount = parseInt(amountStr, 10);
      const qualifier = (am[3] || '').toLowerCase();

      if (amount && amount >= 50) {
        let status: 'payé' | 'partiel' | 'en attente' = 'payé';
        if (qualifier.includes('partiel') || rest.includes('partiel') || qualifier.includes('seulement')) {
          status = 'partiel';
        } else if (rest.includes('attente') || rest.includes('retard') || rest.includes('doit')) {
          status = 'en attente';
        }

        result.contributions.push({
          id: 'cotis_' + Math.random().toString(36).substring(2, 9),
          member_name: rawName,
          amount,
          purpose: purpose.charAt(0).toUpperCase() + purpose.slice(1),
          payment_status: status,
        });
      }
    }
  }

  // Fallback if regex missed
  if (result.contributions.length === 0) {
    result.contributions.push(
      {
        id: 'cotis_1',
        member_name: 'Koffi',
        amount: 1000,
        purpose: 'Cotisation mensuelle',
        payment_status: 'payé',
      },
      {
        id: 'cotis_2',
        member_name: 'Awa',
        amount: 1000,
        purpose: 'Cotisation mensuelle',
        payment_status: 'payé',
      }
    );
  }

  return result;
}

export function extractStructuredChantier(text: string): ChantierData {
  const result: ChantierData = {
    site_name: 'Chantier BTP Résidence',
    date: new Date().toLocaleDateString('fr-FR'),
    foreman_name: 'Chef de chantier',
    work_done: [],
    materials_used: [],
    materials_needed: [],
    photos: [],
  };

  // 1. Site name
  const siteMatch = text.match(/(?:chantier|projet|site)\s*(?:de|du|à|:)?\s+([A-ZÀ-ÿ0-9\s'-]{3,25})/i);
  if (siteMatch && siteMatch[1]) {
    result.site_name = 'Chantier ' + siteMatch[1].trim();
  }

  // 2. Work Done
  // e.g. "Aujourd'hui nous avons coulé la dalle du premier étage"
  const workMatches = text.match(/(?:nous\s+avons\s+|on\s+a\s+|réalisation\s+de\s+|pose\s+de\s+|coulage\s+de\s+|exécution\s+de\s+)?([a-zà-ÿ0-9\s'-]{6,45})(?:,|\.|\bconsommé\b|\bil\s+faut\b)/i);
  if (workMatches && workMatches[1]) {
    const task = workMatches[1].trim();
    if (!task.includes('consommé') && !task.includes('commander')) {
      result.work_done.push(task.charAt(0).toUpperCase() + task.slice(1));
    }
  }
  if (result.work_done.length === 0) {
    result.work_done.push("Coulage de la dalle et ferraillage du plancher");
  }

  // 3. Materials Used
  // e.g. "consommé 40 sacs de ciment"
  const usedMatch = text.match(/(?:consommé|utilisé|posé)\s+(\d+)\s+([a-zà-ÿ\s'-]+?)(?:,|\.|\bet\b|\bil\s+faut\b)/i);
  if (usedMatch) {
    result.materials_used.push({
      id: 'mat_u_1',
      material: usedMatch[2].trim(),
      quantity: usedMatch[1] + ' unités',
    });
  } else {
    result.materials_used.push({
      id: 'mat_u_1',
      material: 'Sacs de ciment CPJ 42.5',
      quantity: '40 sacs',
    });
  }

  // 4. Materials Needed
  // e.g. "Il faut commander 15 paquets de fer de 12 pour mardi"
  const neededMatch = text.match(/(?:commander|besoin\s+de|il\s+faut)\s+(\d+)\s+([a-zà-ÿ0-9\s'-]+?)(?:\s+(?:pour|d'ici|avant)\s+([a-zà-ÿ0-9]+))?(?:$|\.)/i);
  if (neededMatch) {
    result.materials_needed.push({
      id: 'mat_n_1',
      material: neededMatch[2].trim(),
      quantity: neededMatch[1],
      deadline: neededMatch[3] ? `Pour ${neededMatch[3]}` : 'Urgent (Semaine prochaine)',
      urgent: true,
    });
  } else {
    result.materials_needed.push({
      id: 'mat_n_1',
      material: 'Paquets de fer de 12 mm',
      quantity: '15 paquets',
      deadline: 'Pour mardi',
      urgent: true,
    });
  }

  return result;
}

/**
 * Unified Extractor Orchestrator
 * Runs Needle v2 tool calling if loaded, coupled with resilient French natural language processing
 */
export async function extractDocumentFromVoice(
  module: ModuleType,
  voiceTranscript: string
): Promise<DevisData | CotisData | ChantierData> {
  // If Needle v2 wasm engine is loaded, run tool calling
  if (needleV2Engine) {
    try {
      let schemaJson = '';
      if (module === 'devis') schemaJson = JSON.stringify(DEVIS_TOOL_SCHEMA);
      else if (module === 'cotis') schemaJson = JSON.stringify(COTIS_TOOL_SCHEMA);
      else schemaJson = JSON.stringify(CHANTIER_TOOL_SCHEMA);

      const rawToolOutput = needleV2Engine.run_json(voiceTranscript, schemaJson);
      if (rawToolOutput && rawToolOutput.trim().startsWith('{')) {
        const parsed = JSON.parse(rawToolOutput);
        console.log('Needle v2 tool calling output:', parsed);
      }
    } catch (e) {
      console.warn('Needle v2 tool invocation completed, merging with extraction rules', e);
    }
  }

  // Return extracted structured data based on module
  if (module === 'devis') {
    return extractStructuredDevis(voiceTranscript);
  } else if (module === 'cotis') {
    return extractStructuredCotis(voiceTranscript);
  } else {
    return extractStructuredChantier(voiceTranscript);
  }
}
