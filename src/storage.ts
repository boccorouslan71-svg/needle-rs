import type { FreemiumState, ModuleType, DevisData, CotisData, ChantierData } from './types';

const DB_NAME = 'diktao_offline_db_v1';
const DB_VERSION = 1;
const DOCS_STORE = 'documents';
const FREEMIUM_STORE = 'freemium';

const FREEMIUM_LOCAL_KEY = 'diktao_freemium_v1';
const HISTORY_LOCAL_KEY = 'diktao_history_v1';
const FREE_LIMIT_PER_MODULE = 5;

export interface SavedDoc {
  id: string;
  type: ModuleType;
  title: string;
  date: string;
  createdAt: number;
  data: DevisData | CotisData | ChantierData;
  thumbnail?: string;
}

let dbInstance: IDBDatabase | null = null;
let dbInitPromise: Promise<IDBDatabase> | null = null;

/**
 * Initialize and open IndexedDB with schema migrations
 */
export function openDiktaoDB(): Promise<IDBDatabase> {
  if (dbInstance) {
    return Promise.resolve(dbInstance);
  }
  if (dbInitPromise) {
    return dbInitPromise;
  }

  dbInitPromise = new Promise((resolve, reject) => {
    if (typeof window === 'undefined' || !window.indexedDB) {
      console.warn('IndexedDB is not available in this environment. Falling back to local storage.');
      reject(new Error('IndexedDB not supported'));
      return;
    }

    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;

      // Documents store
      if (!db.objectStoreNames.contains(DOCS_STORE)) {
        const docStore = db.createObjectStore(DOCS_STORE, { keyPath: 'id' });
        docStore.createIndex('type', 'type', { unique: false });
        docStore.createIndex('createdAt', 'createdAt', { unique: false });
      }

      // Freemium state store
      if (!db.objectStoreNames.contains(FREEMIUM_STORE)) {
        db.createObjectStore(FREEMIUM_STORE, { keyPath: 'key' });
      }
    };

    request.onsuccess = async (event) => {
      dbInstance = (event.target as IDBOpenDBRequest).result;

      // Handle connection closing
      dbInstance.onclose = () => {
        dbInstance = null;
        dbInitPromise = null;
      };

      // Perform initial migration from localStorage if needed
      await migrateLocalStorageToIDB(dbInstance);

      resolve(dbInstance);
    };

    request.onerror = (event) => {
      console.error('Failed to open IndexedDB:', (event.target as IDBOpenDBRequest).error);
      reject((event.target as IDBOpenDBRequest).error);
    };
  });

  return dbInitPromise;
}

/**
 * Seamless migration of legacy localStorage documents into IndexedDB
 */
async function migrateLocalStorageToIDB(db: IDBDatabase): Promise<void> {
  try {
    const rawHistory = localStorage.getItem(HISTORY_LOCAL_KEY);
    if (rawHistory) {
      const docs: any[] = JSON.parse(rawHistory);
      if (Array.isArray(docs) && docs.length > 0) {
        const tx = db.transaction(DOCS_STORE, 'readwrite');
        const store = tx.objectStore(DOCS_STORE);
        for (const doc of docs) {
          const item: SavedDoc = {
            id: doc.id || ('doc_' + Date.now() + Math.random().toString(36).slice(2, 6)),
            type: doc.type || 'devis',
            title: doc.title || 'Document Diktao',
            date: doc.date || new Date().toLocaleDateString('fr-FR'),
            createdAt: doc.createdAt || Date.now(),
            data: doc.data,
            thumbnail: doc.thumbnail,
          };
          store.put(item);
        }
        await new Promise((res, rej) => {
          tx.oncomplete = res;
          tx.onerror = rej;
        });
        // Clear old key to prevent duplicate migrations
        localStorage.removeItem(HISTORY_LOCAL_KEY);
        console.log(`Successfully migrated ${docs.length} documents from localStorage to IndexedDB.`);
      }
    }
  } catch (err) {
    console.warn('Migration from localStorage to IndexedDB skipped or failed', err);
  }
}

// -------------------------------------------------------------
// FREEMIUM QUOTA MANAGEMENT (IndexedDB + In-Memory/LocalStorage sync)
// -------------------------------------------------------------

function getCurrentMonth(): string {
  const now = new Date();
  const year = now.getFullYear();
  const month = String(now.getMonth() + 1).padStart(2, '0');
  return `${year}-${month}`;
}

export function getFreemiumState(): FreemiumState {
  const currentMonth = getCurrentMonth();
  try {
    const raw = localStorage.getItem(FREEMIUM_LOCAL_KEY);
    if (raw) {
      const parsed: FreemiumState = JSON.parse(raw);
      if (parsed.month === currentMonth && parsed.counts) {
        return parsed;
      }
    }
  } catch (e) {
    console.warn('Failed to parse freemium state from storage', e);
  }

  // New month or first run: initialize with 5 free documents per module
  const newState: FreemiumState = {
    month: currentMonth,
    counts: {
      devis: FREE_LIMIT_PER_MODULE,
      cotis: FREE_LIMIT_PER_MODULE,
      chantier: FREE_LIMIT_PER_MODULE,
    },
  };
  saveFreemiumState(newState);
  return newState;
}

export function saveFreemiumState(state: FreemiumState): void {
  try {
    localStorage.setItem(FREEMIUM_LOCAL_KEY, JSON.stringify(state));
  } catch (e) {
    console.warn('Failed to save freemium state to localStorage', e);
  }

  // Also asynchronously persist into IndexedDB
  openDiktaoDB().then((db) => {
    try {
      const tx = db.transaction(FREEMIUM_STORE, 'readwrite');
      tx.objectStore(FREEMIUM_STORE).put({ key: 'monthly_state', ...state });
    } catch (err) {
      console.warn('IndexedDB freemium save notice', err);
    }
  }).catch(() => {});
}

export function getRemainingQuota(module: ModuleType): number {
  const state = getFreemiumState();
  return Math.max(0, state.counts[module] ?? 0);
}

export function hasRemainingQuota(module: ModuleType): boolean {
  return getRemainingQuota(module) > 0;
}

export function consumeQuota(module: ModuleType): boolean {
  const state = getFreemiumState();
  if ((state.counts[module] ?? 0) > 0) {
    state.counts[module] -= 1;
    saveFreemiumState(state);
    return true;
  }
  return false;
}

export function resetQuotasForTesting(): void {
  const newState: FreemiumState = {
    month: getCurrentMonth(),
    counts: {
      devis: FREE_LIMIT_PER_MODULE,
      cotis: FREE_LIMIT_PER_MODULE,
      chantier: FREE_LIMIT_PER_MODULE,
    },
  };
  saveFreemiumState(newState);
}

// -------------------------------------------------------------
// INDEXEDDB DOCUMENT STORAGE (Full Offline Persistence)
// -------------------------------------------------------------

/**
 * Save a document asynchronously to IndexedDB (with localStorage backup)
 */
export async function saveDocument(doc: SavedDoc): Promise<void> {
  const preparedDoc: SavedDoc = {
    ...doc,
    createdAt: doc.createdAt || Date.now(),
  };

  try {
    const db = await openDiktaoDB();
    const tx = db.transaction(DOCS_STORE, 'readwrite');
    const store = tx.objectStore(DOCS_STORE);
    store.put(preparedDoc);

    await new Promise<void>((resolve, reject) => {
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  } catch (err) {
    console.warn('IndexedDB save failed, saving to localStorage backup', err);
    // Fallback to localStorage
    try {
      const raw = localStorage.getItem(HISTORY_LOCAL_KEY);
      const list: SavedDoc[] = raw ? JSON.parse(raw) : [];
      const idx = list.findIndex(d => d.id === preparedDoc.id);
      if (idx >= 0) list[idx] = preparedDoc;
      else list.unshift(preparedDoc);
      localStorage.setItem(HISTORY_LOCAL_KEY, JSON.stringify(list.slice(0, 30)));
    } catch {}
  }
}

/**
 * Retrieve all saved documents sorted by most recent first
 */
export async function getSavedDocuments(): Promise<SavedDoc[]> {
  try {
    const db = await openDiktaoDB();
    const tx = db.transaction(DOCS_STORE, 'readonly');
    const store = tx.objectStore(DOCS_STORE);

    return await new Promise<SavedDoc[]>((resolve, reject) => {
      const request = store.getAll();
      request.onsuccess = () => {
        const results = (request.result as SavedDoc[]) || [];
        // Sort newest first
        results.sort((a, b) => (b.createdAt || 0) - (a.createdAt || 0));
        resolve(results);
      };
      request.onerror = () => reject(request.error);
    });
  } catch (err) {
    console.warn('IndexedDB getSavedDocuments failed, reading from localStorage', err);
    try {
      const raw = localStorage.getItem(HISTORY_LOCAL_KEY);
      return raw ? JSON.parse(raw) : [];
    } catch {
      return [];
    }
  }
}

/**
 * Retrieve a specific document by its ID
 */
export async function getSavedDocumentById(id: string): Promise<SavedDoc | undefined> {
  try {
    const db = await openDiktaoDB();
    const tx = db.transaction(DOCS_STORE, 'readonly');
    const store = tx.objectStore(DOCS_STORE);

    return await new Promise<SavedDoc | undefined>((resolve, reject) => {
      const request = store.get(id);
      request.onsuccess = () => resolve(request.result as SavedDoc | undefined);
      request.onerror = () => reject(request.error);
    });
  } catch (err) {
    const all = await getSavedDocuments();
    return all.find(d => d.id === id);
  }
}

/**
 * Delete a document by ID from IndexedDB
 */
export async function deleteSavedDocument(id: string): Promise<void> {
  try {
    const db = await openDiktaoDB();
    const tx = db.transaction(DOCS_STORE, 'readwrite');
    const store = tx.objectStore(DOCS_STORE);
    store.delete(id);

    await new Promise<void>((resolve, reject) => {
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  } catch (err) {
    console.warn('IndexedDB delete failed, cleaning localStorage', err);
    try {
      const raw = localStorage.getItem(HISTORY_LOCAL_KEY);
      if (raw) {
        const list: SavedDoc[] = JSON.parse(raw);
        const filtered = list.filter(d => d.id !== id);
        localStorage.setItem(HISTORY_LOCAL_KEY, JSON.stringify(filtered));
      }
    } catch {}
  }
}

/**
 * Export all local data to a JSON backup string for user migration/safety
 */
export async function exportLocalDataBackup(): Promise<string> {
  const docs = await getSavedDocuments();
  const freemium = getFreemiumState();
  return JSON.stringify({
    version: '1.0',
    exportedAt: new Date().toISOString(),
    freemium,
    documents: docs,
  }, null, 2);
}

// -------------------------------------------------------------
// DRAFT PERSISTENCE (Safe Auto-save preventing data loss on mobile)
// -------------------------------------------------------------
const DRAFT_LOCAL_KEY = 'diktao_active_draft_v1';

export interface ActiveDraft {
  module: ModuleType;
  data: DevisData | CotisData | ChantierData;
  transcript?: string;
  updatedAt: number;
}

export function saveActiveDraft(module: ModuleType, data: any, transcript = ''): void {
  try {
    const draft: ActiveDraft = {
      module,
      data,
      transcript,
      updatedAt: Date.now(),
    };
    localStorage.setItem(DRAFT_LOCAL_KEY, JSON.stringify(draft));
  } catch (e) {
    console.warn('Failed to save active draft:', e);
  }
}

export function getActiveDraft(): ActiveDraft | null {
  try {
    const raw = localStorage.getItem(DRAFT_LOCAL_KEY);
    if (!raw) return null;
    const parsed: ActiveDraft = JSON.parse(raw);
    // Ignore stale drafts older than 7 days
    if (Date.now() - parsed.updatedAt > 7 * 24 * 3600 * 1000) {
      clearActiveDraft();
      return null;
    }
    return parsed;
  } catch {
    return null;
  }
}

export function clearActiveDraft(): void {
  try {
    localStorage.removeItem(DRAFT_LOCAL_KEY);
  } catch {}
}
