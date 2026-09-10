// Diktao Types Definition

export type ModuleType = 'devis' | 'cotis' | 'chantier';

export interface DevisItem {
  id: string;
  description: string;
  quantity: number;
  unit_price: number;
}

export interface DevisData {
  client_name: string;
  provider_name: string;
  date: string;
  currency: string;
  items: DevisItem[];
  labor_days?: number;
  labor_price_per_day?: number;
}

export interface DevisCalculations {
  subtotal_items: number;
  subtotal_labor: number;
  total_general: number;
  item_count: number;
}

export interface ContributionItem {
  id: string;
  member_name: string;
  amount: number;
  purpose?: string;
  payment_status: 'payé' | 'partiel' | 'en attente';
}

export interface CotisData {
  association_name: string;
  date: string;
  contributions: ContributionItem[];
}

export interface CotisCalculations {
  total_collected: number;
  total_members_paid: number;
  status_counts: {
    paye: number;
    partiel: number;
    en_attente: number;
  };
}

export interface ChantierMaterial {
  id: string;
  material: string;
  quantity: string;
  deadline?: string;
  urgent?: boolean;
}

export interface ChantierData {
  site_name: string;
  date: string;
  foreman_name?: string;
  work_done: string[];
  materials_used: ChantierMaterial[];
  materials_needed: ChantierMaterial[];
  photos: string[]; // Base64 data URLs
}

export interface FreemiumState {
  month: string; // 'YYYY-MM'
  counts: {
    devis: number;
    cotis: number;
    chantier: number;
  };
}

export type ActiveScreen = 
  | 'splash'
  | 'home'
  | 'voice'
  | 'editor'
  | 'share'
  | 'limit'
  | 'about'
  | 'history';
