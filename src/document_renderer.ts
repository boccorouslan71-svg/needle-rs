import type { DevisData, CotisData, ChantierData, ModuleType } from './types';
import { computeDevisTotals, computeCotisTotals } from './ai_engine';

const CANVAS_WIDTH = 1200;
const CANVAS_HEIGHT = 1700;

// Helper to draw rounded rectangle on 2D context
function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
  fill = true,
  stroke = false
) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
  if (fill) ctx.fill();
  if (stroke) ctx.stroke();
}

// Format numbers with thousands spaces (e.g. 15 000)
function formatMoney(amount: number): string {
  return amount.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ' ');
}

// Draw the watermark footer with Diktao logo
function drawFooterWatermark(ctx: CanvasRenderingContext2D, yPos: number = 1630) {
  ctx.save();
  ctx.strokeStyle = '#E2E8F0';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(80, yPos - 30);
  ctx.lineTo(1120, yPos - 30);
  ctx.stroke();

  // "Généré via" text
  ctx.fillStyle = '#64748B';
  ctx.font = '500 20px system-ui, sans-serif';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  ctx.fillText('Généré via', 80, yPos);

  // Diktao Brand mark
  // Small D shape + megaphone + "Diktao"
  const startX = 200;
  // Deep navy D box
  ctx.fillStyle = '#0D204A';
  roundRect(ctx, startX, yPos - 18, 36, 36, 8, true, false);

  // Orange megaphone triangle
  ctx.fillStyle = '#FF7A00';
  ctx.beginPath();
  ctx.moveTo(startX + 14, yPos);
  ctx.lineTo(startX + 28, yPos - 10);
  ctx.lineTo(startX + 28, yPos + 10);
  ctx.closePath();
  ctx.fill();

  // Wordmark
  ctx.fillStyle = '#0D204A';
  ctx.font = 'bold 24px system-ui, sans-serif';
  ctx.fillText('Diktao', startX + 46, yPos);

  // Security badge on the right
  ctx.fillStyle = '#94A3B8';
  ctx.font = '400 18px system-ui, sans-serif';
  ctx.textAlign = 'right';
  ctx.fillText('Document certifié 100% hors-ligne · Diktao.app', 1120, yPos);

  ctx.restore();
}

/**
 * Render Module 1: Devis-Éclair
 */
export function renderDevisToCanvas(canvas: HTMLCanvasElement, data: DevisData): void {
  canvas.width = CANVAS_WIDTH;
  canvas.height = CANVAS_HEIGHT;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const totals = computeDevisTotals(data);

  // 1. Background
  ctx.fillStyle = '#FFFFFF';
  ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

  // Top decorative brand accent bar (Coral & Navy)
  ctx.fillStyle = '#FF6B4A';
  ctx.fillRect(0, 0, CANVAS_WIDTH, 12);
  ctx.fillStyle = '#1A365D';
  ctx.fillRect(0, 12, CANVAS_WIDTH, 6);

  // 2. Header Area
  ctx.fillStyle = '#1A365D';
  ctx.font = 'bold 44px system-ui, sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText('DEVIS ESTIMATIF', 80, 85);

  // Reference & Date Badge
  ctx.fillStyle = '#F1F5F9';
  roundRect(ctx, 800, 45, 320, 60, 12, true, false);
  ctx.fillStyle = '#475569';
  ctx.font = '500 20px system-ui, sans-serif';
  ctx.textAlign = 'right';
  ctx.fillText(`Réf : DEV-${new Date().getFullYear()}-001`, 1100, 72);
  ctx.fillText(`Date : ${data.date || new Date().toLocaleDateString('fr-FR')}`, 1100, 95);

  // Subtitle / Legal
  ctx.fillStyle = '#64748B';
  ctx.font = '500 20px system-ui, sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText('Prestations de services & Fournitures de matériaux', 80, 120);

  // 3. Provider & Client Information Cards
  // Provider Box (Left)
  ctx.fillStyle = '#F8FAFC';
  ctx.strokeStyle = '#E2E8F0';
  ctx.lineWidth = 2;
  roundRect(ctx, 80, 155, 490, 150, 16, true, true);

  ctx.fillStyle = '#FF6B4A';
  ctx.font = 'bold 16px system-ui, sans-serif';
  ctx.fillText('ÉMIS PAR (PRESTATAIRE / ARTISAN)', 105, 190);

  ctx.fillStyle = '#0F172A';
  ctx.font = 'bold 26px system-ui, sans-serif';
  ctx.fillText(data.provider_name || 'Entreprise Artisanale BTP', 105, 230);

  ctx.fillStyle = '#475569';
  ctx.font = '500 20px system-ui, sans-serif';
  ctx.fillText('Artisanat & Bâtiment Professionnel', 105, 265);

  // Client Box (Right)
  ctx.fillStyle = '#EFF6FF';
  ctx.strokeStyle = '#BFDBFE';
  roundRect(ctx, 630, 155, 490, 150, 16, true, true);

  ctx.fillStyle = '#1D4ED8';
  ctx.font = 'bold 16px system-ui, sans-serif';
  ctx.fillText('DESTINATAIRE (CLIENT)', 655, 190);

  ctx.fillStyle = '#0F172A';
  ctx.font = 'bold 28px system-ui, sans-serif';
  ctx.fillText(data.client_name || 'M. Mensah', 655, 230);

  ctx.fillStyle = '#475569';
  ctx.font = '500 20px system-ui, sans-serif';
  ctx.fillText('Devis sur commande verbale', 655, 265);

  // 4. Materials & Supplies Table Header
  const tableY = 345;
  ctx.fillStyle = '#1A365D';
  roundRect(ctx, 80, tableY, 1040, 52, 10, true, false);

  ctx.fillStyle = '#FFFFFF';
  ctx.font = 'bold 20px system-ui, sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText('N°', 105, tableY + 33);
  ctx.fillText('DÉSIGNATION DES MATÉRIAUX / FOURNITURES', 160, tableY + 33);
  ctx.textAlign = 'center';
  ctx.fillText('QTÉ', 690, tableY + 33);
  ctx.textAlign = 'right';
  ctx.fillText(`P. UNIT (${data.currency})`, 890, tableY + 33);
  ctx.fillText(`TOTAL (${data.currency})`, 1090, tableY + 33);

  // 5. Table Rows
  let currentY = tableY + 52;
  const items = data.items.length > 0 ? data.items : [
    { id: '1', description: 'Fournitures de chantier', quantity: 1, unit_price: 15000 }
  ];

  items.forEach((item, index) => {
    // Alternating zebra row
    ctx.fillStyle = index % 2 === 0 ? '#FFFFFF' : '#F8FAFC';
    ctx.fillRect(80, currentY, 1040, 54);

    ctx.strokeStyle = '#E2E8F0';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(80, currentY + 54);
    ctx.lineTo(1120, currentY + 54);
    ctx.stroke();

    // Row text
    ctx.fillStyle = '#64748B';
    ctx.font = '500 20px system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`${index + 1}`, 105, currentY + 34);

    ctx.fillStyle = '#0F172A';
    ctx.font = 'bold 21px system-ui, sans-serif';
    ctx.fillText(item.description, 160, currentY + 34);

    ctx.fillStyle = '#334155';
    ctx.font = '600 21px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(item.quantity.toString(), 690, currentY + 34);

    ctx.textAlign = 'right';
    ctx.fillText(formatMoney(item.unit_price), 890, currentY + 34);

    ctx.fillStyle = '#0F172A';
    ctx.font = 'bold 21px system-ui, sans-serif';
    const lineTotal = (item.quantity || 0) * (item.unit_price || 0);
    ctx.fillText(formatMoney(lineTotal), 1090, currentY + 34);

    currentY += 54;
  });

  // Items Subtotal Row
  ctx.fillStyle = '#F1F5F9';
  ctx.fillRect(80, currentY, 1040, 50);
  ctx.fillStyle = '#475569';
  ctx.font = '600 20px system-ui, sans-serif';
  ctx.textAlign = 'right';
  ctx.fillText('SOUS-TOTAL FOURNITURES :', 890, currentY + 32);
  ctx.fillStyle = '#0F172A';
  ctx.font = 'bold 22px system-ui, sans-serif';
  ctx.fillText(`${formatMoney(totals.subtotal_items)} ${data.currency}`, 1090, currentY + 32);

  currentY += 80;

  // 6. Labor Section (Main d'œuvre)
  if (data.labor_days && data.labor_days > 0) {
    ctx.fillStyle = '#FFF7ED';
    ctx.strokeStyle = '#FED7AA';
    ctx.lineWidth = 2;
    roundRect(ctx, 80, currentY, 1040, 110, 14, true, true);

    ctx.fillStyle = '#C2410C';
    ctx.font = 'bold 18px system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('MAIN D’ŒUVRE & PRESTATIONS DE SERVICE', 110, currentY + 36);

    ctx.fillStyle = '#0F172A';
    ctx.font = '500 22px system-ui, sans-serif';
    ctx.fillText(`Durée : ${data.labor_days} jours de travail`, 110, currentY + 76);
    ctx.fillText(`Tarif journalier : ${formatMoney(data.labor_price_per_day || 0)} ${data.currency}/jour`, 430, currentY + 76);

    ctx.textAlign = 'right';
    ctx.fillStyle = '#9A3412';
    ctx.font = 'bold 26px system-ui, sans-serif';
    ctx.fillText(`${formatMoney(totals.subtotal_labor)} ${data.currency}`, 1090, currentY + 76);

    currentY += 140;
  } else {
    currentY += 30;
  }

  // 7. Highlighted Grand Total Banner
  ctx.fillStyle = '#1A365D';
  roundRect(ctx, 600, currentY, 520, 120, 18, true, false);

  ctx.fillStyle = '#94A3B8';
  ctx.font = 'bold 18px system-ui, sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText('MONTANT TOTAL DU DEVIS', 635, currentY + 45);

  const amountStr = formatMoney(totals.total_general);
  const currencyStr = data.currency || 'FCFA';

  // Responsive font size for very large totals so it never wraps or clips
  const amountFontSize = amountStr.length > 10 ? 34 : (amountStr.length > 8 ? 38 : 42);
  const amountFont = `bold ${amountFontSize}px system-ui, -apple-system, sans-serif`;

  // Set font and measure width with the EXACT font used for numbers!
  ctx.font = amountFont;
  const amountWidth = ctx.measureText(amountStr).width;

  // Draw total numbers in vibrant coral
  ctx.fillStyle = '#FF6B4A';
  ctx.textAlign = 'left';
  ctx.fillText(amountStr, 635, currentY + 95);

  // Draw currency right after the measured amount with a comfortable gap
  ctx.fillStyle = '#FFFFFF';
  ctx.font = 'bold 22px system-ui, -apple-system, sans-serif';
  ctx.fillText(currencyStr, 635 + amountWidth + 14, currentY + 95);

  // Payment conditions on the left
  ctx.fillStyle = '#0F172A';
  ctx.font = 'bold 20px system-ui, sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText('Conditions de règlement :', 80, currentY + 35);
  ctx.fillStyle = '#64748B';
  ctx.font = '400 18px system-ui, sans-serif';
  ctx.fillText('• Acompte de 50% au démarrage des travaux', 80, currentY + 68);
  ctx.fillText('• Solde à la livraison et validation du chantier', 80, currentY + 98);

  // Signature areas
  const signY = currentY + 165;
  ctx.fillStyle = '#64748B';
  ctx.font = '500 18px system-ui, sans-serif';
  ctx.fillText('Signature de l’Artisan (Bon pour accord) :', 80, signY);
  ctx.fillText('Signature du Client :', 720, signY);

  ctx.strokeStyle = '#CBD5E1';
  ctx.lineWidth = 1.5;
  ctx.setLineDash([5, 5]);
  ctx.beginPath();
  ctx.moveTo(80, signY + 60);
  ctx.lineTo(440, signY + 60);
  ctx.moveTo(720, signY + 60);
  ctx.lineTo(1080, signY + 60);
  ctx.stroke();
  ctx.setLineDash([]);

  // 8. Watermark footer
  drawFooterWatermark(ctx);
}

/**
 * Render Module 2: Cotis-Métier
 */
export function renderCotisToCanvas(canvas: HTMLCanvasElement, data: CotisData): void {
  canvas.width = CANVAS_WIDTH;
  canvas.height = CANVAS_HEIGHT;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const totals = computeCotisTotals(data);

  // Background
  ctx.fillStyle = '#FFFFFF';
  ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

  // Top accent bar
  ctx.fillStyle = '#0D9488'; // Teal / Green theme for treasury
  ctx.fillRect(0, 0, CANVAS_WIDTH, 12);
  ctx.fillStyle = '#1A365D';
  ctx.fillRect(0, 12, CANVAS_WIDTH, 6);

  // Header Area
  ctx.fillStyle = '#1A365D';
  ctx.font = 'bold 42px system-ui, sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText('REGISTRE DES COTISATIONS', 80, 85);

  ctx.fillStyle = '#F1F5F9';
  roundRect(ctx, 800, 45, 320, 60, 12, true, false);
  ctx.fillStyle = '#475569';
  ctx.font = '500 20px system-ui, sans-serif';
  ctx.textAlign = 'right';
  ctx.fillText(`Réf : COT-${new Date().getFullYear()}-001`, 1100, 72);
  ctx.fillText(`Date séance : ${data.date || new Date().toLocaleDateString('fr-FR')}`, 1100, 95);

  // Association Name Card
  ctx.fillStyle = '#F0FDFA';
  ctx.strokeStyle = '#CCFBF1';
  ctx.lineWidth = 2;
  roundRect(ctx, 80, 135, 1040, 100, 16, true, true);

  ctx.fillStyle = '#0F766E';
  ctx.font = 'bold 16px system-ui, sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText('ORGANISATION / MUTUELLE / TONTINE', 110, 170);

  ctx.fillStyle = '#115E59';
  ctx.font = 'bold 30px system-ui, sans-serif';
  ctx.fillText(data.association_name || 'Association Solidarité & Entraide', 110, 208);

  // Summary Metrics Badges (3 boxes)
  const metricY = 260;
  // Box 1: Total collecté
  ctx.fillStyle = '#1A365D';
  roundRect(ctx, 80, metricY, 360, 90, 14, true, false);
  ctx.fillStyle = '#94A3B8';
  ctx.font = 'bold 15px system-ui, sans-serif';
  ctx.fillText('TOTAL COLLECTÉ', 105, metricY + 32);
  ctx.fillStyle = '#FF6B4A';
  ctx.font = 'bold 34px system-ui, sans-serif';
  ctx.fillText(`${formatMoney(totals.total_collected)} FCFA`, 105, metricY + 70);

  // Box 2: Membres contributeurs
  ctx.fillStyle = '#F8FAFC';
  ctx.strokeStyle = '#E2E8F0';
  roundRect(ctx, 470, metricY, 320, 90, 14, true, true);
  ctx.fillStyle = '#64748B';
  ctx.font = 'bold 15px system-ui, sans-serif';
  ctx.fillText('MEMBRES CONTRIBUTEURS', 495, metricY + 32);
  ctx.fillStyle = '#0F172A';
  ctx.font = 'bold 34px system-ui, sans-serif';
  ctx.fillText(`${totals.total_members_paid} personnes`, 495, metricY + 70);

  // Box 3: Cotisations validées
  ctx.fillStyle = '#F8FAFC';
  ctx.strokeStyle = '#E2E8F0';
  roundRect(ctx, 820, metricY, 300, 90, 14, true, true);
  ctx.fillStyle = '#64748B';
  ctx.font = 'bold 15px system-ui, sans-serif';
  ctx.fillText('TOTAL VERSEMENTS', 845, metricY + 32);
  ctx.fillStyle = '#0D9488';
  ctx.font = 'bold 34px system-ui, sans-serif';
  ctx.fillText(`${data.contributions.length} lignes`, 845, metricY + 70);

  // Table Header
  const tableY = 380;
  ctx.fillStyle = '#1A365D';
  roundRect(ctx, 80, tableY, 1040, 52, 10, true, false);

  ctx.fillStyle = '#FFFFFF';
  ctx.font = 'bold 20px system-ui, sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText('N°', 105, tableY + 33);
  ctx.fillText('MEMBRE DE L’ASSOCIATION', 160, tableY + 33);
  ctx.fillText('MOTIF / TYPE DE VERSEMENT', 540, tableY + 33);
  ctx.textAlign = 'center';
  ctx.fillText('STATUT', 890, tableY + 33);
  ctx.textAlign = 'right';
  ctx.fillText('MONTANT (FCFA)', 1090, tableY + 33);

  let currentY = tableY + 52;
  const list = data.contributions.length > 0 ? data.contributions : [
    { id: '1', member_name: 'Koffi', amount: 1000, purpose: 'Cotisation mensuelle', payment_status: 'payé' as const },
    { id: '2', member_name: 'Awa', amount: 1000, purpose: 'Cotisation mensuelle', payment_status: 'payé' as const }
  ];

  list.forEach((item, index) => {
    ctx.fillStyle = index % 2 === 0 ? '#FFFFFF' : '#F8FAFC';
    ctx.fillRect(80, currentY, 1040, 56);

    ctx.strokeStyle = '#E2E8F0';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(80, currentY + 56);
    ctx.lineTo(1120, currentY + 56);
    ctx.stroke();

    // Index
    ctx.fillStyle = '#64748B';
    ctx.font = '500 20px system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`${index + 1}`, 105, currentY + 35);

    // Member name
    ctx.fillStyle = '#0F172A';
    ctx.font = 'bold 22px system-ui, sans-serif';
    ctx.fillText(item.member_name, 160, currentY + 35);

    // Purpose
    ctx.fillStyle = '#475569';
    ctx.font = '500 20px system-ui, sans-serif';
    ctx.fillText(item.purpose || 'Cotisation', 540, currentY + 35);

    // Status pill
    const status = item.payment_status;
    let pillBg = '#DCFCE7'; // green
    let pillText = '#15803D';
    let label = 'PAYÉ';

    if (status === 'partiel') {
      pillBg = '#FEF3C7'; // amber
      pillText = '#B45309';
      label = 'PARTIEL';
    } else if (status === 'en attente') {
      pillBg = '#FEE2E2'; // red
      pillText = '#B91C1C';
      label = 'EN ATTENTE';
    }

    ctx.fillStyle = pillBg;
    roundRect(ctx, 830, currentY + 13, 120, 30, 15, true, false);
    ctx.fillStyle = pillText;
    ctx.font = 'bold 15px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(label, 890, currentY + 33);

    // Amount
    ctx.fillStyle = '#0F172A';
    ctx.font = 'bold 22px system-ui, sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(formatMoney(item.amount), 1090, currentY + 35);

    currentY += 56;
  });

  // Table bottom summary
  ctx.fillStyle = '#1A365D';
  roundRect(ctx, 80, currentY + 30, 1040, 70, 14, true, false);
  ctx.fillStyle = '#FFFFFF';
  ctx.font = 'bold 24px system-ui, sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText('TOTAL ENREGISTRÉ LORS DE CETTE SÉANCE :', 110, currentY + 72);
  ctx.fillStyle = '#FF6B4A';
  ctx.font = 'bold 36px system-ui, sans-serif';
  ctx.textAlign = 'right';
  ctx.fillText(`${formatMoney(totals.total_collected)} FCFA`, 1090, currentY + 74);

  // Signatures
  const signY = currentY + 160;
  ctx.fillStyle = '#475569';
  ctx.font = '600 20px system-ui, sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText('Visa du Trésorier :', 100, signY);
  ctx.fillText('Visa du Président / Secrétaire :', 700, signY);

  ctx.strokeStyle = '#CBD5E1';
  ctx.setLineDash([5, 5]);
  ctx.beginPath();
  ctx.moveTo(100, signY + 60);
  ctx.lineTo(400, signY + 60);
  ctx.moveTo(700, signY + 60);
  ctx.lineTo(1000, signY + 60);
  ctx.stroke();
  ctx.setLineDash([]);

  // Watermark footer
  drawFooterWatermark(ctx);
}

/**
 * Render Module 3: Rapport-Chantier-Pro
 */
export async function renderChantierToCanvas(canvas: HTMLCanvasElement, data: ChantierData): Promise<void> {
  canvas.width = CANVAS_WIDTH;
  canvas.height = CANVAS_HEIGHT;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  // Background
  ctx.fillStyle = '#FFFFFF';
  ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

  // Top accent bar (Amber & Navy)
  ctx.fillStyle = '#F59E0B';
  ctx.fillRect(0, 0, CANVAS_WIDTH, 12);
  ctx.fillStyle = '#1A365D';
  ctx.fillRect(0, 12, CANVAS_WIDTH, 6);

  // Header Area
  ctx.fillStyle = '#1A365D';
  ctx.font = 'bold 42px system-ui, sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText('RAPPORT JOURNALIER DE CHANTIER', 80, 85);

  ctx.fillStyle = '#F1F5F9';
  roundRect(ctx, 800, 45, 320, 60, 12, true, false);
  ctx.fillStyle = '#475569';
  ctx.font = '500 20px system-ui, sans-serif';
  ctx.textAlign = 'right';
  ctx.fillText(`Réf : RAP-${new Date().getFullYear()}-001`, 1100, 72);
  ctx.fillText(`Date : ${data.date || new Date().toLocaleDateString('fr-FR')}`, 1100, 95);

  // Site Header Box
  ctx.fillStyle = '#FFFBEB';
  ctx.strokeStyle = '#FDE68A';
  ctx.lineWidth = 2;
  roundRect(ctx, 80, 130, 1040, 95, 16, true, true);

  ctx.fillStyle = '#D97706';
  ctx.font = 'bold 16px system-ui, sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText('LOCALISATION & IDENTIFICATION DU CHANTIER', 110, 165);

  ctx.fillStyle = '#78350F';
  ctx.font = 'bold 30px system-ui, sans-serif';
  ctx.fillText(data.site_name || 'Chantier Résidence BTP', 110, 202);

  let currentY = 255;

  // Section 1: Travaux Réalisés (List with checkmarks)
  ctx.fillStyle = '#1A365D';
  roundRect(ctx, 80, currentY, 1040, 46, 10, true, false);
  ctx.fillStyle = '#FFFFFF';
  ctx.font = 'bold 20px system-ui, sans-serif';
  ctx.fillText('1. TRAVAUX RÉALISÉS CE JOUR', 105, currentY + 30);

  currentY += 60;
  const works = data.work_done.length > 0 ? data.work_done : ['Travaux exécutés conformément aux plans'];
  works.forEach(task => {
    // Checkmark circle
    ctx.fillStyle = '#DCFCE7';
    ctx.beginPath();
    ctx.arc(115, currentY + 15, 16, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#16A34A';
    ctx.font = 'bold 20px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('✓', 115, currentY + 22);

    ctx.fillStyle = '#1E293B';
    ctx.font = '600 22px system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(task, 145, currentY + 22);

    currentY += 45;
  });

  currentY += 20;

  // Section 2: Matériaux Consommés
  ctx.fillStyle = '#1A365D';
  roundRect(ctx, 80, currentY, 1040, 46, 10, true, false);
  ctx.fillStyle = '#FFFFFF';
  ctx.font = 'bold 20px system-ui, sans-serif';
  ctx.fillText('2. MATÉRIAUX CONSOMMÉS (STOCK SORTI)', 105, currentY + 30);

  currentY += 56;
  const used = data.materials_used.length > 0 ? data.materials_used : [
    { id: '1', material: 'Ciment CPJ 42.5', quantity: '40 sacs' }
  ];

  used.forEach((mat, idx) => {
    ctx.fillStyle = idx % 2 === 0 ? '#F8FAFC' : '#FFFFFF';
    ctx.fillRect(80, currentY, 1040, 44);

    ctx.fillStyle = '#0F172A';
    ctx.font = '600 21px system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`•  ${mat.material}`, 110, currentY + 28);

    ctx.fillStyle = '#0F766E';
    ctx.font = 'bold 21px system-ui, sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(mat.quantity, 1090, currentY + 28);

    currentY += 44;
  });

  currentY += 25;

  // Section 3: Besoins pour la suite (Commandes / Alertes urgentes)
  ctx.fillStyle = '#EA580C';
  roundRect(ctx, 80, currentY, 1040, 46, 10, true, false);
  ctx.fillStyle = '#FFFFFF';
  ctx.font = 'bold 20px system-ui, sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText('3. BESOINS POUR LA SUITE (COMMANDES & APPROVISIONNEMENT)', 105, currentY + 30);

  currentY += 56;
  const needed = data.materials_needed.length > 0 ? data.materials_needed : [
    { id: '1', material: 'Paquets de fer de 12', quantity: '15 paquets', deadline: 'Pour mardi', urgent: true }
  ];

  needed.forEach(item => {
    ctx.fillStyle = '#FFF7ED';
    ctx.strokeStyle = '#FED7AA';
    ctx.lineWidth = 1.5;
    roundRect(ctx, 80, currentY, 1040, 52, 10, true, true);

    ctx.fillStyle = '#9A3412';
    ctx.font = 'bold 21px system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`⚠️  ${item.material} : ${item.quantity}`, 110, currentY + 33);

    if (item.deadline) {
      ctx.fillStyle = '#EA580C';
      ctx.font = 'bold 20px system-ui, sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(`Délai souhaité : ${item.deadline}`, 1090, currentY + 33);
    }

    currentY += 60;
  });

  currentY += 15;

  // Section 4: Galerie Photos Terrain (Photos prises via input capture)
  ctx.fillStyle = '#1A365D';
  roundRect(ctx, 80, currentY, 1040, 46, 10, true, false);
  ctx.fillStyle = '#FFFFFF';
  ctx.font = 'bold 20px system-ui, sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText(`4. PHOTOS TERRAIN RELEVÉES (${data.photos.length} cliché${data.photos.length > 1 ? 's' : ''})`, 105, currentY + 30);

  currentY += 60;

  if (data.photos.length > 0) {
    const photoWidth = 230;
    const photoHeight = 160;
    const gap = 30;

    for (let i = 0; i < Math.min(data.photos.length, 4); i++) {
      const px = 80 + i * (photoWidth + gap);
      const py = currentY;

      try {
        const img = new Image();
        img.src = data.photos[i];
        await new Promise((resolve) => {
          img.onload = resolve;
          img.onerror = resolve;
        });

        // Draw photo with border
        ctx.save();
        roundRect(ctx, px, py, photoWidth, photoHeight, 12, false, false);
        ctx.clip();
        ctx.drawImage(img, px, py, photoWidth, photoHeight);
        ctx.restore();

        ctx.strokeStyle = '#CBD5E1';
        ctx.lineWidth = 2;
        roundRect(ctx, px, py, photoWidth, photoHeight, 12, false, true);

        // Photo index stamp
        ctx.fillStyle = 'rgba(15, 23, 42, 0.7)';
        roundRect(ctx, px + 8, py + photoHeight - 32, 80, 24, 6, true, false);
        ctx.fillStyle = '#FFFFFF';
        ctx.font = 'bold 13px system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(`Photo #${i + 1}`, px + 48, py + photoHeight - 16);
      } catch (err) {
        console.warn('Failed to draw photo on canvas', err);
      }
    }
  } else {
    // Placeholder message
    ctx.fillStyle = '#F8FAFC';
    ctx.strokeStyle = '#E2E8F0';
    ctx.lineWidth = 1.5;
    roundRect(ctx, 80, currentY, 1040, 90, 12, true, true);
    ctx.fillStyle = '#64748B';
    ctx.font = 'italic 20px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Aucune photo jointe. Prenez 1 à 4 clichés directement depuis la caméra de votre téléphone.', 600, currentY + 50);
  }

  // Watermark footer
  drawFooterWatermark(ctx);
}

/**
 * Convert Canvas to PNG Blob
 */
export function getCanvasBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error('Canvas export failed'));
    }, 'image/png', 0.95);
  });
}
