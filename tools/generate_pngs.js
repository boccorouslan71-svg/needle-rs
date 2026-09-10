import fs from 'fs';
import zlib from 'zlib';

function crc32(buf) {
  let table = [];
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let k = 0; k < 8; k++) {
      c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1);
    }
    table[i] = c;
  }
  let crc = 0 ^ (-1);
  for (let i = 0; i < buf.length; i++) {
    crc = (crc >>> 8) ^ table[(crc ^ buf[i]) & 0xff];
  }
  return (crc ^ (-1)) >>> 0;
}

function makePng(width, height, getPixel) {
  const rowSize = width * 4 + 1;
  const raw = Buffer.alloc(rowSize * height);
  for (let y = 0; y < height; y++) {
    const rowOffset = y * rowSize;
    raw[rowOffset] = 0; // Filter 0
    for (let x = 0; x < width; x++) {
      const px = rowOffset + 1 + x * 4;
      const [r, g, b, a] = getPixel(x, y, width, height);
      raw[px] = r;
      raw[px + 1] = g;
      raw[px + 2] = b;
      raw[px + 3] = a;
    }
  }

  const signature = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);

  function chunk(type, data) {
    const len = Buffer.alloc(4);
    len.writeUInt32BE(data.length, 0);
    const typeBuf = Buffer.from(type, 'ascii');
    const crcBuf = Buffer.alloc(4);
    const toCrc = Buffer.concat([typeBuf, data]);
    crcBuf.writeUInt32BE(crc32(toCrc), 0);
    return Buffer.concat([len, typeBuf, data, crcBuf]);
  }

  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8; // Bit depth
  ihdr[9] = 6; // RGBA
  ihdr[10] = 0; // Compression
  ihdr[11] = 0; // Filter
  ihdr[12] = 0; // Interlace

  const idat = zlib.deflateSync(raw, { level: 9 });

  return Buffer.concat([
    signature,
    chunk('IHDR', ihdr),
    chunk('IDAT', idat),
    chunk('IEND', Buffer.alloc(0))
  ]);
}

// Draw the Diktao App Icon: Blue squircle + Soundwave to Document
function renderAppIcon(x, y, w, h) {
  // Normalized 0 to 1
  const nx = x / w;
  const ny = y / h;

  // Squircle check (rx ~ 0.22)
  const r = 0.22;
  const dx = Math.max(0, Math.abs(nx - 0.5) - (0.5 - r));
  const dy = Math.max(0, Math.abs(ny - 0.5) - (0.5 - r));
  const dist = Math.sqrt(dx * dx + dy * dy);
  if (dist > r) {
    return [0, 0, 0, 0]; // Transparent outside
  }

  // Base Blue gradient: #0052FF to #0038E0
  let bgR = Math.round(0 * (1 - ny) + 0 * ny);
  let bgG = Math.round(82 * (1 - ny) + 56 * ny);
  let bgB = Math.round(255 * (1 - ny) + 224 * ny);

  // Sound bars on left
  const bars = [
    { x: 0.14, w: 0.035, y1: 0.46, y2: 0.54 },
    { x: 0.19, w: 0.035, y1: 0.42, y2: 0.58 },
    { x: 0.24, w: 0.035, y1: 0.36, y2: 0.64 },
    { x: 0.29, w: 0.035, y1: 0.31, y2: 0.69 },
    { x: 0.34, w: 0.035, y1: 0.36, y2: 0.64 },
    { x: 0.39, w: 0.035, y1: 0.41, y2: 0.59 },
    { x: 0.44, w: 0.035, y1: 0.44, y2: 0.56 },
  ];

  for (const bar of bars) {
    const rx = bar.w / 2;
    const bdx = Math.abs(nx - bar.x);
    if (bdx <= rx) {
      if (ny >= bar.y1 && ny <= bar.y2) {
        return [255, 255, 255, 255];
      }
      // Pill caps
      const dy1 = ny - bar.y1;
      const dy2 = ny - bar.y2;
      if (bdx * bdx + dy1 * dy1 <= rx * rx || bdx * bdx + dy2 * dy2 <= rx * rx) {
        return [255, 255, 255, 255];
      }
    }
  }

  // 4 Horizontal lines connecting to document
  const hlines = [
    { y: 0.39, x1: 0.48, x2: 0.70 },
    { y: 0.46, x1: 0.49, x2: 0.70 },
    { y: 0.53, x1: 0.49, x2: 0.70 },
    { y: 0.60, x1: 0.48, x2: 0.65 },
  ];
  for (const hl of hlines) {
    const ry = 0.016;
    if (Math.abs(ny - hl.y) <= ry && nx >= hl.x1 && nx <= hl.x2) {
      return [255, 255, 255, 255];
    }
    // Rounded ends
    if (Math.hypot(nx - hl.x1, ny - hl.y) <= ry) {
      return [255, 255, 255, 255];
    }
  }

  // Document sheet body: x from ~0.67 to 0.86, y from ~0.38 to 0.73
  if (nx >= 0.67 && nx <= 0.86 && ny >= 0.38 && ny <= 0.73) {
    // Dog ear fold at top-right corner (nx > 0.76 and ny < 0.48)
    if (nx > 0.76 && ny < 0.48) {
      const foldDiag = (nx - 0.76) + (ny - 0.38);
      if (foldDiag > 0.10) {
        return [bgR, bgG, bgB, 255]; // Cutout to background
      }
    }
    return [255, 255, 255, 255];
  }

  // Folded flap
  if (nx >= 0.76 && nx <= 0.86 && ny >= 0.38 && ny <= 0.48) {
    const foldDiag = (nx - 0.76) + (ny - 0.38);
    if (foldDiag <= 0.10 && (nx >= 0.76 && ny >= 0.38)) {
      return [240, 244, 255, 255];
    }
  }

  return [bgR, bgG, bgB, 255];
}

fs.writeFileSync('public/pwa-192x192.png', makePng(192, 192, renderAppIcon));
fs.writeFileSync('public/pwa-512x512.png', makePng(512, 512, renderAppIcon));
fs.writeFileSync('public/pwa-maskable-512x512.png', makePng(512, 512, renderAppIcon));
fs.writeFileSync('public/apple-touch-icon.png', makePng(180, 180, renderAppIcon));
fs.writeFileSync('public/favicon.png', makePng(64, 64, renderAppIcon));
console.log('PNG generation successful!');
