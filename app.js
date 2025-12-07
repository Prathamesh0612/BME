// NEC - Neural Encryption Corrector - AI-Enhanced Single-Key System
// Client-side implementation with Web Crypto API and TensorFlow.js
let necModel = null;

async function loadNECModel() {
    try {
        // TFJS model ka relative URL
        necModel = await tf.loadLayersModel('./nec_model_tjFs/model.json');
        console.log('✓ NEC AI model loaded successfully');
        updateAIStatus('✓ AI Model Active');
        return true;
    } catch (e) {
        console.log('⚠ NEC AI model not found, using heuristic fallback', e);
        necModel = null;
        updateAIStatus('100% Safe'); // User's status
        return false;
    }
}


function updateAIStatus(status) {
    const statusEl = document.getElementById('ai-status');
    if (statusEl) statusEl.textContent = status;
}

// Initialize app after model loading
document.addEventListener('DOMContentLoaded', async () => {
    await loadNECModel();
    const statusEl = document.getElementById('mode-status'); 
    if (statusEl) {
        statusEl.textContent = necModel ? 'AI Model Active' : 'Heuristic Mode Active';
        statusEl.style.color = necModel ? '#00cc99' : '#ff9900';
    }
    new NECApp();
});

class NECrypto {
    constructor() {
        this.VERSION = 1;
        this.KDF_ITERATIONS = 150000;
        this.SALT_SIZE = 32;
        this.SEED_SIZE = 32;
        this.MAX_FILE_SIZE = 200 * 1024 * 1024;
        this.CHUNK_SIZE = 10 * 1024 * 1024;
        this.BASES = [12, 16, 20, 36];
        this.MIN_CORRUPTION_RATIO = 0.001;
        this.MAX_CORRUPTION_RATIO = 0.01;
        this.MAX_CRYPTO_RANDOM_BYTES = 65536;

        // Compression config
        // Sirf badi files (20MB+) ke liye internal compression
        this.COMPRESSION_THRESHOLD = 20 * 1024 * 1024; // 20MB
        this.COMPRESSION_METHOD = 'deflate';
    }

    // -------- Compression helpers --------
    canCompress() {
        return (typeof pako !== 'undefined' &&
                typeof pako.deflate === 'function' &&
                typeof pako.inflate === 'function');
    }

    async compressBytes(bytes) {
    if (!this.canCompress()) return bytes;
    try {
        if (!(bytes instanceof Uint8Array)) {
            bytes = new Uint8Array(bytes);
        }
        let out = pako.deflate(bytes);
        // pako kabhi plain Array bhi de sakta hai
        if (!(out instanceof Uint8Array)) {
            out = new Uint8Array(out);
        }

        // AGAR KISI wajah se 0-byte aa gaya to compression cancel
        if (!out || out.length === 0) {
            console.warn('compressBytes: got empty output, falling back to raw bytes');
            return bytes;
        }

        return out;
    } catch (e) {
        console.error('compressBytes failed, using raw bytes:', e);
        return bytes;
    }
}

    async decompressBytes(bytes) {
    if (!this.canCompress()) {
        throw new Error('Compression library not available for decompression');
    }
    try {
        if (!(bytes instanceof Uint8Array)) {
            bytes = new Uint8Array(bytes);
        }
        let out = pako.inflate(bytes);
        if (!(out instanceof Uint8Array)) {
            out = new Uint8Array(out);
        }

        if (!out || out.length === 0) {
            // Zlib stream sahi nahi tha
            throw new Error('Decompression produced empty data (0 bytes)');
        }

        return out;
    } catch (e) {
        console.error('decompressBytes failed:', e);
        throw e;
    }
}


    // -------- Header / key --------
    async createCompactKey(
        keyString,
        salt,
        iterations,
        partition,
        bases,
        fileHashHex,
        seed,
        encodedStrings,
        originalBitLength,
        compressed = false,
        originalSize = null
    ) {
        const { masterKey, macKey } = await this.deriveKeysFromKeyString(keyString, salt, iterations);
        const encSeed = this.xorEncryptDecrypt(seed, masterKey);
        const headerObj = {
            version: this.VERSION,
            salt: btoa(String.fromCharCode(...salt)),
            iterations,
            encryptedSeed: btoa(String.fromCharCode(...encSeed)),
            partition,
            bases,
            fileHashHex,
            encodedStrings,
            originalBitLength,
            compressed: !!compressed,
            originalSize: originalSize != null ? originalSize : null,
            compressionMethod: compressed ? this.COMPRESSION_METHOD : null
        };
        const headerBytes = new TextEncoder().encode(JSON.stringify(headerObj));
        const tag = await this.computeHMAC(macKey, headerBytes);
        const combined = new Uint8Array(headerBytes.length + tag.length);
        combined.set(headerBytes, 0);
        combined.set(tag, headerBytes.length);
        const base = this.base85Encode(combined);
        const checksum = (await this.sha256(base)).slice(0, 8);
        return `NEC${checksum}${base}`;
    }

    async parseCompactKeyFromHeader(headerString, keyString) {
        if (!headerString.startsWith('NEC')) throw new Error('Invalid key header format');
        const checksum = headerString.substring(3, 11);
        const base = headerString.substring(11);
        const computed = (await this.sha256(base)).slice(0, 8);
        if (checksum !== computed) throw new Error('Key checksum mismatch');

        const combined = this.base85Decode(base);
        if (combined.length < 32) throw new Error('Invalid header payload');
        const headerLen = combined.length - 32;
        const headerBytes = combined.slice(0, headerLen);
        const expectedTag = combined.slice(headerLen);

        const hdrText = new TextDecoder().decode(headerBytes);
        const headerObj = JSON.parse(hdrText);
        try {
            const encodedInfo = Array.isArray(headerObj.encodedStrings)
                ? headerObj.encodedStrings.map(s => (s ? s.length : 0))
                : [];
            console.log('parseCompactKeyFromHeader:', {
                version: headerObj.version,
                partition: headerObj.partition,
                bases: headerObj.bases,
                fileHashHex: headerObj.fileHashHex,
                encodedLengths: encodedInfo,
                compressed: headerObj.compressed,
                originalSize: headerObj.originalSize
            });
        } catch (e) {
            console.warn('parseCompactKeyFromHeader: logging failed', e);
        }

        const salt = new Uint8Array(Array.from(atob(headerObj.salt)).map(c => c.charCodeAt(0)));
        const iterations = headerObj.iterations;

        const { masterKey, macKey } = await this.deriveKeysFromKeyString(keyString, salt, iterations);
        const ok = await this.verifyHMAC(macKey, headerBytes, expectedTag);
        if (!ok) throw new Error('Key verification failed - invalid key');

        const encSeed = new Uint8Array(Array.from(atob(headerObj.encryptedSeed)).map(c => c.charCodeAt(0)));
        const seed = this.xorEncryptDecrypt(encSeed, masterKey);

        return {
            version: headerObj.version,
            salt,
            iterations,
            partition: headerObj.partition,
            bases: headerObj.bases,
            fileHash: headerObj.fileHashHex,
            seed,
            encodedStrings: headerObj.encodedStrings,
            macKey,
            originalBitLength: headerObj.originalBitLength,
            compressed: !!headerObj.compressed,
            originalSize: headerObj.originalSize != null ? headerObj.originalSize : null,
            compressionMethod: headerObj.compressionMethod || null
        };
    }

    // -------- AI analysis --------
    buildAIFeatures(bytes) {
        const features = [];
        const entropy4k = this.calculateEntropy(bytes.slice(0, Math.min(4096, bytes.length)));
        const entropy64k = this.calculateEntropy(bytes.slice(0, Math.min(65536, bytes.length)));
        features.push(entropy4k, entropy64k);

        const hist = new Array(16).fill(0);
        const sample = bytes.slice(0, Math.min(65536, bytes.length));
        for (const b of sample) hist[Math.floor(b / 16)]++;
        const sum = hist.reduce((a, b) => a + b, 0) || 1;
        features.push(...hist.map(v => v / sum));

        const h = bytes.slice(0, 16);
        const flags = [
            (h[0] === 0xFF && h[1] === 0xD8),
            (h[0] === 0x89 && h[1] === 0x50 && h[2] === 0x4E && h[3] === 0x47),
            (h[0] === 0x47 && h[1] === 0x49 && h[2] === 0x46),
            (h[0] === 0x25 && h[1] === 0x50 && h[2] === 0x44 && h[3] === 0x46),
            (h[0] === 0x50 && h[1] === 0x4B)
        ].map(b => (b ? 1 : 0));
        features.push(...flags);

        const sizeMB = bytes.length / (1024 * 1024);
        const sizeBins = [sizeMB < 1, sizeMB >= 1 && sizeMB < 10, sizeMB >= 10].map(b => (b ? 1 : 0));
        features.push(...sizeBins);

        return features;
    }
    analyzeFile(data) {
    const bytes = new Uint8Array(data);

    if (typeof tf !== 'undefined' && necModel) {
        try {
            const features = this.buildAIFeatures(bytes);
            const x = tf.tensor(features, [1, features.length]);
            const predictions = necModel.predict(x);

            const r_raw = predictions[0].dataSync()[0];
const w_logits = Array.from(predictions[1].dataSync());
const p_raw = Array.from(predictions[2].dataSync());

// 1) Corruption ratio ka CENTRAL value (0.001–0.01 ke beech)
const r_center = Math.max(
    this.MIN_CORRUPTION_RATIO,
    Math.min(this.MAX_CORRUPTION_RATIO, 0.001 + 0.009 * r_raw)
);

// 2) Spread decide karo (file ke size / entropy pe bhi baad me kar sakte ho)
// abhi simple: ±30% ka band
let spread = 0.3 * r_center; // 30% of center
// minimum spread thoda de do so that same file bhi hamesha change ho
const MIN_SPREAD = 0.0005;
if (spread < MIN_SPREAD) spread = MIN_SPREAD;

// min / max range clamp to global bounds
let minR = Math.max(this.MIN_CORRUPTION_RATIO, r_center - spread / 2);
let maxR = Math.min(this.MAX_CORRUPTION_RATIO, r_center + spread / 2);

// edge case: agar kisi wajah se min >= max aa gaya to fixed small window
if (minR >= maxR) {
    minR = Math.max(this.MIN_CORRUPTION_RATIO, r_center - 0.0005);
    maxR = Math.min(this.MAX_CORRUPTION_RATIO, r_center + 0.0005);
}

const w_sum = w_logits.reduce((a, b) => a + Math.exp(b), 0);
const w = w_logits.map(v => Math.exp(v) / w_sum);

// Model se aa raha raw 0–1 use karke partitions ka range
let minP = Math.max(4, Math.min(64, Math.round(4 + 60 * p_raw[0])));  // 4–64
let maxP = Math.max(minP, Math.min(256, Math.round(4 + 252 * p_raw[1]))); // minP–256

return {
    size: bytes.length,
    entropy: this.calculateEntropy(bytes.slice(0, Math.min(1024 * 1024, bytes.length))),
    fileType: this.detectFileType(bytes),

    // ⚠️ yahan ab single value nahi bhej rahe, RANGE bhej rahe
    corruptionRange: { min: minR, max: maxR },
    partitionRange: { min: minP, max: maxP },

    bases: this.BASES,
    strategyWeights: { header: w[0], structure: w[1], random: w[2] },
    aiUsed: true
};

        } catch (e) {
            console.warn('AI prediction failed, using heuristic:', e);
        }
    }

    // Fallback (heuristic)
    return {
        size: bytes.length,
        entropy: this.calculateEntropy(
            bytes.slice(0, Math.min(1024 * 1024, bytes.length))
        ),
        fileType: this.detectFileType(bytes),
        aiUsed: false
    };
}

    calculateEntropy(bytes) {
        const freq = new Array(256).fill(0);
        for (const b of bytes) freq[b]++;
        let H = 0, n = bytes.length;
        for (const c of freq) {
            if (c > 0) {
                const p = c / n;
                H -= p * Math.log2(p);
            }
        }
        return H / 8;
    }

    detectFileType(bytes) {
        const h = bytes.slice(0, 16);
        if (h[0] === 0xFF && h[1] === 0xD8) return 'image/jpeg';
        if (h[0] === 0x89 && h[1] === 0x50 && h[2] === 0x4E && h[3] === 0x47) return 'image/png';
        if (h[0] === 0x47 && h[1] === 0x49 && h[2] === 0x46) return 'image/gif';
        if (h[0] === 0x25 && h[1] === 0x50 && h[2] === 0x44 && h[3] === 0x46) return 'application/pdf';
        if (h[0] === 0x50 && h[1] === 0x4B) return 'application/zip';
        let textScore = 0;
        for (let i = 0; i < Math.min(1024, bytes.length); i++) {
            const b = bytes[i];
            if ((b >= 32 && b <= 126) || b === 9 || b === 10 || b === 13) textScore++;
        }
        if (textScore > bytes.length * 0.8) return 'text/plain';
        return 'application/octet-stream';
    }

    // -------- Crypto + utility --------
    generateRandomBytes(size) {
        const maxChunk = this.MAX_CRYPTO_RANDOM_BYTES || 65536;
        if (size <= maxChunk) {
            const a = new Uint8Array(size);
            crypto.getRandomValues(a);
            return a;
        }
        const out = new Uint8Array(size);
        let offset = 0;
        while (offset < size) {
            const chunk = Math.min(maxChunk, size - offset);
            const tmp = new Uint8Array(chunk);
            crypto.getRandomValues(tmp);
            out.set(tmp, offset);
            offset += chunk;
        }
        return out;
    }

    generateKeyString() {
        const b = this.generateRandomBytes(32);
        let s = '';
        for (let i = 0; i < b.length; i++) s += String.fromCharCode(b[i]);
        return btoa(s);
    }

    bufferToHex(buf) {
        if (buf instanceof ArrayBuffer) buf = new Uint8Array(buf);
        return Array.from(buf).map(b => b.toString(16).padStart(2, '0')).join('');
    }

    base85Encode(data) {
        let binary = '';
        for (let i = 0; i < data.length; i++) binary += String.fromCharCode(data[i]);
        return btoa(binary);
    }

    base85Decode(str) {
        const binary = atob(str);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
        return bytes;
    }

    async sha256(input) {
        let data;

        if (typeof input === 'string') {
            // String ko UTF-8 bytes
            data = new TextEncoder().encode(input);
        } else if (input instanceof ArrayBuffer) {
            // Direct ArrayBuffer
            data = input;
        } else if (ArrayBuffer.isView(input)) {
            // Uint8Array, DataView, etc.
            data = input;
        } else if (input && typeof input.length === 'number') {
            // Plain Array ya array-like (pako.inflate kabhi Array de deta hai)
            data = new Uint8Array(input);
        } else {
            // Fallback – jo bhi hai usko string bana ke hash
            data = new TextEncoder().encode(String(input));
        }

        const h = await crypto.subtle.digest('SHA-256', data);
        return this.bufferToHex(new Uint8Array(h));
    }

    async deriveKeysFromKeyString(keyString, salt, iterations) {
        const pw = new TextEncoder().encode(keyString);
        const baseKey = await crypto.subtle.importKey('raw', pw, { name: 'PBKDF2' }, false, ['deriveBits']);
        const bits = await crypto.subtle.deriveBits(
            { name: 'PBKDF2', salt: salt, iterations: iterations, hash: 'SHA-256' },
            baseKey,
            512
        );
        const buf = new Uint8Array(bits);
        const masterKey = buf.slice(0, 32);
        const macRaw = buf.slice(32, 64);
        const macKey = await crypto.subtle.importKey('raw', macRaw, { name: 'HMAC', hash: 'SHA-256' }, false, ['sign', 'verify']);
        return { masterKey, macKey };
    }

    async computeHMAC(macKey, data) {
        if (!(data instanceof Uint8Array)) data = new Uint8Array(data);
        const sig = await crypto.subtle.sign('HMAC', macKey, data);
        return new Uint8Array(sig);
    }

    async verifyHMAC(macKey, data, expected) {
        const sig = await this.computeHMAC(macKey, data);
        return this.constantTimeEquals(sig, expected);
    }

    xorEncryptDecrypt(data, key) {
        const out = new Uint8Array(data.length);
        for (let i = 0; i < data.length; i++) out[i] = data[i] ^ key[i % key.length];
        return out;
    }

    csprngInt(maxExclusive) {
        if (maxExclusive <= 0) return 0;
        const r = this.generateRandomBytes(4).reduce((a, b) => (a << 8) | b, 0) >>> 0;
        return r % maxExclusive;
    }

    generateRandomPartition(totalBits, parts, maxPart) {
        if (parts <= 0) return [];
        const base = Math.floor(totalBits / parts);
        let rem = totalBits - base * parts;
        const out = new Array(parts).fill(base);
        for (let i = 0; i < parts && rem > 0; i++, rem--) out[i]++;
        return out;
    }

    constantTimeEquals(a, b) {
        if (a.length !== b.length) return false;
        let r = 0;
        for (let i = 0; i < a.length; i++) r |= a[i] ^ b[i];
        return r === 0;
    }

    async generateBitPositions(seed, fileHashHex, chunkId, partition, chunkBitLen) {
        const needed = partition.reduce((s, x) => s + x, 0);
        const positions = new Set();
        if (chunkBitLen <= 0 || needed <= 0) return [];
        let counter = 0;
        while (positions.size < needed) {
            const ctx = new Uint8Array(seed.length + fileHashHex.length + 8 + 4);
            ctx.set(seed, 0);
            ctx.set(new TextEncoder().encode(fileHashHex), seed.length);
            const dv = new DataView(ctx.buffer);
            dv.setUint32(seed.length + fileHashHex.length, chunkId || 0);
            dv.setUint32(seed.length + fileHashHex.length + 4, counter++);
            const hash = new Uint8Array(await crypto.subtle.digest('SHA-256', ctx));
            for (let i = 0; i + 3 < hash.length && positions.size < needed; i += 4) {
                const v = (hash[i] << 24) | (hash[i + 1] << 16) | (hash[i + 2] << 8) | hash[i + 3];
                const pos = Math.abs(v) % chunkBitLen;
                positions.add(pos);
            }
        }
        return Array.from(positions).sort((a, b) => a - b);
    }

    bytesToBits(bytes) {
        const bits = new Array(bytes.length * 8);
        for (let i = 0; i < bytes.length; i++) {
            for (let b = 0; b < 8; b++) bits[i * 8 + b] = (bytes[i] >> (7 - b)) & 1;
        }
        return bits;
    }

    bitsToBytes(bits) {
        const out = new Uint8Array(Math.ceil(bits.length / 8));
        out.fill(0);
        for (let i = 0; i < bits.length; i++) {
            if (bits[i]) {
                const byteIndex = Math.floor(i / 8);
                const bitIndex = i % 8;
                out[byteIndex] |= (1 << (7 - bitIndex));
            }
        }
        return out;
    }

    // BigInt <-> base-N
    bigintToBase(bi, base) {
        const alphabet = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz';
        if (base < 2 || base > alphabet.length) throw new Error('Unsupported base');
        if (bi === 0n) return '0';
        let x = bi < 0n ? -bi : bi;
        let s = '';
        while (x > 0n) {
            const mod = x % BigInt(base);
            s = alphabet[Number(mod)] + s;
            x = x / BigInt(base);
        }
        return s;
    }

    baseToBigint(str, base) {
        const alphabet = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz';
        if (base < 2 || base > alphabet.length) throw new Error('Unsupported base');
        let result = 0n;
        for (let i = 0; i < str.length; i++) {
            const idx = alphabet.indexOf(str[i]);
            if (idx < 0 || idx >= base) throw new Error('Invalid digit for base');
            result = result * BigInt(base) + BigInt(idx);
        }
        return result;
    }

    // Remove bits and encode per-partition into base-N strings (small/normal files)
    async removeBitsAndEncode(bytes, positionsGroups, bases) {
        const bits = this.bytesToBits(bytes);
        const totalBits = bits.length;
        const removedMask = new Array(totalBits).fill(0);
        const encodedStrings = [];
        let outOfRangeCount = 0;

        for (let i = 0; i < positionsGroups.length; i++) {
            const posList = positionsGroups[i];
            const base = bases && bases[i] ? bases[i] : this.BASES[i % this.BASES.length];
            let bi = 0n;
            for (let j = 0; j < posList.length; j++) {
                const p = posList[j];
                if (p < 0 || p >= totalBits) {
                    console.error('removeBitsAndEncode: position out of range', { part: i, idx: j, pos: p, totalBits });
                    outOfRangeCount++;
                    continue;
                }
                const bit = bits[p];
                bi = (bi << 1n) | BigInt(bit);
                removedMask[p] = 1;
            }
            const s = this.bigintToBase(bi, base);
            encodedStrings.push(s);
        }

        const remainingBits = [];
        for (let i = 0; i < bits.length; i++) if (!removedMask[i]) remainingBits.push(bits[i]);
        const remainingBytes = this.bitsToBytes(remainingBits);

        try {
            const actualRemoved = positionsGroups.reduce((a, b) => a + b.length, 0) - outOfRangeCount;
            console.log('removeBitsAndEncode:', {
                totalBits,
                actualRemoved,
                outOfRangeCount,
                remainingBitsLen: remainingBits.length
            });
        } catch (e) {
            console.warn('removeBitsAndEncode: debug log failed', e);
        }
        return { remainingBytes, encodedStrings };
    }

    async decodeAndInsertBits(remainingBytes, positionsGroups, encodedStrings, bases) {
        const remainingBits = this.bytesToBits(remainingBytes);
        const totalBits = positionsGroups.reduce((a, b) => a + b.length, 0) + remainingBits.length;
        const reconstructed = new Array(totalBits).fill(null);

        for (let i = 0; i < positionsGroups.length; i++) {
            const posList = positionsGroups[i];
            const base = bases && bases[i] ? bases[i] : this.BASES[i % this.BASES.length];
            const s = encodedStrings[i] || '0';
            const bi = this.baseToBigint(s, base);
            const needed = posList.length;
            const partBits = new Array(needed).fill(0);

            if (needed > 0) {
                for (let k = 0; k < needed; k++) {
                    const bitPos = needed - 1 - k;
                    partBits[k] = Number((bi >> BigInt(bitPos)) & 1n);
                }
            }
            for (let j = 0; j < posList.length; j++) {
                const p = posList[j];
                if (p < 0 || p >= reconstructed.length) {
                    console.error('decodeAndInsertBits: position out of range', { part: i, idx: j, pos: p, reconstructedLen: reconstructed.length });
                    continue;
                }
                reconstructed[p] = partBits[j];
            }
        }

        let remIdx = 0;
        for (let i = 0; i < reconstructed.length; i++) {
            if (reconstructed[i] === null) {
                reconstructed[i] = remainingBits[remIdx++] || 0;
            }
        }

        const nullCount = reconstructed.reduce((c, v) => c + (v === null ? 1 : 0), 0);
        if (nullCount > 0) console.error('decodeAndInsertBits: reconstructed contains nulls', nullCount);

        const outBytes = this.bitsToBytes(reconstructed);
        try {
            console.log('decodeAndInsertBits:', { totalBits, outBytesLen: outBytes.length });
        } catch (e) {}
        return outBytes;
    }

    generateTestData(size = 1024 * 1024) {
        return this.generateRandomBytes(size);
    }

    // ==================== STREAMING / CHUNKED (bit-level, low memory) ====================
    async streamingRemoveBitsAndEncode(fileBuffer, positionsGroups, bases, yieldInterval = 50000) {
        // fileBuffer: Uint8Array
        const totalBits = fileBuffer.length * 8;
        const totalBytes = fileBuffer.length;
        const encodedStrings = [];

        // STEP 1: encode removed bits per partition (bit-level)
        for (let partIdx = 0; partIdx < positionsGroups.length; partIdx++) {
            const posList = positionsGroups[partIdx];
            const base = bases[partIdx];
            let bi = 0n;

            for (let posIdx = 0; posIdx < posList.length; posIdx++) {
                const p = posList[posIdx];
                if (p < 0 || p >= totalBits) continue;

                const byteIndex = Math.floor(p / 8);
                const bitIndex = p % 8;
                const byte = fileBuffer[byteIndex];
                const bit = (byte >> (7 - bitIndex)) & 1;

                bi = (bi << 1n) | BigInt(bit);

                if (posIdx % yieldInterval === 0) {
                    await new Promise(r => setTimeout(r, 0));
                }
            }

            const s = this.bigintToBase(bi, base);
            encodedStrings.push(s);
        }

        // STEP 2: build set of removed positions
        const removedPositions = new Set();
        for (const posList of positionsGroups) {
            for (const p of posList) removedPositions.add(p);
        }

        // STEP 3: collect remaining bits and pack into bytes
        const remainingBytesArray = [];
        let currentByte = 0;
        let bitCountInCurrent = 0;

        for (let byteIdx = 0; byteIdx < totalBytes; byteIdx++) {
            const byte = fileBuffer[byteIdx];
            for (let bitIdx = 0; bitIdx < 8; bitIdx++) {
                const globalBitPos = byteIdx * 8 + bitIdx;
                if (globalBitPos >= totalBits) break;

                if (removedPositions.has(globalBitPos)) continue;

                const bit = (byte >> (7 - bitIdx)) & 1;
                currentByte = (currentByte << 1) | bit;
                bitCountInCurrent++;

                if (bitCountInCurrent === 8) {
                    remainingBytesArray.push(currentByte);
                    currentByte = 0;
                    bitCountInCurrent = 0;
                }
            }

            if (byteIdx % yieldInterval === 0) {
                await new Promise(r => setTimeout(r, 0));
            }
        }

        if (bitCountInCurrent > 0) {
            currentByte = currentByte << (8 - bitCountInCurrent);
            remainingBytesArray.push(currentByte);
        }

        const remainingBytes = new Uint8Array(remainingBytesArray);

        console.log('streamingRemoveBitsAndEncode: totalBits, removedCount, remainingBytesLen, encodedLengths=', {
            totalBits,
            removedCount: positionsGroups.reduce((a, b) => a + b.length, 0),
            remainingBytesLen: remainingBytes.length,
            encodedLengths: encodedStrings.map(s => s.length)
        });

        return { remainingBytes, encodedStrings };
    }

   async streamingDecodeAndInsertBits(
    remainingBytes,
    positionsGroups,
    encodedStrings,
    bases,
    totalBits,
    yieldInterval = 50000
) {
    // totalBits must be the originalBitLength used at encryption

    // Pehle sirf itna tha:
    // const removedCount = positionsGroups.reduce((a, b) => a + b.length, 0);
    // const neededRemBits = totalBits - removedCount;

    const removedCount = positionsGroups.reduce((a, b) => a + b.length, 0);

    // STEP 1: build position -> bit map from encodedStrings
    const positionToBit = new Map();
    for (let partIdx = 0; partIdx < positionsGroups.length; partIdx++) {
        const posList = positionsGroups[partIdx];
        const base = bases[partIdx];
        const s = encodedStrings[partIdx] || '0';
        const bi = this.baseToBigint(s, base);
        const needed = posList.length;

        if (needed > 0) {
            for (let k = 0; k < needed; k++) {
                const bitPos = needed - 1 - k; // MSB-first
                const bit = Number((bi >> BigInt(bitPos)) & 1n);
                const p = posList[k];
                positionToBit.set(p, bit); // duplicate positions yahan overwrite ho jaate hain
            }
        }
    }

    // ✅ REAL correct value:
    // uniqueRemoved = actual distinct bit positions jo hataye gaye the
    const uniqueRemoved = positionToBit.size;
    const neededRemBits = totalBits - uniqueRemoved;

    // Optional: debug log
    console.log('streamingDecodeAndInsertBits stats:', {
        totalBits,
        removedCount,
        uniqueRemoved,
        neededRemBits,
        remainingBytesLen: remainingBytes.length
    });

    // STEP 2: remaining bits reader
    const remainingReader = {
        bytes: remainingBytes,
        byteIdx: 0,
        bitIdx: 0,
        consumed: 0,
        needed: neededRemBits,
        next() {
            if (this.consumed >= this.needed) return 0; // extra bits sirf padding ban jaayenge
            if (this.byteIdx >= this.bytes.length) {
                this.consumed++;
                return 0;
            }
            const byte = this.bytes[this.byteIdx];
            const bit = (byte >> (7 - this.bitIdx)) & 1;
            this.bitIdx++;
            if (this.bitIdx === 8) {
                this.bitIdx = 0;
                this.byteIdx++;
            }
            this.consumed++;
            return bit;
        }
    };

    // STEP 3: sequentially reconstruct bits and pack into bytes
    const outBytesArray = [];
    let currentByte = 0;
    let bitCountInCurrent = 0;

    for (let pos = 0; pos < totalBits; pos++) {
        const bit = positionToBit.has(pos) ? positionToBit.get(pos) : remainingReader.next();

        currentByte = (currentByte << 1) | bit;
        bitCountInCurrent++;

        if (bitCountInCurrent === 8) {
            outBytesArray.push(currentByte);
            currentByte = 0;
            bitCountInCurrent = 0;
        }

        if (pos % yieldInterval === 0) {
            await new Promise(r => setTimeout(r, 0));
        }
    }

    if (bitCountInCurrent > 0) {
        currentByte = currentByte << (8 - bitCountInCurrent);
        outBytesArray.push(currentByte);
    }

    const outBytes = new Uint8Array(outBytesArray);
    console.log('streamingDecodeAndInsertBits: totalBits, outBytesLen=', {
        totalBits,
        outBytesLen: outBytes.length
    });
    return outBytes;
}

}

// -----------------------------------------------------------------
// CLASS 2: NECApp (The UI controller)
// -----------------------------------------------------------------
class NECApp {
    constructor() {
        this.crypto = new NECrypto();
        this.currentFile = null;
        this.encryptedFile = null;
        this.encryptedData = null;
        this.restoredData = null;
        this.originalFileName = null;

        // File System Access handles
        this.encryptFileHandle = null;
        this.decryptFileHandle = null;

        // sab event listeners / buttons yahin se bind honge
        this.initializeUI();
    }
    async bundleAndEncryptFilesWithLimit(files, maxMb) {
    try {
        if (!window.JSZip) {
            this.showError('Bundling requires JSZip (missing on page).');
            return;
        }

        const maxBytes = maxMb * 1024 * 1024;

        // Total raw size check (NEC ka internal hard limit)
        const totalRaw = files.reduce((s, f) => s + f.size, 0);
        if (totalRaw > this.crypto.MAX_FILE_SIZE) {
            this.showError(
                `Total raw size is too large (${this.formatFileSize(totalRaw)}). ` +
                `Max allowed by NEC: ${this.crypto.MAX_FILE_SIZE / (1024 * 1024)} MB`
            );
            return;
        }

        this.showSuccess(
            `Creating ZIP bundle from ${files.length} files (target ~${maxMb} MB)...`
        );

        // ✅ STEP 1: SAARI FILES ZIP ME DAALO
        const zip = new JSZip();
        for (const f of files) {
            // root level par seedha file name se add kar rahe hain
            zip.file(f.name, f);
        }

        // ✅ STEP 2: ZIP generate karo (DEFLATE compression ke sath)
        const zipBytes = await zip.generateAsync({
            type: 'uint8array',
            compression: 'DEFLATE',
            compressionOptions: { level: 6 }
        });

        const zipSizeMb = zipBytes.length / (1024 * 1024);

        // ✅ STEP 3: agar limit cross ho gayi toh sirf WARNING
        if (zipBytes.length > maxBytes) {
            this.showError(
                `Warning: Bundle size ${zipSizeMb.toFixed(2)} MB ` +
                `is larger than your desired limit of ${maxMb} MB. ` +
                `Encrypting anyway.`
            );
        } else {
            this.showSuccess(
                `Bundled ${files.length} file(s) into ~${zipSizeMb.toFixed(2)} MB ZIP.`
            );
        }

        // Final zip filename
        const bundleName = `NEC_Bundle_${files.length}_files.zip`;

        // ✅ STEP 4: ZIP ko File bana ke normal NEC flow me de do
        const zipFile = new File(
            [zipBytes],
            bundleName,
            { type: 'application/zip' }
        );

        // 1) analyze + UI update
        await this.handleFileSelect(zipFile, 'encrypt');

        // 2) direct encrypt start
        const startEncBtn = document.getElementById('start-encrypt');
        if (startEncBtn) startEncBtn.click();

    } catch (err) {
        console.error('bundleAndEncryptFilesWithLimit error', err);
        this.showError('Bundling with size limit failed: ' + err.message);
    }
}


    initializeUI() {

                    // ----- Bundle & Encrypt (ZIP + compression + desired size) -----
const bundleSelectBtn = document.getElementById('bundle-select-btn');
const bundleFilesInput = document.getElementById('bundle-files');
const bundleEncryptBtn = document.getElementById('bundle-encrypt-btn');

if (bundleSelectBtn && bundleFilesInput) {
    bundleSelectBtn.addEventListener('click', () => bundleFilesInput.click());

    bundleFilesInput.addEventListener('change', () => {
        if (bundleFilesInput.files && bundleFilesInput.files.length > 0) {
            bundleEncryptBtn.removeAttribute('disabled');
            bundleEncryptBtn.textContent =
                `Bundle (ZIP + Compress) & Encrypt (${bundleFilesInput.files.length} files)`;
        } else {
            bundleEncryptBtn.setAttribute('disabled', 'disabled');
            bundleEncryptBtn.textContent = 'Bundle (ZIP + Compress) & Encrypt';
        }
    });
}

if (bundleEncryptBtn && bundleFilesInput) {
    bundleEncryptBtn.addEventListener('click', () => {
        if (bundleFilesInput.files && bundleFilesInput.files.length > 0) {
            const maxMbInput = document.getElementById('bundle-max-mb');
            let maxMb = 50;
            if (maxMbInput && maxMbInput.value) {
                maxMb = Math.max(1, Number(maxMbInput.value) || 50);
            }
            this.bundleAndEncryptFilesWithLimit(
                Array.from(bundleFilesInput.files),
                maxMb
            );
        }
    });
}

        const screenshotBtn = document.getElementById('take-screenshot');
        if (screenshotBtn) {
            screenshotBtn.addEventListener('click', async () => {
                try {
                    const element = document.getElementById('encrypt-results');
                    if (!element) {
                        alert('No encryption results to capture');
                        return;
                    }
                    const canvas = await html2canvas(element, { scale: 2 });
                    canvas.toBlob(blob => {
                        const a = document.createElement('a');
                        a.href = URL.createObjectURL(blob);
                        a.download = `NecEncryptionResults_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.png`;
                        a.click();
                        URL.revokeObjectURL(a.href);
                    });
                } catch (e) {
                    alert('Screenshot failed: ' + e.message);
                }
            });
        }

        const copyKeyBtn = document.querySelector('.copy-btn[data-target="encryption-key"]');
        if (copyKeyBtn) {
            copyKeyBtn.addEventListener('click', (e) => {
                const targetId = e.currentTarget.dataset.target;
                if (targetId) this.copyToClipboard(targetId);
            });
        }

        const saveKeyBtn = document.getElementById('save-key-txt');
        if (saveKeyBtn) {
            saveKeyBtn.addEventListener('click', () => {
                const keyArea = document.getElementById('encryption-key');
                if (!keyArea || !keyArea.value) {
                    alert('No encryption key to save');
                    return;
                }
                const blob = new Blob([keyArea.value], { type: 'text/plain' });
                const a = document.createElement('a');
                a.href = URL.createObjectURL(blob);
                a.download = 'NEC_Encryption_Key_' + new Date().toISOString().slice(0, 19).replace(/:/g, '-') + '.txt';
                a.click();
                URL.revokeObjectURL(a.href);
            });
        }

        document.querySelectorAll('.tab-btn').forEach(btn =>
            btn.addEventListener('click', (e) => this.switchTab(e.target.dataset.tab))
        );

        this.setupFileUpload('encrypt');
        this.setupFileUpload('decrypt');

        const startEnc = document.getElementById('start-encrypt');
        if (startEnc) startEnc.addEventListener('click', () => this.encryptFile());

        const startDec = document.getElementById('start-decrypt');
        if (startDec) startDec.addEventListener('click', () => this.decryptFile());

        const selfTest = document.getElementById('run-self-test');
        if (selfTest) selfTest.addEventListener('click', () => this.runSelfTest());

        const dlEnc = document.getElementById('download-encrypted');
        if (dlEnc) {
            dlEnc.addEventListener('click', () =>
                this.downloadFile(this.encryptedData, `${this.currentFile?.name || 'encrypted'}.nec`)
            );
        }

        const dlRest = document.getElementById('download-restored');
        if (dlRest) {
            dlRest.addEventListener('click', () =>
                this.downloadFile(this.restoredData, this.originalFileName || 'restored.bin')
            );
        }

        // File System Access buttons (optional; only work if IDs exist in HTML)
        const fsOpenEncrypt = document.getElementById('fs-open-encrypt');
        if (fsOpenEncrypt) {
            fsOpenEncrypt.addEventListener('click', () => this.pickFileWithFS('encrypt'));
        }

        const fsOverwriteEncrypted = document.getElementById('fs-overwrite-encrypted');
        if (fsOverwriteEncrypted) {
            fsOverwriteEncrypted.addEventListener('click', () => this.overwriteOriginalWithEncrypted());
        }

        const fsOpenDecrypt = document.getElementById('fs-open-decrypt');
        if (fsOpenDecrypt) {
            fsOpenDecrypt.addEventListener('click', () => this.pickFileWithFS('decrypt'));
        }

        const fsOverwriteDecrypted = document.getElementById('fs-overwrite-decrypted');
        if (fsOverwriteDecrypted) {
            fsOverwriteDecrypted.addEventListener('click', () => this.overwriteOriginalWithDecrypted());
        }

        this.createDemoFileButton();
    }

    createDemoFileButton() {
        const uploadArea = document.getElementById('encrypt-upload');
        if (!uploadArea) return;
        const demoButton = document.createElement('button');
        demoButton.className = 'btn btn--outline demo-btn';
        demoButton.textContent = 'Use Demo File (1KB Text)';
        demoButton.style.marginTop = '1rem';
        demoButton.addEventListener('click', () => {
            const demoText = 'This is a demo file for testing NEC AI-enhanced encryption system.\n'.repeat(25);
            const demoFile = new File([demoText], 'demo.txt', { type: 'text/plain' });
            this.handleFileSelect(demoFile, 'encrypt');
        });
        uploadArea.appendChild(demoButton);
    }

    switchTab(tab) {
        document.querySelectorAll('.tab-btn').forEach(btn =>
            btn.classList.toggle('active', btn.dataset.tab === tab)
        );
        document.querySelectorAll('.tab-content').forEach(c =>
            c.classList.toggle('active', c.id === `${tab}-tab`)
        );
    }

    setupFileUpload(type) {
        const area = document.getElementById(`${type}-upload`);
        const input = document.getElementById(`${type}-file`);
        if (!area || !input) return;

        area.addEventListener('click', (e) => {
            if (e.target.tagName !== 'BUTTON') input.click();
        });
        area.addEventListener('dragover', (e) => {
            e.preventDefault();
            area.classList.add('dragover');
        });
        area.addEventListener('dragleave', (e) => {
            e.preventDefault();
            area.classList.remove('dragover');
        });
        area.addEventListener('drop', (e) => {
            e.preventDefault();
            area.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length) this.handleFileSelect(files[0], type);
        });
        input.addEventListener('change', (e) => {
            if (e.target.files && e.target.files.length) this.handleFileSelect(e.target.files[0], type);
        });
    }

    async handleFileSelect(file, type) {
        if (file.size > this.crypto.MAX_FILE_SIZE) {
            this.showError(`File too large. Maximum size is ${this.crypto.MAX_FILE_SIZE / (1024 * 1024)} MB`);
            return;
        }

        try {
            if (type === 'encrypt') {
                this.currentFile = file;
                await this.analyzeFile(file);
                const btn = document.getElementById('start-encrypt');
                if (btn) btn.classList.remove('hidden');
            } else {
                this.encryptedFile = file;
                const cred = document.getElementById('decrypt-credentials');
                if (cred) cred.classList.remove('hidden');
                const btn = document.getElementById('start-decrypt');
                if (btn) btn.classList.remove('hidden');
            }
        } catch (err) {
            this.showError(`File processing error: ${err.message}`);
        }
    }

    async analyzeFile(file) {
        const data = await file.arrayBuffer();
        const analysis = this.crypto.analyzeFile(data);

        document.getElementById('encrypt-filename').textContent = file.name;
        document.getElementById('encrypt-filesize').textContent = this.formatFileSize(file.size);
        document.getElementById('encrypt-filetype').textContent = analysis.fileType;
        document.getElementById('encrypt-entropy').textContent = analysis.entropy.toFixed(3);

        let ratioText, partText, aiStatus;
        if (analysis.aiUsed) {
            ratioText = `${(analysis.corruptionRatio * 100).toFixed(3)}%`;
            partText = analysis.partitionCount;
            aiStatus = '✓ AI Enhanced';
        } else {
            ratioText = 'Dynamic (on encrypt)';
            partText = 'Dynamic (on encrypt)';
            aiStatus = '⚠ Heuristic Mode';
        }

        document.getElementById('encrypt-ratio').textContent = ratioText;
        document.getElementById('encrypt-partitions').textContent = partText;
        updateAIStatus(aiStatus);

        document.getElementById('encrypt-info').classList.remove('hidden');
        this.fileAnalysis = analysis;
    }

    // -------- Encrypt --------
    // -------- Encrypt --------
async encryptFile() {
    if (!this.currentFile) {
        this.showError('No file selected for encryption.');
        return;
    }
    const t0 = Date.now();
    this.showProgress('encrypt', 0);

    try {
        let corruptionRatio, partitionCount, bases;

        // --- AI ya heuristic se parameters ---
       if (this.fileAnalysis.aiUsed && this.fileAnalysis.corruptionRange && this.fileAnalysis.partitionRange) {
    const { min: minR, max: maxR } = this.fileAnalysis.corruptionRange;
    const { min: minP, max: maxP } = this.fileAnalysis.partitionRange;

    // 1) RANDOM corruption ratio within AI-safe range
    const lowR = Math.max(this.crypto.MIN_CORRUPTION_RATIO, minR);
    const highR = Math.min(this.crypto.MAX_CORRUPTION_RATIO, maxR);

    if (highR > lowR) {
        const rand = this.crypto.generateRandomBytes(4).reduce((a, b) => a * 256 + b, 0) / 0xFFFFFFFF;
        corruptionRatio = lowR + rand * (highR - lowR);
    } else {
        // Fallback: agar range galat aa gaya to centre use karo
        corruptionRatio = lowR;
    }

    // 2) RANDOM partition count within AI-safe range
    const lowP = Math.max(4, Math.floor(minP));
    const highP = Math.min(256, Math.floor(maxP));
    if (highP > lowP) {
        partitionCount = lowP + this.crypto.csprngInt(highP - lowP + 1);
    } else {
        partitionCount = lowP;
    }

    // 3) Bases random shuffle as before
    const shuffledBases = [...this.crypto.BASES].sort(() => 0.5 - Math.random());
    bases = [];
    for (let i = 0; i < partitionCount; i++) {
        bases.push(shuffledBases[i % shuffledBases.length]);
    }

    // UI me show karein approximate range/actual
    const ratioEl = document.getElementById('encrypt-ratio');
    if (ratioEl) {
        ratioEl.textContent =
            `${(corruptionRatio * 100).toFixed(3)}% ` +
            `(range ~${(lowR * 100).toFixed(3)}–${(highR * 100).toFixed(3)}%)`;
    }
    const partsEl = document.getElementById('encrypt-partitions');
    if (partsEl) {
        partsEl.textContent = `${partitionCount} (AI-range: ${lowP}–${highP})`;
    }

} else {
    // --- OLD heuristic branch jaisa ka taisa rehne do ---
    const randomVal =
        this.crypto.generateRandomBytes(4).reduce((a, b) => a * 256 + b, 0) / 0xFFFFFFFF;
    corruptionRatio =
        this.crypto.MIN_CORRUPTION_RATIO +
        randomVal * (this.crypto.MAX_CORRUPTION_RATIO - this.crypto.MIN_CORRUPTION_RATIO);

    partitionCount = 2 + this.crypto.csprngInt(7);

    const shuffledBases = [...this.crypto.BASES].sort(() => 0.5 - Math.random());
    bases = [];
    for (let i = 0; i < partitionCount; i++) {
        bases.push(shuffledBases[i % shuffledBases.length]);
    }

    document.getElementById('encrypt-ratio').textContent = `${(corruptionRatio * 100).toFixed(3)}%`;
    document.getElementById('encrypt-partitions').textContent = partitionCount;
}


        const salt = this.crypto.generateRandomBytes(32);
        const seed = this.crypto.generateRandomBytes(32);
        const keyString = this.crypto.generateKeyString();

        // ---- ORIGINAL DATA READ + HASH ----
        const originalData = new Uint8Array(await this.currentFile.arrayBuffer());
        const fileHash = await this.crypto.sha256(originalData); // hash of original file
        document.getElementById('original-hash').textContent = fileHash;

        const originalSize = originalData.length;

        // --- Decide streaming based on ORIGINAL size ---
        const STREAMING_THRESHOLD = 10 * 1024 * 1024; // 10 MB
        const willUseStreaming = originalSize > STREAMING_THRESHOLD;

        let data = originalData;
        let compressed = false;

        // ✅ Compression sirf chhoti / medium files ke liye (non-streaming)
        if (
            !willUseStreaming &&                                     // streaming nahi hai
            originalSize > this.crypto.COMPRESSION_THRESHOLD &&      // 6MB se bada
            this.crypto.canCompress()                                // pako available
        ) {
            console.log('encrypt: compressing before encryption (small/medium file)...');
            const compressedBytes = await this.crypto.compressBytes(originalData);

            if (
                compressedBytes &&
                compressedBytes.length > 0 &&
                compressedBytes.length < originalSize                // compression effective
            ) {
                data = compressedBytes;
                compressed = true;
                console.log('encrypt: compression applied', {
                    originalSize,
                    compressedSize: data.length
                });
            } else {
                data = originalData;
                compressed = false;
                console.warn('encrypt: compression not effective, using raw data', {
                    originalSize,
                    compressedSize: compressedBytes ? compressedBytes.length : 0
                });
            }
        } else {
            if (willUseStreaming) {
                console.log('encrypt: LARGE file -> streaming ON, compression OFF');
            } else {
                console.log('encrypt: compression skipped (below threshold or pako missing)');
            }
        }

        // --- Bit operations start from here on `data` (maybe compressed) ---
        const totalBits = data.length * 8;

        const partition = this.crypto.generateRandomPartition(
            Math.floor(totalBits * corruptionRatio),
            partitionCount,
            partitionCount
        );

        this.showProgress('encrypt', 25);

        const positionsGroups = [];
        for (let pIdx = 0; pIdx < partition.length; pIdx++) {
            const posList = await this.crypto.generateBitPositions(
                seed,
                fileHash,
                pIdx,
                [partition[pIdx]],
                totalBits
            );
            positionsGroups.push(posList.slice(0, partition[pIdx]));
        }

        let remainingBytes, encodedStrings;

        // 💾 Streaming sirf jab `data.length > 10MB`
        if (data.length > STREAMING_THRESHOLD) {
            console.log('encrypt: using streaming mode for large file');
            const result = await this.crypto.streamingRemoveBitsAndEncode(
                data,
                positionsGroups,
                bases
            );
            remainingBytes = result.remainingBytes;
            encodedStrings = result.encodedStrings;
        } else {
            const result = await this.crypto.removeBitsAndEncode(
                data,
                positionsGroups,
                bases
            );
            remainingBytes = result.remainingBytes;
            encodedStrings = result.encodedStrings;
        }

        const encryptedData = remainingBytes;

        this.showProgress('encrypt', 75);
        const encryptedHash = await this.crypto.sha256(encryptedData);
        document.getElementById('encrypted-hash').textContent = encryptedHash;

        const originalBitLength = data.length * 8; // data may be compressed

        const headerString = await this.crypto.createCompactKey(
            keyString,
            salt,
            this.crypto.KDF_ITERATIONS,
            partition,
            bases,
            fileHash, // original (uncompressed) file ka hash
            seed,
            encodedStrings,
            originalBitLength,
            compressed,
            originalSize
        );

        const preface = `NECHDR\n${headerString}\nENDHDR\n`;
        const prefaceBytes = new TextEncoder().encode(preface);
        const finalData = new Uint8Array(prefaceBytes.length + encryptedData.length);
        finalData.set(prefaceBytes, 0);
        finalData.set(encryptedData, prefaceBytes.length);
        this.encryptedData = finalData;

        this.showProgress('encrypt', 100);

        document.getElementById('encryption-key').value = keyString;
        document.getElementById('encrypt-results').classList.remove('hidden');

        const t1 = Date.now();
        const thr = (data.length / ((t1 - t0) || 1) / 1000) / (1024 * 1024);
        document.getElementById('encrypt-throughput').textContent = `${thr.toFixed(1)} MB/s`;

        const aiNote = this.fileAnalysis.aiUsed ? ' (AI-Enhanced)' : ' (Dynamic Heuristic)';
        const compNote = compressed ? ' + Compressed' : '';
        this.showSuccess(`File encrypted successfully${aiNote}${compNote}! Save your key and the .nec file.`);
        const fsOverwriteEncrypted = document.getElementById('fs-overwrite-encrypted');
        if (fsOverwriteEncrypted && this.encryptFileHandle && this.encryptedData) {
            fsOverwriteEncrypted.removeAttribute('disabled');
}

    } catch (err) {
        this.showError(`Encryption failed: ${err.message}`);
        const bar = document.getElementById('encrypt-progress');
        if (bar) bar.classList.add('hidden');
    }
}

    // -------- Decrypt --------
    async decryptFile() {
        if (!this.encryptedFile) {
            this.showError('No file selected for decryption.');
            return;
        }
        const keyString = document.getElementById('decrypt-key').value.trim();
        if (!keyString) {
            this.showError('Please provide the encryption key');
            return;
        }

        const t0 = Date.now();
        this.showProgress('decrypt', 0);

        try {
            let raw;
            try {
                raw = new Uint8Array(await this.encryptedFile.arrayBuffer());
            } catch (readErr) {
                console.error('decrypt: failed to read encryptedFile.arrayBuffer()', readErr);
                this.showError(
                    'Failed to read the selected file. Please re-select the .nec file (maybe moved or blocked).'
                );
                const input = document.getElementById('decrypt-file');
                if (input) {
                    try {
                        input.value = null;
                        input.click();
                    } catch (e) {}
                }
                return;
            }

            const startMarker = 'NECHDR\n';
            const endMarker = '\nENDHDR\n';
            const startMarkerBytes = new TextEncoder().encode(startMarker);
            const endMarkerBytes = new TextEncoder().encode(endMarker);

            const indexOfSubarray = (buf, pat, from = 0) => {
                const limit = buf.length - pat.length;
                outer: for (let i = from; i <= limit; i++) {
                    for (let j = 0; j < pat.length; j++)
                        if (buf[i + j] !== pat[j]) continue outer;
                    return i;
                }
                return -1;
            };

            let startIdx = indexOfSubarray(raw, startMarkerBytes, 0);
            if (startIdx < 0) throw new Error('Missing key header in file');

            let endIdx = indexOfSubarray(raw, endMarkerBytes, startIdx + startMarkerBytes.length);
            if (endIdx < 0) throw new Error('Corrupted key header in file (END marker missing)');

            const headerBytes = raw.slice(startIdx + startMarkerBytes.length, endIdx);
            const headerString = new TextDecoder().decode(headerBytes);
            const headerBytesLen = endIdx + endMarkerBytes.length;

            const keyData = await this.crypto.parseCompactKeyFromHeader(headerString, keyString);
            this.showProgress('decrypt', 25);

            const remainingBytes = raw.slice(headerBytesLen);

            const positionsGroups = [];
            const totalBits =
                keyData.originalBitLength ||
                (remainingBytes.length * 8 + keyData.partition.reduce((a, b) => a + b, 0));

            for (let pIdx = 0; pIdx < keyData.partition.length; pIdx++) {
                const posList = await this.crypto.generateBitPositions(
                    keyData.seed,
                    keyData.fileHash,
                    pIdx,
                    [keyData.partition[pIdx]],
                    totalBits
                );
                positionsGroups.push(posList.slice(0, keyData.partition[pIdx]));
            }

            let reconstructed;
            let usedStreaming = false;

            if (remainingBytes.length > 10 * 1024 * 1024) {
                console.log('decrypt: using streaming mode for large file');
                usedStreaming = true;
                reconstructed = await this.crypto.streamingDecodeAndInsertBits(
                    remainingBytes,
                    positionsGroups,
                    keyData.encodedStrings || [],
                    keyData.bases,
                    totalBits
                );
            } else {
                reconstructed = await this.crypto.decodeAndInsertBits(
                    remainingBytes,
                    positionsGroups,
                    keyData.encodedStrings || [],
                    keyData.bases
                );
            }

            // For non-streaming (small files) we may need to trim padding bits
            if (!usedStreaming && keyData.originalBitLength && keyData.originalBitLength < reconstructed.length * 8) {
                const bits = this.crypto.bytesToBits(reconstructed);
                const originalBits = bits.slice(0, keyData.originalBitLength);
                reconstructed = this.crypto.bitsToBytes(originalBits);
            }

            // If compressed, decompress now
           // If compressed, decompress now
let restored = reconstructed;
if (keyData.compressed) {
    if (!this.crypto.canCompress()) {
        throw new Error('File was compressed but decompressor is not available');
    }
    console.log('decrypt: decompressing restored data...');
    restored = await this.crypto.decompressBytes(reconstructed);
}

// Yahan 0-byte catch karein
if (!restored || restored.length === 0) {
    throw new Error('Decryption produced empty data (0 bytes). Key mismatch ya internal error.');
}

this.showProgress('decrypt', 80);

            const restoredHash = await this.crypto.sha256(restored);
            document.getElementById('verify-original-hash').textContent = keyData.fileHash;
            document.getElementById('verify-restored-hash').textContent = restoredHash;

            const ok = restoredHash === keyData.fileHash;
            const statusEl = document.getElementById('hash-match-status');
            statusEl.textContent = ok ? 'MATCH ✓' : 'MISMATCH ✗';
            statusEl.className = ok ? 'status status--success' : 'status status--error';

            document.getElementById('decrypt-verification').classList.remove('hidden');
            this.showProgress('decrypt', 100);

            if (ok) this.showSuccess('File decrypted and verified successfully!');
            else this.showError('Hash verification failed - wrong key or corrupted file');

            this.restoredData = restored;
            // restoredData set karne ke baad:
            const fsOverwriteDecrypted = document.getElementById('fs-overwrite-decrypted');
            if (fsOverwriteDecrypted && this.decryptFileHandle && this.restoredData) {
                fsOverwriteDecrypted.removeAttribute('disabled');
            }
            this.originalFileName = this.encryptedFile.name.replace(/\.nec$/i, '') || 'restored.bin';
            // Try smart post-processing (ZIP detection)
            this.tryPostProcessRestored(restored);

            const dlBtn = document.getElementById('download-restored');
            if (dlBtn) {
                dlBtn.classList.remove('hidden');
                dlBtn.removeAttribute('disabled');
                dlBtn.textContent = 'Download Restored File';
            }

            const link = document.getElementById('download-restored-link');
            if (link && this.restoredData) {
                const blob = new Blob([this.restoredData], { type: 'application/octet-stream' });
                const url = URL.createObjectURL(blob);
                link.href = url;
                link.download = this.originalFileName;
                link.classList.remove('hidden');
                link.addEventListener(
                    'click',
                    () => setTimeout(() => URL.revokeObjectURL(url), 1000),
                    { once: true }
                );
            }

            const t1 = Date.now();
            const thr = (restored.length / ((t1 - t0) || 1) / 1000) / (1024 * 1024);
            document.getElementById('decrypt-throughput').textContent = `${thr.toFixed(1)} MB/s`;
        } catch (err) {
            this.showError(`Decryption failed: ${err.message}`);
            const bar = document.getElementById('decrypt-progress');
            if (bar) bar.classList.add('hidden');
        }
    }
        isZipBytes(bytes) {
    if (!(bytes instanceof Uint8Array)) return false;
    if (bytes.length < 4) return false;
    // ZIP magic: 'PK\x03\x04'
    return bytes[0] === 0x50 && bytes[1] === 0x4B && bytes[2] === 0x03 && bytes[3] === 0x04;
}

async tryPostProcessRestored(restored) {
    const bytes = restored instanceof Uint8Array ? restored : new Uint8Array(restored);

    if (!this.isZipBytes(bytes)) {
        console.log('Restored data is not ZIP, skipping bundle extract UI.');
        return;
    }

    if (!window.JSZip) {
        this.showSuccess('Decrypted file is a ZIP bundle. Download and extract it locally.');
        return;
    }

    // Show a small panel in decrypt tab: "Extract Bundle"
    const panelId = 'zip-extract-panel';
    let panel = document.getElementById(panelId);
    if (!panel) {
        const active = document.querySelector('#decrypt-tab'); // ya jo bhi decrypt tab ka ID hai
        if (!active) return;
        panel = document.createElement('div');
        panel.id = panelId;
        panel.style.marginTop = '1rem';
        panel.innerHTML = `
            <div class="card">
                <div class="card__header">
                    <strong>ZIP Bundle Detected</strong>
                </div>
                <div class="card__body">
                    <p>This decrypted file is a ZIP archive. You can either download it or extract contained files here.</p>
                    <button id="download-zip-direct" class="btn btn--outline">Download ZIP</button>
                    <button id="extract-zip-here" class="btn btn--primary">Extract Files Here</button>
                    <div id="zip-file-list" style="margin-top: 0.75rem; font-size: 0.9rem;"></div>
                </div>
            </div>
        `;
        active.appendChild(panel);
    }

    const bytesCopy = bytes; // closure ke liye

    const dlZipBtn = document.getElementById('download-zip-direct');
    const extBtn = document.getElementById('extract-zip-here');

    if (dlZipBtn) {
        dlZipBtn.onclick = () => {
            this.downloadFile(bytesCopy, this.originalFileName || 'bundle.zip');
        };
    }

    if (extBtn) {
        extBtn.onclick = async () => {
            try {
                const zip = await JSZip.loadAsync(bytesCopy);
                const listDiv = document.getElementById('zip-file-list');
                if (!listDiv) return;

                const entries = [];
                zip.forEach((relativePath, file) => {
                    if (!file.dir) entries.push(relativePath);
                });

                if (entries.length === 0) {
                    listDiv.textContent = 'No files found inside ZIP.';
                    return;
                }

                listDiv.innerHTML = `<strong>Files in bundle:</strong><br>` +
                    entries.map(name => `• ${name}`).join('<br>');

                // Optionally: auto-generate download for each file
                for (const name of entries) {
                    const file = zip.file(name);
                    if (!file) continue;

                    const blob = await file.async('blob');
                    const a = document.createElement('a');
                    a.href = URL.createObjectURL(blob);
                    a.download = name;
                    a.textContent = `Download ${name}`;
                    a.style.display = 'block';
                    a.style.marginTop = '4px';
                    listDiv.appendChild(a);
                }

                this.showSuccess('Bundle extracted list ready. Click individual files to download.');

            } catch (e) {
                console.error('ZIP extract error', e);
                this.showError('Failed to extract ZIP: ' + e.message);
            }
        };
    }

    this.showSuccess('Decrypted data is a ZIP bundle. You can extract it here.');
}

    // -------- Self-test --------
    async runSelfTest() {
        const resultsDiv = document.getElementById('self-test-results');
        if (!resultsDiv) return;
        const statusDiv = resultsDiv.querySelector('.test-status');
        const detailsDiv = resultsDiv.querySelector('.test-details');
        if (!statusDiv || !detailsDiv) return;

        resultsDiv.classList.remove('hidden');
        statusDiv.textContent = 'Running self-test...';
        statusDiv.className = 'test-status pulse';

        try {
            const testData = this.crypto.generateTestData(1024 * 1024);
            const originalHash = await this.crypto.sha256(testData);
            const salt = this.crypto.generateRandomBytes(32);
            const seed = this.crypto.generateRandomBytes(32);
            const keyString = this.crypto.generateKeyString();
            const partition = this.crypto.generateRandomPartition(1000, 3, 5);
            const bases = [12, 16, 20];

            const positionsGroups = [];
            const totalBits = testData.length * 8;
            for (let pIdx = 0; pIdx < partition.length; pIdx++) {
                const posList = await this.crypto.generateBitPositions(
                    seed,
                    originalHash,
                    pIdx,
                    [partition[pIdx]],
                    totalBits
                );
                positionsGroups.push(posList.slice(0, partition[pIdx]));
            }

            const { remainingBytes, encodedStrings } = await this.crypto.removeBitsAndEncode(
                testData,
                positionsGroups,
                bases
            );

            const originalBitLength = testData.length * 8;
            const headerString = await this.crypto.createCompactKey(
                keyString,
                salt,
                this.crypto.KDF_ITERATIONS,
                partition,
                bases,
                originalHash,
                seed,
                encodedStrings,
                originalBitLength,
                false,
                testData.length
            );
            const keyData = await this.crypto.parseCompactKeyFromHeader(headerString, keyString);

            const positionsGroups2 = [];
            for (let pIdx = 0; pIdx < keyData.partition.length; pIdx++) {
                const posList = await this.crypto.generateBitPositions(
                    keyData.seed,
                    keyData.fileHash,
                    pIdx,
                    [keyData.partition[pIdx]],
                    totalBits
                );
                positionsGroups2.push(posList.slice(0, keyData.partition[pIdx]));
            }

            let restoredData = await this.crypto.decodeAndInsertBits(
                remainingBytes,
                positionsGroups2,
                keyData.encodedStrings || [],
                keyData.bases
            );

            if (keyData.originalBitLength && keyData.originalBitLength < restoredData.length * 8) {
                const bits = this.crypto.bytesToBits(restoredData);
                const originalBits = bits.slice(0, keyData.originalBitLength);
                restoredData = this.crypto.bitsToBytes(originalBits);
            }

            const restoredHash = await this.crypto.sha256(restoredData);
            const hashMatch = originalHash === restoredHash;
            const seedMatch = this.crypto.constantTimeEquals(seed, keyData.seed);
            const dataMatch = this.crypto.constantTimeEquals(testData, restoredData);
            const aiStatus = necModel ? 'Available' : 'Fallback Mode';

            if (hashMatch && dataMatch && seedMatch) {
                statusDiv.textContent = 'SELF-TEST PASSED ✓';
                statusDiv.className = 'test-status';
                statusDiv.style.color = 'var(--color-success)';
                detailsDiv.innerHTML = `
                    <strong>Test Results:</strong><br>
                    Test Size: 1 MB<br>
                    AI Model: ${aiStatus}<br>
                    Original Hash: ${originalHash.substring(0, 16)}...<br>
                    Restored Hash: ${restoredHash.substring(0, 16)}...<br>
                    Key Length: ${headerString.length} chars<br>
                    Partition: [${partition.join(', ')}]<br>
                    Data Match: ${dataMatch ? '✓' : '✗'}<br>
                    Hash Match: ${hashMatch ? '✓' : '✗'}<br>
                    Seed Recovery: ${seedMatch ? '✓' : '✗'}
                `;
                this.showSuccess('Self-test completed successfully!');
            } else {
                statusDiv.textContent = 'SELF-TEST FAILED ✗';
                statusDiv.className = 'test-status';
                statusDiv.style.color = 'var(--color-error)';
                detailsDiv.innerHTML = `
                    Data Match: ${dataMatch ? '✓' : '✗'}<br>
                    Hash Match: ${hashMatch ? '✓' : '✗'}<br>
                    Seed Recovery: ${seedMatch ? '✓' : '✗'}
                `;
                this.showError('Self-test failed - cryptographic operations may be compromised');
            }
        } catch (err) {
            const resultsDiv2 = document.getElementById('self-test-results');
            if (resultsDiv2) {
                const statusDiv2 = resultsDiv2.querySelector('.test-status');
                const detailsDiv2 = resultsDiv2.querySelector('.test-details');
                if (statusDiv2) {
                    statusDiv2.textContent = 'SELF-TEST ERROR ✗';
                    statusDiv2.className = 'test-status';
                    statusDiv2.style.color = 'var(--color-error)';
                }
                if (detailsDiv2) detailsDiv2.textContent = `Error: ${err.message}`;
            }
            this.showError(`Self-test error: ${err.message}`);
        }
    }

    // -------- File System Access helpers --------
    async pickFileWithFS(type) {
        if (!window.showOpenFilePicker) {
            alert('Your browser does not support File System Access API. Use normal upload instead.');
            return;
        }

        try {
            const [handle] = await window.showOpenFilePicker({
                multiple: false
            });
            const file = await handle.getFile();

            if (type === 'encrypt') {
                this.encryptFileHandle = handle;
            } else if (type === 'decrypt') {
                this.decryptFileHandle = handle;
            }

            await this.handleFileSelect(file, type);
        } catch (e) {
            console.error('File picker cancelled or failed', e);
        }
    }

    async overwriteOriginalWithEncrypted() {
        if (!this.encryptFileHandle || !this.encryptedData) {
            this.showError('No original file handle or encrypted data available');
            return;
        }

        try {
            const writable = await this.encryptFileHandle.createWritable();
            await writable.write(this.encryptedData);
            await writable.close();
            this.showSuccess('Original file overwritten with encrypted data (with your permission).');
        } catch (e) {
            console.error('overwriteOriginalWithEncrypted error', e);
            this.showError('Failed to overwrite original file: ' + e.message);
        }
    }

    async overwriteOriginalWithDecrypted() {
        if (!this.decryptFileHandle || !this.restoredData) {
            this.showError('No original file handle or restored data available');
            return;
        }

        try {
            const writable = await this.decryptFileHandle.createWritable();
            await writable.write(this.restoredData);
            await writable.close();
            this.showSuccess('Original file overwritten with decrypted data (with your permission).');
        } catch (e) {
            console.error('overwriteOriginalWithDecrypted error', e);
            this.showError('Failed to overwrite original file: ' + e.message);
        }
    }

    // -------- UI utilities --------
    showProgress(type, percent) {
        const fill = document.getElementById(`${type}-progress-fill`);
        const text = document.getElementById(`${type}-progress-text`);
        const bar = document.getElementById(`${type}-progress`);
        if (!fill || !text || !bar) return;
        bar.classList.remove('hidden');
        fill.style.width = `${percent}%`;
        text.textContent = `${Math.round(percent)}%`;
    }

    showError(message) {
        this.showMessage(message, 'error');
    }
    showSuccess(message) {
        this.showMessage(message, 'success');
    }

    showMessage(message, type) {
        document.querySelectorAll('.error-message, .success-message').forEach(el => el.remove());
        const div = document.createElement('div');
        div.className = `${type}-message fade-in`;
        div.textContent = message;
        div.style.cssText = `
            padding: 12px; margin: 12px 0; border-radius: 6px; font-size: 14px;
            background: rgba(${type === 'error' ? '192, 21, 47' : '33, 128, 141'}, 0.1);
            border: 1px solid rgba(${type === 'error' ? '192, 21, 47' : '33, 128, 141'}, 0.3);
            color: var(--color-${type === 'error' ? 'error' : 'success'});
        `;
        const active = document.querySelector('.tab-content.active');
        if (active) {
            active.insertBefore(div, active.firstChild);
        }
        setTimeout(() => {
            if (div.parentNode) div.parentNode.removeChild(div);
        }, 5000);
    }

    async copyToClipboard(elementId) {
        const el = document.getElementById(elementId);
        const btn = document.querySelector(`[data-target="${elementId}"]`);
        if (!el || !btn) return;

        try {
            await navigator.clipboard.writeText(el.value);
            const orig = btn.innerHTML;
            btn.textContent = 'Copied!';
            setTimeout(() => (btn.innerHTML = orig), 2000);
        } catch {
            el.select();
            document.execCommand('copy');
            const orig = btn.innerHTML;
            btn.textContent = 'Copied!';
            setTimeout(() => (btn.innerHTML = orig), 2000);
        }
    }

    downloadFile(data, filename) {
    if (!data) {
        this.showError('No data to download');
        return;
    }
    if (data.length === 0) {
        console.error('downloadFile: buffer is empty (0 bytes)', { filename });
        this.showError('Download failed: internal buffer is empty (0 bytes). Check encryption/decryption result.');
        return;
    }

    const blob = new Blob([data], { type: 'application/octet-stream' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
}


