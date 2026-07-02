/**
 * Gemini Watermark Removal Engine
 * 
 * Supports two methods:
 * 1. Reverse Alpha Blending - Mathematical precision for known watermark patterns
 * 2. LaMa AI Inpainting - AI-powered removal using ONNX model
 * 
 * Zero external dependencies (except ONNX Runtime for LaMa method)
 * 
 * Core detection logic adapted from:
 * https://github.com/GargantuaX/gemini-watermark-remover (MIT License)
 */

const WatermarkEngine = (function() {
    'use strict';

    // ==================== Constants ====================
    const ALPHA_NOISE_FLOOR = 3 / 255;
    const ALPHA_THRESHOLD = 0.002;
    const MAX_ALPHA = 0.99;
    const LOGO_VALUE = 255;
    const ALPHA_GAIN_CANDIDATES = Object.freeze([0.6, 1, 1.1, 1.15, 1.3, 0.45, 0.7, 0.85, 0.55]);
    const LARGE_MARGIN_ALPHA_GAIN_CANDIDATES = Object.freeze([0.25, 0.3, 0.35, 0.4, 0.55, 0.6, 0.7, 0.85, 1]);
    const MIN_ACCEPTED_CONFIDENCE = 0.08;
    const MIN_ACCEPTED_IMPROVEMENT = 0.04;
    const MAX_NEAR_BLACK_RATIO_INCREASE = 0.12;
    const SEARCH_CONFIG = {
        searchAreaRatio: 0.25,
        minConfidence: 0.3,
    };
    const MASK_CONFIG = {
        dilatePx: 8,
        featherPx: 3,
    };
    const LAMA_MODEL_URL = "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx?download=1";
    const MODEL_SIZE = 512;
    const LAMA_ROI_CONFIG = Object.freeze({
        minSize: 512,
        maxSize: 768,
        contextScale: 6,
        blendFeatherPx: 10,
    });
    const MODEL_VERSION = "v2";
    const WATERMARK_CONFIG_BY_TIER = Object.freeze({
        '0.5k': Object.freeze({ logoSize: 48, marginRight: 32, marginBottom: 32 }),
        '1k': Object.freeze({ logoSize: 96, marginRight: 64, marginBottom: 64 }),
        '2k': Object.freeze({ logoSize: 96, marginRight: 64, marginBottom: 64 }),
        '4k': Object.freeze({ logoSize: 96, marginRight: 64, marginBottom: 64 }),
        '2k-new-margin': Object.freeze({
            logoSize: 96,
            marginRight: 192,
            marginBottom: 192,
            alphaVariant: '20260520',
        }),
    });
    const GEMINI_3X_CURRENT_1K_WATERMARK_CONFIG = Object.freeze({
        logoSize: 48,
        marginRight: 32,
        marginBottom: 32,
    });
    const GEMINI_3X_LEGACY_1K_WATERMARK_CONFIG = Object.freeze({
        logoSize: 96,
        marginRight: 64,
        marginBottom: 64,
    });
    const GEMINI_3X_CURRENT_1K_LARGE_MARGIN_WATERMARK_CONFIG = Object.freeze({
        logoSize: 48,
        marginRight: 96,
        marginBottom: 96,
    });
    const GEMINI_3X_V2_SMALL_WATERMARK_CONFIG = Object.freeze({
        logoSize: 36,
        marginRight: 96,
        marginBottom: 96,
        alphaVariant: 'v2',
    });
    const KNOWN_FIXED_GEMINI_WATERMARK_CONFIGS_BY_SIZE = Object.freeze({
        '1408x768': Object.freeze([
            Object.freeze({ logoSize: 46, marginRight: 32, marginBottom: 32, fixedVariant: true }),
        ]),
    });
    const OFFICIAL_GEMINI_IMAGE_SIZES = Object.freeze([
        ...createGeminiSizeEntries('gemini-3.x-image', '0.5k', [
            ['1:1', 512, 512], ['1:4', 256, 1024], ['1:8', 192, 1536],
            ['2:3', 424, 632], ['3:2', 632, 424], ['3:4', 448, 600],
            ['4:1', 1024, 256], ['4:3', 600, 448], ['4:5', 464, 576],
            ['5:4', 576, 464], ['8:1', 1536, 192], ['9:16', 384, 688],
            ['16:9', 688, 384], ['21:9', 792, 168],
        ]),
        ...createGeminiSizeEntries('gemini-3.x-image', '1k', [
            ['1:1', 1024, 1024], ['1:4', 512, 2048], ['1:8', 384, 3072],
            ['2:3', 848, 1264], ['3:2', 1264, 848], ['3:4', 896, 1200],
            ['4:1', 2048, 512], ['4:3', 1200, 896], ['4:5', 928, 1152],
            ['5:4', 1152, 928], ['8:1', 3072, 384], ['9:16', 768, 1376],
            ['16:9', 1376, 768], ['16:9', 1408, 768], ['21:9', 1584, 672],
        ]),
        ...createGeminiSizeEntries('gemini-3.x-image', '2k', [
            ['1:1', 2048, 2048], ['1:4', 1024, 4096], ['1:8', 768, 6144],
            ['2:3', 1696, 2528], ['3:2', 2528, 1696], ['3:4', 1792, 2400],
            ['4:1', 4096, 1024], ['4:3', 2400, 1792], ['4:5', 1856, 2304],
            ['5:4', 2304, 1856], ['8:1', 6144, 768], ['9:16', 1536, 2752],
            ['16:9', 2752, 1536], ['21:9', 3168, 1344],
        ]),
        ...createGeminiSizeEntries('gemini-3.x-image', '2k-new-margin', [
            ['16:9', 2816, 1536],
        ]),
        ...createGeminiSizeEntries('gemini-3.x-image', '4k', [
            ['1:1', 4096, 4096], ['1:4', 2048, 8192], ['1:8', 1536, 12288],
            ['2:3', 3392, 5056], ['3:2', 5056, 3392], ['3:4', 3584, 4800],
            ['4:1', 8192, 2048], ['4:3', 4800, 3584], ['4:5', 3712, 4608],
            ['5:4', 4608, 3712], ['8:1', 12288, 1536], ['9:16', 3072, 5504],
            ['16:9', 5504, 3072], ['21:9', 6336, 2688],
        ]),
        ...createGeminiSizeEntries('gemini-2.5-flash-image', '1k', [
            ['1:1', 1024, 1024], ['2:3', 832, 1248], ['3:2', 1248, 832],
            ['3:4', 864, 1184], ['4:3', 1184, 864], ['4:5', 896, 1152],
            ['5:4', 1152, 896], ['9:16', 768, 1344], ['16:9', 1344, 768],
            ['21:9', 1536, 672],
        ]),
    ]);
    const OFFICIAL_GEMINI_IMAGE_SIZE_INDEX = (() => {
        const index = new Map();
        for (const entry of OFFICIAL_GEMINI_IMAGE_SIZES) {
            const key = `${entry.width}x${entry.height}`;
            if (!index.has(key)) {
                index.set(key, entry);
            }
        }
        return index;
    })();

    // ==================== State ====================
    let alphaMap48 = null;
    let alphaMap96 = null;
    let alphaMap96NewMargin = null;
    let alphaMap36V2 = null;
    const alphaMapCache = new Map();
    const negativeAlphaMapCache = new WeakMap();
    let initialized = false;
    let lamaWorker = null;
    let lamaReady = false;
    let lamaEP = 'wasm';
    let lamaLoadPromise = null;
    let lamaStatusCallback = null;

    // ==================== Utility Functions ====================

    /**
     * Load an image from a URL
     */
    function loadImageFromUrl(url) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = () => reject(new Error(`Failed to load image from ${url}`));
            img.src = url;
        });
    }

    /**
     * Load an image from a File object
     */
    function loadImageFromFile(file) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            const url = URL.createObjectURL(file);
            img.onload = () => {
                URL.revokeObjectURL(url);
                resolve(img);
            };
            img.onerror = () => {
                URL.revokeObjectURL(url);
                reject(new Error("Failed to load image"));
            };
            img.src = url;
        });
    }

    async function loadFloat32AlphaMapFromUrl(url, expectedLength) {
        const response = await fetch(url, { cache: 'force-cache' });
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const buffer = await response.arrayBuffer();
        if (buffer.byteLength !== expectedLength * 4) {
            throw new Error(`Expected ${expectedLength * 4} bytes, got ${buffer.byteLength}`);
        }

        return new Float32Array(buffer);
    }

    /**
     * Calculate alpha map from a watermark template image
     */
    function calculateAlphaMap(imageData) {
        const { width, height, data } = imageData;
        const alphaMap = new Float32Array(width * height);

        for (let i = 0; i < alphaMap.length; i++) {
            const idx = i * 4;
            const r = data[idx];
            const g = data[idx + 1];
            const b = data[idx + 2];
            const maxChannel = Math.max(r, g, b);
            alphaMap[i] = maxChannel / 255.0;
        }

        return alphaMap;
    }

    /**
     * Calculate correlation score between watermark template and image region
     */
    function calculateCorrelation(imageData, alphaMap, startX, startY, size) {
        const { width, data } = imageData;

        let avgBrightness = 0;
        let count = 0;

        for (let row = 0; row < size; row++) {
            for (let col = 0; col < size; col++) {
                const imgX = startX + col;
                const imgY = startY + row;

                if (imgX < 0 || imgX >= imageData.width || imgY < 0 || imgY >= imageData.height) {
                    continue;
                }

                const imgIdx = (imgY * width + imgX) * 4;
                const brightness = (data[imgIdx] + data[imgIdx + 1] + data[imgIdx + 2]) / 3;
                avgBrightness += brightness;
                count++;
            }
        }

        if (count === 0) return 0;
        avgBrightness /= count;

        let correlation = 0;
        let alphaSum = 0;

        for (let row = 0; row < size; row++) {
            for (let col = 0; col < size; col++) {
                const imgX = startX + col;
                const imgY = startY + row;

                if (imgX < 0 || imgX >= imageData.width || imgY < 0 || imgY >= imageData.height) {
                    continue;
                }

                const imgIdx = (imgY * width + imgX) * 4;
                const alphaIdx = row * size + col;
                const alpha = Math.abs(alphaMap[alphaIdx]);

                if (alpha > 0.05) {
                    const brightness = (data[imgIdx] + data[imgIdx + 1] + data[imgIdx + 2]) / 3;
                    const brightnessDeviation = brightness - avgBrightness;

                    if (brightnessDeviation > 0) {
                        correlation += alpha * (brightnessDeviation / 255);
                    }
                    alphaSum += alpha;
                }
            }
        }

        if (alphaSum === 0) return 0;
        return correlation / alphaSum;
    }

    /**
     * Search for watermark in the image using template matching
     */
    function searchWatermark(imageData, alphaMap, size) {
        const { width, height } = imageData;

        const searchWidth = Math.floor(width * SEARCH_CONFIG.searchAreaRatio);
        const searchHeight = Math.floor(height * SEARCH_CONFIG.searchAreaRatio);

        const startSearchX = width - searchWidth;
        const startSearchY = height - searchHeight;

        let bestX = width - size - 32;
        let bestY = height - size - 32;
        let bestScore = -Infinity;

        const step = Math.max(1, Math.floor(size / 8));

        // Coarse search
        for (let y = startSearchY; y <= height - size; y += step) {
            for (let x = startSearchX; x <= width - size; x += step) {
                const score = calculateCorrelation(imageData, alphaMap, x, y, size);
                if (score > bestScore) {
                    bestScore = score;
                    bestX = x;
                    bestY = y;
                }
            }
        }

        // Fine search around best position
        const refineRange = step * 2;
        for (let y = bestY - refineRange; y <= bestY + refineRange; y++) {
            for (let x = bestX - refineRange; x <= bestX + refineRange; x++) {
                if (x < 0 || x > width - size || y < 0 || y > height - size) continue;
                const score = calculateCorrelation(imageData, alphaMap, x, y, size);
                if (score > bestScore) {
                    bestScore = score;
                    bestX = x;
                    bestY = y;
                }
            }
        }

        return {
            x: bestX,
            y: bestY,
            confidence: Math.max(0, Math.min(1, bestScore * 2)),
        };
    }

    /**
     * Apply reverse alpha blending to remove watermark.
     * Formula: original = (watermarked - alpha * logoValue) / (1 - alpha)
     */
    function removeWatermark(imageData, alphaMap, position, watermarkSize, options = {}) {
        const { width, height, data } = imageData;
        const regionWidth = position.width || watermarkSize;
        const regionHeight = position.height || watermarkSize;
        const mapSize = watermarkSize || regionWidth;
        const alphaGain = Number.isFinite(options.alphaGain) && options.alphaGain > 0
            ? options.alphaGain
            : 1;

        for (let row = 0; row < regionHeight; row++) {
            for (let col = 0; col < regionWidth; col++) {
                const imgX = position.x + col;
                const imgY = position.y + row;

                if (imgX < 0 || imgX >= width || imgY < 0 || imgY >= height) {
                    continue;
                }

                const imgIdx = (imgY * width + imgX) * 4;
                const alphaIdx = row * mapSize + col;
                const rawAlpha = alphaMap[alphaIdx] || 0;
                const alphaMagnitude = Math.abs(rawAlpha);
                const signalAlpha = Math.max(0, alphaMagnitude - ALPHA_NOISE_FLOOR) * alphaGain;

                if (signalAlpha < ALPHA_THRESHOLD) {
                    continue;
                }

                const logoValue = Number.isFinite(options.logoValue)
                    ? options.logoValue
                    : (rawAlpha < 0 ? 0 : LOGO_VALUE);
                const alpha = Math.min(alphaMagnitude * alphaGain, MAX_ALPHA);
                const oneMinusAlpha = 1.0 - alpha;

                for (let c = 0; c < 3; c++) {
                    const watermarked = data[imgIdx + c];
                    const original = (watermarked - alpha * logoValue) / oneMinusAlpha;
                    data[imgIdx + c] = Math.max(0, Math.min(255, Math.round(original)));
                }
            }
        }
    }

    function createGeminiSizeEntries(modelFamily, resolutionTier, rows) {
        return rows.map(([aspectRatio, width, height]) => ({
            modelFamily,
            resolutionTier,
            aspectRatio,
            width,
            height,
        }));
    }

    function normalizeDimension(value) {
        const numeric = Number(value);
        if (!Number.isFinite(numeric)) return null;
        const rounded = Math.round(numeric);
        return rounded > 0 ? rounded : null;
    }

    function clamp(value, min, max) {
        return Math.max(min, Math.min(max, value));
    }

    function buildConfigKey(config) {
        return [
            config.logoSize,
            config.marginRight,
            config.marginBottom,
            config.alphaVariant || 'default',
            config.fixedVariant ? 'fixed' : 'standard',
        ].join(':');
    }

    function matchOfficialGeminiImageSize(width, height) {
        const normalizedWidth = normalizeDimension(width);
        const normalizedHeight = normalizeDimension(height);
        if (!normalizedWidth || !normalizedHeight) return null;
        return OFFICIAL_GEMINI_IMAGE_SIZE_INDEX.get(`${normalizedWidth}x${normalizedHeight}`) || null;
    }

    function getEntryConfig(entry) {
        if (entry?.modelFamily === 'gemini-3.x-image' && entry.resolutionTier === '1k') {
            return GEMINI_3X_CURRENT_1K_WATERMARK_CONFIG;
        }
        return WATERMARK_CONFIG_BY_TIER[entry?.resolutionTier] || null;
    }

    function detectWatermarkConfigBySize(imageWidth, imageHeight) {
        const match = matchOfficialGeminiImageSize(imageWidth, imageHeight);
        const officialConfig = getEntryConfig(match);
        if (officialConfig) {
            return { ...officialConfig };
        }

        if (imageWidth > 1024 && imageHeight > 1024) {
            return {
                logoSize: 96,
                marginRight: 64,
                marginBottom: 64,
            };
        }

        return {
            logoSize: 48,
            marginRight: 32,
            marginBottom: 32,
        };
    }

    function calculateWatermarkPosition(imageWidth, imageHeight, config) {
        const { logoSize, marginRight, marginBottom } = config;
        return {
            x: imageWidth - marginRight - logoSize,
            y: imageHeight - marginBottom - logoSize,
            width: logoSize,
            height: logoSize,
        };
    }

    function isRegionInsideImage(imageData, region) {
        return region.x >= 0 &&
            region.y >= 0 &&
            region.x + region.width <= imageData.width &&
            region.y + region.height <= imageData.height;
    }

    function createNewMarginVariantConfig(baseConfig, width, height) {
        if (!baseConfig || baseConfig.logoSize !== 96) return null;
        if (baseConfig.marginRight === 192 && baseConfig.marginBottom === 192) return null;

        const config = {
            logoSize: 96,
            marginRight: 192,
            marginBottom: 192,
            alphaVariant: '20260520',
        };
        const position = calculateWatermarkPosition(width, height, config);
        return position.x >= 0 && position.y >= 0 ? config : null;
    }

    function createCurrentLargeMarginVariantConfig(baseConfig, width, height, { allowAnyBase = false } = {}) {
        if (!allowAnyBase && (!baseConfig || baseConfig.logoSize !== 48)) return null;
        if (baseConfig?.marginRight === 96 && baseConfig?.marginBottom === 96) return null;

        const config = { ...GEMINI_3X_CURRENT_1K_LARGE_MARGIN_WATERMARK_CONFIG };
        const position = calculateWatermarkPosition(width, height, config);
        return position.x >= 0 && position.y >= 0 ? config : null;
    }

    function createV2SmallVariantConfig(width, height) {
        const normalizedWidth = normalizeDimension(width);
        const normalizedHeight = normalizeDimension(height);
        if (!normalizedWidth || !normalizedHeight) return null;
        if (Math.max(normalizedWidth, normalizedHeight) > 2048) return null;

        const longSide = Math.max(normalizedWidth, normalizedHeight);
        const shortSide = Math.min(normalizedWidth, normalizedHeight);
        const sourceLongDim = shortSide >= 566 ? 2752 : (shortSide >= 550 ? 2816 : 2848);
        const margin = Math.round(192 * (longSide / sourceLongDim));
        const config = {
            ...GEMINI_3X_V2_SMALL_WATERMARK_CONFIG,
            marginRight: margin,
            marginBottom: margin,
        };
        const position = calculateWatermarkPosition(normalizedWidth, normalizedHeight, config);
        return position.x >= 0 && position.y >= 0 ? config : null;
    }

    function createProjectedConfig(baseConfig, scaleX, scaleY, {
        minLogoSize,
        maxLogoSize,
        roundLogoSize = Math.round,
    }) {
        const logoSize = clamp(
            roundLogoSize(baseConfig.logoSize * ((scaleX + scaleY) / 2)),
            minLogoSize,
            maxLogoSize
        );
        return {
            logoSize,
            marginRight: Math.max(8, Math.round(baseConfig.marginRight * scaleX)),
            marginBottom: Math.max(8, Math.round(baseConfig.marginBottom * scaleY)),
            ...(baseConfig.alphaVariant ? { alphaVariant: baseConfig.alphaVariant } : {}),
        };
    }

    function collectNearOfficialProjectedEntries(width, height, limit = 3) {
        const normalizedWidth = normalizeDimension(width);
        const normalizedHeight = normalizeDimension(height);
        if (!normalizedWidth || !normalizedHeight) return [];

        const targetAspectRatio = normalizedWidth / normalizedHeight;
        const entries = [];

        for (const official of OFFICIAL_GEMINI_IMAGE_SIZES) {
            const baseConfig = getEntryConfig(official);
            if (!baseConfig) continue;

            const scaleX = normalizedWidth / official.width;
            const scaleY = normalizedHeight / official.height;
            const scale = (scaleX + scaleY) / 2;
            const officialAspectRatio = official.width / official.height;
            const relativeAspectRatioDelta = Math.abs(targetAspectRatio - officialAspectRatio) / officialAspectRatio;
            const scaleMismatchRatio = Math.abs(scaleX - scaleY) / Math.max(scaleX, scaleY);
            if (relativeAspectRatioDelta > 0.02 || scaleMismatchRatio > 0.12) continue;

            const projections = [{ config: baseConfig, source: `${official.width}x${official.height}` }];
            if (official.modelFamily === 'gemini-3.x-image' && official.resolutionTier === '1k') {
                projections.push({
                    config: GEMINI_3X_CURRENT_1K_LARGE_MARGIN_WATERMARK_CONFIG,
                    source: `${official.width}x${official.height}-large-margin`,
                    roundLogoSize: Math.ceil,
                });
            }

            for (const projection of projections) {
                const config = createProjectedConfig(projection.config, scaleX, scaleY, {
                    minLogoSize: 24,
                    maxLogoSize: 192,
                    roundLogoSize: projection.roundLogoSize || Math.round,
                });
                const position = calculateWatermarkPosition(normalizedWidth, normalizedHeight, config);
                if (position.x < 0 || position.y < 0) continue;

                entries.push({
                    config,
                    source: `near-official:${projection.source}`,
                    sourcePriority: 4,
                    score:
                        relativeAspectRatioDelta * 100 +
                        scaleMismatchRatio * 20 +
                        Math.abs(Math.log2(Math.max(scale, 1e-6))),
                });
            }
        }

        return entries
            .sort((a, b) => a.score - b.score)
            .slice(0, limit);
    }

    function resolveWatermarkSearchEntries(width, height) {
        const entries = [];
        const addEntry = (config, source, sourcePriority) => {
            if (!config) return;
            const position = calculateWatermarkPosition(width, height, config);
            if (position.x < 0 || position.y < 0) return;
            entries.push({ config: { ...config }, source, sourcePriority });
        };

        const defaultConfig = detectWatermarkConfigBySize(width, height);
        const exactMatch = matchOfficialGeminiImageSize(width, height);
        addEntry(defaultConfig, 'default-config', 0);

        const knownFixed = KNOWN_FIXED_GEMINI_WATERMARK_CONFIGS_BY_SIZE[`${width}x${height}`] || [];
        knownFixed.forEach((config) => addEntry(config, 'known-fixed-size', 2));

        if (exactMatch) {
            const officialConfig = getEntryConfig(exactMatch);
            addEntry(officialConfig, 'official-size', 0);

            if (exactMatch.modelFamily === 'gemini-3.x-image' && exactMatch.resolutionTier === '1k') {
                addEntry(
                    createCurrentLargeMarginVariantConfig(officialConfig, width, height),
                    '202606-large-margin',
                    1
                );
                addEntry(createV2SmallVariantConfig(width, height), 'allenk-v2-small', 2);
                addEntry(GEMINI_3X_LEGACY_1K_WATERMARK_CONFIG, 'legacy-96px', 3);
            } else {
                addEntry(createNewMarginVariantConfig(officialConfig, width, height), '20260520-new-margin', 3);
            }
        } else {
            collectNearOfficialProjectedEntries(width, height).forEach((entry) => {
                addEntry(entry.config, entry.source, entry.sourcePriority);
            });
            addEntry(createNewMarginVariantConfig(defaultConfig, width, height), 'unknown-size-new-margin', 2);
            addEntry(
                createCurrentLargeMarginVariantConfig(defaultConfig, width, height, { allowAnyBase: true }),
                'unknown-size-large-margin',
                1
            );
        }

        const deduped = [];
        const seen = new Set();
        for (const entry of entries.sort((a, b) => a.sourcePriority - b.sourcePriority)) {
            const key = buildConfigKey(entry.config);
            if (seen.has(key)) continue;
            seen.add(key);
            deduped.push(entry);
        }
        return deduped;
    }

    function interpolateAlphaMap(sourceAlpha, sourceSize, targetSize) {
        if (targetSize <= 0) return new Float32Array(0);
        if (sourceSize === targetSize) return new Float32Array(sourceAlpha);

        const out = new Float32Array(targetSize * targetSize);
        const scale = (sourceSize - 1) / Math.max(1, targetSize - 1);

        for (let y = 0; y < targetSize; y++) {
            const sy = y * scale;
            const y0 = Math.floor(sy);
            const y1 = Math.min(sourceSize - 1, y0 + 1);
            const fy = sy - y0;

            for (let x = 0; x < targetSize; x++) {
                const sx = x * scale;
                const x0 = Math.floor(sx);
                const x1 = Math.min(sourceSize - 1, x0 + 1);
                const fx = sx - x0;

                const p00 = sourceAlpha[y0 * sourceSize + x0];
                const p10 = sourceAlpha[y0 * sourceSize + x1];
                const p01 = sourceAlpha[y1 * sourceSize + x0];
                const p11 = sourceAlpha[y1 * sourceSize + x1];
                const top = p00 + (p10 - p00) * fx;
                const bottom = p01 + (p11 - p01) * fx;
                out[y * targetSize + x] = top + (bottom - top) * fy;
            }
        }

        return out;
    }

    function resolveAlphaMapForSize(size) {
        if (size === 48) return alphaMap48;
        if (size === 96) return alphaMap96;

        const key = String(size);
        if (alphaMapCache.has(key)) return alphaMapCache.get(key);

        const interpolated = interpolateAlphaMap(alphaMap96, 96, size);
        alphaMapCache.set(key, interpolated);
        return interpolated;
    }

    function resolveAlphaMapForConfig(config) {
        if (config.alphaVariant === '20260520' && alphaMap96NewMargin) {
            return alphaMap96NewMargin;
        }
        if (config.alphaVariant === 'v2' && config.logoSize === 36 && alphaMap36V2) {
            return alphaMap36V2;
        }
        return resolveAlphaMapForSize(config.logoSize);
    }

    function createNegativeAlphaMap(alphaMap) {
        if (!alphaMap) return null;
        const cached = negativeAlphaMapCache.get(alphaMap);
        if (cached) return cached;

        const negative = new Float32Array(alphaMap.length);
        for (let i = 0; i < alphaMap.length; i++) {
            negative[i] = -alphaMap[i];
        }
        negativeAlphaMapCache.set(alphaMap, negative);
        return negative;
    }

    function cloneImageData(imageData) {
        return new ImageData(
            new Uint8ClampedArray(imageData.data),
            imageData.width,
            imageData.height
        );
    }

    function createRegionImageData(sourceImageData, position) {
        const region = {
            width: position.width,
            height: position.height,
            data: new Uint8ClampedArray(position.width * position.height * 4),
        };

        for (let row = 0; row < position.height; row++) {
            const srcStart = ((position.y + row) * sourceImageData.width + position.x) * 4;
            const srcEnd = srcStart + position.width * 4;
            const dstStart = row * position.width * 4;
            region.data.set(sourceImageData.data.subarray(srcStart, srcEnd), dstStart);
        }

        return region;
    }

    function toRegionGrayscale(imageData, region) {
        const size = region.size || region.width || region.height;
        if (!size || size <= 0) return new Float32Array(0);
        if (region.x < 0 || region.y < 0 || region.x + size > imageData.width || region.y + size > imageData.height) {
            return new Float32Array(0);
        }

        const out = new Float32Array(size * size);
        for (let row = 0; row < size; row++) {
            for (let col = 0; col < size; col++) {
                const idx = ((region.y + row) * imageData.width + (region.x + col)) * 4;
                out[row * size + col] =
                    (0.2126 * imageData.data[idx] +
                        0.7152 * imageData.data[idx + 1] +
                        0.0722 * imageData.data[idx + 2]) / 255;
            }
        }
        return out;
    }

    function meanAndVariance(values) {
        let sum = 0;
        for (let i = 0; i < values.length; i++) sum += values[i];
        const mean = values.length > 0 ? sum / values.length : 0;

        let sq = 0;
        for (let i = 0; i < values.length; i++) {
            const d = values[i] - mean;
            sq += d * d;
        }

        return {
            mean,
            variance: values.length > 0 ? sq / values.length : 0,
        };
    }

    function normalizedCrossCorrelation(a, b) {
        if (a.length !== b.length || a.length === 0) return 0;

        const statsA = meanAndVariance(a);
        const statsB = meanAndVariance(b);
        const den = Math.sqrt(statsA.variance * statsB.variance) * a.length;
        if (den < 1e-8) return 0;

        let num = 0;
        for (let i = 0; i < a.length; i++) {
            num += (a[i] - statsA.mean) * (b[i] - statsB.mean);
        }
        return num / den;
    }

    function sobelMagnitude(values, width, height) {
        const out = new Float32Array(width * height);

        for (let y = 1; y < height - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                const i = y * width + x;
                const gx =
                    -values[i - width - 1] - 2 * values[i - 1] - values[i + width - 1] +
                    values[i - width + 1] + 2 * values[i + 1] + values[i + width + 1];
                const gy =
                    -values[i - width - 1] - 2 * values[i - width] - values[i - width + 1] +
                    values[i + width - 1] + 2 * values[i + width] + values[i + width + 1];
                out[i] = Math.sqrt(gx * gx + gy * gy);
            }
        }

        return out;
    }

    function computeRegionSpatialCorrelation(imageData, alphaMap, position) {
        const patch = toRegionGrayscale(imageData, position);
        if (patch.length === 0 || patch.length !== alphaMap.length) return 0;
        return normalizedCrossCorrelation(patch, alphaMap);
    }

    function computeRegionGradientCorrelation(imageData, alphaMap, position) {
        const patch = toRegionGrayscale(imageData, position);
        const size = position.size || position.width || position.height;
        if (patch.length === 0 || patch.length !== alphaMap.length || !size || size <= 2) return 0;

        const patchGrad = sobelMagnitude(patch, size, size);
        const alphaGrad = sobelMagnitude(alphaMap, size, size);
        return normalizedCrossCorrelation(patchGrad, alphaGrad);
    }

    function scoreRegion(imageData, alphaMap, position) {
        const spatialScore = computeRegionSpatialCorrelation(imageData, alphaMap, position);
        const gradientScore = computeRegionGradientCorrelation(imageData, alphaMap, position);
        return { spatialScore, gradientScore };
    }

    function calculateNearBlackRatio(imageData, position) {
        let nearBlack = 0;
        let total = 0;

        for (let row = 0; row < position.height; row++) {
            for (let col = 0; col < position.width; col++) {
                const x = position.x + col;
                const y = position.y + row;
                if (x < 0 || y < 0 || x >= imageData.width || y >= imageData.height) continue;

                const idx = (y * imageData.width + x) * 4;
                const r = imageData.data[idx];
                const g = imageData.data[idx + 1];
                const b = imageData.data[idx + 2];
                if (r <= 5 && g <= 5 && b <= 5) {
                    nearBlack++;
                }
                total++;
            }
        }

        return total > 0 ? nearBlack / total : 0;
    }

    function getAlphaGainCandidates(entry) {
        const config = entry.config;
        const isCurrentLargeMargin =
            config.logoSize === 48 &&
            config.marginRight >= 90 &&
            config.marginBottom >= 90;

        return isCurrentLargeMargin
            ? LARGE_MARGIN_ALPHA_GAIN_CANDIDATES
            : ALPHA_GAIN_CANDIDATES;
    }

    function createCandidateRegionAfterRemoval(originalImageData, alphaMap, position, alphaGain) {
        const regionImageData = createRegionImageData(originalImageData, position);
        removeWatermark(regionImageData, alphaMap, {
            x: 0,
            y: 0,
            width: position.width,
            height: position.height,
        }, position.width, { alphaGain });
        return regionImageData;
    }

    function evaluateWatermarkCandidate(originalImageData, entry, alphaMap, alphaGain) {
        const position = calculateWatermarkPosition(
            originalImageData.width,
            originalImageData.height,
            entry.config
        );
        if (!isRegionInsideImage(originalImageData, position)) return null;

        const originalScores = scoreRegion(originalImageData, alphaMap, position);
        const baselineNearBlackRatio = calculateNearBlackRatio(originalImageData, position);
        const regionImageData = createCandidateRegionAfterRemoval(originalImageData, alphaMap, position, alphaGain);
        const regionPosition = { x: 0, y: 0, width: position.width, height: position.height };
        const processedScores = scoreRegion(regionImageData, alphaMap, regionPosition);
        const nearBlackRatio = calculateNearBlackRatio(regionImageData, regionPosition);
        const nearBlackIncrease = nearBlackRatio - baselineNearBlackRatio;
        const improvement = originalScores.spatialScore - processedScores.spatialScore;
        const gradientIncrease = processedScores.gradientScore - originalScores.gradientScore;
        const confidence = clamp(
            Math.max(0, originalScores.spatialScore) * 0.55 +
            Math.max(0, originalScores.gradientScore) * 0.35 +
            Math.max(0, improvement) * 0.1,
            0,
            1
        );
        const originalEvidence =
            originalScores.spatialScore >= 0.05 ||
            originalScores.gradientScore >= 0.08 ||
            confidence >= MIN_ACCEPTED_CONFIDENCE;
        const strongOriginalEvidence =
            originalScores.spatialScore >= 0.9 &&
            originalScores.gradientScore >= 0.9;
        const residualAcceptable =
            Math.abs(processedScores.spatialScore) <= 0.45 ||
            processedScores.gradientScore <= originalScores.gradientScore + 0.04;
        const accepted =
            (
                originalEvidence &&
                improvement >= MIN_ACCEPTED_IMPROVEMENT &&
                residualAcceptable &&
                nearBlackIncrease <= MAX_NEAR_BLACK_RATIO_INCREASE
            ) ||
            (
                strongOriginalEvidence &&
                improvement >= 0.5 &&
                nearBlackIncrease <= 0.4
            );
        const validationCost =
            Math.abs(processedScores.spatialScore) +
            Math.max(0, processedScores.gradientScore) * 0.6 +
            Math.max(0, nearBlackIncrease) * 3 +
            Math.max(0, gradientIncrease) * 0.25;

        return {
            accepted,
            source: entry.source,
            sourcePriority: entry.sourcePriority,
            config: entry.config,
            position,
            size: position.width,
            alphaMap,
            alphaGain,
            confidence,
            validationCost,
            improvement,
            nearBlackIncrease,
            strongOriginalEvidence,
            originalSpatialScore: originalScores.spatialScore,
            originalGradientScore: originalScores.gradientScore,
            processedSpatialScore: processedScores.spatialScore,
            processedGradientScore: processedScores.gradientScore,
        };
    }

    function pickBetterCandidate(currentBest, candidate) {
        if (!candidate) return currentBest;
        if (!currentBest) return candidate;
        if (candidate.accepted !== currentBest.accepted) {
            return candidate.accepted ? candidate : currentBest;
        }

        if (candidate.accepted) {
            const candidateIsV2Small = candidate.config?.alphaVariant === 'v2' && candidate.size === 36;
            const currentIsV2Small = currentBest.config?.alphaVariant === 'v2' && currentBest.size === 36;
            if (candidateIsV2Small !== currentIsV2Small) {
                const v2Candidate = candidateIsV2Small ? candidate : currentBest;
                const otherCandidate = candidateIsV2Small ? currentBest : candidate;
                const shouldPreferV2 =
                    v2Candidate.confidence >= 0.12 &&
                    Math.abs(v2Candidate.processedSpatialScore) <= 0.12 &&
                    !otherCandidate.strongOriginalEvidence;
                if (shouldPreferV2) {
                    return v2Candidate;
                }
            }
            if (candidate.strongOriginalEvidence !== currentBest.strongOriginalEvidence) {
                return candidate.strongOriginalEvidence ? candidate : currentBest;
            }
            const candidateOriginalScore =
                Math.max(0, candidate.originalSpatialScore) +
                Math.max(0, candidate.originalGradientScore) * 0.8;
            const currentOriginalScore =
                Math.max(0, currentBest.originalSpatialScore) +
                Math.max(0, currentBest.originalGradientScore) * 0.8;
            if (
                candidateOriginalScore >= currentOriginalScore + 0.35 &&
                candidate.confidence >= currentBest.confidence + 0.25 &&
                Math.abs(candidate.processedSpatialScore) <= 0.35
            ) {
                return candidate;
            }
            const costDelta = candidate.validationCost - currentBest.validationCost;
            if (Math.abs(costDelta) > 0.005) {
                return costDelta < 0 ? candidate : currentBest;
            }
            if (candidate.sourcePriority !== currentBest.sourcePriority) {
                return candidate.sourcePriority < currentBest.sourcePriority ? candidate : currentBest;
            }
            return candidate.improvement > currentBest.improvement ? candidate : currentBest;
        }

        return candidate.confidence > currentBest.confidence ? candidate : currentBest;
    }

    function evaluateEntryWithPolarity(originalImageData, entry) {
        const alphaMap = resolveAlphaMapForConfig(entry.config);
        if (!alphaMap) return null;

        let best = null;
        for (const alphaGain of getAlphaGainCandidates(entry)) {
            best = pickBetterCandidate(
                best,
                evaluateWatermarkCandidate(originalImageData, entry, alphaMap, alphaGain)
            );
        }

        const negativeAlphaMap = createNegativeAlphaMap(alphaMap);
        for (const alphaGain of getAlphaGainCandidates(entry)) {
            const darkCandidate = evaluateWatermarkCandidate(
                originalImageData,
                {
                    ...entry,
                    source: `${entry.source}+dark`,
                    sourcePriority: entry.sourcePriority + 1,
                },
                negativeAlphaMap,
                alphaGain
            );
            best = pickBetterCandidate(best, darkCandidate);
        }

        return best;
    }

    function shouldRefineCandidatePosition(candidate, entry) {
        if (!candidate || !entry) return false;
        if (candidate.strongOriginalEvidence) return false;
        if (String(entry.source).includes('+refined')) return false;

        return candidate.confidence < 0.45 ||
            String(entry.source).includes('unknown-size') ||
            String(entry.source).includes('near-official');
    }

    function refineEntryByCorrelation(originalImageData, entry) {
        if (!entry?.config) return null;

        const alphaMap = resolveAlphaMapForConfig(entry.config);
        if (!alphaMap) return null;

        const size = entry.config.logoSize;
        const anchor = calculateWatermarkPosition(
            originalImageData.width,
            originalImageData.height,
            entry.config
        );
        if (!isRegionInsideImage(originalImageData, anchor)) return null;

        const anchorScore = Math.abs(computeRegionSpatialCorrelation(originalImageData, alphaMap, anchor));
        const searchRadius = size >= 96 ? 96 : 56;
        let best = {
            x: anchor.x,
            y: anchor.y,
            score: anchorScore,
        };

        const scan = (fromX, toX, fromY, toY, step) => {
            for (let y = fromY; y <= toY; y += step) {
                for (let x = fromX; x <= toX; x += step) {
                    const position = { x, y, width: size, height: size };
                    if (!isRegionInsideImage(originalImageData, position)) continue;
                    const score = Math.abs(computeRegionSpatialCorrelation(originalImageData, alphaMap, position));
                    if (score > best.score) {
                        best = { x, y, score };
                    }
                }
            }
        };

        scan(
            Math.max(0, anchor.x - searchRadius),
            Math.min(originalImageData.width - size, anchor.x + searchRadius),
            Math.max(0, anchor.y - searchRadius),
            Math.min(originalImageData.height - size, anchor.y + searchRadius),
            8
        );
        scan(
            Math.max(0, best.x - 10),
            Math.min(originalImageData.width - size, best.x + 10),
            Math.max(0, best.y - 10),
            Math.min(originalImageData.height - size, best.y + 10),
            2
        );

        const moved = Math.abs(best.x - anchor.x) > 1 || Math.abs(best.y - anchor.y) > 1;
        if (!moved || best.score < 0.35 || best.score < anchorScore + 0.12) {
            return null;
        }

        return {
            config: {
                ...entry.config,
                marginRight: originalImageData.width - best.x - size,
                marginBottom: originalImageData.height - best.y - size,
            },
            source: `${entry.source}+refined`,
            sourcePriority: entry.sourcePriority,
        };
    }

    function sanitizeWatermarkInfo(candidate) {
        if (!candidate) {
            return {
                size: 48,
                position: { x: 0, y: 0, width: 48, height: 48 },
                confidence: 0,
                detected: false,
            };
        }

        return {
            size: candidate.size,
            position: candidate.position,
            confidence: candidate.confidence,
            detected: candidate.accepted === true,
            methodSource: candidate.source,
            alphaGain: candidate.alphaGain,
            alphaVariant: candidate.config?.alphaVariant || null,
            spatialScore: candidate.originalSpatialScore,
            gradientScore: candidate.originalGradientScore,
        };
    }

    function rankWatermarkCandidates(originalImageData) {
        const candidates = [];
        const entries = resolveWatermarkSearchEntries(originalImageData.width, originalImageData.height);

        for (const entry of entries) {
            const candidate = evaluateEntryWithPolarity(originalImageData, entry);
            if (candidate) candidates.push(candidate);
            if (shouldRefineCandidatePosition(candidate, entry)) {
                const refinedEntry = refineEntryByCorrelation(originalImageData, entry);
                if (refinedEntry) {
                    const refinedCandidate = evaluateEntryWithPolarity(originalImageData, refinedEntry);
                    if (refinedCandidate) candidates.push(refinedCandidate);
                }
            }
        }

        const adaptive = detectAdaptiveWatermarkRegion(originalImageData);
        if (adaptive) {
            const candidate = evaluateEntryWithPolarity(originalImageData, adaptive);
            if (candidate) candidates.push(candidate);
        }

        return candidates;
    }

    function detectBestWatermark(originalImageData) {
        let best = null;
        for (const candidate of rankWatermarkCandidates(originalImageData)) {
            best = pickBetterCandidate(best, candidate);
        }
        return best;
    }

    function detectAdaptiveWatermarkRegion(imageData) {
        const fallback48 = searchWatermark(imageData, alphaMap48, 48);
        const fallback96 = searchWatermark(imageData, alphaMap96, 96);
        const bestFallback = fallback96.confidence > fallback48.confidence
            ? { size: 96, result: fallback96 }
            : { size: 48, result: fallback48 };

        if (bestFallback.result.confidence < SEARCH_CONFIG.minConfidence) {
            return null;
        }

        return {
            config: {
                logoSize: bestFallback.size,
                marginRight: imageData.width - bestFallback.result.x - bestFallback.size,
                marginBottom: imageData.height - bestFallback.result.y - bestFallback.size,
            },
            source: 'adaptive-search',
            sourcePriority: 8,
        };
    }

    /**
     * Dilate a binary mask by radius pixels
     */
    function dilateBinaryMask(maskCanvas, radius) {
        const r = Math.max(1, radius | 0);
        const w = maskCanvas.width;
        const h = maskCanvas.height;

        const srcCtx = maskCanvas.getContext('2d', { willReadFrequently: true });
        const src = srcCtx.getImageData(0, 0, w, h).data;

        // Precompute disc offsets
        const offsets = [];
        for (let dy = -r; dy <= r; dy++) {
            for (let dx = -r; dx <= r; dx++) {
                if (dx * dx + dy * dy <= r * r) {
                    offsets.push({ dx, dy });
                }
            }
        }

        const out = new Uint8ClampedArray(w * h * 4);
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                let hit = false;
                for (let k = 0; k < offsets.length; k++) {
                    const nx = x + offsets[k].dx;
                    const ny = y + offsets[k].dy;
                    if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                    const a = src[((ny * w + nx) << 2) + 3];
                    if (a > 127) {
                        hit = true;
                        break;
                    }
                }
                const o = (y * w + x) << 2;
                out[o] = 255;
                out[o + 1] = 255;
                out[o + 2] = 255;
                out[o + 3] = hit ? 255 : 0;
            }
        }

        const outCanvas = document.createElement('canvas');
        outCanvas.width = w;
        outCanvas.height = h;
        outCanvas.getContext('2d').putImageData(new ImageData(out, w, h), 0, 0);
        return outCanvas;
    }

    /**
     * Build mask from watermark alpha map for LaMa inpainting
     */
    function buildMaskFromAlphaMap(alphaMap, watermarkSize, position, imgW, imgH) {
        const mask = document.createElement('canvas');
        mask.width = imgW;
        mask.height = imgH;
        const mctx = mask.getContext('2d');
        const dst = mctx.createImageData(imgW, imgH);
        const d = dst.data;

        // Initialize all pixels as transparent (no inpainting needed)
        for (let i = 0; i < d.length; i += 4) {
            d[i] = 255;
            d[i + 1] = 255;
            d[i + 2] = 255;
            d[i + 3] = 0;
        }

        // Mark watermark area as white (needs inpainting)
        const alphaThreshold = 0.05;
        for (let row = 0; row < watermarkSize; row++) {
            for (let col = 0; col < watermarkSize; col++) {
                const imgX = position.x + col;
                const imgY = position.y + row;

                if (imgX < 0 || imgX >= imgW || imgY < 0 || imgY >= imgH) continue;

                const alphaIdx = row * watermarkSize + col;
                const alpha = Math.abs(alphaMap[alphaIdx]);

                if (alpha > alphaThreshold) {
                    const pixelIdx = (imgY * imgW + imgX) * 4;
                    d[pixelIdx + 3] = 255;
                }
            }
        }

        mctx.putImageData(dst, 0, 0);

        // Dilate mask
        const dilated = dilateBinaryMask(mask, MASK_CONFIG.dilatePx);

        // Feather and re-binarize
        const final = document.createElement('canvas');
        final.width = imgW;
        final.height = imgH;
        const fctx = final.getContext('2d');

        if (MASK_CONFIG.featherPx > 0) {
            fctx.filter = `blur(${MASK_CONFIG.featherPx}px)`;
            fctx.drawImage(dilated, 0, 0);
            fctx.filter = 'none';

            const fi = fctx.getImageData(0, 0, imgW, imgH);
            const fd = fi.data;
            for (let i = 0; i < fd.length; i += 4) {
                const a = fd[i + 3] / 255;
                const bin = a > 0.4 ? 255 : 0;
                fd[i] = 255;
                fd[i + 1] = 255;
                fd[i + 2] = 255;
                fd[i + 3] = bin;
            }
            fctx.putImageData(fi, 0, 0);
        } else {
            fctx.drawImage(dilated, 0, 0);
        }

        return final;
    }

    function calculateLamaRoi(watermarkInfo, imgW, imgH) {
        const position = watermarkInfo.position;
        const watermarkSize = Math.max(
            watermarkInfo.size || 0,
            position.width || 0,
            position.height || 0
        );
        const minSide = Math.min(LAMA_ROI_CONFIG.minSize, imgW, imgH);
        const maxSide = Math.min(LAMA_ROI_CONFIG.maxSize, imgW, imgH);
        const desiredSide = Math.ceil(watermarkSize * LAMA_ROI_CONFIG.contextScale);
        const side = Math.round(clamp(desiredSide, minSide, maxSide));
        const centerX = position.x + position.width / 2;
        const centerY = position.y + position.height / 2;

        return {
            x: Math.round(clamp(centerX - side / 2, 0, imgW - side)),
            y: Math.round(clamp(centerY - side / 2, 0, imgH - side)),
            width: side,
            height: side,
        };
    }

    function cropCanvas(sourceCanvas, rect) {
        const crop = document.createElement('canvas');
        crop.width = rect.width;
        crop.height = rect.height;
        const cctx = crop.getContext('2d');
        cctx.imageSmoothingEnabled = false;
        cctx.drawImage(
            sourceCanvas,
            rect.x,
            rect.y,
            rect.width,
            rect.height,
            0,
            0,
            rect.width,
            rect.height
        );
        return crop;
    }

    function resizeCanvasToImageData(sourceCanvas, targetSize) {
        const resized = document.createElement('canvas');
        resized.width = targetSize;
        resized.height = targetSize;
        const rctx = resized.getContext('2d');
        rctx.imageSmoothingEnabled = true;
        rctx.imageSmoothingQuality = 'high';
        rctx.drawImage(sourceCanvas, 0, 0, targetSize, targetSize);
        return rctx.getImageData(0, 0, targetSize, targetSize);
    }

    function createFeatheredMaskImageData(maskCanvas, featherPx) {
        const blendMask = document.createElement('canvas');
        blendMask.width = maskCanvas.width;
        blendMask.height = maskCanvas.height;
        const bctx = blendMask.getContext('2d');

        if (featherPx > 0) {
            bctx.filter = `blur(${featherPx}px)`;
        }
        bctx.drawImage(maskCanvas, 0, 0);
        bctx.filter = 'none';

        return bctx.getImageData(0, 0, blendMask.width, blendMask.height);
    }

    function mergeInpaintedRoi(originalImageData, roiImageData, roiMaskCanvas, roi) {
        if (roiImageData.width !== roi.width || roiImageData.height !== roi.height) {
            throw new Error(`Unexpected LaMa ROI output size: ${roiImageData.width}x${roiImageData.height}`);
        }

        const processedImageData = cloneImageData(originalImageData);
        const original = originalImageData.data;
        const output = processedImageData.data;
        const inpainted = roiImageData.data;
        const blendMask = createFeatheredMaskImageData(roiMaskCanvas, LAMA_ROI_CONFIG.blendFeatherPx).data;

        for (let row = 0; row < roi.height; row++) {
            for (let col = 0; col < roi.width; col++) {
                const roiIdx = (row * roi.width + col) * 4;
                const alpha = blendMask[roiIdx + 3] / 255;
                if (alpha <= 0.003) continue;

                const fullIdx = ((roi.y + row) * originalImageData.width + roi.x + col) * 4;
                const keep = 1 - alpha;

                output[fullIdx] = Math.round(original[fullIdx] * keep + inpainted[roiIdx] * alpha);
                output[fullIdx + 1] = Math.round(original[fullIdx + 1] * keep + inpainted[roiIdx + 1] * alpha);
                output[fullIdx + 2] = Math.round(original[fullIdx + 2] * keep + inpainted[roiIdx + 2] * alpha);
                output[fullIdx + 3] = original[fullIdx + 3];
            }
        }

        return processedImageData;
    }

    // ==================== LaMa Worker Code ====================
    const LAMA_WORKER_CODE = `
importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.webgpu.min.js');

let session = null;
let modelEP = null;

function log(message) {
    try {
        self.postMessage({ type: 'log', message });
    } catch (e) {
        console.log(message);
    }
}

const DB_NAME = 'lama-cache';
const STORE = 'models';
const MODEL_VERSION = '${MODEL_VERSION}';

function idbOpen() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open(DB_NAME, 1);
        request.onupgradeneeded = () => {
            const db = request.result;
            if (!db.objectStoreNames.contains(STORE)) {
                db.createObjectStore(STORE);
            }
        };
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
    });
}

async function idbGet(key) {
    const db = await idbOpen();
    return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE, 'readonly');
        const store = tx.objectStore(STORE);
        const request = store.get(key);
        request.onsuccess = () => resolve(request.result || null);
        request.onerror = () => reject(request.error);
    });
}

async function idbSet(key, value) {
    const db = await idbOpen();
    return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE, 'readwrite');
        const store = tx.objectStore(STORE);
        const request = store.put(value, key);
        request.onsuccess = () => resolve();
        request.onerror = () => reject(request.error);
    });
}

async function fetchModelWithCache(url) {
    const key = MODEL_VERSION + '::' + url;
    
    try {
        const cached = await idbGet(key);
        if (cached) {
            const buf = await cached.arrayBuffer();
            log('Using cached ONNX model');
            return buf;
        }
    } catch (e) {
        log('Cache read failed: ' + e);
    }
    
    log('Downloading LaMa model (~200MB)...');
    const resp = await fetch(url, { mode: 'cors', credentials: 'omit', cache: 'force-cache' });
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    
    const buf = await resp.arrayBuffer();
    
    try {
        await idbSet(key, new Blob([buf], { type: 'application/octet-stream' }));
        log('Model cached into IndexedDB');
    } catch (e) {
        log('Cache write failed: ' + e);
    }
    
    return buf;
}

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
ort.env.wasm.numThreads = 1;

async function tryInitWebGPU() {
    if (!('gpu' in self.navigator)) return false;
    try {
        const adapter = await self.navigator.gpu.requestAdapter();
        if (!adapter) return false;
        const device = await adapter.requestDevice();
        ort.env.webgpu.adapter = adapter;
        ort.env.webgpu.device = device;
        return true;
    } catch (e) {
        return false;
    }
}

function canvasToImage01_CHW(canvas) {
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const { width, height } = canvas;
    const { data } = ctx.getImageData(0, 0, width, height);
    const area = width * height;
    const arr = new Float32Array(3 * area);
    
    for (let i = 0; i < area; i++) {
        const p = i * 4;
        arr[i] = data[p] / 255;
        arr[i + area] = data[p + 1] / 255;
        arr[i + 2 * area] = data[p + 2] / 255;
    }
    
    return { data: arr, shape: [1, 3, height, width] };
}

function canvasToMask01_CHW(canvas) {
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const { width, height } = canvas;
    const { data } = ctx.getImageData(0, 0, width, height);
    const area = width * height;
    const arr = new Float32Array(area);
    
    for (let i = 0; i < area; i++) {
        const p = i * 4;
        const lum = (0.2126 * data[p] + 0.7152 * data[p + 1] + 0.0722 * data[p + 2]) / 255;
        arr[i] = lum > 0.5 ? 1 : 0;
    }
    
    return { data: arr, shape: [1, 1, height, width] };
}

function tensorCHW_toRGBA_255(chw, W, H) {
    const area = W * H;
    const rgba = new Uint8ClampedArray(area * 4);
    
    for (let i = 0; i < area; i++) {
        let r = chw[i];
        let g = chw[i + area];
        let b = chw[i + 2 * area];
        
        r = Math.max(0, Math.min(255, r));
        g = Math.max(0, Math.min(255, g));
        b = Math.max(0, Math.min(255, b));
        
        const o = i * 4;
        rgba[o] = r;
        rgba[o + 1] = g;
        rgba[o + 2] = b;
        rgba[o + 3] = 255;
    }
    
    return rgba;
}

self.onmessage = async (e) => {
    const { type } = e.data || {};
    
    if (type === 'load') {
        try {
            log('Starting model load...');
            const modelBuffer = await fetchModelWithCache(e.data.modelUrl);
            
            const providers = [];
            if (await tryInitWebGPU()) {
                providers.push('webgpu');
                log('WebGPU available');
            }
            providers.push('wasm');
            
            let lastError = null;
            for (const ep of providers) {
                try {
                    log('Trying execution provider: ' + ep);
                    session = await ort.InferenceSession.create(modelBuffer, { executionProviders: [ep] });
                    modelEP = ep;
                    break;
                } catch (err) {
                    lastError = err;
                    log('EP ' + ep + ' failed: ' + err.message);
                }
            }
            
            if (!session) {
                throw new Error('Failed to initialize any execution provider: ' + (lastError?.message || lastError));
            }
            
            log('Model loaded with EP: ' + modelEP);
            self.postMessage({ type: 'loaded', ep: modelEP });
        } catch (err) {
            self.postMessage({ type: 'error', error: 'Model load failed: ' + err.message });
        }
    }
    
    if (type === 'run') {
        if (!session) {
            self.postMessage({ type: 'error', error: 'Session not ready' });
            return;
        }
        
        try {
            const { imgRGBA, maskRGBA, modelSize, outW, outH } = e.data;
            
            const img512 = new OffscreenCanvas(modelSize, modelSize);
            img512.getContext('2d').putImageData(
                new ImageData(new Uint8ClampedArray(imgRGBA), modelSize, modelSize), 
                0, 0
            );
            
            const mask512 = new OffscreenCanvas(modelSize, modelSize);
            mask512.getContext('2d').putImageData(
                new ImageData(new Uint8ClampedArray(maskRGBA), modelSize, modelSize), 
                0, 0
            );
            
            const tImg = canvasToImage01_CHW(img512);
            const tMask = canvasToMask01_CHW(mask512);
            
            const imgT = new ort.Tensor('float32', tImg.data, tImg.shape);
            const maskT = new ort.Tensor('float32', tMask.data, tMask.shape);
            
            const feeds = {};
            if (session.inputNames?.includes('image') && session.inputNames?.includes('mask')) {
                feeds['image'] = imgT;
                feeds['mask'] = maskT;
            } else {
                const names = session.inputNames || ['image', 'mask'];
                feeds[names[0]] = imgT;
                feeds[names[1] || 'mask'] = maskT;
            }
            
            log('Running inference...');
            
            const results = await session.run(feeds);
            const outName = session.outputNames?.[0] || Object.keys(results)[0];
            const outT = results[outName];
            
            const W = outT.dims[3];
            const H = outT.dims[2];
            const outData = outT.data instanceof Float32Array ? outT.data : await outT.getData();
            
            const rgba512 = tensorCHW_toRGBA_255(outData, W, H);
            
            const out512 = new OffscreenCanvas(W, H);
            out512.getContext('2d').putImageData(new ImageData(rgba512, W, H), 0, 0);
            
            const outFull = new OffscreenCanvas(outW, outH);
            const octx = outFull.getContext('2d');
            octx.imageSmoothingEnabled = true;
            octx.imageSmoothingQuality = 'high';
            octx.drawImage(out512, 0, 0, W, H, 0, 0, outW, outH);
            
            const outImg = octx.getImageData(0, 0, outW, outH);
            self.postMessage(
                { type: 'result', rgba: outImg.data.buffer, w: outW, h: outH, ep: modelEP },
                [outImg.data.buffer]
            );
        } catch (err) {
            self.postMessage({ type: 'error', error: 'Inference failed: ' + err.message });
        }
    }
};
`;

    // ==================== Public API ====================

    function updateLamaStatus(partial) {
        if (lamaStatusCallback) {
            lamaStatusCallback({
                loading: false,
                ready: lamaReady,
                error: null,
                executionProvider: lamaEP,
                progress: '',
                ...partial,
            });
        }
    }

    return {
        /**
         * Initialize the engine by loading alpha maps
         */
        async initialize() {
            if (initialized) return;

            try {
                // Load 48x48 alpha map
                const img48 = await loadImageFromUrl("bg_48.png");
                const canvas48 = document.createElement("canvas");
                canvas48.width = img48.width;
                canvas48.height = img48.height;
                const ctx48 = canvas48.getContext("2d");
                ctx48.drawImage(img48, 0, 0);
                const imageData48 = ctx48.getImageData(0, 0, img48.width, img48.height);
                alphaMap48 = calculateAlphaMap(imageData48);

                // Load 96x96 alpha map
                const img96 = await loadImageFromUrl("bg_96.png");
                const canvas96 = document.createElement("canvas");
                canvas96.width = img96.width;
                canvas96.height = img96.height;
                const ctx96 = canvas96.getContext("2d");
                ctx96.drawImage(img96, 0, 0);
                const imageData96 = ctx96.getImageData(0, 0, img96.width, img96.height);
                alphaMap96 = calculateAlphaMap(imageData96);
                alphaMapCache.clear();

                try {
                    const img96NewMargin = await loadImageFromUrl("bg_96_20260520.png");
                    const canvas96NewMargin = document.createElement("canvas");
                    canvas96NewMargin.width = img96NewMargin.width;
                    canvas96NewMargin.height = img96NewMargin.height;
                    const ctx96NewMargin = canvas96NewMargin.getContext("2d");
                    ctx96NewMargin.drawImage(img96NewMargin, 0, 0);
                    const imageData96NewMargin = ctx96NewMargin.getImageData(
                        0,
                        0,
                        img96NewMargin.width,
                        img96NewMargin.height
                    );
                    alphaMap96NewMargin = calculateAlphaMap(imageData96NewMargin);
                } catch (variantError) {
                    console.warn("Optional Gemini 20260520 alpha template was not loaded:", variantError);
                    alphaMap96NewMargin = null;
                }

                try {
                    alphaMap36V2 = await loadFloat32AlphaMapFromUrl("bg_36_v2.bin", 36 * 36);
                } catch (variantError) {
                    console.warn("Optional Gemini V2 36px alpha template was not loaded:", variantError);
                    alphaMap36V2 = null;
                }

                initialized = true;
            } catch (error) {
                console.error("Failed to initialize WatermarkEngine:", error);
                throw error;
            }
        },

        /**
         * Set callback for LaMa status updates
         */
        setLamaStatusCallback(callback) {
            lamaStatusCallback = callback;
        },

        /**
         * Initialize LaMa worker and load model
         */
        async initializeLama() {
            if (lamaReady) return;
            if (lamaLoadPromise) return lamaLoadPromise;

            lamaLoadPromise = new Promise((resolve, reject) => {
                updateLamaStatus({ loading: true, progress: 'Initializing worker...' });

                const blob = new Blob([LAMA_WORKER_CODE], { type: "application/javascript" });
                const url = URL.createObjectURL(blob);
                lamaWorker = new Worker(url);

                lamaWorker.onmessage = (msg) => {
                    const { type } = msg.data || {};

                    if (type === 'log') {
                        console.log('🧠 LaMa Worker:', msg.data.message);
                        updateLamaStatus({ progress: msg.data.message });
                        return;
                    }

                    if (type === 'loaded') {
                        lamaReady = true;
                        lamaEP = msg.data.ep || 'wasm';
                        updateLamaStatus({
                            loading: false,
                            ready: true,
                            executionProvider: lamaEP,
                            progress: `Model ready (${lamaEP})`,
                        });
                        resolve();
                    } else if (type === 'error') {
                        const error = msg.data.error;
                        updateLamaStatus({ loading: false, error, progress: '' });
                        reject(new Error(error));
                    }
                };

                lamaWorker.postMessage({ type: 'load', modelUrl: LAMA_MODEL_URL });
            });

            return lamaLoadPromise;
        },

        /**
         * Check if LaMa is ready
         */
        isLamaReady() {
            return lamaReady;
        },

        /**
         * Detect watermark in image
         */
        detectWatermark(imageData) {
            if (!initialized) {
                throw new Error("WatermarkEngine not initialized");
            }

            const best = detectBestWatermark(imageData);
            return {
                ...sanitizeWatermarkInfo(best),
                alphaMap: best?.alphaMap || null,
            };
        },

        /**
         * Process image using Alpha Blending method
         */
        async processWithAlphaBlending(img, startTime) {
            const canvas = document.createElement("canvas");
            canvas.width = img.width;
            canvas.height = img.height;
            const ctx = canvas.getContext("2d");
            ctx.drawImage(img, 0, 0);

            const originalImageData = ctx.getImageData(0, 0, img.width, img.height);
            const processedImageData = new ImageData(
                new Uint8ClampedArray(originalImageData.data),
                originalImageData.width,
                originalImageData.height
            );

            const watermarkInfo = this.detectWatermark(originalImageData);

            if (watermarkInfo.detected) {
                const alphaMap = watermarkInfo.alphaMap || (watermarkInfo.size === 48 ? alphaMap48 : alphaMap96);
                removeWatermark(processedImageData, alphaMap, watermarkInfo.position, watermarkInfo.size, {
                    alphaGain: watermarkInfo.alphaGain,
                });
            }
            const { alphaMap: _alphaMap, ...publicWatermarkInfo } = watermarkInfo;

            return {
                success: true,
                originalImageData,
                processedImageData,
                watermarkInfo: publicWatermarkInfo,
                processingTime: performance.now() - startTime,
                method: 'alpha-blending',
            };
        },

        /**
         * Process image using LaMa AI inpainting
         */
        async processWithLama(img, startTime) {
            if (!lamaWorker || !lamaReady) {
                throw new Error("LaMa worker not ready. Call initializeLama() first.");
            }

            const canvas = document.createElement("canvas");
            canvas.width = img.width;
            canvas.height = img.height;
            const ctx = canvas.getContext("2d");
            ctx.drawImage(img, 0, 0);

            const originalImageData = ctx.getImageData(0, 0, img.width, img.height);
            const watermarkInfo = this.detectWatermark(originalImageData);
            const alphaMap = watermarkInfo.alphaMap || (watermarkInfo.size === 48 ? alphaMap48 : alphaMap96);

            if (!watermarkInfo.detected || !alphaMap) {
                const { alphaMap: _alphaMap, ...publicWatermarkInfo } = watermarkInfo;
                return {
                    success: true,
                    originalImageData,
                    processedImageData: cloneImageData(originalImageData),
                    watermarkInfo: publicWatermarkInfo,
                    processingTime: performance.now() - startTime,
                    method: 'lama-ai',
                };
            }

            // Build mask
            const maskCanvas = buildMaskFromAlphaMap(
                alphaMap,
                watermarkInfo.size,
                watermarkInfo.position,
                img.width,
                img.height
            );
            const lamaInputImageData = cloneImageData(originalImageData);
            const lamaPrecleanAlphaGain = Math.max(1, watermarkInfo.alphaGain || 1);
            removeWatermark(lamaInputImageData, alphaMap, watermarkInfo.position, watermarkInfo.size, {
                alphaGain: lamaPrecleanAlphaGain,
            });
            const lamaInputCanvas = document.createElement("canvas");
            lamaInputCanvas.width = img.width;
            lamaInputCanvas.height = img.height;
            lamaInputCanvas.getContext("2d").putImageData(lamaInputImageData, 0, 0);

            const lamaRoi = calculateLamaRoi(watermarkInfo, img.width, img.height);
            const roiCanvas = cropCanvas(lamaInputCanvas, lamaRoi);
            const roiMaskCanvas = cropCanvas(maskCanvas, lamaRoi);
            const imgData512 = resizeCanvasToImageData(roiCanvas, MODEL_SIZE);
            const maskData512 = resizeCanvasToImageData(roiMaskCanvas, MODEL_SIZE);
            const { alphaMap: _alphaMap, ...publicWatermarkInfo } = watermarkInfo;
            publicWatermarkInfo.lamaRoi = { ...lamaRoi };
            publicWatermarkInfo.lamaPrecleaned = true;
            publicWatermarkInfo.lamaPrecleanAlphaGain = lamaPrecleanAlphaGain;

            return new Promise((resolve, reject) => {
                const handler = (msg) => {
                    const { type } = msg.data || {};

                    if (type === 'log') {
                        console.log('🧠 LaMa:', msg.data.message);
                        return;
                    }

                    if (type === 'result') {
                        lamaWorker.removeEventListener('message', handler);
                        try {
                            const { rgba, w, h } = msg.data;
                            const out = new Uint8ClampedArray(rgba);
                            const roiImageData = new ImageData(out, w, h);
                            const processedImageData = mergeInpaintedRoi(
                                lamaInputImageData,
                                roiImageData,
                                roiMaskCanvas,
                                lamaRoi
                            );

                            resolve({
                                success: true,
                                originalImageData,
                                processedImageData,
                                watermarkInfo: publicWatermarkInfo,
                                processingTime: performance.now() - startTime,
                                method: 'lama-ai',
                            });
                        } catch (err) {
                            reject(err);
                        }
                    } else if (type === 'error') {
                        lamaWorker.removeEventListener('message', handler);
                        reject(new Error(msg.data.error));
                    }
                };

                lamaWorker.addEventListener('message', handler);

                lamaWorker.postMessage(
                    {
                        type: 'run',
                        imgRGBA: imgData512.data.buffer,
                        maskRGBA: maskData512.data.buffer,
                        modelSize: MODEL_SIZE,
                        outW: lamaRoi.width,
                        outH: lamaRoi.height,
                    },
                    [imgData512.data.buffer, maskData512.data.buffer]
                );
            });
        },

        /**
         * Process a single image file
         */
        async processImage(file, method = 'alpha-blending') {
            const startTime = performance.now();

            await this.initialize();

            const img = await loadImageFromFile(file);

            if (method === 'lama-ai') {
                await this.initializeLama();
                return this.processWithLama(img, startTime);
            } else {
                return this.processWithAlphaBlending(img, startTime);
            }
        },

        /**
         * Convert ImageData to Blob
         */
        imageDataToBlob(imageData, type = "image/png", quality = 1.0) {
            return new Promise((resolve, reject) => {
                const canvas = document.createElement("canvas");
                canvas.width = imageData.width;
                canvas.height = imageData.height;
                const ctx = canvas.getContext("2d");
                ctx.putImageData(imageData, 0, 0);

                canvas.toBlob(
                    (blob) => {
                        if (blob) {
                            resolve(blob);
                        } else {
                            reject(new Error("Failed to create blob"));
                        }
                    },
                    type,
                    quality
                );
            });
        },

        /**
         * Dispose resources
         */
        dispose() {
            if (lamaWorker) {
                lamaWorker.terminate();
                lamaWorker = null;
                lamaReady = false;
                lamaLoadPromise = null;
            }
        }
    };
})();

if (typeof window !== 'undefined') {
    window.WatermarkEngine = WatermarkEngine;
}
