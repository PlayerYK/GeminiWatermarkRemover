# Gemini Watermark Remover

[中文](README.md) | [Try Online](https://aitoo.app/tools/gemini-watermark-remover/)

A standalone, minimal-dependency tool to remove Gemini AI watermarks from images. Works 100% client-side in the browser.

## Features

- **Two removal methods:**
  - ⚡ **Alpha Blending** - Instant, mathematical precision, no download required
  - 🧠 **LaMa AI** - AI-powered inpainting for complex backgrounds (~200MB model)
- **Zero server dependencies** - All processing happens in your browser
- **Supports JPEG, PNG, WebP** formats
- **Batch processing** with individual downloads

## Quick Start

1. Open `index.html` in a modern browser (Chrome, Firefox, Edge, Safari)
2. Select removal method
3. Drop or select images
4. Click "Remove Watermarks"
5. Download processed images

> **Note:** For local development, you may need to serve files via a local server due to CORS restrictions. Use `python -m http.server 8000` or similar.

## How It Works

### Gemini Watermark Structure

Gemini AI adds a semi-transparent "✦" (four-pointed star) watermark in the bottom-right corner of generated images. Known visible watermark layouts now include:
- **48×48 pixels** for standard and large-margin layouts
- **96×96 pixels** for standard layouts
- **96×96 pixels with newer margins** (for example 2816×1536 output)
- **36×36 pixels** for the V2 small watermark profile

The watermark is applied using **alpha blending** - a standard image compositing technique:

```
watermarked_pixel = original_pixel × (1 - α) + watermark_pixel × α
```

Where `α` (alpha) is the transparency value of the watermark at each pixel.

### Method 1: Reverse Alpha Blending (Quick)

Since we know the exact watermark pattern (captured in `bg_48.png`, `bg_96.png`, `bg_96_20260520.png`, and `bg_36_v2.bin`), we can mathematically reverse the blending:

```
original_pixel = (watermarked_pixel - α × 255) / (1 - α)
```

**Algorithm steps:**

1. **Load alpha maps** - Pre-computed transparency values from watermark templates
2. **Generate candidate anchors** - Use Gemini's official size catalog, newer margins, V2 small watermarks, and near-official scaled layouts
3. **Score candidates** - Use spatial correlation, gradient correlation, residual scoring, and multiple alpha strengths
4. **Apply reverse blending** - For each pixel in the watermark region, calculate the original value

**Pros:**
- Instant processing (< 50ms)
- Perfect reconstruction when watermark matches template
- No external dependencies

**Cons:**
- Requires exact watermark template
- May leave artifacts if image was resized/compressed after watermarking

### Method 2: LaMa AI Inpainting

[LaMa (Large Mask Inpainting)](https://github.com/advimman/lama) is a neural network designed to fill in missing or masked regions of images by understanding the surrounding context.

**Algorithm steps:**

1. **Detect watermark** - Same template matching as above
2. **Generate mask** - Create a binary mask marking the watermark area (with dilation for safety margin)
3. **Alpha pre-clean** - First weaken the watermark with reverse alpha blending so LaMa is less likely to regenerate a star-shaped ghost
4. **Crop local ROI** - Crop a sufficiently large region around the watermark to avoid downscaling the whole image
5. **Run LaMa model** - The ONNX model predicts what pixels should be under the mask from local context
6. **Blend result** - Softly merge only the masked ROI area back into the original image

**Pros:**
- Better for complex textures/patterns
- Works even if watermark template doesn't perfectly match
- Preserves non-watermark pixels instead of resizing the whole image through the model
- Handles edge cases where alpha blending leaves artifacts

**Cons:**
- Requires ~200MB model download (cached in IndexedDB)
- Slower processing (1-5 seconds depending on hardware)
- Results are AI-generated, not mathematically perfect

### Watermark Detection

Both methods use **template correlation** to find the watermark:

1. Generate candidate anchors from Gemini's official dimensions and known variants
2. Fall back to adaptive bottom-right search when needed
3. Score each candidate using spatial/gradient correlation and post-removal residuals
4. Choose among 48px, 96px, newer 96px, and V2 36px alpha templates

## File Structure

```
GeminiWatermarkRemover/
├── index.html      # Main HTML page
├── style.css       # Minimal dark theme styling
├── engine.js       # Core watermark removal engine
├── app.js          # UI interaction logic
├── bg_48.png       # 48×48 watermark alpha template
├── bg_96.png       # 96×96 watermark alpha template
├── bg_96_20260520.png # newer-margin 96×96 alpha template
├── bg_36_v2.bin    # 36×36 V2 watermark alpha template
├── README.md       # Chinese docs
└── README.en.md    # This file
```

## Dependencies

**Alpha Blending method:** None (pure JavaScript)

**LaMa AI method:** 
- [ONNX Runtime Web](https://onnxruntime.ai/) - Loaded from CDN at runtime
- [LaMa ONNX Model](https://huggingface.co/Carve/LaMa-ONNX) - Downloaded and cached on first use

## Browser Compatibility

- Chrome 90+ (WebGPU support for faster AI inference)
- Firefox 90+
- Safari 15+
- Edge 90+

## Technical Notes

### Alpha Map Generation

The PNG template files contain the watermark pattern on a black background. The alpha value for each pixel is derived from the maximum RGB channel:

```javascript
alpha = max(R, G, B) / 255
```

`bg_36_v2.bin` is a `Float32Array` binary export of the reference project's embedded V2 36px alpha map.

### LaMa Model Details

- Model: `lama_fp32.onnx` (~200MB)
- Input: alpha-precleaned local ROI resized to 512×512 RGB image + 512×512 binary mask
- Output: inpainted local ROI, blended back into the original image through the mask
- Execution: WebGPU (if available) or WebAssembly fallback

### Mask Generation

For LaMa inpainting, the mask is generated from the alpha map:
1. Mark pixels where `alpha > 0.05` as needing inpainting
2. Dilate mask by 4 pixels to ensure complete coverage
3. Apply 3px Gaussian blur and re-binarize for smooth edges

## Limitations

- Only removes **visible** Gemini watermarks (the semi-transparent star logo)
- Does **NOT** remove invisible/steganographic watermarks (SynthID)
- Best results with original, uncompressed images from Gemini
- Thumbnails or heavily compressed images may not work perfectly

## Credits

- Core detection and candidate-selection logic adapted from [GargantuaX/gemini-watermark-remover](https://github.com/GargantuaX/gemini-watermark-remover) (MIT License)
- Reverse alpha blending formula based on [gemini-watermark-remover](https://github.com/journey-ad/gemini-watermark-remover) (MIT License)
- LaMa model from [Carve/LaMa-ONNX](https://huggingface.co/Carve/LaMa-ONNX) on Hugging Face

## License

MIT License - See original repositories for model licensing terms.
