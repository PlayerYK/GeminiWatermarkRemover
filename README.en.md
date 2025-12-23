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

Gemini AI adds a semi-transparent "✦" (four-pointed star) watermark in the bottom-right corner of generated images. The watermark comes in two sizes:
- **48×48 pixels** for smaller images
- **96×96 pixels** for larger images

The watermark is applied using **alpha blending** - a standard image compositing technique:

```
watermarked_pixel = original_pixel × (1 - α) + watermark_pixel × α
```

Where `α` (alpha) is the transparency value of the watermark at each pixel.

### Method 1: Reverse Alpha Blending (Quick)

Since we know the exact watermark pattern (captured in `bg_48.png` and `bg_96.png`), we can mathematically reverse the blending:

```
original_pixel = (watermarked_pixel - α × 255) / (1 - α)
```

**Algorithm steps:**

1. **Load alpha maps** - Pre-computed transparency values from watermark templates
2. **Detect watermark position** - Template matching using correlation scoring in the bottom-right search area
3. **Apply reverse blending** - For each pixel in the watermark region, calculate the original value

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
3. **Run LaMa model** - The ONNX model predicts what pixels should be under the mask based on surrounding context
4. **Blend result** - Upscale model output to original resolution

**Pros:**
- Better for complex textures/patterns
- Works even if watermark template doesn't perfectly match
- Handles edge cases where alpha blending leaves artifacts

**Cons:**
- Requires ~200MB model download (cached in IndexedDB)
- Slower processing (1-5 seconds depending on hardware)
- Results are AI-generated, not mathematically perfect

### Watermark Detection

Both methods use **template correlation** to find the watermark:

1. Search the bottom-right 25% of the image
2. For each candidate position, calculate correlation score:
   - Higher score = brighter pixels where watermark template is more opaque
3. Refine search around best match
4. Report confidence based on correlation strength

## File Structure

```
GeminiWatermarkRemover/
├── index.html      # Main HTML page
├── style.css       # Minimal dark theme styling
├── engine.js       # Core watermark removal engine
├── app.js          # UI interaction logic
├── bg_48.png       # 48×48 watermark alpha template
├── bg_96.png       # 96×96 watermark alpha template
└── README.md       # This file
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

The `bg_48.png` and `bg_96.png` files contain the watermark pattern on a black background. The alpha value for each pixel is derived from the maximum RGB channel:

```javascript
alpha = max(R, G, B) / 255
```

### LaMa Model Details

- Model: `lama_fp32.onnx` (~200MB)
- Input: 512×512 RGB image + 512×512 binary mask
- Output: 512×512 RGB inpainted image
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

- Alpha blending algorithm based on [gemini-watermark-remover](https://github.com/journey-ad/gemini-watermark-remover) (MIT License)
- LaMa model from [Carve/LaMa-ONNX](https://huggingface.co/Carve/LaMa-ONNX) on Hugging Face

## License

MIT License - See original repositories for model licensing terms.

