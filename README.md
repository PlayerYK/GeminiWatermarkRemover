# Gemini 水印去除工具

[English](README.en.md) | [在线使用](https://aitoo.app/tools/gemini-watermark-remover/)

一个独立的、最小依赖的工具，用于去除 Gemini AI 生成图片中的✦水印。100% 在浏览器端运行。

## 功能特点

- **两种去除方法：**
  - ⚡ **Alpha 混合** - 即时处理，数学精确，无需下载
  - 🧠 **LaMa AI** - AI 智能修复，适合复杂背景（需下载约 200MB 模型）
- **零服务器依赖** - 所有处理都在浏览器中完成
- **支持 JPEG、PNG、WebP** 格式
- **批量处理** 并支持单独下载

## 快速开始

1. 用现代浏览器打开 `index.html`（Chrome、Firefox、Edge、Safari）
2. 选择去除方法
3. 拖放或选择图片
4. 点击「Remove Watermarks」
5. 下载处理后的图片

> **注意：** 本地开发时，由于 CORS 限制，可能需要通过本地服务器提供文件。可使用 `python -m http.server 8000` 或类似工具。

## 工作原理

### Gemini 水印结构

Gemini AI 在生成的图片右下角添加半透明的「✦」（四角星）水印。水印有两种尺寸：
- **48×48 像素** 用于较小的图片
- **96×96 像素** 用于较大的图片

水印使用 **Alpha 混合** 技术应用 —— 一种标准的图像合成技术：

```
带水印像素 = 原始像素 × (1 - α) + 水印像素 × α
```

其中 `α`（alpha）是水印在每个像素点的透明度值。

### 方法一：逆向 Alpha 混合（快速）

由于我们知道精确的水印图案（保存在 `bg_48.png` 和 `bg_96.png` 中），可以通过数学方法逆向混合：

```
原始像素 = (带水印像素 - α × 255) / (1 - α)
```

**算法步骤：**

1. **加载 Alpha 映射** - 从水印模板预计算的透明度值
2. **检测水印位置** - 在右下角搜索区域使用相关性评分进行模板匹配
3. **应用逆向混合** - 对水印区域的每个像素计算原始值

**优点：**
- 即时处理（< 50ms）
- 当水印与模板匹配时可完美重建
- 无外部依赖

**缺点：**
- 需要精确的水印模板
- 如果图片在添加水印后被缩放/压缩，可能留下痕迹

### 方法二：LaMa AI 修复

[LaMa（Large Mask Inpainting）](https://github.com/advimman/lama) 是一个神经网络，通过理解周围上下文来填充图像中缺失或被遮罩的区域。

**算法步骤：**

1. **检测水印** - 与上述相同的模板匹配
2. **生成遮罩** - 创建标记水印区域的二值遮罩（带膨胀以增加安全边距）
3. **运行 LaMa 模型** - ONNX 模型根据周围上下文预测遮罩下应该是什么像素
4. **混合结果** - 将模型输出放大到原始分辨率

**优点：**
- 更适合复杂的纹理/图案
- 即使水印模板不完全匹配也能工作
- 可处理 Alpha 混合留下痕迹的边缘情况

**缺点：**
- 需要下载约 200MB 模型（缓存在 IndexedDB 中）
- 处理较慢（1-5 秒，取决于硬件）
- 结果是 AI 生成的，非数学精确

### 水印检测

两种方法都使用 **模板相关性** 来查找水印：

1. 搜索图片右下角 25% 的区域
2. 对每个候选位置计算相关性分数：
   - 分数越高 = 水印模板越不透明的地方像素越亮
3. 在最佳匹配周围精细搜索
4. 根据相关性强度报告置信度

## 文件结构

```
GeminiWatermarkRemover/
├── index.html      # 主 HTML 页面
├── style.css       # 简约深色主题样式
├── engine.js       # 核心水印去除引擎
├── app.js          # UI 交互逻辑
├── bg_48.png       # 48×48 水印 Alpha 模板
├── bg_96.png       # 96×96 水印 Alpha 模板
├── README.md       # 英文说明
└── README.zh-CN.md # 本文件
```

## 依赖

**Alpha 混合方法：** 无（纯 JavaScript）

**LaMa AI 方法：**
- [ONNX Runtime Web](https://onnxruntime.ai/) - 运行时从 CDN 加载
- [LaMa ONNX 模型](https://huggingface.co/Carve/LaMa-ONNX) - 首次使用时下载并缓存

## 浏览器兼容性

- Chrome 90+（支持 WebGPU 以加速 AI 推理）
- Firefox 90+
- Safari 15+
- Edge 90+

## 技术细节

### Alpha 映射生成

`bg_48.png` 和 `bg_96.png` 文件包含黑色背景上的水印图案。每个像素的 Alpha 值由最大 RGB 通道得出：

```javascript
alpha = max(R, G, B) / 255
```

### LaMa 模型详情

- 模型：`lama_fp32.onnx`（约 200MB）
- 输入：512×512 RGB 图像 + 512×512 二值遮罩
- 输出：512×512 RGB 修复图像
- 执行：WebGPU（如可用）或 WebAssembly 回退

### 遮罩生成

对于 LaMa 修复，遮罩从 Alpha 映射生成：
1. 标记 `alpha > 0.05` 的像素为需要修复
2. 膨胀遮罩 4 像素以确保完全覆盖
3. 应用 3px 高斯模糊并重新二值化以获得平滑边缘

## 局限性

- 仅去除 **可见的** Gemini 水印（半透明星形标志）
- **不能** 去除不可见/隐写水印（SynthID）
- 使用 Gemini 原始未压缩图片效果最佳
- 缩略图或高度压缩的图片可能效果不佳

## 致谢

- Alpha 混合算法基于 [gemini-watermark-remover](https://github.com/journey-ad/gemini-watermark-remover)（MIT 许可证）
- LaMa 模型来自 Hugging Face 上的 [Carve/LaMa-ONNX](https://huggingface.co/Carve/LaMa-ONNX)

## 许可证

MIT 许可证 - 模型许可条款请参见原始仓库。

