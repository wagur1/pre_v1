# AdaVCM: Adaptive Video Preprocessing Techniques for Optimizing Video Coding for Machines (VCM)

Official repository for the paper:  
**"Adaptive Video Preprocessing Techniques for Optimizing Video Coding for Machines (VCM)"**

Target venues: *Journal of Visual Communication and Image Representation (JVCI)* / *Multimedia Tools and Applications (MTAP)* (Q2) / *IEEE Access* (Q2).

---

## 1. Problem Statement & Motivation

Video Coding for Machines (VCM, standardized under **ISO/IEC 23888 MPEG-AI**) shifts the compression objective from human perceptual fidelity (PSNR, SSIM, VMAF) to downstream machine analytics (mAP, MOTA, top-1 accuracy). 

### The Failure of Conventional Learned Editors
Previous works attempting end-to-end learned neural video editors before standard codecs suffer from three major defects:
1. **Feature Distortion & mAP Collapse:** Neural autoencoders distort object boundaries and texture distributions, degrading detector accuracy by 20% to 25% even before quantization.
2. **Sharp Boundary High-Frequency Penalties:** Naive hard ROI masks introduce artificial step discontinuities, causing DCT block-coding overhead that cancels out bitrate savings.
3. **Temporal Blindness:** Independent frame preprocessing ignores inter-frame temporal correlation, resulting in large motion compensation residuals in P/B frames.

---

## 2. Proposed Architecture: AdaVCM

`AdaVCM` addresses these issues by strictly separating **task-critical foreground invariance** from **adaptive background redundancy elimination**:

```
Input Video X ──► [ 1. Spatio-Temporal Salience & Motion Estimator (ST-SME) ]
                               │
                               ▼
                      Soft Weight Map W(x,y,t)
                               │
        ┌──────────────────────┴──────────────────────┐
        ▼                                             ▼
[ 2. Adaptive Policy Net ]                    [ 3. Temporal Background Regularizer ]
  Predicts optimal:                             Stabilizes non-salient background across
  - Sigma (spatial blur)                        frames to minimize inter residuals
  - Alpha (temporal decay)                      X_bg(t) = α X_bg(t-1) + (1-α) X_bg(t)
        │                                             │
        └──────────────────────┬──────────────────────┘
                               ▼
               [ 4. Boundary-Aware Soft Filter ]
                 - Core objects: 100% bit-exact pass-through (Zero degradation)
                 - Boundaries: Continuous non-linear sigmoid transition
                 - Background: High-frequency suppression
                               │
                               ▼
                      Preprocessed Video X'
                               │
                               ▼
             Standard Codec (H.264 / H.265 / VVC) ──► Downstream Vision Model (YOLO/Faster R-CNN)
```

### Key Modules:
* **ST-SME:** Fuses spatial object detections/saliency with inter-frame motion vectors and applies soft morphological dilation for continuous boundary buffers.
* **Boundary-Aware Filter:** Separable Gaussian filter with smooth transition weighting $W_{smooth} = W^\gamma$, completely eliminating edge blocking artifacts.
* **Temporal Background Regularizer (TBR):** Stabilizes static background across consecutive frames, slashing motion estimation bitrate in P/B frames.
* **Adaptive Policy Network:** Lightweight controller (~30k parameters) solving the Rate-Accuracy Lagrangian optimization problem:
  $$\max_\theta \left[ \text{TaskAccuracy}(X') - \lambda \cdot \mathcal{R}_{estimate}(X') \right]$$

---

## 3. Installation & Requirements

```bash
git clone https://github.com/wagur1/pre_v1.git
cd pre_v1
pip install -r requirements.txt
```

---

## 4. Quickstart

### Run Unit Tests
```bash
pytest -v
```

### Train Policy Network Locally or Synthetically
```bash
python train.py --synthetic --epochs 5 --device cpu
```

### Train on COCO Dataset
```bash
python train.py --config configs/default.yaml
```

### Evaluate Rate-Accuracy and BD-Rate
```bash
python evaluate.py --codec h264 --qps 27,32,38,43 --synthetic
```

---

## 5. Kaggle Training Integration

To train at scale with Kaggle GPU resources, use the automated pusher:
```bash
python ops/push_kaggle.py --pool C:/Users/Wagur1/Downloads/pool.json --account wagur124705
```

---

## 6. Citation & Literature Anchors

* Bajić, *Rate-Accuracy Bounds in Visual Coding for Machines*, IEEE MIPR 2025.
* Zhao et al., *Learned Video Pre-processing for Machines*, DCC 2024.
* Lu et al., *Learned Neural Pre-processing for Machine Vision*, IEEE TCSVT 2024.
* Li & Rhee, *Dual-Region Preprocessing for Machine-Friendly JPEG*, ITC-CSCC 2025.
* Różek et al., *Video Coding for Machines using Object Analysis and Standard Video Codecs*, IEEE VCIP 2023.