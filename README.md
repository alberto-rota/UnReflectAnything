<div align="center">

<h1>UnReflectAnything</h1>

<h3>RGB-Only Highlight Removal by Rendering Synthetic Specular Supervision</h3>

<p>
<b>Alberto Rota<sup>1*</sup>, Mert Kiray<sup>2</sup>, Mert Asim Karaoglu<sup>2</sup>, Patrick Ruhkamp<sup>2</sup>,<br>
Elena De Momi<sup>1</sup>, Nassir Navab<sup>2</sup>, Benjamin Busam<sup>2</sup></b><br>
<sup>1</sup>Politecnico di Milano &nbsp;&nbsp; <sup>2</sup>Technical University of Munich
</p>

<p>
<a href="#" style="padding:6px 14px;border:1px solid #555;border-radius:6px;text-decoration:none;margin:4px;display:inline-block">📄 Paper</a>
<a href="#" style="padding:6px 14px;border:1px solid #555;border-radius:6px;text-decoration:none;margin:4px;display:inline-block">💻 Code</a>

</p>

<img src="assets/header.png" alt="method overview" width="90%"/>

</div>

---

## Abstract

<div style="max-width:900px;margin:auto">

Specular highlights distort appearance, obscure texture, and hinder geometric reasoning in both natural and surgical imagery.  
We present **UnReflectAnything**, an RGB-only framework that removes highlights from a single image by predicting a highlight map together with a reflection-free diffuse reconstruction.  

The model leverages a frozen vision transformer encoder to extract multi-scale features, a lightweight head to localize specular regions, and a **token-level inpainting module** that restores corrupted feature patches prior to image decoding.  

To address the lack of paired supervision, we introduce a **Virtual Highlight Synthesis** pipeline that renders physically plausible specularities using monocular geometry, Fresnel-aware shading, and randomized lighting. This enables training on arbitrary RGB images while preserving geometric consistency.  

UnReflectAnything generalizes across natural and surgical domains—where non-Lambertian surfaces and non-uniform lighting produce severe highlights—and achieves competitive performance with state-of-the-art methods on multiple benchmarks.

</div>

---

## Key Contributions
- **Virtual Highlight Synthesis** from monocular geometry enabling paired supervision from any RGB image  
- **Token-space diffuse inpainting** of DINOv3 features prior to image reconstruction  
- **RGB-only inference** without polarization sensors or paired ground truth  
- Strong generalization to both **natural and endoscopic imagery**  
- Improved robustness in downstream **correspondence and pose estimation** tasks  

---

<!-- ## Method Overview

<img src="assets/method.png" alt="method diagram" width="100%"/>

**Pipeline.**  
Given a single RGB image, UnReflectAnything predicts a highlight localization mask and performs token-space inpainting on frozen ViT features to recover a reflection-free diffuse reconstruction. Training supervision is obtained via physically based synthetic highlight rendering using monocular geometry.

---

## Results

<img src="assets/results.png" alt="qualitative results" width="100%"/>

Qualitative results on **natural scenes** and **endoscopic imagery**, demonstrating robust highlight removal under severe specularities and complex lighting conditions.

---
 -->
