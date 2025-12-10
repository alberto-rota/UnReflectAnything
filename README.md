<div align="center">

<p>
<b>Alberto Rota<sup>1</sup>, Mert Kiray<sup>2,3</sup>, Mert Asim Karaoglu<sup>4,2</sup>, Patrick Ruhkamp<sup>2</sup>,<br>
Elena De Momi<sup>1</sup>, Nassir Navab<sup>2,3</sup>, Benjamin Busam<sup>2,3</sup></b><br>

<sup>1</sup>Politecnico di Milano &nbsp;&nbsp; <sup>2</sup>Technical University of Munich &nbsp;&nbsp; <sup>3</sup>Munich Center for Machine Learning (MCML) &nbsp;&nbsp; <sup>4</sup>ImFusion
</p> 

 <p>
<a href="#" style="padding:10px 20px;background:linear-gradient(135deg, #667eea 0%, #764ba2 100%);color:#fff;border-radius:8px;text-decoration:none;margin:6px;display:inline-block;font-weight:600;box-shadow:0 4px 6px rgba(0,0,0,0.1);transition:transform 0.2s,box-shadow 0.2s" onmouseover="this.style.transform='translateY(-2px)';this.style.boxShadow='0 6px 12px rgba(0,0,0,0.15)'" onmouseout="this.style.transform='translateY(0)';this.style.boxShadow='0 4px 6px rgba(0,0,0,0.1)'">📄 Paper</a> 
<a href="https://github.com/alberto-rota/UnReflectAnything" style="padding:10px 20px;background:linear-gradient(135deg, #11998e 0%, #38ef7d 100%);color:#fff;border-radius:8px;text-decoration:none;margin:6px;display:inline-block;font-weight:600;box-shadow:0 4px 6px rgba(0,0,0,0.1);transition:transform 0.2s,box-shadow 0.2s" onmouseover="this.style.transform='translateY(-2px)';this.style.boxShadow='0 6px 12px rgba(0,0,0,0.15)'" onmouseout="this.style.transform='translateY(0)';this.style.boxShadow='0 4px 6px rgba(0,0,0,0.1)'">💻 Code</a>

</p> 

<img src="assets/header.png" alt="method overview" width="90%"/>

</div>

---

## Abstract

<div style="max-width:900px;margin:auto; text-align:justify">
Specular highlights distort appearance, obscure texture, and hinder geometric reasoning in both natural and surgical imagery. We present UnReflectAnything, an RGB-only framework that removes highlights from a single image by predicting a highlight map together with a reflection-free diffuse reconstruction. The model uses a frozen vision transformer encoder to extract multi-scale features, a lightweight head to localize specular regions, and a token-level inpainting module that restores corrupted feature patches before producing the final diffuse image. To overcome the lack of paired supervision, we introduce a Virtual Highlight Synthesis pipeline that renders physically plausible specularities using monocular geometry, Fresnel-aware shading, and randomized lighting which enables training on arbitrary RGB images with correct geometric structure. UnReflectAnything generalizes across natural and surgical domains where non-Lambertian surfaces and non-uniform lighting create severe highlights and it achieves competitive performance with state-of-the-art results on several benchmarks
</div>

<!-- --- -->

<!-- ## Key Contributions
- **Virtual Highlight Synthesis** from monocular geometry enabling paired supervision from any RGB image  
- **Token-space diffuse inpainting** of DINOv3 features prior to image reconstruction  
- **RGB-only inference** without polarization sensors or paired ground truth  
- Strong generalization to both **natural and endoscopic imagery**  
- Improved robustness in downstream **correspondence and pose estimation** tasks   -->

<!-- --- -->

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
