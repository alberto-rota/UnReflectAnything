<div align="center">

# UnReflectAnything: RGB-Only Highlight Removal by Rendering Synthetic Specular Supervision

<img src="assets/header.png" alt="Header" width="100%"/>

**Alberto Rota¹*, Mert Kiray², Mert Asim Karaoglu², Patrick Ruhkamp²,  Elena De Momi¹, Nassir Navab², Benjamin Busam²**  
¹ Politecnico di Milano ² Technical University of Munich

</div>

<div align="center">

## Abstract

Specular highlights distort appearance, obscure texture, and hinder geometric reasoning in both natural and surgical imagery. We present **UnReflectAnything**, an RGB-only framework that removes highlights from a single image by predicting a highlight map together with a reflection-free diffuse reconstruction. The model uses a frozen vision transformer encoder to extract multi-scale features, a lightweight head to localize specular regions, and a token-level inpainting module that restores corrupted feature patches before producing the final diffuse image. To overcome the lack of paired supervision, we introduce a **Virtual Highlight Synthesis** pipeline that renders physically plausible specularities using monocular geometry, Fresnel-aware shading, and randomized lighting. This enables training on arbitrary RGB images while preserving correct geometric structure. UnReflectAnything generalizes across natural and surgical domains—where non-Lambertian surfaces and non-uniform lighting create severe highlights—and achieves competitive performance with state-of-the-art methods on several benchmarks.

</div>

## Key Contributions
- Virtual Highlight Synthesis from monocular geometry for paired supervision from any RGB image  
- Token-space diffuse inpainting of DINOv3 features before image decoding  
- RGB-only inference without polarization sensors or paired ground truth  
- Strong generalization to natural and endoscopic imagery  
- Improved robustness in downstream correspondence and pose estimation tasks  
