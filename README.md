# UnReflectAnything  
**RGB-Only Highlight Removal by Rendering Synthetic Specular Supervision**

![Header](assets/header.png)

**Alberto Rota¹*, Mert Kiray², Mert Asim Karaoglu², Patrick Ruhkamp²,  
Elena De Momi¹, Nassir Navab², Benjamin Busam²**  
¹ Politecnico di Milano ² Technical University of Munich

## Abstract
UnReflectAnything is an RGB-only framework for single-image specular highlight removal. It predicts a soft highlight map and reconstructs a reflection-free diffuse image by inpainting corrupted feature tokens extracted with a frozen DINOv3 Vision Transformer. To enable training without paired data, the method introduces Virtual Highlight Synthesis, which renders physically plausible specularities using monocular geometry, Fresnel-aware shading, and randomized lighting. The approach generalizes across natural and surgical domains and improves downstream geometric consistency. :contentReference[oaicite:0]{index=0}

## Key Contributions
- Virtual Highlight Synthesis from monocular geometry for paired supervision from any RGB image  
- Token-space diffuse inpainting of DINOv3 features before image decoding  
- RGB-only inference without polarization sensors or paired ground truth  
- Strong generalization to natural and endoscopic imagery  
- Improved robustness in downstream correspondence and pose estimation tasks :contentReference[oaicite:1]{index=1}
