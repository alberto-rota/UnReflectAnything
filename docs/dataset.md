# Dataset Structure

The UnReflectAnything dataset module supports three main polarization datasets for reflection removal and 6D pose estimation:
## Supported Datasets

### HOUSECAT6D
6D object pose estimation with polarization data

- **Paper**: "HouseCat6D -- A Large-Scale Multi-Modal Category Level 6D Object Perception Dataset with Household Objects in Realistic Scenarios" by HyunJun Jung et al.
- **Project Page**: https://sites.google.com/view/housecat6d/
- **GitHub Repository**: https://github.com/junggy/housecat6d

### SCRREAM
Specular reflection removal dataset

- **Paper**: "Single-Image Specular Highlight Removal via Real-World Dataset Construction" by Zhongqi Wu et al.
- **GitHub Repository**: https://github.com/jianweiguo/SpecularityNet-PSD

### PolaRGB
Polarization-guided RGB processing dataset

- **Paper**: "PolarFree: Polarization-based Reflection-free Imaging" by Mingdeng Yao et al.
- **GitHub Repository**: https://github.com/mdyao/polarfree

## Directory Structure

Each dataset follows a standardized directory structure with scene-based organization. The module supports two different polarization data formats, which are automatically detected based on the file naming convention.

### Format 1: Single Polarization Files

For datasets where each polarization image is stored as a single file (typically containing 4 quadrants), in a counter-clockwise arrangement:

```tree
DATASET_ROOT/
├── scene_001/
│   ├── rgb/
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   └── ...
│   ├── pol/
│   │   ├── 000000.png        # Single file with 4 polarization quadrants
│   │   ├── 000001.png
│   │   └── ...
│   ├── specular/             # Optional: ground truth specular components
│   ├── diffuse/              # Optional: ground truth diffuse components  
│   ├── normals/              # Optional: surface normal maps
│   └── intrinsics.txt        # Camera intrinsics matrix
├── scene_002/
│   └── ... (same structure)
└── ...
```

### Format 2: Separate Polarization Files

For datasets where each polarization angle is stored in separate files:

```
DATASET_ROOT/
├── scene_001/
│   ├── rgb/
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   └── ...
│   ├── pol/
│   │   ├── 000000_000.png    # 0° polarization
│   │   ├── 000000_045.png    # 45° polarization
│   │   ├── 000000_090.png    # 90° polarization
│   │   ├── 000000_135.png    # 135° polarization
│   │   ├── 000001_000.png
│   │   ├── 000001_045.png
│   │   ├── 000001_090.png
│   │   ├── 000001_135.png
│   │   └── ...
│   ├── specular/             # Optional: ground truth specular components
│   ├── diffuse/              # Optional: ground truth diffuse components
│   ├── normals/              # Optional: surface normal maps
│   └── intrinsics.txt        # Camera intrinsics matrix
├── scene_002/
│   └── ... (same structure)
└── ...
```
