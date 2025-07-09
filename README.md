# Infinity-Tomographic-Reconstruction
Tomographic reconstruction for diferent imaging modalities (PET, SPECT, CT and Compton)
Image Reconstruction Framework for PET/CT and SPECT

## 🚀 Overview

This repository provides modular image reconstruction framework designed for Positron Emission Tomography (PET),
Computed Tomography (CT), and Single Photon Emission Computed Tomography (SPECT). 

You can define the **Device** type
 + PET
 + CT
 + SPECT
 + Compton

**Modules and Submodules**
 + LYSO and SPiM
 + Collimators and CZT
 + Generic

**Geometry**
   + EasyPET, 
   + Cylindrical, Polygonal, Planar

**Projectors**
   + ***PET***
     + Uniforme Tube of response,
     + OD-RT-VF
     + Solid Angle (has limitations)
     + Multivariate Gaussian PSF
   + ***SPECT and CT*** 
     + Pyramidal uniform phantom
   +  ***Compton***
      + Conical - No GPU optimizer sibling

**GPU optimizer - Multiple kernel**
   + LM - MLEM
   + MLEM
   + LM - MRP
   + MRP


## 🛠️ Installation
```bash
    pip install git+https://github.com/DRIM-UA/infinity-tomographic-reconstruction.git
```

## 📖 Documentation

Some guides and API references are available in the docs/ directory or on the documentation site.

https://crispim-enc.github.io/Infinity-Tomographic-Reconstruction/

## 🛡️ License

This project is licensed under the CC BY-NC-SA 4.0 License. See the LICENSE file for details.

## 🏆 Acknowledgments

    Funding from 2019.143964.BD grant - FCT

## 📬 Contact

    Project Lead: Pedro Encarnação
    Website: https://crispim-enc.github.io/Infinity-Tomographic-Reconstruction/
   
## ⚠️ Disclaimer

This repository is provided for non-commercial research and educational purposes only. It contains or builds upon
components that are licensed under terms which prohibit commercial use.

The author:

+ Does not authorize the use of this repository, in whole or in part, in any commercial application or product.

+ Shall not be held responsible for any direct or indirect consequences arising from unauthorized commercial use.

+ Disclaims all liability for any outcomes, conclusions, or decisions—particularly those of a medical or clinical 
nature—derived from the use of this repository.

+ Makes no guarantees regarding the accuracy, reliability, or fitness of the code for any specific application.

+ Has not received any financial compensation related to the use or distribution of this repository.

By using this repository, you agree to comply with all applicable license terms, 
including the CC BY-NC-SA 4.0 license under which this project is distributed.
