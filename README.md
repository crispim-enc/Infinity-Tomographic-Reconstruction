# Infinity-Tomographic-Reconstruction
<div align="center">
<img src="docs/_static/logo.png" alt="toor Logo" width="300"/>
</div>
Tomographic reconstruction for diferent imaging modalities (PET, SPECT, CT and Compton)
Image Reconstruction Framework for PET/CT and SPECT


## 🚀 Overview

This repository provides modular image reconstruction framework designed for Positron Emission Tomography (PET),
Computed Tomography (CT), and Single Photon Emission Computed Tomography (SPECT). 


You can define the **Device** type
 + PET
 + CT
 + SPECT
 

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


**GPU optimizer - Multiple kernel**
   + LM - MLEM
   + MLEM
   + LM - MRP
   + MRP

⚠️ This project is under active development. Some features may be incomplete or change without notice.

## 🛠️ Installation
Prerequisites:
+ Cuda Toolkit 11.3 or higher, 
+ Python 3.8, pip, cl compiler.*

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

## 📖 Cite

If you use this software in your research please cite the following articles accordingly:


NEMA Phantoms, NEMA IQ analysis and simulations present in Zenodo

+ Encarnação, P. M. C. C., Correia, P. M. M., Ribeiro, F. M., & Veloso, J. F. C. A. (2025). 
Timing performance evaluation of a dual-axis rotational PET system according to NEMA NU 4-2008 standards: A simulation study. Biomedical Physics & Engineering Express, 11(3), 035012. https://doi.org/10.1088/2057-1976/adc5f5
 
PET Reconstruction and ODRTVF projector
+ Encarnação, P. M. C. C., Correia, P. M. M., Silva, A. L. M., Ribeiro, F. M., Castro, I. F., & Veloso, J. F. C. A. (2025).
A modified orthogonal-distance ray-tracer method applied to dual rotation PET systems. Physics in Medicine & Biology, 70(2), 025021. https://doi.org/10.1088/1361-6560/ada718


Pyramidal Projector
+ P. M. C. C. Encarnação et al., "Individual and Simultaneous Imaging of ⁹⁹mTc and ¹⁷⁷Lu With a Preclinical Broad Energy-Spectrum CZT-Based SPECT," in 
IEEE Transactions on Radiation and Plasma Medical Sciences, vol. 9, no. 5, pp. 564-577, May 2025, doi: 10.1109/TRPMS.2025.3527874. https://ieeexplore.ieee.org/document/10836142

Any use this code should cite the P. M. C. C. Encarnação PhD thesis.
+ Encarnação, P. M. C. C. (2025). Intelligent scanning method for adaptive Positron Emission Tomography 
[PhD thesis, University of Aveiro]. Repositório Institucional da Universidade de Aveiro.http://hdl.handle.net/10773/44507

If you use any of the simulation data in the Zenodo repository, please cite also the Zenodo record:
+ Encarnação, P. M. C. C., Correia, P. M. M., Silva, A., Ribeiro, F. M., Castro, I. F., & Veloso, J. F. C. A. (2025). 
A Modified Orthogonal-Distance Ray-Tracer Method Applied to Dual Rotation PET Systems (raw data) [Data set]. 
In A modified orthogonal-distance ray-tracer method applied to dual rotation PET systems (1.0.0). Zenodo. https://doi.org/10.5281/zenodo.12761721

+ Encarnação, P., Correia, P., de Meneses Ribeiro, F., & Veloso, J. (2024). Timing Performance Evaluation of a Dual-Axis Rotational PET System According to NEMA NU 4-2008 Standards: A Simulation Study [Data set]. In Timing Performance Evaluation of a Dual-Axis Rotational PET System According to NEMA NU 4-2008 Standards: A Simulation Study. Zenodo. https://doi.org/10.5281/zenodo.12773305

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

+ Has not received any financial compensation related to the use or distribution of this repository by third parties.

By using this repository, you agree to comply with all applicable license terms, 
including the CC BY-NC-SA 4.0 license under which this project is distributed 
