# Fesom-C Model - Regional / Coatsal / Deep-Water Ocean Simulations

This repository contains the source code and configuration for the Fesom-C model used in our studies.  
If you use this model or these utilities, please cite our publication:

Kuznetsov, I., Rabe, B., Androsov, A., Fang, Y.-C., Hoppmann, M., Quintanilla-Zurita, A., Harig, S., Tippenhauer, S., Schulz, K., Mohrholz, V., Fer, I., Fofonova, V., and Janout, M.: Dynamical reconstruction of the upper-ocean state in the central Arctic during the winter period of the MOSAiC expedition, Ocean Sci., 20, 759–777, https://doi.org/10.5194/os-20-759-2024, 2024.

## Model Description

FESOM-C implements a finite-volume, cell-vertex discretization on unstructured meshes. Its core features include:

- Hybrid unstructured meshes combining **triangular and quadrilateral cells**, giving flexibility in mesh geometry and resolution. This allows dense resolution in areas of interest and coarser elsewhere. 
- A **terrain-following vertical (sigma) coordinate system**, suitable for coastal, shelf, or variable bathymetry regions. 
- High-order horizontal advection schemes and implicit vertical advection/viscosity.
- Support for wetting/drying, tidal forcing, open boundaries, river inflow (as streams or open-boundary conditions), ...

## Features and Adaptations (as used in this work)
- Use of hybrid meshes so that a high-resolution “core” region (region of interest) is surrounded by coarser-resolution zones, minimizing boundary influence while avoiding nested grids.  
- Vertical representation via sigma coordinates, enabling realistic representation of vertical stratification, bathymetry, and coastal topography.  
- Barotropic (sea-level) computations modified: since our focus is on deep-water regions where bottom friction is minimal and barotropic mode is not dominant, we simplified the semi-implicit sea-level solver by omitting the block of averaged (barotropic) equations.  
- Sea-ice thermodynamics;
- Effect of sea-ice on surface-layer dynamics: parameterized via ocean–ice friction. 
- Parallelization and I/O: we used MPI-based parallel computing, and for boundary condition I/O (open-surface boundaries), we implemented parallel input/output using the PnetCDF library — offering efficient, scalable I/O suitable for large coastal domains.  

## Publications Using This Version
- Kuznetsov, I., Rabe, B., Androsov, A., Fang, Y.-C., Hoppmann, M., Quintanilla-Zurita, A., Harig, S., Tippenhauer, S., Schulz, K., Mohrholz, V., Fer, I., Fofonova, V., and Janout, M.: Dynamical reconstruction of the upper-ocean state in the central Arctic during the winter period of the MOSAiC expedition, Ocean Sci., 20, 759–777, https://doi.org/10.5194/os-20-759-2024, 2024. 
- Debolskaya, E.I., Kuznetsov, I.S. & Androsov, A.A. Studying Hydrophysical Processes in Summer and Winter Periods in the Tidal Arctic Estuary of the Indiga River Using a Mathematical Model FESOM-C. Water Resour 52, 147–171 (2025). https://doi.org/10.1134/S0097807824603674
- Debolskaya, E.I., Kuznetsov, I.S. & Androsov, A.A. Numerical Simulation of Hydrodynamic Processes in Indiga Bay. Power Technol Eng 56, 691–697 (2023). https://doi.org/10.1007/s10749-023-01575-z
- Kuznetsov, I., Androsov, A., Fofonova, V., Danilov, S., Rakowsky, N., Harig, S., Wiltshire, K.H. Evaluation and Application of Newly Designed Finite Volume Coastal Model FESOM-C, Effect of Variable Resolution in the Southeastern North Sea. Water 2020, 12, 1412. https://doi.org/10.3390/w12051412 

## License

The model is distributed under the GNU GPL-3.0 license. 

<img width="600" height="415" alt="image" src="https://github.com/user-attachments/assets/18bad9bb-9810-478c-b394-0d3370d452d4" />
<img width="3164" height="1183" alt="image" src="https://github.com/user-attachments/assets/c70f54e6-da0a-4fc6-b4e3-5c2c790c1289" />

<img width="425" height="169" alt="image" src="https://github.com/user-attachments/assets/fa1994be-f94f-4084-9954-16f48574de43" />
<img width="425" height="627" alt="image" src="https://github.com/user-attachments/assets/efa4bd58-1207-4b35-9bef-2cf547daf66d" />
