# A finite-volume based physics-informed Fourier neural operator network for parametric learning of subsurface flow

This study introduces a novel finite-volume based physics-informed Fourier neural operator (FV-PIFNO) (https://doi.org/10.1016/j.advwatres.2025.105087)  for parametric learning of subsurface flow in heterogeneous porous media. The existing physics-informed neural operators struggle with heterogeneous parameter fields due to challenges in automatic differentiation, thus their applicability to parametric learning of subsurface flow remains limited. To address these limitations, FV-PIFNO integrates finite volume method (FVM) discretization of governing equations into the physics-informed loss function, bypassing automatic differentiation (AD) related issues and ensuring flux continuity across heterogeneous domains. A gated Fourier neural operator (Gated-FNO) with space-frequency cooperative filtering is developed to enhance feature extraction and noise suppression. The framework is validated through 2D and 3D heterogeneous reservoir models, demonstrating superior performance in scenarios involving sparse data, variable permeability ratios, and diverse correlation lengths. Results show that FV-PIFNO achieves higher accuracy and robustness compared to data-driven counterparts, particularly under extreme data scarcity. The method’s ability to generalize across untrained parameter spaces and maintain physical consistency in velocity fields highlights its potential as an efficient surrogate model for subsurface heterogeneous flow applications. It should be noted that the present work only considers the steady-state subsurface flow problems, and the unsteady-state problems will be addressed in future work.

# FV-PIFNO-Data
 The the dataset used in FV-PIFNO model training is available at https://drive.google.com/drive/folders/14o40TUStynrOySdDWEqR6Ks7HNEn681m?usp=drive_link
# Citation
@article{yan2025finite,
  title={A Finite-Volume Based Physics-Informed Fourier Neural Operator Network for Parametric Learning of Subsurface Flow},
  author={Yan, Xia and Lin, Jingqi and Ju, Yafeng and Zhang, Qi and Zhang, Zhao and Zhang, Liming and Yao, Jun and Zhang, Kai},
  journal={Advances in Water Resources},
  pages={105087},
  year={2025},
  publisher={Elsevier}
}
