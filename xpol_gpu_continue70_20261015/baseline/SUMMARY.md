# TiO2 voxel3d X/Y polarizer run

- period: 4.000 um
- grid: 400 x 400 (10.0 nm pixel)
- layers: 100 x 10.0 nm = 1.000 um
- wavelengths: 400, 500, 600, 700 nm
- Fourier orders: +/-10
- objective mode: softmin; maximize X total transmission and suppress Y total transmission
- evaluation: periodic RCWA total transmission, not finite-aperture FDTD or measured reflection
- layer ordering: index 0 at top, last layer on substrate; support gate runs bottom to top
- support projection: posts_vented; applied before every binary forward solve
- posts_vented adds substrate posts and opens enclosed air; local_grounded deletes unsupported components
- this projection does not certify minimum neck/gap or full process printability
- objective changed in a separate branch; global step numbers retained, Adam moments reset; see objective_branch.json
- state.pt contains post-update theta; snapshot arrays and best_state.pt match evaluated pre-update geometry

## Best binary candidate

- step: 180
- threshold: 0.5
- F: 0.383171
- mean T_X: 0.790602
- min T_X: 0.775509
- mean T_Y: 0.289043
- max T_Y: 0.291117
- mean delta: 0.501559
- min delta: 0.484392
- mean ER: 4.370 dB
- min ER: 4.255 dB

| wavelength | T_X | T_Y | delta | contrast | ER dB |
|---:|---:|---:|---:|---:|---:|
| 400 | 0.7755 | 0.2911 | 0.4844 | 0.4541 | 4.26 |
| 500 | 0.7870 | 0.2911 | 0.4959 | 0.4600 | 4.32 |
| 600 | 0.7992 | 0.2896 | 0.5096 | 0.4681 | 4.41 |
| 700 | 0.8006 | 0.2844 | 0.5163 | 0.4758 | 4.50 |
