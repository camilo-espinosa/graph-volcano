correr : python scripts/01b_prepare_cross-volcano_data.py
despues verificar que edge info se calcula para los otros volcanes tmbn

**1.- correr zero shot para todas las ablations**

hay que verificar que la info the edges es cargada en ablations, zero shot y cross volcano training
**2.- correr cross volcano LOO para todas las ablations**
**3.- antes de seguir, si MPNN es mejorq ue Unet para cross volcano, proseguimos con continuous, si no, hay que ver otra arquitectura o scope para el artículo**
**6.- object detection**
boundaries by heatmap, then integration on original segmentation, we only train the head
