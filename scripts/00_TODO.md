**1.- One last iteration to see if we can beat UNet**
1a.- distance to crater information
1b



**3.- antes de seguir, si MPNN es mejorq ue Unet para cross volcano, proseguimos con continuous, si no, hay que ver otra arquitectura o scope para el artículo**
What this implies for design (your "more general recommendations")

Scientific Question	Switch
Does global temporal context help?	Bottleneck Attention
Is fixed station identity necessary?	Shared Station Encoder
Do early station interactions help?	Early PairConv
Do repeated station interactions help?	Hierarchical PairConv
Does learned message weighting improve PairConv?	Sum vs Attention aggregation
Does station attention improve PairConv?	Station Attention

**6.- object detection**
boundaries by heatmap, then integration on original segmentation, we only train the head
