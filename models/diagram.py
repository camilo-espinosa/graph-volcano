from models.UNet_GraphSAGE import UNet_GraphSAGE
import torch

import os

os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"
GRAPH_LEVELS = [3, 4]
V5_FULL_KWARGS = {
    "graph_levels": GRAPH_LEVELS,
    "attention_pool_mode": "all_levels",
    "use_bottleneck_attention": True,
    "graph_norm_type": "graphnorm",
    "node_feature_mode": "both",
    "graph_backend": "graphsage",
    "use_message_passing": True,
    "virtual_node_pool_mode": "learned",
    "bottleneck_virtual_node_pool_mode": "learned",
    "use_skip_graph": True,
    "init_features": 16,
    "depth": 5,
}
model = UNet_GraphSAGE(
    in_channels=1,
    out_channels=6,
    **V5_FULL_KWARGS,
)


model.eval()

dummy_input = torch.randn(1, 8, 8192)  # batch size

# path = "C:\\CAMILO\\Volcanes_UFRO\\CODES\\graph-volcano\\results\\model.onnx"
# torch.onnx.export(
#     model,
#     dummy_input,
#     path,
#     export_params=True,
#     opset_version=17,
#     do_constant_folding=True,
#     input_names=["input"],
#     output_names=["output"],
#     dynamic_axes={
#         "input": {0: "batch_size"},
#         "output": {0: "batch_size"}
#     }
# )

# print("Exported model.onnx")


from torchview import draw_graph

graph = draw_graph(model, input_size=(1, 8, 8192), expand_nested=True, depth=1)
graph.visual_graph.render("architecture_depth_1_nested")
