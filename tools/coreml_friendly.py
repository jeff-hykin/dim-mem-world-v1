"""Rewrites an exported SigLIP2 vision graph so ONNX Runtime's CoreML provider runs all of it (same math):

- the patch conv's auto_pad=VALID becomes explicit zero pads (the CoreML ML Program builder drops VALID's 'pad');
- the closing L2 normalize (ReduceL2 -> Clip -> Expand -> Div) becomes Mul -> ReduceSum -> Sqrt -> Clip -> Div:
  CoreML has no ReduceL2, so the GPU part ended on a no-op Reshape that CoreML optimizes away, and its output went missing.

    uv run --no-project --with onnx python tools/coreml_friendly.py <in.onnx> <out.onnx>
"""
import sys

import numpy
import onnx
from onnx import helper, numpy_helper


def rewrite(model: onnx.ModelProto) -> onnx.ModelProto:
    graph = model.graph
    for node in graph.node:
        if node.op_type == "Conv":
            auto_pad = next((a for a in node.attribute if a.name == "auto_pad"), None)
            if auto_pad is not None and auto_pad.s == b"VALID":
                node.attribute.remove(auto_pad)
                node.attribute.append(helper.make_attribute("pads", [0, 0, 0, 0]))

    reduce = next(node for node in graph.node if node.op_type == "ReduceL2")
    clip = next(node for node in graph.node if node.op_type == "Clip" and node.input[0] == reduce.output[0])
    expand = next(node for node in graph.node if node.op_type == "Expand" and node.input[0] == clip.output[0])
    divide = next(node for node in graph.node if node.op_type == "Div" and node.input[1] == expand.output[0])
    vectors = reduce.input[0]
    axes = reduce.input[1] if len(reduce.input) > 1 else None
    keepdims = next((a.i for a in reduce.attribute if a.name == "keepdims"), 1)
    axes_attribute = next((a for a in reduce.attribute if a.name == "axes"), None)
    if axes is None:
        # opset < 18 kept the axes as an attribute; ReduceSum (opset >= 13) takes them as an input
        axes = "coreml_friendly_axes"
        graph.initializer.append(numpy_helper.from_array(numpy.array(list(axes_attribute.ints), dtype=numpy.int64), axes))
    squared = helper.make_node("Mul", [vectors, vectors], ["coreml_friendly_squared"], name="coreml_friendly/Mul")
    summed = helper.make_node("ReduceSum", ["coreml_friendly_squared", axes], ["coreml_friendly_summed"], name="coreml_friendly/ReduceSum", keepdims=keepdims)
    root = helper.make_node("Sqrt", ["coreml_friendly_summed"], [reduce.output[0]], name="coreml_friendly/Sqrt")
    # Div broadcasts, so the Expand goes
    divide.input[1] = clip.output[0]
    index = list(graph.node).index(reduce)
    for node in (reduce, expand):
        graph.node.remove(node)
    for offset, node in enumerate((squared, summed, root)):
        graph.node.insert(index + offset, node)
    onnx.checker.check_model(model)
    return model


if __name__ == "__main__":
    source, target = sys.argv[1], sys.argv[2]
    onnx.save(rewrite(onnx.load(source)), target)
