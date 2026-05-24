# scripts/inspect_onnx.py
"""
Loads artifacts/model.onnx and prints:
  - input / output tensor specs
  - number of nodes in the graph
  - total parameter count (initializers)
"""

import sys
from pathlib import Path

import numpy as np
import onnx


def main(model_path: str = "artifacts/model.onnx") -> None:
    path = Path(model_path)
    if not path.exists():
        print(f"Model not found: {path}", file=sys.stderr)
        sys.exit(1)

    model = onnx.load(str(path))
    graph = model.graph

    # Input / output specs
    print("=== Inputs ===")
    for inp in graph.input:
        shape = [d.dim_value if d.dim_value else d.dim_param
                 for d in inp.type.tensor_type.shape.dim]
        dtype = onnx.TensorProto.DataType.Name(inp.type.tensor_type.elem_type)
        print(f"  {inp.name}: {dtype}{shape}")

    print("\n=== Outputs ===")
    for out in graph.output:
        shape = [d.dim_value if d.dim_value else d.dim_param
                 for d in out.type.tensor_type.shape.dim]
        dtype = onnx.TensorProto.DataType.Name(out.type.tensor_type.elem_type)
        print(f"  {out.name}: {dtype}{shape}")

    # Node count
    node_count = len(graph.node)
    print("\n=== Graph ===")
    print(f"  Nodes : {node_count}")

    # Separate learned parameters from fixed buffers.
    # Buffers (Hanning window, BN running stats) are ONNX initializers but not trained weights.
    # We identify them by cross-referencing with PyTorch: anything not a gradient-carrying
    # parameter is a buffer. As a conservative proxy: 1-D initializers <= 512 elements that
    # aren't weight/bias pairs are likely BN running stats; the hanning_window is named explicitly.
    buffer_names = {"hanning_window"}  # add any other known buffer names here
    learned = sum(
        np.prod(init.dims)
        for init in graph.initializer
        if init.name not in buffer_names
        and not init.name.endswith("running_mean")
        and not init.name.endswith("running_var")
        and not init.name.endswith("num_batches_tracked")
    )
    total = sum(np.prod(init.dims) for init in graph.initializer)
    print(f"  Learned params : {learned:,}")
    print(f"  Total (incl. buffers): {total:,}")

    # Op breakdown
    from collections import Counter
    op_counts = Counter(n.op_type for n in graph.node)
    print("\n=== Op types ===")
    for op, count in sorted(op_counts.items()):
        print(f"  {op}: {count}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/model.onnx")
    args = parser.parse_args()
    main(args.model)
