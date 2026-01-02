"""Export Flax ActorNet parameters (Dense kernels/biases) to a C header.

Expected checkpoint format (pickle):
    params = {"actor": agent.actor_states.params}

This script:
- loads pickle ckpt
- extracts Dense_0..Dense_{num_layers} params from actor tree
- ignores actor_logstd
- writes a .h with static const float arrays

Flax nn.Dense parameter shapes:
- kernel: (in_features, out_features)
- bias:   (out_features,)
Storage convention in generated header:
- W is row-major flattened with index W[i*out_dim + j] = kernel[i, j]
- b is b[j]

Then in C you can do:
    for (j=0; j<out_dim; ++j) {
        float s = b[j];
        for (i=0; i<in_dim; ++i) s += x[i] * W[i*out_dim + j];
        y[j] = s;
    }
"""

from __future__ import annotations

import argparse
import datetime as _dt
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _dense_names_in_actor(actor_tree: Dict[str, Any]) -> List[str]:
    """Return sorted dense module names present."""
    dense_names = [k for k in actor_tree.keys() if k.startswith("Dense_")]

    # Sort by the integer suffix
    def _idx(name: str) -> int:
        try:
            return int(name.split("_", 1)[1])
        except Exception:
            return 10**9

    dense_names.sort(key=_idx)
    return dense_names


def _extract_dense_params(actor_tree: Dict[str, Any]) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Extract list of (kernel, bias) from Dense_0..Dense_N in order."""
    dense_names = _dense_names_in_actor(actor_tree)
    if not dense_names:
        raise ValueError("No Dense_* modules found in actor params tree.")

    layers: List[Tuple[np.ndarray, np.ndarray]] = []
    for name in dense_names:
        node = actor_tree[name]
        if "kernel" not in node or "bias" not in node:
            raise ValueError(f"Module {name} missing kernel/bias keys: {node.keys()}")

        k = np.asarray(node["kernel"], dtype=np.float32)  # (in, out)
        b = np.asarray(node["bias"], dtype=np.float32)  # (out,)
        if k.ndim != 2:
            raise ValueError(f"{name}.kernel expected 2D, got {k.shape}")
        if b.ndim != 1 or b.shape[0] != k.shape[1]:
            raise ValueError(f"{name}.bias shape {b.shape} not compatible with kernel {k.shape}")

        layers.append((k, b))

    return layers


def _c_float_literal(x: float) -> str:
    """Format float with enough precision and an explicit 'f' suffix."""
    # Use repr-like precision; avoid scientific notation issues by using %.9g
    s = f"{x:.9g}"
    # Ensure it contains a decimal or exponent to be treated as float literal in C
    if ("e" not in s) and ("E" not in s) and ("." not in s):
        s += ".0"
    return s + "f"


def _format_c_array_1d(name: str, arr: np.ndarray, cols: int = 8) -> str:
    """Format a 1D float32 array into C initializer."""
    flat = arr.reshape(-1).astype(np.float32)
    lines = []
    for i in range(0, flat.size, cols):
        chunk = ", ".join(_c_float_literal(float(v)) for v in flat[i : i + cols])
        lines.append("  " + chunk + ("," if i + cols < flat.size else ""))
    body = "\n".join(lines)
    return f"static const float {name}[{flat.size}] = {{\n{body}\n}};\n"


def _format_dense_layer(layer_idx: int, k: np.ndarray, b: np.ndarray) -> str:
    in_dim, out_dim = k.shape
    # Row-major flatten: W[i*out + j] = k[i, j]
    w_flat = k.reshape(-1)  # numpy default row-major is what we want here
    parts = []
    parts.append(f"// Layer {layer_idx}: in={in_dim}, out={out_dim}\n")
    parts.append(f"static const int ACTOR_L{layer_idx}_IN  = {in_dim};\n")
    parts.append(f"static const int ACTOR_L{layer_idx}_OUT = {out_dim};\n")
    parts.append(_format_c_array_1d(f"actor_W{layer_idx}", w_flat))
    parts.append(_format_c_array_1d(f"actor_b{layer_idx}", b))
    parts.append("\n")
    return "".join(parts)


def export_header(
    ckpt_path: Path, out_path: Path, header_guard: str, prefix_comment: str | None = None
) -> None:
    """Export actor Dense params from Flax checkpoint to C header."""
    with open(ckpt_path, "rb") as f:
        params = pickle.load(f)

    actor_params = params["actor"]

    # In Flax TrainState params are typically {"params": {...}}
    if isinstance(actor_params, dict) and "params" in actor_params:
        actor_tree = actor_params["params"]
    else:
        actor_tree = actor_params

    # actor_tree contains Dense_* and actor_logstd. We ignore actor_logstd.
    layers = _extract_dense_params(actor_tree)

    # --- Derive basic network metadata from extracted Dense layers ---
    # ActorNet layout: [Dense_0 .. Dense_{N-2}] are hidden, last Dense is output.
    if len(layers) < 1:
        raise ValueError("No Dense layers found; cannot export header.")

    input_size = int(layers[0][0].shape[0])  # first kernel in-dim
    output_size = int(layers[-1][0].shape[1])  # last kernel out-dim
    num_dense = int(len(layers))  # total Dense modules
    num_hidden_layers = max(num_dense - 1, 0)  # hidden Dense count
    hidden_size = int(layers[0][0].shape[1]) if num_hidden_layers > 0 else 0

    # Validate hidden layers have consistent hidden_size (optional but recommended)
    for li, (k, b) in enumerate(layers[:-1]):  # exclude output layer
        if int(k.shape[1]) != hidden_size:
            raise ValueError(
                f"Hidden layer width mismatch at layer {li}: "
                f"expected out_dim={hidden_size}, got {k.shape[1]}"
            )

    # --- Parameter counting and memory footprint ---
    # Count only Dense params (kernel + bias) because we ignore actor_logstd for deployment.
    total_params = 0
    for k, b in layers:
        total_params += int(k.size) + int(b.size)

    bytes_per_param = 4  # float32
    total_bytes = total_params * bytes_per_param
    total_kib = total_bytes / 1024.0

    # Sanity: last layer output is usually act_dim=4, but we don't hard-require.

    now = _dt.datetime.now().isoformat(timespec="seconds")
    guard = header_guard.upper().replace("-", "_")
    if not guard.endswith("_H"):
        guard += "_H"

    lines: List[str] = []
    lines.append(f"// Auto-generated from: {ckpt_path}\n")
    lines.append(f"// Generated at: {now}\n")
    if prefix_comment:
        lines.append(f"// {prefix_comment}\n")

    # --- Network metadata in comments ---
    lines.append("//\n")
    lines.append("// Actor network metadata (derived from Dense params, excluding actor_logstd):\n")
    lines.append(f"// - input_size   : {input_size}\n")
    lines.append(f"// - output_size  : {output_size}\n")
    lines.append(f"// - num_layers   : {num_hidden_layers} (hidden Dense layers)\n")
    lines.append(f"// - hidden_size  : {hidden_size}\n")
    lines.append(f"// - dense_layers : {num_dense} (including output Dense)\n")
    lines.append(f"// - total_params : {total_params} (kernel+bias only)\n")
    lines.append(f"// - param_bytes  : {total_bytes} bytes (float32)\n")
    lines.append(f"// - param_size   : {total_kib:.3f} KiB (float32)\n")
    lines.append("//\n")

    # --- Storage convention comments ---
    lines.append("// Storage convention:\n")
    lines.append("// - Flax Dense kernel shape is (in_features, out_features)\n")
    lines.append("// - We store actor_Wk as row-major flattened kernel:\n")
    lines.append("//     actor_Wk[i*out + j] == kernel[i, j]\n")
    lines.append("// - Bias stored as actor_bk[j]\n")
    lines.append("//\n\n")

    # Header guard + extern "C"
    lines.append(f"#ifndef {guard}\n")
    lines.append(f"#define {guard}\n\n")

    lines.append('#ifdef __cplusplus\nextern "C" {\n#endif\n\n')

    # --- Export metadata as C constants too (useful for asserts / tooling) ---
    lines.append(f"static const int ACTOR_INPUT_SIZE  = {input_size};\n")
    lines.append(f"static const int ACTOR_OUTPUT_SIZE = {output_size};\n")
    lines.append(f"static const int ACTOR_HIDDEN_SIZE = {hidden_size};\n")
    lines.append(f"static const int ACTOR_NUM_LAYERS  = {num_hidden_layers};\n")
    lines.append(f"static const int ACTOR_NUM_DENSE   = {num_dense};\n")
    lines.append(f"static const int ACTOR_TOTAL_PARAMS = {total_params};\n")
    lines.append(f"static const int ACTOR_PARAM_BYTES  = {total_bytes};\n\n")

    # Dense layers
    for li, (k, b) in enumerate(layers):
        lines.append(_format_dense_layer(li, k, b))

    # Footer
    lines.append("#ifdef __cplusplus\n}\n#endif\n\n")
    lines.append(f"#endif  // {guard}\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    """Main."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--exp_name",
        type=str,
        default="bptt_rprv",
        help="Experiment name to locate the pickle ckpt (your .ckpt file)",
    )
    ap.add_argument(
        "--out", type=str, default="actor_params.h", help="Output header path, e.g. actor_params.h"
    )
    ap.add_argument("--guard", type=str, default="ACTOR_PARAMS_H", help="Header guard macro")
    ap.add_argument(
        "--comment",
        type=str,
        default="ActorNet params for deterministic inference (mean only).",
        help="Extra comment line in header",
    )
    args = ap.parse_args()

    model_path = Path(__file__).parents[1] / f"saves/{args.exp_name}_model.ckpt"
    out_path = Path(__file__).parents[1] / f"saves/{args.exp_name}_policy_params.h"

    export_header(
        ckpt_path=model_path,
        out_path=out_path,
        header_guard=args.guard,
        prefix_comment=args.comment,
    )
    print(f"Wrote header: {out_path}")


if __name__ == "__main__":
    main()
