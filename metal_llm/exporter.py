"""
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List
import os
import shutil
import subprocess

import torch
try:
    import coremltools as ct
except Exception:
    ct = None


@dataclass
class ExportConfig:
    model_id: str
    output_dir: str
    precision: str = "float16"  # "float16" | "float32"
    select_layers: Optional[List[int]] = None  # None -> full export
    convert_attention_only: bool = False
    opset_version: int = 13


def export_coreml(cfg: ExportConfig) -> str:
    """
    Export a HF model (or subgraph) to CoreML via torchscript->mlprogram path.
    - precision: fp16/fp32
    - select_layers: export decoder blocks subset (if supported by model)
    - convert_attention_only: attempt to export only attention modules
    """
    if ct is None:
        raise RuntimeError("coremltools is not installed. Please `pip install coremltools`.\n")

    os.makedirs(cfg.output_dir, exist_ok=True)

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, torch_dtype=torch.float32)
    model.eval()

    # Prepare a minimal forward for export (single step)
    hidden_size = getattr(model.config, "hidden_size", getattr(model.config, "d_model", 4096))
    seq_len = 16
    B = 1
    dummy = torch.randint(0, getattr(model.config, "vocab_size", 32000), (B, seq_len))

    def forward_export(input_ids: torch.Tensor):
        out = model(input_ids)
        return out.logits

    traced = torch.jit.trace(forward_export, (dummy,))
    traced = torch.jit.freeze(traced)

    compute_precision = ct.precision.FLOAT32 if cfg.precision == "float32" else ct.precision.FLOAT16
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="input_ids", shape=dummy.shape, dtype=ct.precision.INT32)],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=compute_precision,
    )

    out_path = os.path.join(cfg.output_dir, "MetalLLM.mlpackage")
    mlmodel.save(out_path)
    return cfg.output_dir


def generate_swift_package(output_dir: str) -> str:
    """
    Scaffold a Swift Package that wraps the MLModel.
    Layout:
    - Package.swift
    - Sources/MetalLLMKit/MetalLLMKit.swift
    - Resources/MetalLLM.mlpackage (symlink or copy)
    """
    pkg_dir = os.path.join(output_dir, "SwiftPackage")
    src_dir = os.path.join(pkg_dir, "Sources", "MetalLLMKit")
    res_dir = os.path.join(pkg_dir, "Resources")
    os.makedirs(src_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    # Copy/symlink the mlpackage if present
    mlpackage_src = os.path.join(output_dir, "MetalLLM.mlpackage")
    mlpackage_dst = os.path.join(res_dir, "MetalLLM.mlpackage")
    if os.path.exists(mlpackage_src) and not os.path.exists(mlpackage_dst):
        try:
            os.symlink(mlpackage_src, mlpackage_dst)
        except Exception:
            shutil.copytree(mlpackage_src, mlpackage_dst)

    package_swift = f"""// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MetalLLMKit",
    platforms: [
        .iOS(.v16), .macOS(.v13)
    ],
    products: [
        .library(name: "MetalLLMKit", targets: ["MetalLLMKit"]),
    ],
    targets: [
        .target(
            name: "MetalLLMKit",
            resources: [
                .process("../Resources/MetalLLM.mlpackage")
            ]
        )
    ]
)
"""
    with open(os.path.join(pkg_dir, "Package.swift"), "w") as f:
        f.write(package_swift)

    kit_swift = """
import CoreML

public final class MetalLLMKit {
    private let model: MLModel

    public init() throws {
        let url = Bundle.module.url(forResource: "MetalLLM", withExtension: "mlpackage")!
        let compiled = try MLModel.compileModel(at: url)
        self.model = try MLModel(contentsOf: compiled)
    }

    public func predict(inputIds: MLMultiArray) throws -> MLMultiArray {
        let input = MLDictionaryFeatureProvider(dictionary: ["input_ids": inputIds])
        let out = try model.prediction(from: input)
        return out.featureValue(for: "output")!.multiArrayValue!
    }
}
"""
    with open(os.path.join(src_dir, "MetalLLMKit.swift"), "w") as f:
        f.write(kit_swift)

    return pkg_dir


