"""Maintainer tool: export the three SigLIP2 members (per-patch vision + text towers) to ONNX.

The app never runs this. Its output is uploaded once as the `models-v1` GitHub release, and the Rust server
downloads those files at runtime. Run it with:

    uv run --no-project --with torch==2.6.0 --with transformers==4.53.3 --with onnx --with onnxruntime \
        --with pillow --with sentencepiece --with numpy python tools/export_models.py <out_dir> [<reference_image>]

Per patch: every token of the vision tower goes through the attention-pooling head ALONE (MaskCLIP), which lands
it in the text-aligned space, then it is L2-normalized. NaFlex members take the position embedding already
resized to the frame's patch grid as an input (the server resizes the 16x16 grid in `pos_grid.bin`), so the graph
has no data-dependent interpolation in it.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from transformers import AutoModel, AutoProcessor

MEMBERS = {
    "base224": "google/siglip2-base-patch16-224",
    "base_naflex": "google/siglip2-base-patch16-naflex",
    "so400m_naflex": "google/siglip2-so400m-patch16-naflex",
}
TEXT_LENGTH = 64
OPSET = 17


class FixedPatches(nn.Module):
    def __init__(self, vision):
        super().__init__()
        self.vision = vision

    def forward(self, pixel_values):
        hidden = self.vision(pixel_values=pixel_values).last_hidden_state
        batch, patches, dim = hidden.shape
        per_patch = self.vision.head(hidden.reshape(batch * patches, 1, dim)).reshape(batch, patches, -1)
        return F.normalize(per_patch, dim=-1)


class NaflexPatches(nn.Module):
    def __init__(self, vision):
        super().__init__()
        self.vision = vision

    def forward(self, patches, pos):
        embeddings = self.vision.embeddings.patch_embedding(patches) + pos
        hidden = self.vision.encoder(inputs_embeds=embeddings).last_hidden_state
        hidden = self.vision.post_layernorm(hidden)
        count, dim = hidden.shape[1], hidden.shape[2]
        per_patch = self.vision.head(hidden.reshape(count, 1, dim)).reshape(1, count, -1)
        return F.normalize(per_patch, dim=-1)


class Text(nn.Module):
    def __init__(self, text_model):
        super().__init__()
        self.text_model = text_model

    def forward(self, input_ids):
        return F.normalize(self.text_model(input_ids=input_ids).pooler_output, dim=-1)


def naflex_size(height, width, patch, budget, eps=1e-5):
    """transformers' get_image_size_for_max_num_patches (the server reimplements it)."""

    def scaled(scale, size):
        return int(max(patch, math.ceil(size * scale / patch) * patch))

    low, high = eps / 10, 100.0
    while high - low >= eps:
        scale = (low + high) / 2
        if (scaled(scale, height) / patch) * (scaled(scale, width) / patch) <= budget:
            low = scale
        else:
            high = scale
    return scaled(low, height), scaled(low, width)


def resized_pos(grid, rows, cols):
    """transformers' Siglip2VisionEmbeddings.resize_positional_embeddings for one image."""
    as_image = grid.permute(2, 0, 1).unsqueeze(0)
    resized = F.interpolate(as_image, size=(rows, cols), mode="bilinear", align_corners=False, antialias=True)
    return resized.reshape(grid.shape[-1], rows * cols).transpose(0, 1).unsqueeze(0)


def max_abs(a, b):
    return float(np.abs(np.asarray(a) - np.asarray(b)).max())


def export_member(key, repo, out, image):
    model = AutoModel.from_pretrained(repo).eval()
    processor = AutoProcessor.from_pretrained(repo)
    naflex = "naflex" in repo
    vision = model.vision_model
    meta = {"repo": repo, "naflex": naflex, "dim": model.config.vision_config.hidden_size, "text_length": TEXT_LENGTH}
    meta["patch"] = model.config.vision_config.patch_size
    reference = {}
    with torch.inference_mode():
        if naflex:
            grid_side = int(vision.embeddings.position_embedding_size)
            grid = vision.embeddings.position_embedding.weight.reshape(grid_side, grid_side, -1).float()
            (out / f"{key}.pos_grid.bin").write_bytes(grid.numpy().astype("<f4").tobytes())
            meta["pos_grid_side"] = grid_side
            budget = 1024
            meta["max_patches"] = budget
            inputs = processor(images=[image], return_tensors="pt", max_num_patches=budget)
            rows, cols = (int(v) for v in inputs["spatial_shapes"][0])
            assert (rows * 16, cols * 16) == naflex_size(image.height, image.width, 16, budget), "naflex_size drifted"
            patches = inputs["pixel_values"][:, : rows * cols]
            pos = resized_pos(grid, rows, cols)
            wrapper = NaflexPatches(vision).eval()
            # the HF path, as hype ran it: full vision forward with its own mask and resize
            hidden = vision(
                pixel_values=inputs["pixel_values"],
                attention_mask=inputs["pixel_attention_mask"],
                spatial_shapes=inputs["spatial_shapes"],
            ).last_hidden_state[0, : rows * cols]
            truth = F.normalize(vision.head(hidden.reshape(rows * cols, 1, -1)).reshape(rows * cols, -1), dim=-1)
            mine = wrapper(patches, pos)[0]
            print(key, "wrapper vs HF max abs", max_abs(mine, truth))
            torch.onnx.export(
                wrapper,
                (patches, pos),
                str(out / f"{key}.vision.onnx"),
                input_names=["patches", "pos"],
                output_names=["embeddings"],
                dynamic_axes={"patches": {1: "n"}, "pos": {1: "n"}, "embeddings": {1: "n"}},
                opset_version=OPSET,
            )
            session = ort.InferenceSession(str(out / f"{key}.vision.onnx"))
            onnx_out = session.run(None, {"patches": patches.numpy(), "pos": pos.numpy()})[0][0]
            print(key, "onnx vs torch max abs", max_abs(onnx_out, mine))
            reference.update(rows=rows, cols=cols, pos_row0=pos[0, 0, :8].tolist(), pos_last=pos[0, -1, :8].tolist())
        else:
            size = model.config.vision_config.image_size
            meta["image_size"] = size
            inputs = processor(images=[image], return_tensors="pt")
            wrapper = FixedPatches(vision).eval()
            mine = wrapper(inputs["pixel_values"])[0]
            torch.onnx.export(
                wrapper,
                (inputs["pixel_values"],),
                str(out / f"{key}.vision.onnx"),
                input_names=["pixel_values"],
                output_names=["embeddings"],
                dynamic_axes={"pixel_values": {0: "batch"}, "embeddings": {0: "batch"}},
                opset_version=OPSET,
            )
            session = ort.InferenceSession(str(out / f"{key}.vision.onnx"))
            onnx_out = session.run(None, {"pixel_values": inputs["pixel_values"].numpy()})[0][0]
            print(key, "onnx vs torch max abs", max_abs(onnx_out, mine))
            side = size // meta["patch"]
            reference.update(rows=side, cols=side)
        reference["patch0"] = mine[0, :8].tolist()
        reference["patch_mid"] = mine[mine.shape[0] // 2, :8].tolist()
        reference["mean"] = mine.mean(0)[:8].tolist()

        text = Text(model.text_model).eval()
        tokens = processor(text=["a traffic cone"], return_tensors="pt", padding="max_length", max_length=TEXT_LENGTH)
        ids = tokens["input_ids"]
        torch.onnx.export(
            text,
            (ids,),
            str(out / f"{key}.text.onnx"),
            input_names=["input_ids"],
            output_names=["embedding"],
            dynamic_axes={"input_ids": {0: "batch"}, "embedding": {0: "batch"}},
            opset_version=OPSET,
        )
        text_truth = text(ids)[0]
        session = ort.InferenceSession(str(out / f"{key}.text.onnx"))
        print(key, "text onnx vs torch", max_abs(session.run(None, {"input_ids": ids.numpy()})[0][0], text_truth))
        reference["text_ids"] = ids[0].tolist()
        reference["text"] = text_truth[:8].tolist()
        reference["cone_score_patch0"] = float(mine[0] @ text_truth)
    tokenizer_file = Path(processor.tokenizer.vocab_file).parent / "tokenizer.json"
    (out / f"{key}.tokenizer.json").write_bytes(tokenizer_file.read_bytes())
    (out / f"{key}.json").write_text(json.dumps(meta, indent=2))
    return reference


PROBES = ["a traffic cone", "a fire extinguisher", "a chair", "a person walking", "the door to the kitchen"]


def quantize_text(key, out):
    """Text towers are mostly their 256k-token embedding table. The table is stored fp16 (a Gather then a Cast back to
    fp32) and every MatMul weight int8 per output channel, ~3x smaller; the probes below must keep cosine >= 0.995
    with the fp32 tower (int8 on the table too measured 0.990, too close)."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    from onnxruntime.quantization import QuantType, quantize_dynamic

    source = out / f"{key}.text.onnx"
    halved = out / f"{key}.text.f16table.onnx"
    target = out / f"{key}.text.small.onnx"
    model = onnx.load(str(source))
    initializers = {init.name: init for init in model.graph.initializer}
    for index, node in enumerate(list(model.graph.node)):
        table = initializers.get(node.input[0]) if node.op_type == "Gather" else None
        if table is None or table.dims[0] < 100_000:
            continue
        values = numpy_helper.to_array(table).astype(np.float16)
        table.CopyFrom(numpy_helper.from_array(values, table.name))
        original = node.output[0]
        node.output[0] = original + "_f16"
        cast = helper.make_node("Cast", [node.output[0]], [original], to=TensorProto.FLOAT)
        model.graph.node.insert(list(model.graph.node).index(node) + 1, cast)
    onnx.save_model(model, str(halved), save_as_external_data=True, all_tensors_to_one_file=True, location=halved.name + ".data")
    quantize_dynamic(str(halved), str(target), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"], per_channel=True)
    processor = AutoProcessor.from_pretrained(MEMBERS[key])
    ids = processor(text=PROBES, return_tensors="np", padding="max_length", max_length=TEXT_LENGTH)["input_ids"]
    full = ort.InferenceSession(str(source)).run(None, {"input_ids": ids})[0]
    small = ort.InferenceSession(str(target)).run(None, {"input_ids": ids})[0]
    cosine = (full * small).sum(-1) / np.linalg.norm(full, axis=-1) / np.linalg.norm(small, axis=-1)
    print(key, "small text cosine", cosine.round(4).tolist(), "bytes", target.stat().st_size)
    assert cosine.min() >= 0.995, f"{key}: small text tower drifted ({cosine})"
    return cosine.min()


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    image = Image.open(sys.argv[2]).convert("RGB") if len(sys.argv) > 2 else Image.new("RGB", (640, 360), (200, 90, 30))
    if "--quantize-only" not in sys.argv:
        references = {key: export_member(key, repo, out, image) for key, repo in MEMBERS.items()}
        references["image"] = {"width": image.width, "height": image.height}
        (out / "reference.json").write_text(json.dumps(references, indent=1))
    for key in MEMBERS:
        quantize_text(key, out)


if __name__ == "__main__":
    main()
