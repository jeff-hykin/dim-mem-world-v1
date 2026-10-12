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
    """Text towers are mostly their 256k-token embedding table. Every big weight is stored fp16: the table is gathered
    in fp16 and its rows cast to fp32 (so RAM holds it at half size); every other big weight is cast to fp32 once at
    load. Weight-only, so the arithmetic is fp32 and the result doesn't depend on which kernels an ONNX Runtime build
    fuses (dynamic int8 measured cosine 0.998 fused but 0.88-0.95 unfused, which is what some builds run).
    The probes below must keep cosine >= 0.999 with the fp32 tower."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    source = out / f"{key}.text.onnx"
    target = out / f"{key}.text.small.onnx"
    model = onnx.load(str(source))
    graph = model.graph
    initializers = {init.name: init for init in graph.initializer}
    tables = set()
    for node in list(graph.node):
        table = initializers.get(node.input[0]) if node.op_type == "Gather" else None
        if table is None or table.dims[0] < 100_000:
            continue
        tables.add(table.name)
        original = node.output[0]
        node.output[0] = original + "_f16"
        graph.node.insert(list(graph.node).index(node) + 1, helper.make_node("Cast", [node.output[0]], [original], to=TensorProto.FLOAT))
    casts = []
    for init in graph.initializer:
        if init.data_type != TensorProto.FLOAT or int(np.prod(init.dims)) < 65_536:
            continue
        values = numpy_helper.to_array(init).astype(np.float16)
        if init.name in tables:
            init.CopyFrom(numpy_helper.from_array(values, init.name))
            continue
        name = init.name
        init.CopyFrom(numpy_helper.from_array(values, name + "_f16"))
        casts.append(helper.make_node("Cast", [name + "_f16"], [name], to=TensorProto.FLOAT))
    for cast in reversed(casts):
        graph.node.insert(0, cast)
    onnx.save_model(model, str(target))
    processor = AutoProcessor.from_pretrained(MEMBERS[key])
    ids = processor(text=PROBES, return_tensors="np", padding="max_length", max_length=TEXT_LENGTH)["input_ids"]
    full = ort.InferenceSession(str(source)).run(None, {"input_ids": ids})[0]
    worst = 1.0
    for level in [ort.GraphOptimizationLevel.ORT_DISABLE_ALL, ort.GraphOptimizationLevel.ORT_ENABLE_ALL]:
        options = ort.SessionOptions()
        options.graph_optimization_level = level
        small = ort.InferenceSession(str(target), options).run(None, {"input_ids": ids})[0]
        cosine = (full * small).sum(-1) / np.linalg.norm(full, axis=-1) / np.linalg.norm(small, axis=-1)
        worst = min(worst, float(cosine.min()))
    print(key, "small text worst cosine (both optimization levels)", round(worst, 6), "bytes", target.stat().st_size)
    assert worst >= 0.999, f"{key}: small text tower drifted ({worst})"
    return worst


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


if __name__ == "__main__" and "--fixture" not in sys.argv and "--owl" not in sys.argv:
    main()


def fixture(image_path, out_path):
    """The test fixture the Rust tests compare against: per member, the full embeddings of a few patches, the grid
    mean and the text "a traffic cone", computed the way hype does (transformers' processor + full vision tower)."""
    image = Image.open(image_path).convert("RGB")
    result = {"image": {"width": image.width, "height": image.height}}
    for key, repo in MEMBERS.items():
        model = AutoModel.from_pretrained(repo).eval()
        processor = AutoProcessor.from_pretrained(repo)
        vision = model.vision_model
        with torch.inference_mode():
            if "naflex" in repo:
                inputs = processor(images=[image], return_tensors="pt", max_num_patches=1024)
                rows, cols = (int(v) for v in inputs["spatial_shapes"][0])
                hidden = vision(
                    pixel_values=inputs["pixel_values"],
                    attention_mask=inputs["pixel_attention_mask"],
                    spatial_shapes=inputs["spatial_shapes"],
                ).last_hidden_state[0, : rows * cols]
            else:
                inputs = processor(images=[image], return_tensors="pt")
                hidden = vision(pixel_values=inputs["pixel_values"]).last_hidden_state[0]
                rows = cols = int(round(hidden.shape[0] ** 0.5))
            grid = F.normalize(vision.head(hidden.reshape(rows * cols, 1, -1)).reshape(rows * cols, -1), dim=-1)
            tokens = processor(text=["a traffic cone"], return_tensors="pt", padding="max_length", max_length=TEXT_LENGTH)
            text = F.normalize(model.text_model(input_ids=tokens["input_ids"]).pooler_output, dim=-1)[0]
        picks = [0, rows * cols // 3, rows * cols // 2, rows * cols - 1]
        result[key] = {
            "rows": rows,
            "cols": cols,
            "patches": {str(i): grid[i].tolist() for i in picks},
            "mean": grid.mean(0).tolist(),
            "text_ids": tokens["input_ids"][0].tolist(),
            "text": text.tolist(),
        }
    Path(out_path).write_text(json.dumps(result))


if __name__ == "__main__" and "--fixture" in sys.argv:
    fixture(sys.argv[2], sys.argv[3])


OWL_REPO = "google/owlv2-base-patch16-ensemble"


class OwlVision(nn.Module):
    """The question-independent half of OWLv2, run once per keyframe at generation: per patch (60x60 = 3600) its box,
    its normalized class embedding and its logit shift/scale. A question then only needs the text tower and a dot."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        feature_map = self.model.image_embedder(pixel_values=pixel_values)[0]
        batch, h, w, dim = feature_map.shape
        image_feats = feature_map.reshape(batch, h * w, dim)
        boxes = self.model.box_predictor(image_feats, feature_map)
        head = self.model.class_head
        class_embeds = head.dense0(image_feats)
        class_embeds = class_embeds / (torch.linalg.norm(class_embeds, dim=-1, keepdim=True) + 1e-6)
        shift = head.logit_shift(image_feats)[..., 0]
        scale = (head.elu(head.logit_scale(image_feats)) + 1)[..., 0]
        return class_embeds, shift, scale, boxes


class OwlText(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        embeds = self.model.owlv2.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        return embeds / (torch.linalg.norm(embeds, dim=-1, keepdim=True) + 1e-6)


def export_owl(out, image_path):
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    model = Owlv2ForObjectDetection.from_pretrained(OWL_REPO).eval()
    processor = Owlv2Processor.from_pretrained(OWL_REPO)
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[["a traffic cone"]], images=image, return_tensors="pt")
    with torch.inference_mode():
        truth = model(**inputs)
        vision, text = OwlVision(model).eval(), OwlText(model).eval()
        class_embeds, shift, scale, boxes = vision(inputs["pixel_values"])
        query = text(inputs["input_ids"], inputs["attention_mask"])
        logits = (class_embeds[0] @ query[0] + shift[0]) * scale[0]
        print("owl split logits vs full", max_abs(logits, truth.logits[0, :, 0]), "boxes", max_abs(boxes, truth.pred_boxes))
        torch.onnx.export(vision, (inputs["pixel_values"],), str(out / "owlv2.vision.onnx"), input_names=["pixel_values"],
                          output_names=["class_embeds", "logit_shift", "logit_scale", "boxes"], opset_version=OPSET)
        torch.onnx.export(text, (inputs["input_ids"], inputs["attention_mask"]), str(out / "owlv2.text.onnx"),
                          input_names=["input_ids", "attention_mask"], output_names=["embedding"],
                          dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"}, "embedding": {0: "batch"}},
                          opset_version=OPSET)
    session = ort.InferenceSession(str(out / "owlv2.vision.onnx"))
    got = session.run(None, {"pixel_values": inputs["pixel_values"].numpy()})
    text_session = ort.InferenceSession(str(out / "owlv2.text.onnx"))
    got_query = text_session.run(None, {"input_ids": inputs["input_ids"].numpy(), "attention_mask": inputs["attention_mask"].numpy()})[0]
    onnx_logits = (got[0][0] @ got_query[0] + got[1][0]) * got[2][0]
    print("owl onnx logits vs full", max_abs(onnx_logits, truth.logits[0, :, 0]), "best score", float(torch.sigmoid(truth.logits.max())))
    print("owl input_ids", inputs["input_ids"][0].tolist(), "mask", inputs["attention_mask"][0].tolist())
    processor.tokenizer.backend_tokenizer.save(str(out / "owlv2.tokenizer.json"))


if __name__ == "__main__" and "--owl" in sys.argv:
    export_owl(Path(sys.argv[2]), sys.argv[3])
