"""Maintainer tool: the four image models as native CoreML (fp16 ML Programs) for Macs. The app never runs this; its
output is uploaded once as the `models-v3` release (each package tarred), and the server compiles a package on first use
and loads it in about a second. ONNX Runtime's CoreML provider recompiled the ONNX graphs on every start (minutes).

    uv run --no-project --with torch==2.6.0 --with transformers==4.53.3 --with coremltools==9.0 --with numpy \
        --with pillow --with sentencepiece python tools/export_coreml.py <out_dir>

Same math as the ONNX export (tools/export_models.py), checked here against PyTorch on random inputs. The per-patch
pooling head is written out: attention over a single key has softmax 1, so a patch's head output is
out_proj(v_proj(patch)) followed by the head's residual MLP (no q/k projections, nothing data-dependent to convert).
"""
import subprocess
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from export_models import MEMBERS, OwlVision  # noqa: E402

MAX_PATCHES = 1024


class PerPatchHead(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head
        dim = head.attention.embed_dim
        self.v_weight = nn.Parameter(head.attention.in_proj_weight[2 * dim :].detach().clone())
        self.v_bias = nn.Parameter(head.attention.in_proj_bias[2 * dim :].detach().clone())

    def forward(self, hidden):
        pooled = self.head.attention.out_proj(F.linear(hidden, self.v_weight, self.v_bias))
        return F.normalize(pooled + self.head.mlp(self.head.layernorm(pooled)), dim=-1)


class Fixed(nn.Module):
    def __init__(self, vision):
        super().__init__()
        self.vision = vision
        self.head = PerPatchHead(vision.head)

    def forward(self, pixel_values):
        return self.head(self.vision(pixel_values=pixel_values).last_hidden_state)


class Naflex(nn.Module):
    def __init__(self, vision):
        super().__init__()
        self.vision = vision
        self.head = PerPatchHead(vision.head)

    def forward(self, patches, pos, mask):
        # always MAX_PATCHES long (a fixed shape keeps CoreML on the GPU); `mask` is 1 for a frame's patches, 0 for the
        # padding after them, which no patch attends to
        embeddings = self.vision.embeddings.patch_embedding(patches) + pos
        bias = ((1.0 - mask) * -1e4)[:, None, None, :]
        hidden = self.vision.post_layernorm(self.vision.encoder(inputs_embeds=embeddings, attention_mask=bias).last_hidden_state)
        return self.head(hidden)


class Owl(nn.Module):
    """export_models.OwlVision at the fixed 960x960 input, its shape arithmetic written as constants (the converter
    can't take image_embedder's): the class token broadcasts by multiplication and the box bias is the model's own."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        vision = self.model.owlv2.vision_model
        embeds = vision.post_layernorm(vision(pixel_values=pixel_values).last_hidden_state)
        image_feats = self.model.layer_norm(embeds[:, 1:, :] * embeds[:, :1, :])
        boxes = torch.sigmoid(self.model.box_head(image_feats) + self.model.box_bias)
        head = self.model.class_head
        class_embeds = head.dense0(image_feats)
        class_embeds = class_embeds / (torch.linalg.norm(class_embeds, dim=-1, keepdim=True) + 1e-6)
        shift = head.logit_shift(image_feats)[..., 0]
        scale = (head.elu(head.logit_scale(image_feats)) + 1)[..., 0]
        return class_embeds, shift, scale, boxes


def cosines(a, b):
    a, b = np.asarray(a, dtype=np.float64).reshape(-1, a.shape[-1]), np.asarray(b, dtype=np.float64).reshape(-1, b.shape[-1])
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)


def convert(module, examples, inputs, outputs, path, exported=False):
    # OWLv2's vision tower only converts through torch.export (its traced graph keeps int casts of shape tensors)
    if exported:
        traced = torch.export.export(module, examples).run_decompositions({})
        inputs = None
    else:
        with torch.inference_mode():
            traced = torch.jit.trace(module, examples)
    model = ct.convert(
        traced,
        inputs=inputs,
        outputs=[ct.TensorType(name, dtype=np.float32) for name in outputs],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS14,
    )
    model.save(str(path))
    return model


def check(model, feed, truth, names):
    got = model.predict({k: v.numpy() for k, v in feed.items()})
    for name, expected in zip(names, truth):
        actual = got[name]
        if expected.ndim >= 2 and expected.shape[-1] > 1:
            values = cosines(actual, expected.numpy())
            print(f"  {name}: cosine mean {values.mean():.4f}, worst {values.min():.4f}")
        else:
            print(f"  {name}: max abs {np.abs(actual - expected.numpy()).max():.4f}")


def tar(path):
    target = path.parent / (path.name + ".tar")
    subprocess.run(["tar", "-cf", str(target), "-C", str(path.parent), path.name], check=True)
    return target


def main():
    from transformers import AutoModel, Owlv2ForObjectDetection

    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    for key, repo in MEMBERS.items():
        vision = AutoModel.from_pretrained(repo, attn_implementation="eager").eval().vision_model
        dim = vision.config.hidden_size
        path = out / f"{key}.vision.mlpackage"
        if "naflex" in repo:
            module = Naflex(vision).eval()
            patch_values = vision.config.patch_size**2 * 3
            n = 1008
            patches, pos = torch.randn(1, n, patch_values), torch.randn(1, n, dim) * 0.02
            pad = MAX_PATCHES - n
            feed = {
                "patches": F.pad(patches, (0, 0, 0, pad)),
                "pos": F.pad(pos, (0, 0, 0, pad)),
                "mask": torch.cat([torch.ones(1, n), torch.zeros(1, pad)], dim=1),
            }
            with torch.inference_mode():
                unpadded = module(patches, pos, torch.ones(1, n))[0]
                padded = module(*feed.values())[0][:n]
            print(key, "padded vs unpadded max abs", float((padded - unpadded).abs().max()))
            inputs = [
                ct.TensorType("patches", shape=(1, MAX_PATCHES, patch_values)),
                ct.TensorType("pos", shape=(1, MAX_PATCHES, dim)),
                ct.TensorType("mask", shape=(1, MAX_PATCHES)),
            ]
        else:
            module = Fixed(vision).eval()
            size = vision.config.image_size
            feed = {"pixel_values": torch.randn(1, 3, size, size)}
            inputs = [ct.TensorType("pixel_values", shape=(1, 3, size, size))]
        with torch.inference_mode():
            truth = [module(*feed.values())[0]]
        print(key)
        model = convert(module, tuple(feed.values()), inputs, ["embeddings"], path)
        check(model, feed, truth, ["embeddings"])
        print("  ", tar(path))
    if (out / "owlv2.vision.mlpackage.tar").exists():
        return
    owl = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").eval()
    module = Owl(owl).eval()
    feed = {"pixel_values": torch.randn(1, 3, 960, 960)}
    names = ["class_embeds", "logit_shift", "logit_scale", "boxes"]
    with torch.inference_mode():
        truth = [t[0] for t in OwlVision(owl).eval()(*feed.values())]
        mine = [t[0] for t in module(*feed.values())]
    print("owlv2 rewritten vs OwlVision max abs", max(float((a - b).abs().max()) for a, b in zip(mine, truth)))
    path = out / "owlv2.vision.mlpackage"
    print("owlv2")
    model = convert(module, tuple(feed.values()), [ct.TensorType("pixel_values", shape=(1, 3, 960, 960))], names, path, exported=True)
    check(model, feed, truth, names)
    print("  ", tar(path))


if __name__ == "__main__":
    main()
