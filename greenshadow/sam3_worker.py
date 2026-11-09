# -*- coding: utf-8 -*-
"""
greenshadow.sam3_worker

SAM 3 (Meta, "Segment Anything with Concepts") text-prompted instance
segmentation, through the Hugging Face ``transformers`` port
(``Sam3Model`` / ``Sam3Processor``, transformers >= 5):

    segmenter = Sam3Segmenter("facebook/sam3")
    found = segmenter.segment([image], ["chimney", "skylight"], threshold=0.4)
    # found[0]: [(prompt index, score, boolean mask), ...]

This file imports nothing but numpy, torch and transformers, so that it can
run from an interpreter of its own - torch's CUDA wheels pin CUDA libraries
the rest of greenshadow (numba-cuda) may need at other versions. Run as a
script, it serves ``segment`` calls on its standard input and output
(``Sam3Client`` is the other end):

    /path/to/sam3-venv/bin/python sam3_worker.py

The checkpoint ``facebook/sam3`` is gated on the Hugging Face Hub: request
access on its page, then ``hf auth login`` in that interpreter. Licence: the
SAM License (commercial use allowed; attribution in publications; trade
controls; the licence travels with the weights).
"""

import os
import pickle
import struct
import sys
import traceback

import numpy as np


class Sam3Segmenter:
    """
    A SAM 3 image model on one device.

    Args:
        model: Hugging Face model id or local directory.
        device: "cuda", "cuda:1", "cpu"... None: CUDA when available.
        dtype: "bfloat16", "float16", "float32". None: bfloat16 on CUDA,
            float32 on the CPU.
        model_obj, processor: an already built model and processor (tests).
    """

    def __init__(self, model="facebook/sam3", device=None, dtype=None, *,
                 model_obj=None, processor=None):
        import torch
        from transformers import Sam3Model, Sam3Processor
        from transformers.utils import logging as hf_logging

        # Deprecation notes and the weight-loading bar, at the margin of the
        # caller's log; what was loaded is in info().
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        self.torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if dtype is None:
            dtype = "bfloat16" if self.device.type == "cuda" else "float32"
        self.dtype = getattr(torch, dtype)
        self.processor = processor or Sam3Processor.from_pretrained(model)
        self.model = (model_obj or Sam3Model.from_pretrained(model, dtype=self.dtype))
        self.model = self.model.to(self.device, self.dtype).eval()
        self.image_size = int(self.processor.image_processor.size["height"])
        self._text = {}

    def info(self):
        return {"device": str(self.device), "dtype": str(self.dtype).replace("torch.", ""),
                "image_size": self.image_size}

    def _text_features(self, prompt):
        # One text encoding per prompt for the life of the model.
        if prompt not in self._text:
            tokens = self.processor(text=[prompt], return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                features = self.model.get_text_features(**tokens)
            self._text[prompt] = (features.pooler_output, tokens.attention_mask)
        return self._text[prompt]

    def segment(self, images, prompts, *, threshold=0.4, mask_threshold=0.5):
        """
        Every instance of every prompt on each image.

        Args:
            images: uint8 arrays (rows, cols, 3).
            prompts: noun phrases.
            threshold: lowest score kept.
            mask_threshold: mask logit probability that is inside.

        Returns:
            Per image, a list of (prompt index, score, bool mask at the
            image's size).
        """
        from transformers.modeling_outputs import BaseModelOutputWithPooling

        torch = self.torch
        inputs = self.processor(images=list(images), return_tensors="pt")
        sizes = inputs["original_sizes"].tolist()
        pixels = inputs["pixel_values"].to(self.device, self.dtype)
        found = [[] for _ in images]
        with torch.inference_mode():
            vision = self.model.get_vision_features(pixel_values=pixels)
            for p, prompt in enumerate(prompts):
                features, attention = self._text_features(prompt)
                n = len(images)
                text = BaseModelOutputWithPooling(pooler_output=features.expand(n, -1, -1))
                outputs = self.model(vision_embeds=vision, text_embeds=text,
                                     attention_mask=attention.expand(n, -1))
                results = self.processor.post_process_instance_segmentation(
                    outputs, threshold=threshold, mask_threshold=mask_threshold,
                    target_sizes=sizes)
                for i, result in enumerate(results):
                    scores = result["scores"].float().cpu().numpy()
                    masks = result["masks"].cpu().numpy().astype(bool)
                    for score, mask in zip(scores, masks):
                        if mask.any():
                            found[i].append((p, float(score), mask))
        return found


# ============================================================
# WORKER PROTOCOL
# ============================================================
# Length-prefixed pickles: a request {"op": ..., ...}, a reply {"ok": bool,
# ...}. Masks travel bit-packed.

def _send(stream, message):
    data = pickle.dumps(message, protocol=5)
    stream.write(struct.pack("<Q", len(data)))
    stream.write(data)
    stream.flush()


def _receive(stream):
    head = stream.read(8)
    if len(head) < 8:
        raise EOFError("SAM3 worker pipe closed")
    (size,) = struct.unpack("<Q", head)
    return pickle.loads(stream.read(size))


def _pack(found):
    return [[(p, score, mask.shape, np.packbits(mask)) for p, score, mask in image]
            for image in found]


def _unpack(packed):
    return [[(p, score, np.unpackbits(bits, count=shape[0] * shape[1]).reshape(shape).astype(bool))
             for p, score, shape, bits in image] for image in packed]


class Sam3Client:
    """
    ``Sam3Segmenter`` in another interpreter, the same ``segment`` and
    ``info``. ``python``: that interpreter (with torch and transformers).
    """

    def __init__(self, python, model="facebook/sam3", device=None, dtype=None):
        import subprocess
        self.process = subprocess.Popen(
            [python, os.path.abspath(__file__)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self._info = self._call({"op": "init", "model": model, "device": device, "dtype": dtype})
        self.image_size = self._info["image_size"]

    def _call(self, request):
        _send(self.process.stdin, request)
        reply = _receive(self.process.stdout)
        if not reply.get("ok"):
            raise RuntimeError("SAM3 worker: " + reply.get("error", "failed"))
        return reply["value"]

    def info(self):
        return dict(self._info)

    def segment(self, images, prompts, *, threshold=0.4, mask_threshold=0.5):
        packed = self._call({"op": "segment", "images": [np.ascontiguousarray(i) for i in images],
                             "prompts": list(prompts), "threshold": threshold,
                             "mask_threshold": mask_threshold})
        return _unpack(packed)

    def close(self):
        if self.process.poll() is None:
            try:
                _send(self.process.stdin, {"op": "exit"})
            except (BrokenPipeError, OSError):
                pass
            self.process.wait(timeout=30)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _serve():
    # The protocol owns the real stdout; anything the libraries print goes
    # to stderr.
    channel_in = sys.stdin.buffer
    channel_out = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    segmenter = None
    while True:
        try:
            request = _receive(channel_in)
        except EOFError:
            return
        op = request.get("op")
        if op == "exit":
            return
        try:
            if op == "init":
                segmenter = Sam3Segmenter(request["model"], request["device"], request["dtype"])
                value = segmenter.info()
            elif op == "segment":
                value = _pack(segmenter.segment(
                    request["images"], request["prompts"], threshold=request["threshold"],
                    mask_threshold=request["mask_threshold"]))
            else:
                raise ValueError(f"unknown request {op!r}")
            _send(channel_out, {"ok": True, "value": value})
        except Exception:
            _send(channel_out, {"ok": False, "error": traceback.format_exc()})


if __name__ == "__main__":
    _serve()
