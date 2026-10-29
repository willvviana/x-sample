"""Inference runtime: backend adapters, probes, hardware detection, model registry.

Modules that import onnxruntime (`adapters`, `inference_check`, `probe`) are only
imported inside worker processes, never by the API process.
"""
