"""KOLAM-R Research Demonstration Application.

Neural Inverse Program Synthesis and Generative Grammar Recovery for Structured Kolam Patterns.
End-to-End Live Pipeline:
Input Raster Image -> Autoregressive Vision-to-Grammar Synthesizer (Stage 4)
                   -> Analysis-by-Synthesis Turtle Execution (Stage 5)
                   -> Topological Invariant Verification (Stage 6)
                   -> Continuous B-Spline Parametric Modeling (Stage 7)
                   -> Baseline Multi-Task Parameter CNN (Stage 3 Ablation)
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any

# Ensure Kolam project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from PIL import Image
import streamlit as st
import torch

from kolam_r.generator import KolamGenerator
from kolam_r.lsystem.engine import LSystemEngine
from kolam_r.lsystem.rules import ProductionRule, get_rule, list_rules, RULE_REGISTRY, RULES_BY_ID
from kolam_r.schema import KolamParams
from kolam_r.topology.betti import compute_graph_betti_numbers, extract_all_topological_invariants
from kolam_r.topology.graph_extractor import extract_skeleton_graph
from kolam_r.topology.skeleton import skeletonize_zhang_suen
from kolam_r.turtle.interpreter import TurtleInterpreter
from kolam_r.curvefit.fit_curves import fit_skeleton_graph_curves
from kolam_r.curvefit.rasterize_curves import rasterize_curves
from kolam_r.reconstruction.pipeline import ReconstructionPipeline, ReconstructionResult
from kolam_r.reconstruction.metrics import (
    compute_reconstruction_metrics,
    compute_ssim,
    compute_psnr,
    compute_iou,
    compute_ncc,
)

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom Academic Dark Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="KOLAM-R | Neural Inverse Program Synthesis",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b0f19;
        color: #f8fafc;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #f8fafc !important;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }
    .research-header {
        border-bottom: 2px solid #1e293b;
        padding-bottom: 14px;
        margin-bottom: 20px;
    }
    .badge-primary {
        background-color: #1e3a8a;
        color: #93c5fd;
        border: 1px solid #3b82f6;
        padding: 5px 12px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 8px;
    }
    .badge-live {
        background-color: #064e3b;
        color: #6ee7b7;
        border: 1px solid #059669;
        padding: 5px 12px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 8px;
        margin-left: 8px;
    }
    .scope-box {
        background-color: #0f172a;
        border-left: 4px solid #38bdf8;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 20px;
        font-size: 0.88rem;
        color: #cbd5e1;
        line-height: 1.6;
    }
    .card {
        background-color: #131d31 !important;
        color: #f8fafc !important;
        border: 1px solid #1e293b !important;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
    }
    .card strong, .card td, .card li, .card span {
        color: #f8fafc !important;
    }
    .grammar-card {
        background-color: #090d16 !important;
        color: #38bdf8 !important;
        border: 1px solid #1e3a8a !important;
        border-left: 5px solid #38bdf8 !important;
        padding: 16px;
        border-radius: 8px;
        font-family: 'Consolas', 'Courier New', Courier, monospace;
        font-size: 1.05rem;
        margin: 12px 0;
        letter-spacing: 0.02em;
    }
    .metric-box {
        text-align: center;
        background: #131d31 !important;
        border: 1px solid #1e293b !important;
        border-radius: 6px;
        padding: 12px 8px;
    }
    .metric-value {
        font-size: 1.35rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #94a3b8 !important;
        text-transform: uppercase;
        font-weight: 600;
    }
    .token-badge {
        display: inline-block;
        background-color: #1e293b;
        color: #f1f5f9;
        border: 1px solid #334155;
        border-radius: 4px;
        padding: 2px 6px;
        margin: 2px 3px;
        font-family: monospace;
        font-size: 0.82rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2. Pipeline Resource Loader (Cached Once)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained Stage 4 & Stage 3 neural checkpoints...")
def load_reconstruction_pipeline() -> ReconstructionPipeline:
    """Load neural models and synthesis pipeline into cached resource."""
    grammar_pt = ROOT_DIR / "checkpoints" / "best_grammar_model.pt"
    baseline_pt = ROOT_DIR / "checkpoints" / "best_val_model.pt"
    return ReconstructionPipeline(
        grammar_model_path=grammar_pt if grammar_pt.exists() else None,
        baseline_model_path=baseline_pt if baseline_pt.exists() else None,
        device="cpu",
    )

# -----------------------------------------------------------------------------
# 3. Image Preprocessing & Standardization Utilities
# -----------------------------------------------------------------------------
def preprocess_input_image(pil_img: Image.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert input image to standardized representations:
    1. gray_256: 256x256 display image
    2. gray_64: 64x64 model input image
    3. stroke_mask_256: binary mask excluding background and dot grid
    """
    gray = pil_img.convert("L")
    arr = np.array(gray, dtype=np.float32)

    # Invert if dark strokes on light background (Kolam canonical is white strokes on dark floor)
    border_pixels = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    if np.median(border_pixels) > 120:
        arr = 255.0 - arr

    # Contrast normalization
    p_low, p_high = np.percentile(arr, (1, 99))
    if p_high > p_low:
        arr = np.clip((arr - p_low) / (p_high - p_low) * 255.0, 0, 255)

    arr_u8 = arr.astype(np.uint8)

    # Square center-crop bounding box if foreground present
    thresh = max(30, int(np.mean(arr_u8) + 0.3 * np.std(arr_u8)))
    coords = np.argwhere(arr_u8 > thresh)
    if coords.size > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        cropped = arr_u8[y0:y1, x0:x1]
        h, w = cropped.shape
        max_dim = max(h, w)
        pad_y = (max_dim - h) // 2
        pad_x = (max_dim - w) // 2
        square = np.pad(
            cropped,
            ((pad_y, max_dim - h - pad_y), (pad_x, max_dim - w - pad_x)),
            mode="constant",
            constant_values=0,
        )
        gray_256_pil = Image.fromarray(square).resize((256, 256), Image.Resampling.BILINEAR)
        gray_64_pil = Image.fromarray(square).resize((64, 64), Image.Resampling.BILINEAR)
    else:
        gray_256_pil = Image.fromarray(arr_u8).resize((256, 256), Image.Resampling.BILINEAR)
        gray_64_pil = Image.fromarray(arr_u8).resize((64, 64), Image.Resampling.BILINEAR)

    gray_256 = np.array(gray_256_pil, dtype=np.uint8)
    gray_64 = np.array(gray_64_pil, dtype=np.uint8)

    # Clean stroke mask (threshold > 200 isolates 255 stroke from 128 dot grid)
    stroke_mask_256 = (gray_256 > 200).astype(np.uint8) * 255
    if np.sum(stroke_mask_256) < 100:  # Fallback for faint non-dot inputs
        stroke_mask_256 = (gray_256 > 30).astype(np.uint8) * 255

    return gray_256, gray_64, stroke_mask_256

# -----------------------------------------------------------------------------
# 4. Header & Scientific Scope Statement
# -----------------------------------------------------------------------------
st.markdown('<div class="research-header">', unsafe_allow_html=True)
st.markdown(
    '<span class="badge-primary">KOLAM-R Full Production System</span>'
    '<span class="badge-live">Live Neural Inference (Stages 3–7 Active)</span>',
    unsafe_allow_html=True,
)
st.title("KOLAM-R")
st.subheader("Neural Inverse Program Synthesis of Generative L-System Grammars for Kolam Patterns")
st.markdown('</div>', unsafe_allow_html=True)

# Permanent Scope Statement (Non-Classification Framing)
st.markdown(
    """
    <div class="scope-box">
        <strong>📌 Mathematical Scope & Model Generalization Boundaries:</strong><br>
        KOLAM-R performs <em>program synthesis</em>: given an input visual pattern, the trained Vision-to-Grammar
        Transformer (Stage 4) autoregressively generates symbolic L-system grammar rules
        (<code>AXIOM: &omega; ; RULES: &alpha; &rarr; &beta;</code>) and continuous turtle geometry parameters
        token-by-token. <strong>It does not perform pattern classification into a pre-registered catalog.</strong>
        Canonical rule identifiers (e.g. R01–R06) are provided strictly as academic reference aids to compare
        synthesized programs with classical literature.
        <br><br>
        <strong>Empirical Generalization Boundary:</strong> In testing so far, the model has not been observed to generate a grammar structurally distinct from its 6 training templates — true open-vocabulary generalization is unverified. When presented with unseen or out-of-registry geometry, the autoregressive decoder reproduces whichever known canonical template (R01–R06) most closely matches its visual feature embeddings. This finding is consistent with the model's training dataset, which contained parameter and geometric variations of only 6 fixed templates — never a genuinely different grammar structure. A model trained on a closed set of templates has no empirical basis to compose novel ones, confirming this behavior reflects a training-data ceiling rather than necessarily an architectural limitation. Testing genuine open-ended generalization would require training on a broader grammar space, which is a natural next step, not yet attempted.
        <br><br>
        <strong>Confidence vs. Correctness:</strong> Reported token confidence reflects the decoder's internal conditional sequence certainty, <em>not</em> verified ground-truth correctness. Empirical evaluation demonstrates that token confidence remains clustered in a narrow band (~94%–97%) regardless of whether the output is correct or verifiably mismatched with ground truth.
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 5. Sidebar: Dataset Statistics & Model Inspection
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("🔬 System & Model Diagnostics")
    st.caption("Inspecting live neural checkpoints and Stage 1-2 training data.")

    stats_file = ROOT_DIR / "data" / "stats" / "dataset_statistics.json"
    if stats_file.exists():
        with open(stats_file, "r") as f:
            stats = json.load(f)
        st.metric("Total Generated Dataset", f"{stats.get('total_images', 0):,} samples")
        st.metric("Canonical Rule Library", len(stats.get("parameter_distributions", {}).get("production_rule_id", {})))
    
    grammar_ckpt = ROOT_DIR / "checkpoints" / "best_grammar_model.pt"
    val_ckpt = ROOT_DIR / "checkpoints" / "best_val_model.pt"

    st.markdown("---")
    st.markdown("### 🧠 Active Checkpoints")
    if grammar_ckpt.exists():
        st.success(f"✓ **Stage 4:** `best_grammar_model.pt` ({grammar_ckpt.stat().st_size / (1024*1024):.1f} MB)")
    else:
        st.error("✗ Stage 4 checkpoint missing!")

    if val_ckpt.exists():
        st.success(f"✓ **Stage 3:** `best_val_model.pt` ({val_ckpt.stat().st_size / (1024*1024):.1f} MB)")
    else:
        st.error("✗ Stage 3 checkpoint missing!")

    st.markdown("---")
    st.markdown("### 📐 Pipeline Stages")
    st.markdown(
        """
        - **Stage 1:** Parametric L-System Generator
        - **Stage 2:** Dual Topological Representations
        - **Stage 3:** Baseline Multi-Task CNN (Ablation)
        - **Stage 4:** Autoregressive Vision-to-Grammar Transformer
        - **Stage 5:** Turtle Analysis-by-Synthesis Reconstructor
        - **Stage 6:** Dual-Complex Homology Invariant Validator
        - **Stage 7:** Continuous B-Spline Parametric Modeling ($K = |E|$)
        """
    )

# -----------------------------------------------------------------------------
# 6. Input Selection Controls
# -----------------------------------------------------------------------------
pipeline = load_reconstruction_pipeline()
generator = KolamGenerator()

st.markdown("### 📥 Select or Upload an Input Pattern")

col_mode1, col_mode2 = st.columns([1.5, 1])

with col_mode1:
    st.markdown("##### ⚡ Benchmark Library Patterns (Ground-Truth Known)")
    col_b1, col_b2, col_b3, col_b4, col_b5, col_b6 = st.columns(6)
    btn_r01 = col_b1.button("Krishna Anklets\n(R01, d=2)", use_container_width=True)
    btn_r02 = col_b2.button("Snake Kolam\n(R02, d=2)", use_container_width=True)
    btn_r03 = col_b3.button("Kolam Tile\n(R03, d=2)", use_container_width=True)
    btn_r04 = col_b4.button("Mango Leaf\n(R04, d=2)", use_container_width=True)
    btn_r05 = col_b5.button("Hilbert Meander\n(R05, d=2)", use_container_width=True)
    btn_r06 = col_b6.button("Branching Floral\n(R06, d=2)", use_container_width=True)

with col_mode2:
    st.markdown("##### 📤 External Image Input")
    uploaded_file = st.file_uploader(
        "Upload any PNG or JPG Kolam image:",
        type=["png", "jpg", "jpeg"],
        help="Upload an arbitrary or real-world Kolam image for program synthesis.",
    )

# Canonical benchmark configurations
CANONICAL_BENCHMARKS = {
    "R01": KolamParams(production_rule_id="R01", recursion_depth=2, symmetry="D4", angle=45.0, grid_size=5, motif="M1"),
    "R02": KolamParams(production_rule_id="R02", recursion_depth=2, symmetry="D2", angle=90.0, grid_size=5, motif="M1"),
    "R03": KolamParams(production_rule_id="R03", recursion_depth=2, symmetry="D4", angle=45.0, grid_size=5, motif="M1"),
    "R04": KolamParams(production_rule_id="R04", recursion_depth=2, symmetry="D2", angle=45.0, grid_size=5, motif="M1"),
    "R05": KolamParams(production_rule_id="R05", recursion_depth=2, symmetry="C4", angle=90.0, grid_size=5, motif="M1"),
    "R06": KolamParams(production_rule_id="R06", recursion_depth=2, symmetry="D4", angle=25.0, grid_size=5, motif="M1"),
}

# Determine active image source
is_uploaded_source = False
if uploaded_file is not None:
    try:
        pil_input = Image.open(uploaded_file)
        # Verify image integrity
        pil_input.verify()
        uploaded_file.seek(0)
        pil_input = Image.open(uploaded_file)
        active_label = f"Uploaded Image: {uploaded_file.name}"
        ground_truth_rule = None
        is_uploaded_source = True
    except Exception as e:
        st.error(f"⚠️ Unreadable or corrupted image file: '{uploaded_file.name}'. Error details: {e}. Falling back to default canonical benchmark.")
        active_label = "Canonical Benchmark: Kolam Tile (R03, d=2) [Corrupted Upload Fallback]"
        res_gen = generator.generate(CANONICAL_BENCHMARKS["R03"])
        pil_input = Image.fromarray(res_gen.image_256)
        ground_truth_rule = "R03"
elif btn_r01:
    active_label = "Canonical Benchmark: Krishna Anklets (R01, d=2)"
    res_gen = generator.generate(CANONICAL_BENCHMARKS["R01"])
    pil_input = Image.fromarray(res_gen.image_256)
    ground_truth_rule = "R01"
elif btn_r02:
    active_label = "Canonical Benchmark: Snake Kolam (R02, d=2)"
    res_gen = generator.generate(CANONICAL_BENCHMARKS["R02"])
    pil_input = Image.fromarray(res_gen.image_256)
    ground_truth_rule = "R02"
elif btn_r04:
    active_label = "Canonical Benchmark: Mango Leaf (R04, d=2)"
    res_gen = generator.generate(CANONICAL_BENCHMARKS["R04"])
    pil_input = Image.fromarray(res_gen.image_256)
    ground_truth_rule = "R04"
elif btn_r05:
    active_label = "Canonical Benchmark: Hilbert Meander (R05, d=2)"
    res_gen = generator.generate(CANONICAL_BENCHMARKS["R05"])
    pil_input = Image.fromarray(res_gen.image_256)
    ground_truth_rule = "R05"
elif btn_r06:
    active_label = "Canonical Benchmark: Branching Floral (R06, d=2)"
    res_gen = generator.generate(CANONICAL_BENCHMARKS["R06"])
    pil_input = Image.fromarray(res_gen.image_256)
    ground_truth_rule = "R06"
else:
    # Default: Canonical R03 (Kolam Tile)
    active_label = "Canonical Benchmark: Kolam Tile (R03, d=2)"
    res_gen = generator.generate(CANONICAL_BENCHMARKS["R03"])
    pil_input = Image.fromarray(res_gen.image_256)
    ground_truth_rule = "R03"

# Preprocess image
gray_256, gray_64, stroke_mask_256 = preprocess_input_image(pil_input)

# Check for blank / empty upload
if np.std(gray_256) < 1.0:
    st.warning("⚠️ Notice: The active input image is completely uniform or blank. No pattern strokes were detected; the models will process the blank uniform canvas.")

# Quick sample selector expander for convenience
with st.expander("📁 Or Load Pre-Packaged Test Images (`app/safe_upload_samples/` & `data/raw/`)", expanded=False):
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    if col_s1.button("Sample R01 Photo", use_container_width=True):
        p = ROOT_DIR / "app" / "safe_upload_samples" / "sample_r01_krishna_anklets.jpg"
        if p.exists():
            pil_input = Image.open(p)
            gray_256, gray_64, stroke_mask_256 = preprocess_input_image(pil_input)
            active_label = "External Sample: sample_r01_krishna_anklets.jpg"
    if col_s2.button("Sample R02 Photo", use_container_width=True):
        p = ROOT_DIR / "app" / "safe_upload_samples" / "sample_r02_snake_kolam.jpg"
        if p.exists():
            pil_input = Image.open(p)
            gray_256, gray_64, stroke_mask_256 = preprocess_input_image(pil_input)
            active_label = "External Sample: sample_r02_snake_kolam.jpg"
    if col_s3.button("Sample R03 Photo", use_container_width=True):
        p = ROOT_DIR / "app" / "safe_upload_samples" / "sample_r03_kolam_tile.jpg"
        if p.exists():
            pil_input = Image.open(p)
            gray_256, gray_64, stroke_mask_256 = preprocess_input_image(pil_input)
            active_label = "External Sample: sample_r03_kolam_tile.jpg"
    if col_s4.button("Raw Dataset K000013", use_container_width=True):
        p = ROOT_DIR / "data" / "raw" / "images" / "K000013.png"
        if p.exists():
            pil_input = Image.open(p)
            gray_256, gray_64, stroke_mask_256 = preprocess_input_image(pil_input)
            active_label = "Raw Dataset: K000013.png"

# -----------------------------------------------------------------------------
# 7. Live Neural Inference Execution (Stages 4, 3, 6, 7)
# -----------------------------------------------------------------------------
with st.spinner("Executing neural inverse program synthesis and topology extraction..."):
    # Stage 4: Vision-to-Grammar Transformer
    res_grammar = pipeline.reconstruct_from_grammar(gray_64, use_discovered_depth=True)

    # Stage 3: Baseline Parameter CNN
    res_cnn = pipeline.reconstruct_from_baseline_cnn(gray_64)

    # Stage 6: Stroke-only Topology Extraction (threshold=200 excludes 128 dot-grid pixels)
    b0_orig, b1_orig = compute_graph_betti_numbers(gray_256, threshold=200)
    b0_grammar, b1_grammar = compute_graph_betti_numbers(res_grammar.image, threshold=200)
    b0_cnn, b1_cnn = compute_graph_betti_numbers(res_cnn.image, threshold=200)

    # Skeleton images for visual display
    skel_orig = skeletonize_zhang_suen(stroke_mask_256 > 0) * 255
    skel_grammar = skeletonize_zhang_suen(res_grammar.mask > 0) * 255
    skel_cnn = skeletonize_zhang_suen(res_cnn.mask > 0) * 255

    # Stage 7: Continuous Parametric Curve Fitting (K = |E|)
    stroke_skel_bool = skeletonize_zhang_suen(stroke_mask_256 > 0)
    stroke_graph = extract_skeleton_graph(stroke_skel_bool)
    fitted_curves = fit_skeleton_graph_curves(stroke_graph, linearity_threshold=0.5)
    curve_raster = rasterize_curves(fitted_curves, image_shape=(256, 256), stroke_width=2)
    curve_skel = skeletonize_zhang_suen(curve_raster > 100)
    curve_graph = extract_skeleton_graph(curve_skel)
    b0_curve, b1_curve = curve_graph.compute_betti_numbers()

st.markdown("---")
st.markdown(f"#### Active Input: `{active_label}`")

# -----------------------------------------------------------------------------
# 8. SECTION 1: Preprocessing & Normalized Representations (Stage 2)
# -----------------------------------------------------------------------------
with st.expander("🔍 Stage 2: Dual Normalized Representations (Grayscale, Stroke Mask & Skeleton)", expanded=False):
    col_pr1, col_pr2, col_pr3 = st.columns(3)
    col_pr1.image(gray_256, caption="Normalized Input (256x256)", use_container_width=True, clamp=True)
    col_pr2.image(stroke_mask_256, caption="Isolated Stroke Mask (>200 threshold excludes dots)", use_container_width=True, clamp=True)
    col_pr3.image(skel_orig, caption=f"Zhang-Suen Stroke Skeleton (β₀={b0_orig}, β₁={b1_orig})", use_container_width=True, clamp=True)

# -----------------------------------------------------------------------------
# 9. SECTION 2: Recovered L-System Generative Grammar (Stage 4 Headline)
# -----------------------------------------------------------------------------
st.markdown("### 📐 Stage 4: Recovered Generative Grammar (Primary Synthesized Program)")

col_g_main, col_g_side = st.columns([1.5, 1])

with col_g_main:
    st.markdown("##### Synthesized L-System Grammar String (Autoregressive Decoder Output):")
    st.markdown(
        f'<div class="grammar-card">{res_grammar.grammar_string}</div>',
        unsafe_allow_html=True,
    )

    # Status Badges
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    with col_stat1:
        if res_grammar.is_syntactically_valid:
            st.markdown('<span style="color:#22c55e; font-weight:700; font-size:0.95rem;">✓ Valid L-System AST Syntax</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span style="color:#ef4444; font-weight:700; font-size:0.95rem;">✗ Syntax Error: {res_grammar.error_message}</span>', unsafe_allow_html=True)

    with col_stat2:
        conf_val = res_grammar.token_confidence if res_grammar.token_confidence is not None else 0.0
        st.markdown(f'<span style="color:#38bdf8; font-weight:700; font-size:0.95rem;">Token Confidence: {conf_val*100:.1f}%</span>', unsafe_allow_html=True)
        st.caption("⚠️ Reflects model probability certainty, not verified ground-truth correctness.")

    with col_stat3:
        can_rule = res_grammar.parameters.get("closest_canonical_rule", "Unknown")
        can_name = res_grammar.parameters.get("closest_canonical_name", "Custom")
        can_sim = res_grammar.parameters.get("canonical_similarity", 0.0)
        if can_sim == 1.0:
            st.caption(f"Template: **Verbatim match to {can_rule} ({can_name})** [0 token diffs]")
        else:
            st.caption(f"Ref Family: **{can_name} ({can_rule})** [{can_sim*100:.0f}% token overlap]")

    with st.expander(f"🔤 Inspect Generated Token Sequence ({len(res_grammar.parameters.get('tokens', []))} tokens)", expanded=False):
        tokens = res_grammar.parameters.get("tokens", [])
        tokens_html = "".join([f'<span class="token-badge">{t}</span>' for t in tokens])
        st.markdown(tokens_html, unsafe_allow_html=True)
        st.caption("Decoded step-by-step by the autoregressive cross-attention transformer decoder over the 24-token vocabulary.")

with col_g_side:
    st.markdown(
        f"""
        <div class="card">
            <h5 style="margin-top:0; color:#38bdf8 !important;">Synthesized Geometry Parameters</h5>
            <table style="width:100%; font-size:0.86rem; line-height:1.9;">
                <tr><td><strong>Discovered Depth ($d$):</strong></td><td><code>d = {res_grammar.parameters.get('depth')}</code> (Protocol B NCC Search in [1, 4])</td></tr>
                <tr><td><strong>Symmetry Group:</strong></td><td><code>{res_grammar.parameters.get('symmetry')}</code></td></tr>
                <tr><td><strong>Turning Angle (&theta;):</strong></td><td><code>{res_grammar.parameters.get('angle', 0.0):.2f}&deg;</code></td></tr>
                <tr><td><strong>Dot Grid Size:</strong></td><td><code>{res_grammar.parameters.get('grid_size')} &times; {res_grammar.parameters.get('grid_size')}</code></td></tr>
                <tr><td><strong>Turtle Segments:</strong></td><td><code>{len(res_grammar.segments)}</code> vectors</td></tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# 10. SECTION 3: Baseline Multi-Task CNN Comparison (Stage 3 Ablation)
# -----------------------------------------------------------------------------
with st.expander("⚖️ Compare with Direct Parameter Baseline CNN (Stage 3 Ablation)", expanded=False):
    col_c1, col_c2 = st.columns([1, 1.2])
    with col_c1:
        st.markdown(
            f"""
            <div class="card">
                <h5 style="margin-top:0;">Baseline CNN Classification</h5>
                <p style="font-size:0.88rem;">
                <strong>Top-1 Rule:</strong> <code>{res_cnn.parameters.get('rule_id')}</code><br>
                <strong>Predicted Depth:</strong> <code>{res_cnn.parameters.get('depth')}</code><br>
                <strong>Predicted Symmetry:</strong> <code>{res_cnn.parameters.get('symmetry')}</code><br>
                <strong>Predicted Angle:</strong> <code>{res_cnn.parameters.get('angle', 0.0):.2f}&deg;</code>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_c2:
        st.markdown("##### Softmax Probabilities Across Canonical Rules:")
        probs = res_cnn.parameters.get("rule_probabilities", {})
        for r_id, p_val in probs.items():
            r_name = RULES_BY_ID[r_id].name
            st.progress(float(p_val), text=f"{r_id} — {r_name}: {p_val*100:.1f}%")

# -----------------------------------------------------------------------------
# 11. SECTION 4: Visual Reconstruction Comparison (Stage 5 Analysis-by-Synthesis)
# -----------------------------------------------------------------------------
st.markdown("### 🔄 Visual Reconstruction Comparison (Stage 5)")

# Difference map between input and Stage 4 reconstruction
diff_map = np.abs(gray_64.astype(np.float32) - res_grammar.image.astype(np.float32)).astype(np.uint8)

col_v1, col_v2, col_v3, col_v4 = st.columns(4)
with col_v1:
    st.image(gray_64, caption="Input Target (64x64)", use_container_width=True, clamp=True)
with col_v2:
    st.image(res_grammar.image, caption="Stage 4 Grammar Synthesis", use_container_width=True, clamp=True)
with col_v3:
    st.image(res_cnn.image, caption=f"Stage 3 CNN Lookup ({res_cnn.parameters.get('rule_id')})", use_container_width=True, clamp=True)
with col_v4:
    st.image(diff_map, caption="Absolute Difference Map", use_container_width=True, clamp=True)

# -----------------------------------------------------------------------------
# 12. SECTION 5: Quantitative Reconstruction Fidelity Metrics
# -----------------------------------------------------------------------------
st.markdown("### 📊 Reconstruction Fidelity Metrics (Live Session Image)")

col_mf1, col_mf2, col_mf3, col_mf4 = st.columns(4)
with col_mf1:
    ssim_g = res_grammar.metrics.ssim if res_grammar.metrics else 0.0
    ssim_c = res_cnn.metrics.ssim if res_cnn.metrics else 0.0
    st.markdown(
        f'<div class="metric-box"><div class="metric-value" style="color:#38bdf8;">{ssim_g:.4f}</div>'
        f'<div class="metric-label">Structural SSIM (CNN: {ssim_c:.4f})</div></div>',
        unsafe_allow_html=True,
    )
with col_mf2:
    psnr_g = res_grammar.metrics.psnr if res_grammar.metrics else 0.0
    psnr_c = res_cnn.metrics.psnr if res_cnn.metrics else 0.0
    st.markdown(
        f'<div class="metric-box"><div class="metric-value" style="color:#38bdf8;">{psnr_g:.1f} dB</div>'
        f'<div class="metric-label">PSNR (CNN: {psnr_c:.1f} dB)</div></div>',
        unsafe_allow_html=True,
    )
with col_mf3:
    iou_g = res_grammar.metrics.iou if res_grammar.metrics else 0.0
    iou_c = res_cnn.metrics.iou if res_cnn.metrics else 0.0
    st.markdown(
        f'<div class="metric-box"><div class="metric-value" style="color:#38bdf8;">{iou_g:.4f}</div>'
        f'<div class="metric-label">Binary IoU (CNN: {iou_c:.4f})</div></div>',
        unsafe_allow_html=True,
    )
with col_mf4:
    chamf_g = res_grammar.metrics.chamfer_distance if res_grammar.metrics else 0.0
    chamf_c = res_cnn.metrics.chamfer_distance if res_cnn.metrics else 0.0
    st.markdown(
        f'<div class="metric-box"><div class="metric-value" style="color:#38bdf8;">{chamf_g:.2f} px</div>'
        f'<div class="metric-label">Chamfer Dist (CNN: {chamf_c:.2f} px)</div></div>',
        unsafe_allow_html=True,
    )

if np.std(gray_256) < 1.0:
    st.info("ℹ️ **Degenerate Uniform Input:** The input canvas has zero variance (blank field). SSIM is mathematically reported as 0.0000; the standard unregularized background-stabilization artifact (~0.52) has been explicitly suppressed to prevent misleading similarity claims.")

# -----------------------------------------------------------------------------
# 13. SECTION 6: Topological Homology Validation (Stage 6)
# -----------------------------------------------------------------------------
st.markdown("### 🕸️ Stage 6: Topological Homology Invariants")

col_topo_img1, col_topo_img2, col_topo_img3 = st.columns(3)
with col_topo_img1:
    st.image(skel_orig, caption=f"Input Skeleton (β₀={b0_orig}, β₁={b1_orig})", use_container_width=True, clamp=True)
with col_topo_img2:
    st.image(skel_grammar, caption=f"Stage 4 Recon Skeleton (β₀={b0_grammar}, β₁={b1_grammar})", use_container_width=True, clamp=True)
with col_topo_img3:
    st.image(skel_cnn, caption=f"Stage 3 Recon Skeleton (β₀={b0_cnn}, β₁={b1_cnn})", use_container_width=True, clamp=True)

topo_grammar_match = (b0_orig == b0_grammar and b1_orig == b1_grammar)
topo_badge = (
    '<span style="color:#22c55e; font-weight:700;">✓ Exact Topological Homology Preserved (β₀ and β₁ match)</span>'
    if topo_grammar_match
    else '<span style="color:#f59e0b; font-weight:700;">≈ Homology Discrepancy (Connectedness or cycle rank differs)</span>'
)

st.markdown(
    f"""
    <div class="card">
        <strong>Dual-Complex Graph Betti Summary:</strong>
        <ul style="margin-top:6px; margin-bottom:8px; line-height:1.8;">
            <li><strong>Connected Components (&beta;₀):</strong> Input = <code>{b0_orig}</code> &rarr; Stage 4 Recon = <code>{b0_grammar}</code> &rarr; Stage 3 Recon = <code>{b0_cnn}</code></li>
            <li><strong>Independent Stroke Cycles (&beta;₁):</strong> Input = <code>{b1_orig}</code> &rarr; Stage 4 Recon = <code>{b1_grammar}</code> &rarr; Stage 3 Recon = <code>{b1_cnn}</code></li>
            <li><strong>Stage 4 Homology Status:</strong> {topo_badge}</li>
        </ul>
        <span style="font-size:0.80rem; color:#94a3b8;">
        <strong>Scientific Invariant Separation:</strong> Topological invariants measure structural loop closure and connectedness.
        Exact homology (&beta;₀, &beta;₁) can be preserved even when pixel alignment metrics (SSIM, IoU) are modest due to rasterization discretization or stroke width variation.
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 14. SECTION 7: Continuous Parametric Curve Reconstruction (Stage 7)
# -----------------------------------------------------------------------------
st.markdown("### 📈 Stage 7: Continuous Parametric Curve Reconstruction ($K = |E|$)")
st.markdown(
    """
    <span style="font-size:0.88rem; color:#cbd5e1;">
    Dual continuous representation pathway. Decomposes the medial stroke skeleton into an undirected branch graph
    $G=(V, E)$ and fits exactly one parametric equation per branch edge ($K = |E|$). Open branches are modeled
    as endpoint-constrained cubic polynomials, while isolated closed loops use periodic cubic B-splines.
    </span>
    """,
    unsafe_allow_html=True,
)

num_curves = len(fitted_curves)
linear_count = sum(1 for c in fitted_curves if c.degree == 1)
cubic_count = sum(1 for c in fitted_curves if c.degree == 3 and not c.is_periodic)
spline_count = sum(1 for c in fitted_curves if c.is_periodic)

col_cv1, col_cv2, col_cv3 = st.columns([1, 1, 1.4])
with col_cv1:
    st.image(stroke_mask_256, caption=f"Clean Stroke Target (|E|={len(stroke_graph.edges)})", use_container_width=True, clamp=True)
with col_cv2:
    st.image(curve_raster, caption=f"Continuous Curve Rasterization ({num_curves} Equations)", use_container_width=True, clamp=True)
with col_cv3:
    k_equal = (num_curves == len(stroke_graph.edges))
    curve_topo_match = (b0_orig == b0_curve and b1_orig == b1_curve)
    c_badge = (
        '<span style="color:#22c55e; font-weight:700;">✓ Stroke Homology Preserved</span>'
        if curve_topo_match
        else '<span style="color:#f59e0b; font-weight:700;">≈ Homology Shift (Raster Aliasing at Junctions)</span>'
    )
    st.markdown(
        f"""
        <div class="card">
            <strong>Parametric Curve System:</strong>
            <ul style="margin-top:6px; margin-bottom:8px; line-height:1.8;">
                <li><strong>Equation Count ($K$):</strong> <code>{num_curves}</code> curves (<em>$K = |E|$: {'✓ Holds' if k_equal else '✗ Discrepancy'}</em>)</li>
                <li><strong>Composition:</strong> <code>{linear_count}</code> linear, <code>{cubic_count}</code> cubic poly, <code>{spline_count}</code> periodic B-spline</li>
                <li><strong>Topology:</strong> &beta;₀: <code>{b0_orig} &rarr; {b0_curve}</code>, &beta;₁: <code>{b1_orig} &rarr; {b1_curve}</code></li>
                <li><strong>Homology Status:</strong> {c_badge}</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.expander(f"📜 Inspect Fitted Parametric Equations (Showing first 5 of {num_curves})", expanded=False):
    for i, c in enumerate(fitted_curves[:5]):
        if c.is_periodic:
            st.text(f"Curve {i+1:03d} | Periodic Cubic B-Spline | Max Error: {c.max_fitting_error:.4f} px | t in [0.0, 1.0]")
        elif c.degree == 1:
            st.text(f"Curve {i+1:03d} | Linear: x(t)={c.coefficients_x[0]:.2f}+{c.coefficients_x[1]:.2f}t, y(t)={c.coefficients_y[0]:.2f}+{c.coefficients_y[1]:.2f}t | Max Err: {c.max_fitting_error:.4f} px")
        else:
            st.text(f"Curve {i+1:03d} | Cubic Polynomial | deg={c.degree} | Max Err: {c.max_fitting_error:.4f} px | Mean Err: {c.mean_fitting_error:.4f} px")
