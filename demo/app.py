import streamlit as st
import torch
import numpy as np
import imageio.v2 as imageio
import torch.nn.functional as F
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Video Authenticity Analyzer",
    page_icon="🔬",
    layout="wide"
)

st.markdown("""
<style>
.main { background-color: #0f1117; }
.block-container { padding-top: 2rem; }
h1, h2, h3 { color: white; }

.metric-card {
    border-radius: 10px;
    padding: 18px 20px;
    height: 100%;
    margin-bottom: 8px;
}
.verdict-box {
    border-radius: 12px;
    padding: 24px 28px;
    margin: 12px 0 20px 0;
}
.step-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 600;
    color: #555;
    margin-bottom: 4px;
}
.section-heading {
    font-size: 1.25rem;
    font-weight: 700;
    color: #fff;
    margin: 0 0 6px 0;
}
.section-sub {
    font-size: 0.88rem;
    color: #888;
    margin: 0 0 20px 0;
    line-height: 1.6;
}
.chart-caption {
    font-size: 0.82rem;
    color: #666;
    font-style: italic;
    margin-top: 6px;
    padding-left: 2px;
}
.map-label {
    color: #ccc;
    font-size: 0.9rem;
    font-weight: 600;
    margin-bottom: 4px;
}
.map-sub {
    color: #666;
    font-size: 0.82rem;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)


# ── Computation ─────────────────────────────────────────────────────────────

def resize_frame(frame, size):
    """Resize (C,H,W) to size×size via bilinear interpolation."""
    return F.interpolate(
        frame.unsqueeze(0), size=(size, size), mode='bilinear', align_corners=False
    ).squeeze(0)

def apply_gaussian_blur(video, kernel_size=3, sigma=0.5):
    x = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
    gauss = torch.exp(-x.pow(2) / (2 * sigma**2))
    kernel = gauss[:, None] * gauss[None, :]
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    T, C, H, W = video.shape
    video = video.view(T*C, 1, H, W)
    video = F.pad(video, (1, 1, 1, 1), mode='reflect')
    video = F.conv2d(video, kernel)
    video = video.view(T, C, H, W)
    return video

def load_video(path, size=512, max_frames=48):
    """
    Preprocessing per BAB 3.1.3: 512×512, max_frames=48, Gaussian blur.
    Frames uniformly sampled across the full video — no stride artifacts.
    """
    reader = imageio.get_reader(path, format="ffmpeg")
    all_frames = []
    for frame in reader:
        frame = frame.astype(np.float32) / 127.5 - 1.0
        frame = torch.from_numpy(frame).permute(2, 0, 1)
        frame = resize_frame(frame, size)
        all_frames.append(frame)
    reader.close()

    if len(all_frames) < 10:
        raise ValueError(f"Only {len(all_frames)} frames could be read. Check video codec.")

    n = len(all_frames)
    if n <= max_frames:
        frames = all_frames
    else:
        indices = np.linspace(0, n - 1, max_frames, dtype=int)
        frames = [all_frames[i] for i in indices]

    video = torch.stack(frames)
    video = apply_gaussian_blur(video)
    return video

def compute_spatiotemporal_gradient(video):
    gray = video.mean(dim=1)
    Ix = (gray[:, :, 2:] - gray[:, :, :-2]) / 2
    Iy = (gray[:, 2:, :] - gray[:, :-2, :]) / 2
    Ix = Ix[:, 1:-1, :]
    Iy = Iy[:, :, 1:-1]
    It = (gray[2:] - gray[:-2]) / 2
    Ix = Ix[1:-1]
    Iy = Iy[1:-1]
    It = It[:, 1:-1, 1:-1]
    return Ix, Iy, It

def compute_probability_field(Ix, Iy, It):
    grad_mag = torch.sqrt(Ix**2 + Iy**2 + It**2 + 1e-8)
    total = grad_mag.sum(dim=(1, 2), keepdim=True)
    G = grad_mag / (total + 1e-8)
    return G

def compute_log_probability(G):
    return torch.log(G + 1e-6)

def compute_nsg(log_p, lambda_val=1e-3):
    """NSG (Eq 2.7): g(x,t) = ∂x log p / (−∂t log p + λ)."""
    spatial_grad = (log_p[:, :, 2:] - log_p[:, :, :-2]) / 2
    temporal_grad = (log_p[2:] - log_p[:-2]) / 2
    spatial_grad = spatial_grad[1:-1]
    temporal_grad = temporal_grad[:, :, 1:-1]
    nsg = spatial_grad / (-temporal_grad + lambda_val)
    nsg = torch.clamp(nsg, -10, 10)
    return nsg

def compute_conservation_error(G):
    """Mean Σ|G(t+1) − G(t)| over time (Eq 2.5)."""
    delta = torch.abs(G[1:] - G[:-1])
    return delta.sum(dim=(1, 2)).mean().item()

def compute_temporal_continuity(G):
    """Mean per-pixel |dG/dt| (Eq 2.6)."""
    dG_dt = torch.abs(G[1:] - G[:-1])
    return dG_dt.mean().item()

def compute_nsg_variance(nsg):
    """Temporal variance per spatial location, averaged over space (Eq 2.8)."""
    return torch.var(nsg, dim=0).mean().item()

def compute_divergence(nsg):
    """Mean |∇·NSG| using torch.gradient (Eq 2.9)."""
    d_dx = torch.gradient(nsg, dim=2)[0]
    d_dy = torch.gradient(nsg, dim=1)[0]
    return torch.abs(d_dx + d_dy).mean().item()

def compute_spatial_entropy(G, epsilon=1e-10):
    """Mean spatial entropy of G per frame."""
    entropy = -torch.sum(G * torch.log(G + epsilon), dim=(1, 2))
    return entropy.mean().item()

def compute_gradient_energy(Ix, Iy, It):
    """Mean total gradient energy per frame."""
    grad_mag = torch.sqrt(Ix**2 + Iy**2 + It**2 + 1e-8)
    return grad_mag.sum(dim=(1, 2)).mean().item()

def compute_directional_coherence(Ix, Iy):
    """Mean cosine similarity between consecutive-frame gradient directions."""
    vx1, vy1 = Ix[:-1], Iy[:-1]
    vx2, vy2 = Ix[1:], Iy[1:]
    dot = vx1 * vx2 + vy1 * vy2
    mag1 = torch.sqrt(vx1**2 + vy1**2 + 1e-8)
    mag2 = torch.sqrt(vx2**2 + vy2**2 + 1e-8)
    return (dot / (mag1 * mag2 + 1e-8)).mean().item()

def compute_temporal_curve(nsg):
    return nsg.abs().mean(dim=(1, 2)).cpu().numpy()


# ── UI helpers ───────────────────────────────────────────────────────────────

def _status(metric, value):
    thresholds = {
        # ascending value = worse
        "conservation": [(0.45, "green", "Normal"),    (0.85, "yellow", "Elevated"),  (1e9, "red", "High")],
        "continuity":   [(5e-7, "green", "Smooth"),    (5e-6, "yellow", "Uneven"),    (1e9, "red", "Abrupt")],
        "variance":     [(5.0,  "green", "Stable"),    (12.0, "yellow", "Variable"),  (1e9, "red", "High")],
        "divergence":   [(1.0,  "green", "Physical"),  (3.0,  "yellow", "Irregular"), (1e9, "red", "Implausible")],
        "coherence":    [(0.20, "green", "Natural"),   (0.35, "yellow", "Elevated"),  (1e9, "red", "High")],
        # ascending threshold = better (lower value = suspicious)
        "entropy":      [(11.60, "red", "Low"),        (11.85, "yellow", "Moderate"), (1e9, "green", "Normal")],
        "energy":       [(15000, "red", "Low"),        (28000, "yellow", "Moderate"), (1e9, "green", "Strong")],
    }
    for t, color, label in thresholds[metric]:
        if value <= t:
            return color, label
    return "red", "High"

def _hex(c):
    return {"green": "#2ea043", "yellow": "#d4a017", "red": "#e05252"}.get(c, "#888")

def _bg(c):
    return {"green": "#0d2b1a", "yellow": "#2b2500", "red": "#2b0d0d"}.get(c, "#1a1a1a")

def render_metric_card(label, description, value_str, key, raw):
    c, badge = _status(key, raw)
    accent, bg = _hex(c), _bg(c)
    st.markdown(f"""
    <div class="metric-card" style="background:{bg}; border-left:4px solid {accent};">
        <div style="font-size:0.72rem; color:#666; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:8px;">{label}</div>
        <div style="font-size:1.45rem; font-weight:700; color:white; margin-bottom:6px;">{value_str}</div>
        <span style="background:{accent}22; color:{accent}; font-size:0.7rem; font-weight:600;
                     padding:2px 10px; border-radius:20px; margin-bottom:10px; display:inline-block;">{badge}</span>
        <div style="font-size:0.82rem; color:#888; line-height:1.5; margin-top:8px;">{description}</div>
    </div>
    """, unsafe_allow_html=True)

def render_verdict(directional_coherence, spatial_entropy, variance, conservation, divergence):
    score = 0

    # Primary: Directional Coherence (largest effect size in thesis data, Cohen's d ≈ 1.05)
    # Fake videos have systematically higher DC — unnaturally uniform motion directions
    if directional_coherence > 0.50:    score += 3
    elif directional_coherence > 0.30:  score += 2
    elif directional_coherence > 0.20:  score += 1

    # Secondary: Spatial Entropy (Cohen's d ≈ 0.84; real videos richer, more diverse)
    if spatial_entropy < 11.65:    score += 2
    elif spatial_entropy < 11.80:  score += 1

    # Supporting: extreme values only
    if variance > 15.0:    score += 1
    if divergence > 3.5:   score += 1

    if score >= 4:
        verdict, c = "LIKELY SYNTHETIC", "red"
        explanation = (
            "Motion direction patterns are unusually uniform — a hallmark of AI-generated or manipulated video. "
            "Natural camera footage produces chaotic, varied motion that follows unpredictable physical laws. "
            "This video's physics signature does not match that pattern."
        )
    elif score >= 2:
        verdict, c = "INCONCLUSIVE", "yellow"
        explanation = (
            "Some motion patterns deviate from what is typical for natural camera footage, but not "
            "strongly enough to reach a definitive conclusion. Results warrant closer manual review."
        )
    else:
        verdict, c = "LIKELY AUTHENTIC", "green"
        explanation = (
            "Motion physics patterns are consistent with natural camera footage. "
            "The video shows the diverse, chaotic motion characteristic of real-world recording. "
            "No significant kinematic anomalies detected."
        )

    accent, bg = _hex(c), _bg(c)
    st.markdown(f"""
    <div class="verdict-box" style="background:{bg}; border:2px solid {accent};">
        <div style="font-size:0.72rem; color:{accent}; text-transform:uppercase;
                    letter-spacing:0.12em; font-weight:600; margin-bottom:8px;">Analysis Result</div>
        <div style="font-size:2.1rem; font-weight:800; color:{accent}; margin-bottom:12px;">{verdict}</div>
        <div style="font-size:0.95rem; color:#ccc; line-height:1.65;">{explanation}</div>
    </div>
    """, unsafe_allow_html=True)

def hr():
    st.markdown("<hr style='border:none; border-top:1px solid #1e2030; margin:28px 0;'>",
                unsafe_allow_html=True)

def plotly_base():
    return dict(
        template='plotly_dark',
        paper_bgcolor='#0f1117',
        plot_bgcolor='#0f1117',
        font=dict(color='#777', size=11),
        margin=dict(l=10, r=10, t=10, b=36),
        xaxis=dict(gridcolor='#1c1f2e', linecolor='#222'),
        yaxis=dict(gridcolor='#1c1f2e', linecolor='#222'),
    )


# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.markdown("## Video Analyzer")
st.sidebar.markdown(
    "<div style='color:#555; font-size:0.82rem; margin-bottom:20px;'>"
    "Physics-informed synthetic video detection</div>",
    unsafe_allow_html=True
)

uploaded_file = st.sidebar.file_uploader("Upload MP4 Video", type=["mp4"])

frame_idx = 10
if uploaded_file is not None:
    st.sidebar.markdown("---")
    frame_idx = st.sidebar.slider("Frame to Inspect", 0, 20, 10,
                                  help="Controls which frame is shown in the spatial maps below.")
    st.sidebar.markdown(
        "<div style='color:#444; font-size:0.78rem; margin-top:8px; line-height:1.5;'>"
        "Drag to examine different moments in the video.</div>",
        unsafe_allow_html=True
    )


# ── Page header ──────────────────────────────────────────────────────────────

st.markdown(
    "<h1 style='font-size:1.9rem; margin-bottom:4px;'>Video Authenticity Analyzer</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<div style='color:#555; font-size:0.92rem; margin-bottom:24px;'>"
    "Detects AI-generated or manipulated video using Normalized Spatiotemporal Gradient (NSG) "
    "physics analysis</div>",
    unsafe_allow_html=True
)


# ── No upload: landing ────────────────────────────────────────────────────────

if uploaded_file is None:
    hr()
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div style='background:#141720; border-radius:12px; padding:24px;'>
            <div style='color:#4a9eff; font-size:0.72rem; text-transform:uppercase;
                        letter-spacing:0.1em; font-weight:600; margin-bottom:14px;'>How it works</div>
            <div style='color:#bbb; font-size:0.9rem; line-height:1.75;'>
                This tool analyzes <strong style='color:#ddd;'>motion physics</strong> in video frames.
                Real cameras capture movement that follows natural physical laws.
                AI-generated or manipulated videos violate these laws in ways invisible to the human eye —
                but detectable through mathematical analysis of spatiotemporal gradients.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div style='background:#141720; border-radius:12px; padding:24px;'>
            <div style='color:#4a9eff; font-size:0.72rem; text-transform:uppercase;
                        letter-spacing:0.1em; font-weight:600; margin-bottom:14px;'>What to upload</div>
            <div style='color:#bbb; font-size:0.9rem; line-height:1.75;'>
                Upload any <strong style='color:#ddd;'>MP4 video</strong>.
                Best results with face or human body footage.
                The analysis uses up to 48 frames sampled evenly — longer videos are handled automatically.
                Sample videos are in the <code style='color:#888;'>data/</code> folder if you want to try.
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown(
        "<div style='color:#333; font-size:0.85rem; text-align:center; margin-top:40px;'>"
        "Upload a video using the sidebar panel to begin</div>",
        unsafe_allow_html=True
    )

# ── Analysis ──────────────────────────────────────────────────────────────────

else:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())

    st.video("temp_video.mp4")

    with st.spinner("Analyzing motion physics... (this may take 20–40 seconds)"):
        video = load_video("temp_video.mp4")
        Ix, Iy, It = compute_spatiotemporal_gradient(video)
        G = compute_probability_field(Ix, Iy, It)
        log_p = compute_log_probability(G)
        nsg = compute_nsg(log_p)

        conservation         = compute_conservation_error(G)
        continuity           = compute_temporal_continuity(G)
        variance             = compute_nsg_variance(nsg)
        divergence           = compute_divergence(nsg)
        spatial_entropy      = compute_spatial_entropy(G)
        gradient_energy      = compute_gradient_energy(Ix, Iy, It)
        directional_coherence = compute_directional_coherence(Ix, Iy)
        dG_dt                = torch.abs(G[1:] - G[:-1])

    hr()

    # ── Step 1: Verdict ──────────────────────────────────────────────────────
    st.markdown("<div class='step-label'>Step 1 of 4 — Overall Assessment</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-heading'>Verdict</div>", unsafe_allow_html=True)
    render_verdict(directional_coherence, spatial_entropy, variance, conservation, divergence)

    hr()

    # ── Step 2: Metric cards ─────────────────────────────────────────────────
    st.markdown("<div class='step-label'>Step 2 of 4 — Physics Integrity Scores</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-heading'>Kinematic Health Indicators</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-sub'>Seven measurements of how well this video obeys natural motion physics. "
        "Each is scored independently — together they form the basis of the verdict above.</div>",
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card(
            "Directional Coherence",
            "How uniformly motion directions repeat across frames. AI-generated video tends to have unnaturally consistent directions — real footage is more chaotic.",
            f"{directional_coherence:.4f}", "coherence", directional_coherence
        )
    with c2:
        render_metric_card(
            "Spatial Entropy",
            "Diversity of motion intensity across the frame. Real video shows rich, varied texture — low entropy suggests artificially smooth or uniform generation.",
            f"{spatial_entropy:.4f}", "entropy", spatial_entropy
        )
    with c3:
        render_metric_card(
            "NSG Variance",
            "Temporal consistency of the physics field at each pixel. Captures how much the motion field fluctuates over time.",
            f"{variance:.4f}", "variance", variance
        )
    with c4:
        render_metric_card(
            "NSG Divergence",
            "Whether motion patterns obey physical flow laws. High values mean physically implausible movement patterns.",
            f"{divergence:.4f}", "divergence", divergence
        )

    c5, c6, c7 = st.columns(3)
    with c5:
        render_metric_card(
            "Conservation Error",
            "Stability of total motion energy across frames. Measures how much the overall motion intensity changes from one frame to the next.",
            f"{conservation:.6f}", "conservation", conservation
        )
    with c6:
        render_metric_card(
            "Temporal Continuity",
            "Smoothness of frame-to-frame probability transitions. Abrupt or jittery changes are a signal of unnatural editing.",
            f"{continuity:.2e}", "continuity", continuity
        )
    with c7:
        render_metric_card(
            "Gradient Energy",
            "Total motion energy captured from spatial and temporal gradients. Real footage typically carries more raw movement energy than synthetic generation.",
            f"{gradient_energy:,.0f}", "energy", gradient_energy
        )

    hr()

    # ── Step 3: Temporal curve ───────────────────────────────────────────────
    st.markdown("<div class='step-label'>Step 3 of 4 — Temporal Motion Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-heading'>Physics Anomalies Over Time</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-sub'>Tracks the average physics violation score for each frame. "
        "Spikes reveal moments where the video's motion suddenly defies natural laws — "
        "a common artifact of AI generation or face-swapping.</div>",
        unsafe_allow_html=True
    )

    curve = compute_temporal_curve(nsg)
    x = np.arange(len(curve))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=curve,
        mode="lines",
        line=dict(width=2.5, color="#4a9eff"),
        fill='tozeroy',
        fillcolor="rgba(74,158,255,0.07)",
        hovertemplate='Frame %{x}: score = %{y:.4f}<extra></extra>'
    ))
    fig.update_layout(
        xaxis_title="Frame",
        yaxis_title="Physics Violation Score",
        height=280,
        **plotly_base()
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        "<div class='chart-caption'>Flat = consistent natural motion (authentic signal). "
        "Large spikes = frames with abnormal physics (suspicious signal).</div>",
        unsafe_allow_html=True
    )

    hr()

    # ── Step 4: Spatial maps ─────────────────────────────────────────────────
    safe_nsg  = min(frame_idx, nsg.shape[0] - 1)
    safe_cont = min(frame_idx, dG_dt.shape[0] - 1)

    st.markdown("<div class='step-label'>Step 4 of 4 — Spatial Physics Map</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-heading'>Where Physics Breaks Down</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='section-sub'>Heat maps showing <em>where</em> in the frame physics violations occur "
        f"(frame {frame_idx}). Use the sidebar slider to inspect different moments. "
        f"Brighter regions are areas of concern.</div>",
        unsafe_allow_html=True
    )

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("<div class='map-label'>Motion Physics Field (NSG)</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='map-sub'>Bright areas = pixels where motion violates natural physics</div>",
            unsafe_allow_html=True
        )
        fig_heat = px.imshow(
            nsg[safe_nsg].cpu().numpy(),
            color_continuous_scale='inferno',
            origin='lower',
            labels=dict(color='NSG'),
        )
        fig_heat.update_layout(
            height=400,
            paper_bgcolor='#0f1117',
            margin=dict(l=0, r=0, t=0, b=0),
            coloraxis_showscale=False,
        )
        fig_heat.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
        fig_heat.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
        st.plotly_chart(fig_heat, use_container_width=True)

    with col_m2:
        st.markdown("<div class='map-label'>Frame Transition Map</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='map-sub'>Bright patches = pixels that changed unnaturally between frames</div>",
            unsafe_allow_html=True
        )
        fig_cont = px.imshow(
            dG_dt[safe_cont].cpu().numpy(),
            color_continuous_scale='magma',
            origin='lower',
            labels=dict(color='dG/dt'),
        )
        fig_cont.update_layout(
            height=400,
            paper_bgcolor='#0f1117',
            margin=dict(l=0, r=0, t=0, b=0),
            coloraxis_showscale=False,
        )
        fig_cont.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
        fig_cont.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
        st.plotly_chart(fig_cont, use_container_width=True)

    hr()

    # ── Technical details ────────────────────────────────────────────────────
    with st.expander("Technical Details — How This Analysis Works"):
        st.markdown("""
**Normalized Spatiotemporal Gradient (NSG)**

NSG models video as a probability flow field. For each pixel at each frame, it measures how the
probability of observing a given motion intensity changes over space and time. Natural video produces
a smooth, consistent field. AI-generated or manipulated video introduces subtle discontinuities in
this field — detectable even when the video looks convincing to the human eye.

**Analysis pipeline**
1. Resize all frames to 512×512 via bilinear interpolation, normalize to [−1, 1], apply Gaussian blur
2. Extract per-pixel spatial gradients (Ix, Iy) and temporal gradient (It) via finite differences
3. Build a normalized probability field G from gradient magnitudes
4. Compute NSG = ∂(log G)/∂x / (−∂(log G)/∂t + λ), clamped to [−10, 10]
5. Aggregate seven scalar metrics across the video

**Key discriminating metrics (from thesis statistical analysis)**
- **Directional Coherence**: Strongest signal (Cohen's d ≈ 1.05). Measures cosine similarity between consecutive-frame gradient directions. Synthetic videos have unnaturally uniform motion directions (mean ≈ 0.37) vs real videos (mean ≈ 0.15).
- **Spatial Entropy**: Second strongest (Cohen's d ≈ 0.84). Real videos have richer, more diverse pixel texture (mean ≈ 11.99) vs synthetic (mean ≈ 11.75).

**Detection logic**
- Directional Coherence > 0.50 → strong synthetic signal (+3)
- Directional Coherence > 0.30 → elevated (+2)
- Spatial Entropy < 11.65 → elevated synthetic signal (+2)
- Score ≥ 4 → Likely Synthetic; Score ≥ 2 → Inconclusive; else → Likely Authentic

**Limitations**: Works best on face or human body video. Heavy compression, static scenes,
or very short clips may reduce accuracy. This is a research prototype, not a production classifier.
        """)
