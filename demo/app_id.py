import streamlit as st
import torch
import numpy as np
import imageio.v2 as imageio
import torch.nn.functional as F
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Pendeteksi Keaslian Video",
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


# ── Komputasi ─────────────────────────────────────────────────────────────

def resize_frame(frame, size):
    """Resize (C,H,W) ke size×size dengan bilinear interpolation."""
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
    Preprocessing sesuai BAB 3.1.3: 512×512, max_frames=48, Gaussian blur.
    Frame diambil secara merata dari seluruh video — tanpa artefak stride.
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
        raise ValueError(f"Hanya {len(all_frames)} frame yang terbaca. Periksa codec video.")

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
    """NSG (Pers. 2.7): g(x,t) = ∂x log p / (−∂t log p + λ)."""
    spatial_grad = (log_p[:, :, 2:] - log_p[:, :, :-2]) / 2
    temporal_grad = (log_p[2:] - log_p[:-2]) / 2
    spatial_grad = spatial_grad[1:-1]
    temporal_grad = temporal_grad[:, :, 1:-1]
    nsg = spatial_grad / (-temporal_grad + lambda_val)
    nsg = torch.clamp(nsg, -10, 10)
    return nsg

def compute_conservation_error(G):
    """Rata-rata Σ|G(t+1) − G(t)| terhadap waktu (Pers. 2.5)."""
    delta = torch.abs(G[1:] - G[:-1])
    return delta.sum(dim=(1, 2)).mean().item()

def compute_temporal_continuity(G):
    """Rata-rata per-piksel |dG/dt| (Pers. 2.6)."""
    dG_dt = torch.abs(G[1:] - G[:-1])
    return dG_dt.mean().item()

def compute_nsg_variance(nsg):
    """Varians temporal per lokasi spasial, dirata-rata atas ruang (Pers. 2.8)."""
    return torch.var(nsg, dim=0).mean().item()

def compute_divergence(nsg):
    """Rata-rata |∇·NSG| menggunakan torch.gradient (Pers. 2.9)."""
    d_dx = torch.gradient(nsg, dim=2)[0]
    d_dy = torch.gradient(nsg, dim=1)[0]
    return torch.abs(d_dx + d_dy).mean().item()

def compute_spatial_entropy(G, epsilon=1e-10):
    """Rata-rata entropi spasial G per frame."""
    entropy = -torch.sum(G * torch.log(G + epsilon), dim=(1, 2))
    return entropy.mean().item()

def compute_gradient_energy(Ix, Iy, It):
    """Rata-rata total energi gradien per frame."""
    grad_mag = torch.sqrt(Ix**2 + Iy**2 + It**2 + 1e-8)
    return grad_mag.sum(dim=(1, 2)).mean().item()

def compute_directional_coherence(Ix, Iy):
    """Rata-rata cosine similarity antara arah gradien frame berurutan."""
    vx1, vy1 = Ix[:-1], Iy[:-1]
    vx2, vy2 = Ix[1:], Iy[1:]
    dot = vx1 * vx2 + vy1 * vy2
    mag1 = torch.sqrt(vx1**2 + vy1**2 + 1e-8)
    mag2 = torch.sqrt(vx2**2 + vy2**2 + 1e-8)
    return (dot / (mag1 * mag2 + 1e-8)).mean().item()

def compute_temporal_curve(nsg):
    return nsg.abs().mean(dim=(1, 2)).cpu().numpy()

def compute_second_order_score(G):
    return torch.abs(G[2:] - 2 * G[1:-1] + G[:-2]).mean().item()

def compute_second_order_curve(G):
    second = torch.abs(G[2:] - 2 * G[1:-1] + G[:-2])
    return second.mean(dim=(1, 2)).cpu().numpy()


# ── Helper UI ───────────────────────────────────────────────────────────────

def _status(metric, value):
    thresholds = {
        # nilai naik = lebih buruk
        "conservation": [(0.45, "green", "Normal"),       (0.85, "yellow", "Meningkat"),       (1e9, "red", "Tinggi")],
        "continuity":   [(5e-7, "green", "Halus"),        (5e-6, "yellow", "Tidak Rata"),      (1e9, "red", "Mendadak")],
        "variance":     [(5.0,  "green", "Stabil"),       (12.0, "yellow", "Bervariasi"),      (1e9, "red", "Tinggi")],
        "divergence":   [(1.0,  "green", "Wajar"),        (3.0,  "yellow", "Tidak Beraturan"), (1e9, "red", "Tidak Masuk Akal")],
        "coherence":    [(0.20, "green", "Natural"),      (0.35, "yellow", "Meningkat"),       (1e9, "red", "Tinggi")],
        # nilai naik = lebih baik (nilai rendah = mencurigakan)
        "entropy":      [(11.60, "red", "Rendah"),        (11.85, "yellow", "Sedang"),         (1e9, "green", "Normal")],
        "energy":       [(15000, "red", "Rendah"),        (28000, "yellow", "Sedang"),         (1e9, "green", "Kuat")],
        "second_order": [(5e-6,  "green", "Halus"),        (1e-5,  "yellow", "Meningkat"),      (1e9, "red", "Tinggi")],
    }
    for t, color, label in thresholds[metric]:
        if value <= t:
            return color, label
    return "red", "Tinggi"

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

    # Utama: Directional Coherence (efek terbesar dalam data tesis, Cohen's d ≈ 1.05)
    # Video palsu memiliki DC lebih tinggi — arah gerak yang seragam secara tidak wajar
    if directional_coherence > 0.50:    score += 3
    elif directional_coherence > 0.30:  score += 2
    elif directional_coherence > 0.20:  score += 1

    # Sekunder: Spatial Entropy (Cohen's d ≈ 0.84; video asli lebih kaya tekstur)
    if spatial_entropy < 11.65:    score += 2
    elif spatial_entropy < 11.80:  score += 1

    # Pendukung: hanya nilai ekstrem
    if variance > 15.0:    score += 1
    if divergence > 3.5:   score += 1

    if score >= 4:
        verdict, c = "KEMUNGKINAN VIDEO PALSU", "red"
        explanation = (
            "Pola arah gerak terlalu seragam dan terstruktur — ciri khas video buatan AI atau yang dimanipulasi. "
            "Video dari kamera nyata menghasilkan gerak yang lebih acak dan tidak terduga sesuai hukum fisika. "
            "Tanda fisika video ini tidak sesuai dengan pola tersebut."
        )
    elif score >= 2:
        verdict, c = "TIDAK DAPAT DISIMPULKAN", "yellow"
        explanation = (
            "Beberapa pola gerak menyimpang dari video kamera nyata, namun belum cukup kuat untuk kesimpulan pasti. "
            "Disarankan untuk dilakukan peninjauan lebih lanjut secara manual."
        )
    else:
        verdict, c = "KEMUNGKINAN VIDEO ASLI", "green"
        explanation = (
            "Pola fisika gerak konsisten dengan rekaman kamera nyata. "
            "Video menunjukkan gerak yang beragam dan acak, khas dari pengambilan gambar di dunia nyata. "
            "Tidak ada kejanggalan kinematik yang signifikan terdeteksi."
        )

    accent, bg = _hex(c), _bg(c)
    st.markdown(f"""
    <div class="verdict-box" style="background:{bg}; border:2px solid {accent};">
        <div style="font-size:0.72rem; color:{accent}; text-transform:uppercase;
                    letter-spacing:0.12em; font-weight:600; margin-bottom:8px;">Hasil Analisis</div>
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

st.sidebar.markdown("## Analisis Video")
st.sidebar.markdown(
    "<div style='color:#555; font-size:0.82rem; margin-bottom:20px;'>"
    "Pendeteksi video palsu berbasis fisika gerak</div>",
    unsafe_allow_html=True
)

uploaded_file = st.sidebar.file_uploader("Unggah Video MP4", type=["mp4"])

frame_idx = 10
if uploaded_file is not None:
    st.sidebar.markdown("---")
    frame_idx = st.sidebar.slider("Frame yang Ingin Dilihat", 0, 20, 10,
                                  help="Mengatur frame mana yang ditampilkan pada peta visual di bawah.")
    st.sidebar.markdown(
        "<div style='color:#444; font-size:0.78rem; margin-top:8px; line-height:1.5;'>"
        "Geser untuk memeriksa bagian berbeda dari video.</div>",
        unsafe_allow_html=True
    )


# ── Header halaman ──────────────────────────────────────────────────────────

st.markdown(
    "<h1 style='font-size:1.9rem; margin-bottom:4px;'>Pendeteksi Keaslian Video</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<div style='color:#555; font-size:0.92rem; margin-bottom:24px;'>"
    "Mendeteksi video buatan AI atau yang dipalsukan menggunakan analisis pola gerak fisika (NSG)</div>",
    unsafe_allow_html=True
)


# ── Halaman awal (belum ada unggahan) ────────────────────────────────────────

if uploaded_file is None:
    hr()
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div style='background:#141720; border-radius:12px; padding:24px;'>
            <div style='color:#4a9eff; font-size:0.72rem; text-transform:uppercase;
                        letter-spacing:0.1em; font-weight:600; margin-bottom:14px;'>Cara Kerja</div>
            <div style='color:#bbb; font-size:0.9rem; line-height:1.75;'>
                Alat ini menganalisis <strong style='color:#ddd;'>pola pergerakan</strong> pada setiap frame video.
                Video dari kamera nyata memiliki pergerakan yang mengikuti hukum fisika alami.
                Video buatan AI atau yang dipalsukan melanggar hukum-hukum tersebut
                dengan cara yang tidak terlihat oleh mata manusia —
                namun dapat dideteksi melalui analisis matematika.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div style='background:#141720; border-radius:12px; padding:24px;'>
            <div style='color:#4a9eff; font-size:0.72rem; text-transform:uppercase;
                        letter-spacing:0.1em; font-weight:600; margin-bottom:14px;'>Video yang Cocok</div>
            <div style='color:#bbb; font-size:0.9rem; line-height:1.75;'>
                Unggah video <strong style='color:#ddd;'>MP4</strong> apapun.
                Hasil terbaik dengan video wajah atau tubuh manusia.
                Analisis menggunakan hingga 48 frame yang diambil merata — video lebih panjang ditangani otomatis.
                Video contoh tersedia di folder <code style='color:#888;'>data/</code> untuk dicoba.
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown(
        "<div style='color:#333; font-size:0.85rem; text-align:center; margin-top:40px;'>"
        "Unggah video melalui panel samping untuk memulai analisis</div>",
        unsafe_allow_html=True
    )

# ── Analisis ──────────────────────────────────────────────────────────────────

else:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())

    st.video("temp_video.mp4")

    with st.spinner("Menganalisis pola fisika gerak... (proses ini memerlukan 20–40 detik)"):
        video = load_video("temp_video.mp4")
        Ix, Iy, It = compute_spatiotemporal_gradient(video)
        G = compute_probability_field(Ix, Iy, It)
        log_p = compute_log_probability(G)
        nsg = compute_nsg(log_p)

        conservation          = compute_conservation_error(G)
        continuity            = compute_temporal_continuity(G)
        variance              = compute_nsg_variance(nsg)
        divergence            = compute_divergence(nsg)
        spatial_entropy       = compute_spatial_entropy(G)
        gradient_energy       = compute_gradient_energy(Ix, Iy, It)
        directional_coherence = compute_directional_coherence(Ix, Iy)
        dG_dt                 = torch.abs(G[1:] - G[:-1])
        second_order          = compute_second_order_score(G)
        second_order_curve_data = compute_second_order_curve(G)
        second_order_map      = torch.abs(G[2:] - 2 * G[1:-1] + G[:-2])

    hr()

    # ── Langkah 1: Hasil Penilaian ───────────────────────────────────────────
    st.markdown("<div class='step-label'>Langkah 1 dari 4 — Penilaian Keseluruhan</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-heading'>Hasil Penilaian</div>", unsafe_allow_html=True)
    render_verdict(directional_coherence, spatial_entropy, variance, conservation, divergence)

    hr()

    # ── Langkah 2: Kartu Metrik ──────────────────────────────────────────────
    st.markdown("<div class='step-label'>Langkah 2 dari 4 — Skor Ukuran Fisika</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-heading'>Tujuh Ukuran Kewajaran Gerak</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-sub'>Delapan pengukuran seberapa baik video ini mematuhi hukum fisika gerak alami. "
        "Masing-masing diukur secara terpisah — bersama-sama menjadi dasar penilaian di atas.</div>",
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card(
            "Konsistensi Arah Gerak",
            "Seberapa seragam arah gerakan berulang antar frame. Video buatan AI cenderung memiliki arah gerak yang terlalu konsisten — rekaman nyata lebih acak dan beragam.",
            f"{directional_coherence:.4f}", "coherence", directional_coherence
        )
    with c2:
        render_metric_card(
            "Entropi Spasial",
            "Keberagaman intensitas gerak di seluruh frame. Video asli menunjukkan tekstur yang kaya dan bervariasi — nilai rendah menandakan hasil generasi yang terlalu halus atau seragam.",
            f"{spatial_entropy:.4f}", "entropy", spatial_entropy
        )
    with c3:
        render_metric_card(
            "Keseragaman Pola NSG",
            "Konsistensi temporal medan fisika pada setiap piksel. Mengukur seberapa besar medan gerak berfluktuasi terhadap waktu.",
            f"{variance:.4f}", "variance", variance
        )
    with c4:
        render_metric_card(
            "Kewajaran Aliran Gerak",
            "Apakah pola gerak mengikuti hukum fisika aliran. Nilai tinggi berarti ada pergerakan yang tidak masuk akal secara fisika.",
            f"{divergence:.4f}", "divergence", divergence
        )

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        render_metric_card(
            "Stabilitas Energi Gerak",
            "Kestabilan total energi gerak antar frame. Mengukur seberapa besar intensitas gerak keseluruhan berubah dari satu frame ke frame berikutnya.",
            f"{conservation:.6f}", "conservation", conservation
        )
    with c6:
        render_metric_card(
            "Kehalusan Pergantian Frame",
            "Kelancaran transisi probabilitas antar frame. Perubahan mendadak atau kasar adalah tanda adanya penyuntingan yang tidak wajar.",
            f"{continuity:.2e}", "continuity", continuity
        )
    with c7:
        render_metric_card(
            "Energi Gradien Total",
            "Total energi gerak yang tertangkap dari gradien spasial dan temporal. Rekaman nyata biasanya memiliki energi gerak mentah lebih besar dibanding video sintetis.",
            f"{gradient_energy:,.0f}", "energy", gradient_energy
        )
    with c8:
        render_metric_card(
            "Kontinuitas Orde Dua",
            "Akselerasi perubahan medan probabilitas antar frame. Mengukur laju perubahan transisi temporal — nilai tinggi menandakan pola tidak wajar yang umum pada video buatan AI.",
            f"{second_order:.2e}", "second_order", second_order
        )

    hr()

    # ── Langkah 3: Kurva Temporal ────────────────────────────────────────────
    st.markdown("<div class='step-label'>Langkah 3 dari 4 — Analisis Gerak Per Waktu</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-heading'>Kejanggalan Fisika dari Waktu ke Waktu</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-sub'>Grafik ini menunjukkan rata-rata skor pelanggaran fisika untuk setiap frame. "
        "Lonjakan pada grafik menandai momen di mana gerak video tiba-tiba tidak sesuai hukum alam — "
        "hal ini umum ditemukan pada video buatan AI atau hasil face-swap.</div>",
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
        hovertemplate='Frame %{x}: skor = %{y:.4f}<extra></extra>'
    ))
    fig.update_layout(
        xaxis_title="Frame",
        yaxis_title="Skor Kejanggalan Gerak",
        height=280,
        **plotly_base()
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        "<div class='chart-caption'>Garis datar = pola gerak konsisten (sinyal video asli). "
        "Lonjakan besar = frame dengan gerak tidak wajar (sinyal mencurigakan).</div>",
        unsafe_allow_html=True
    )

    curve2 = second_order_curve_data
    x2 = np.arange(len(curve2))
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=x2, y=curve2,
        mode="lines",
        line=dict(width=2.5, color="#a78bfa"),
        fill='tozeroy',
        fillcolor="rgba(167,139,250,0.07)",
        hovertemplate='Frame %{x}: skor = %{y:.4e}<extra></extra>'
    ))
    fig2.update_layout(
        xaxis_title="Frame",
        yaxis_title="Skor Orde Dua",
        height=220,
        **plotly_base()
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown(
        "<div class='chart-caption'>Kontinuitas temporal orde dua — mengukur akselerasi perubahan medan probabilitas. "
        "Nilai tinggi menandakan diskontinuitas temporal yang tidak wajar secara fisika.</div>",
        unsafe_allow_html=True
    )

    hr()

    # ── Langkah 4: Peta Spasial ──────────────────────────────────────────────
    safe_nsg  = min(frame_idx, nsg.shape[0] - 1)
    safe_cont = min(frame_idx, dG_dt.shape[0] - 1)
    safe_so   = min(frame_idx, second_order_map.shape[0] - 1)

    st.markdown("<div class='step-label'>Langkah 4 dari 4 — Peta Visual Kejanggalan</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-heading'>Di Mana Gerak Tidak Wajar Terjadi</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='section-sub'>Peta panas yang menunjukkan <em>di mana</em> dalam frame kejanggalan fisika terjadi "
        f"(frame {frame_idx}). Gunakan slider di panel samping untuk memeriksa frame berbeda. "
        f"Area lebih terang menunjukkan bagian yang perlu diperhatikan.</div>",
        unsafe_allow_html=True
    )

    col_m1, col_m2, col_m3 = st.columns(3)

    with col_m1:
        st.markdown("<div class='map-label'>Medan Pola Gerak Fisika (NSG)</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='map-sub'>Area terang = piksel di mana gerak melanggar hukum fisika alami</div>",
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
        st.markdown("<div class='map-label'>Peta Perubahan Antar Frame</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='map-sub'>Bercak terang = piksel yang berubah secara tidak wajar antar frame</div>",
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

    with col_m3:
        st.markdown("<div class='map-label'>Peta Kontinuitas Orde Dua</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='map-sub'>Bercak terang = piksel dengan akselerasi temporal yang tidak wajar</div>",
            unsafe_allow_html=True
        )
        fig_so = px.imshow(
            second_order_map[safe_so].cpu().numpy(),
            color_continuous_scale='viridis',
            origin='lower',
            labels=dict(color='∂²G/∂t²'),
        )
        fig_so.update_layout(
            height=400,
            paper_bgcolor='#0f1117',
            margin=dict(l=0, r=0, t=0, b=0),
            coloraxis_showscale=False,
        )
        fig_so.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
        fig_so.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
        st.plotly_chart(fig_so, use_container_width=True)

    hr()

    # ── Detail Teknis ────────────────────────────────────────────────────────
    with st.expander("Detail Teknis — Cara Kerja Analisis Ini"):
        st.markdown("""
**Normalized Spatiotemporal Gradient (NSG)**

NSG memodelkan video sebagai medan aliran probabilitas. Untuk setiap piksel pada setiap frame,
NSG mengukur seberapa besar perubahan probabilitas kemunculan intensitas gerak tertentu terhadap
ruang dan waktu. Video nyata menghasilkan medan yang halus dan konsisten. Video buatan AI atau
yang dimanipulasi menghasilkan diskontinuitas kecil pada medan ini — terdeteksi meskipun video
terlihat meyakinkan secara visual.

**Alur analisis**
1. Ubah ukuran semua frame ke 512×512 dengan interpolasi bilinear, normalisasi ke [−1, 1], terapkan Gaussian blur
2. Hitung gradien spasial per piksel (Ix, Iy) dan gradien temporal (It) menggunakan beda hingga
3. Bangun medan probabilitas ternormalisasi G dari magnitudo gradien
4. Hitung NSG = ∂(log G)/∂x / (−∂(log G)/∂t + λ), dibatasi pada rentang [−10, 10]
5. Agregasi delapan metrik skalar dari seluruh video

**Metrik diskriminan utama (dari analisis statistik tesis)**
- **Konsistensi Arah Gerak**: Sinyal terkuat (Cohen's d ≈ 1,05). Mengukur kemiripan cosine antara arah gradien frame berurutan. Video sintetis memiliki arah yang terlalu seragam (rata-rata ≈ 0,37) vs video asli (rata-rata ≈ 0,15).
- **Entropi Spasial**: Kedua terkuat (Cohen's d ≈ 0,84). Video asli memiliki tekstur lebih kaya dan beragam (rata-rata ≈ 11,99) vs sintetis (rata-rata ≈ 11,75).
- **Kontinuitas Orde Dua**: Sinyal pendukung. Mengukur turunan temporal orde dua dari G — 'akselerasi' perubahan medan probabilitas. Video sintetis menunjukkan nilai sedikit lebih tinggi karena diskontinuitas temporal yang tidak wajar secara fisika.

**Logika deteksi**
- Konsistensi Arah Gerak > 0,50 → sinyal kuat video palsu (+3)
- Konsistensi Arah Gerak > 0,30 → sinyal meningkat (+2)
- Entropi Spasial < 11,65 → sinyal video palsu meningkat (+2)
- Skor ≥ 4 → Kemungkinan Palsu; Skor ≥ 2 → Tidak Dapat Disimpulkan; lainnya → Kemungkinan Asli

**Keterbatasan**: Hasil terbaik pada video wajah atau tubuh manusia. Kompresi berat, adegan statis,
atau video yang sangat pendek dapat mengurangi akurasi. Ini adalah prototipe penelitian, bukan sistem klasifikasi produksi.
        """)
