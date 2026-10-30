import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ============================================================
# LTE / 5G QAM DEMO
# Giữ nguyên các nội dung cốt lõi: phát & ánh xạ, kênh AWGN /
# Rayleigh, thu Euclidean, BER / SER và chòm sao tín hiệu.
# ============================================================

st.set_page_config(
    page_title="LTE/5G | QAM Simulation Lab",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Giao diện
st.markdown("""
<style>
    .stApp { background: linear-gradient(180deg, #f5f8fc 0%, #edf2f8 100%); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #101d35 0%, #172b4d 100%);
    }
    [data-testid="stSidebar"] * { color: #f4f7fb !important; }
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] *,
    [data-testid="stSidebar"] input { color: #172033 !important; }
    .hero {
        padding: 1.45rem 1.7rem; border-radius: 18px; color: white;
        background: linear-gradient(115deg, #102544 0%, #175b86 55%, #168c9b 100%);
        margin-bottom: 1rem; box-shadow: 0 10px 28px rgba(20,50,85,.16);
    }
    .hero h1 { font-size: 2rem; margin: 0 0 .45rem 0; }
    .hero p { margin: 0; color: #e4f3fb; font-size: 1rem; }
    .section-label {
        color: #54708f; text-transform: uppercase; letter-spacing: .11em;
        font-size: .75rem; font-weight: 700; margin-bottom: .25rem;
    }
    div[data-testid="stMetric"] {
        background: white; border: 1px solid #e4ebf3; padding: 14px 16px;
        border-radius: 14px; box-shadow: 0 4px 14px rgba(25,50,80,.045);
    }
    div[data-testid="stTabs"] button { font-weight: 650; }
    .note {
        background: #eaf3fb; border-left: 4px solid #168c9b;
        padding: .8rem 1rem; border-radius: 8px; color: #243b53;
    }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="section-label" style="color:#b9e7f2">COMMUNICATION SYSTEMS • SIMULATION LAB</div>
  <h1>📡 Mô phỏng và so sánh kỹ thuật điều chế QPSK / QAM</h1>
  <p>Đồ án LTE/5G — mô phỏng phát, truyền qua kênh fading, thu và đánh giá BER/SER.</p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    "Hệ thống tích hợp toàn bộ module: **Phát & Ánh xạ → Kênh truyền & Fading "
    "→ Thu, giải điều chế Euclidean → Đo BER/SER**."
)

# ------------------------- XỬ LÝ TÍN HIỆU -------------------------

def tao_chom_sao_va_map(M):
    """Tạo chòm sao chuẩn hóa và ánh xạ chỉ số symbol sang chuỗi bit."""
    if M == 4:
        bits_map = [
            np.array([0, 0]), np.array([0, 1]),
            np.array([1, 0]), np.array([1, 1]),
        ]
        points = np.array([
            (1 + 1j) / np.sqrt(2),
            (1 - 1j) / np.sqrt(2),
            (-1 + 1j) / np.sqrt(2),
            (-1 - 1j) / np.sqrt(2),
        ])
        return points, bits_map

    m = int(np.sqrt(M))
    k = int(np.log2(M))
    levels = np.arange(-(m - 1), m, 2)
    levels = levels / np.sqrt(2 * (M - 1) / 3)
    in_phase, quadrature = np.meshgrid(levels, levels)
    points = in_phase.flatten() + 1j * quadrature.flatten()

    bits_map = []
    for i in range(M):
        bits_val = [(i >> bit_idx) & 1 for bit_idx in range(k - 1, -1, -1)]
        bits_map.append(np.array(bits_val))
    return points, bits_map


def dieu_che_bits(bits, M):
    k = int(np.log2(M))
    remainder = len(bits) % k
    if remainder:
        bits = np.pad(bits, (0, k - remainder), mode="constant")
    bits_reshaped = bits.reshape((-1, k))
    powers = 2 ** np.arange(k - 1, -1, -1)
    symbol_indices = np.dot(bits_reshaped, powers)
    points, _ = tao_chom_sao_va_map(M)
    return symbol_indices, points[symbol_indices]


def kenh_truyen(tx_symbols, snr_db, M, channel_type, rng=None):
    """Mô phỏng AWGN hoặc Rayleigh; cân bằng kênh ở phía thu."""
    if rng is None:
        rng = np.random.default_rng()
    k = int(np.log2(M))
    snr_lin = 10 ** (snr_db / 10.0)
    noise_var = 1 / (k * snr_lin)
    noise = np.sqrt(noise_var / 2) * (
        rng.standard_normal(len(tx_symbols)) +
        1j * rng.standard_normal(len(tx_symbols))
    )
    if channel_type == "Rayleigh Fading":
        h = (rng.standard_normal(len(tx_symbols)) +
             1j * rng.standard_normal(len(tx_symbols))) / np.sqrt(2)
        # Tránh chia cho giá trị fading gần bằng 0
        h = np.where(np.abs(h) < 1e-12, 1e-12 + 0j, h)
        return (h * tx_symbols + noise) / h
    return tx_symbols + noise


def qam_demodulate(received_symbols, constellation_points, bit_mapping):
    demod_bits = []
    for start in range(0, len(received_symbols), 2000):
        end = min(start + 2000, len(received_symbols))
        distances = np.abs(
            received_symbols[start:end, None] -
            constellation_points[None, :]
        ) ** 2
        min_indices = np.argmin(distances, axis=1)
        for idx in min_indices:
            demod_bits.extend(bit_mapping[idx])
    return np.asarray(demod_bits, dtype=int)


def calculate_error_rates(tx_bits, rx_bits, tx_symbol_indices, rx_symbol_indices):
    bit_count = min(len(tx_bits), len(rx_bits))
    symbol_count = min(len(tx_symbol_indices), len(rx_symbol_indices))
    ber = (np.count_nonzero(tx_bits[:bit_count] != rx_bits[:bit_count]) / bit_count
           if bit_count else 0.0)
    ser = (np.count_nonzero(
        tx_symbol_indices[:symbol_count] != rx_symbol_indices[:symbol_count]
    ) / symbol_count if symbol_count else 0.0)
    return ber, ser


def simulate(M, channel, n_bits, snr_values, seed=2026):
    rng = np.random.default_rng(seed)
    tx_bits = rng.integers(0, 2, n_bits)
    tx_indices, tx_symbols = dieu_che_bits(tx_bits, M)
    points, bit_mapping = tao_chom_sao_va_map(M)
    ber_values, ser_values = [], []
    for snr in snr_values:
        rx_symbols = kenh_truyen(tx_symbols, snr, M, channel, rng)
        rx_bits = qam_demodulate(rx_symbols, points, bit_mapping)
        rx_indices = np.argmin(
            np.abs(rx_symbols[:, None] - points[None, :]) ** 2, axis=1
        )
        ber, ser = calculate_error_rates(
            tx_bits, rx_bits, tx_indices, rx_indices
        )
        ber_values.append(ber)
        ser_values.append(ser)
    return tx_bits, tx_symbols, points, np.asarray(ber_values), np.asarray(ser_values)


# ------------------------- THANH ĐIỀU KHIỂN -------------------------

st.sidebar.markdown("## ⚙️ Cấu hình mô phỏng")
st.sidebar.caption("THIẾT LẬP HỆ THỐNG LTE / 5G")
selected_m = st.sidebar.selectbox(
    "Kỹ thuật điều chế", [4, 16, 64, 256],
    format_func=lambda x: "QPSK" if x == 4 else f"{x}-QAM",
)
selected_channel = st.sidebar.selectbox(
    "Kênh truyền", ["AWGN", "Rayleigh Fading"]
)
snr_db_slider = st.sidebar.slider(
    "SNR quan sát chòm sao (dB)", -5, 30, 15, 1
)
N_bits = st.sidebar.selectbox(
    "Số lượng bit mô phỏng", [10000, 20000, 30000, 40000, 50000], index=1
)
seed = st.sidebar.number_input("Seed ngẫu nhiên", min_value=0, max_value=999999,
                               value=2026, step=1)
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 Gợi ý AMC")
if snr_db_slider < 5:
    st.sidebar.warning("SNR thấp (< 5 dB): ưu tiên **QPSK** để tăng độ bền.")
elif snr_db_slider < 15:
    st.sidebar.info("SNR trung bình (5–15 dB): cân nhắc **16-QAM / 64-QAM**.")
else:
    st.sidebar.success("SNR cao (≥ 15 dB): có thể cân nhắc **256-QAM** để tăng thông lượng.")
st.sidebar.caption("Gợi ý AMC mang tính tham khảo; ngưỡng thực tế phụ thuộc yêu cầu BLER, mã hóa và điều kiện kênh.")

run_button = st.sidebar.button("▶  Chạy mô phỏng", type="primary", use_container_width=True)

# ------------------------- DASHBOARD -------------------------

if "simulation" not in st.session_state:
    st.session_state.simulation = None

if run_button:
    snr_db_range = np.arange(-5, 31, 2)
    with st.spinner("Đang mô phỏng tín hiệu và tính BER/SER..."):
        result = simulate(selected_m, selected_channel, N_bits, snr_db_range, int(seed))
        st.session_state.simulation = {
            "result": result, "snr_range": snr_db_range,
            "M": selected_m, "channel": selected_channel,
            "n_bits": N_bits, "seed": int(seed),
        }

if st.session_state.simulation is None:
    st.markdown("""
    <div class="note">
    <b>Sẵn sàng mô phỏng.</b> Chọn cấu hình ở thanh bên trái, sau đó nhấn
    <b>Chạy mô phỏng</b> để xem đường cong BER/SER và chòm sao tín hiệu.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("#### Quy trình hệ thống")
    c1, c2, c3, c4 = st.columns(4)
    c1.info("**01 — Phát**\n\nSinh bit và ánh xạ lên chòm sao.")
    c2.info("**02 — Kênh**\n\nAWGN hoặc Rayleigh Fading.")
    c3.info("**03 — Thu**\n\nCân bằng kênh và giải điều chế Euclidean.")
    c4.info("**04 — Đánh giá**\n\nTính BER và SER theo SNR.")
else:
    data = st.session_state.simulation
    tx_bits, tx_symbols, points, ber_list, ser_list = data["result"]
    snr_db_range = data["snr_range"]
    M, channel = data["M"], data["channel"]
    mod_name = "QPSK" if M == 4 else f"{M}-QAM"
    idx = int(np.argmin(np.abs(snr_db_range - snr_db_slider)))
    # Chạy riêng chòm sao với SNR đang chọn, không thay đổi dữ liệu đường cong.
    constellation_rng = np.random.default_rng(int(seed) + 77)
    sample_count = min(500, len(tx_symbols))
    sample_rx = kenh_truyen(
        tx_symbols[:sample_count], snr_db_slider, M, channel, constellation_rng
    )
    rx_bits_at_point = qam_demodulate(sample_rx, points, tao_chom_sao_va_map(M)[1])
    tx_sample_bits = tx_bits[:len(rx_bits_at_point)]
    sample_ber = (np.mean(tx_sample_bits != rx_bits_at_point)
                  if len(tx_sample_bits) else 0.0)

    st.markdown(f"### Kết quả mô phỏng — {mod_name} / {channel}")
    st.caption(f"{data['n_bits']:,} bit • SNR quét từ -5 đến 30 dB, bước 2 dB • Seed {data['seed']}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Điều chế", mod_name)
    m2.metric("Kênh truyền", channel)
    m3.metric("BER tại điểm SNR gần nhất", f"{ber_list[idx]:.3e}",
              help=f"SNR trên đường cong: {snr_db_range[idx]} dB")
    m4.metric("SER tại điểm SNR gần nhất", f"{ser_list[idx]:.3e}",
              help=f"SNR trên đường cong: {snr_db_range[idx]} dB")

    tab1, tab2, tab3 = st.tabs(["📈 Hiệu năng BER / SER", "✳️ Chòm sao tín hiệu", "📋 Dữ liệu kết quả"])
    with tab1:
        fig, ax = plt.subplots(figsize=(9, 4.8), dpi=120)
        ax.semilogy(snr_db_range, np.maximum(ber_list, 1e-7), "o-",
                    linewidth=2.2, markersize=5, label="BER (mô phỏng)")
        ax.semilogy(snr_db_range, np.maximum(ser_list, 1e-7), "s--",
                    linewidth=2.0, markersize=5, label="SER (mô phỏng)")
        ax.axvline(snr_db_range[idx], color="#64748b", linestyle=":", alpha=.8)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("Tỷ lệ lỗi (thang log)")
        ax.set_title(f"Hiệu năng {mod_name} qua kênh {channel}")
        ax.grid(True, which="both", linestyle="--", alpha=.28)
        ax.legend(frameon=True)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.caption("Giá trị BER/SER bằng 0 được hiển thị tại mức sàn 10⁻⁷ để có thể quan sát trên trục log.")
    with tab2:
        fig2, ax2 = plt.subplots(figsize=(7, 6), dpi=120)
        ax2.scatter(points.real, points.imag, s=65, marker="+",
                    linewidths=2.0, label="Điểm chòm sao lý tưởng")
        ax2.scatter(sample_rx.real, sample_rx.imag, s=18, alpha=.55,
                    label="Tín hiệu thu")
        ax2.set_xlabel("Phần thực (I)")
        ax2.set_ylabel("Phần ảo (Q)")
        ax2.set_title(f"Chòm sao {mod_name} tại SNR = {snr_db_slider} dB")
        ax2.grid(True, linestyle="--", alpha=.3)
        ax2.set_aspect("equal", adjustable="datalim")
        ax2.legend()
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        st.caption(f"BER trên mẫu minh họa ({sample_count} symbol): {sample_ber:.3e}")
    with tab3:
        st.dataframe({
            "SNR (dB)": snr_db_range,
            "BER": ber_list,
            "SER": ser_list,
        }, use_container_width=True, hide_index=True)
        csv = "SNR_dB,BER,SER\n" + "\n".join(
            f"{s},{b:.10g},{e:.10g}" for s, b, e in zip(snr_db_range, ber_list, ser_list)
        )
        st.download_button("⬇️ Tải dữ liệu CSV", csv, "lte5g_qam_results.csv",
                           "text/csv", use_container_width=False)

st.markdown("---")
st.caption("LTE/5G QAM Simulation Lab • Đồ án mô phỏng kỹ thuật điều chế số")
