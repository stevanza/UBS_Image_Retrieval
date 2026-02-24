import streamlit as st
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os

# --- 1. KONFIGURASI HALAMAN & TEMA ---
st.set_page_config(
    page_title="UBS Jewelry Search System",
    page_icon="💎",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp { background-color: #fcfcfc; }
    .main-title {
        font-family: 'Playfair Display', serif;
        color: #000000 !important;
        text-align: center;
        font-size: 3.5rem;
        font-weight: bold;
        margin-top: -2rem;
        margin-bottom: 0.5rem;
    }
    p, .stMarkdown, [data-testid="stWidgetLabel"] p, .stCaption, .stAlert p, h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
    }
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        color: #000000 !important;
    }
    .stImage {
        border-radius: 15px;
        transition: transform .3s;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    .stImage:hover { transform: scale(1.03); }
    [data-testid="column"] {
        padding: 15px;
        background: white;
        border-radius: 15px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    [data-testid="stCaptionContainer"] {
        color: #333333 !important;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. KONFIGURASI PATH ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "Gianjay/clip_jewelery_finetuned" 
DATABASE_FEATS = "clip_database.pth"
DATASET_ROOT = "dataset"

HF_TOKEN = st.secrets.get("HF_TOKEN")

# --- 3. FUNGSI LOAD ASSETS ---
@st.cache_resource
def load_assets():
    with st.spinner("Menginisialisasi sistem..."):
        model = CLIPModel.from_pretrained(MODEL_PATH).to(DEVICE).eval()
        processor = CLIPProcessor.from_pretrained(MODEL_PATH)
        
        if not os.path.exists(DATABASE_FEATS):
            st.error(f"File {DATABASE_FEATS} tidak ditemukan!")
            st.stop()
            
        checkpoint = torch.load(DATABASE_FEATS, map_location=DEVICE)
        
        all_paths = checkpoint['paths']
        categories = set()
        path_category_map = [] 

        for p in all_paths:
            p = p.replace("\\", "/")
            if "dataset/" in p:
                parts = p.split("dataset/")[-1].split("/")
                cat = parts[0] if len(parts) > 0 else "Unknown"
            else:
                parts = p.split("/")
                cat = parts[-3] if len(parts) > 2 else "Unknown"
            
            categories.add(cat)
            path_category_map.append(cat)

        return model, processor, checkpoint['features'], checkpoint['paths'], sorted(list(categories)), path_category_map

model, processor, db_features, db_paths, available_categories, db_categories = load_assets()

# --- 4. DASHBOARD UTAMA ---
st.markdown("<h1 class='main-title'>UBS JEWELRY SEARCH</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #000000; font-size: 1.2rem; margin-bottom: 2rem; font-weight: 600;'>Content-Based Image Retrieval System</p>", unsafe_allow_html=True)

col_empty_left, col_input, col_empty_right = st.columns([0.5, 3, 0.5])

with col_input:
    uploaded_file = st.file_uploader("Unggah Foto Perhiasan", type=['jpg', 'jpeg', 'png'])
    
    # Grid Filter Baru
    col_f1, col_f2, col_f3 = st.columns([1.5, 1, 1.5])
    with col_f1:
        selected_category = st.selectbox(
            "Filter Kategori", 
            options=["Semua Kategori"] + available_categories,
            index=0
        )
    with col_f2:
        top_k = st.select_slider("Jumlah Hasil", options=[3, 6], value=6)
    with col_f3:
        # Slider Range Kemiripan
        sim_range = st.slider(
            "Range Kemiripan (%)",
            min_value=0,
            max_value=100,
            value=(30, 100),
            help="Hanya menampilkan hasil dengan persentase kemiripan di dalam rentang ini."
        )

# --- 5. LOGIKA PENCARIAN ---
if uploaded_file:
    st.divider()
    
    col_q, col_res = st.columns([1, 3], gap="large")
    
    with col_q:
        st.markdown("### Gambar Query")
        query_img = Image.open(uploaded_file).convert("RGB")
        st.image(query_img, width='stretch', caption=f"{uploaded_file.name}")
        
        # Status Filter
        st.info(f"Hasil Pencarian Kategori:\n**{selected_category}**")

    with col_res:
        valid_results = [] 

        with st.status("Menganalisis kemiripan visual..."):
            inputs = processor(images=query_img, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                # FIX: Ambil fitur dan pastikan tipenya Tensor
                outputs = model.get_image_features(**inputs)
                
                # Mengakses tensor dasar dari wrapper BaseModelOutputWithPooling
                raw_feat = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
                
                # Normalisasi L2
                query_feat = raw_feat / raw_feat.norm(dim=-1, keepdim=True)
            
            # Hitung Cosine Similarity
            scores = (query_feat @ db_features.to(DEVICE).T).squeeze(0)
            
            # Masking kategori jika dipilih
            if selected_category != "Semua Kategori":
                for idx, cat in enumerate(db_categories):
                    if cat != selected_category:
                        scores[idx] = -1.0 

            # Ambil kandidat lebih banyak untuk difilter berdasarkan skor
            candidates_k = min(len(db_paths), 100) 
            top_scores, top_indices = torch.topk(scores, k=candidates_k)

            for score, idx in zip(top_scores, top_indices):
                idx_val = idx.item()
                score_val = score.item()
                score_percentage = score_val * 100
                
                if score_val == -1.0: continue

                # --- FILTER RANGE PERSENTASE ---
                if not (sim_range[0] <= score_percentage <= sim_range[1]):
                    continue

                raw_path = db_paths[idx_val].replace("\\", "/")
                if "dataset/" in raw_path:
                    relative_path = raw_path.split("dataset/")[-1]
                    local_path = os.path.join(DATASET_ROOT, relative_path)
                else:
                    parts = raw_path.split("/")
                    local_path = os.path.join(DATASET_ROOT, parts[-3], parts[-2], parts[-1])

                if not os.path.exists(local_path): continue 

                product_name_folder = os.path.basename(os.path.dirname(local_path))
                category_name = db_categories[idx_val]
                
                valid_results.append({
                    "path": local_path,
                    "score": score_val,
                    "name": product_name_folder,
                    "category": category_name
                })

                if len(valid_results) >= top_k: break

        # --- HEADER HASIL ---
        h_col1, h_col2 = st.columns([2, 1])
        with h_col1:
            st.markdown(f"### 🔍 Hasil Pencarian ({len(valid_results)})")
        with h_col2:
             if len(valid_results) > 0:
                highest = valid_results[0]['score'] * 100
                lowest = valid_results[-1]['score'] * 100
                st.markdown(f"<h4 style='text-align: right; color: #4CAF50;'>Range: {lowest:.1f}% - {highest:.1f}%</h4>", unsafe_allow_html=True)

        # --- DISPLAY GRID ---
        if len(valid_results) == 0:
            st.warning(f"Tidak ada produk yang memenuhi kriteria kemiripan {sim_range[0]}% - {sim_range[1]}%.")
        else:
            for i in range(0, len(valid_results), 3):
                cols_grid = st.columns(3)
                for j in range(3):
                    if i + j < len(valid_results):
                        item = valid_results[i + j]
                        with cols_grid[j]:
                            st.image(item["path"], width='stretch', caption=item["name"])
                            st.metric(label="Kemiripan", value=f"{item['score']*100:.2f}%")
                            st.caption(f"📂 {item['category']}")

else:
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("Silakan unggah gambar untuk memulai pencarian visual.")

st.divider()
st.markdown("<p style='text-align: center; color: #000000; font-weight: bold;'>© 2026 UBS Image Retrieval System </p>", unsafe_allow_html=True)