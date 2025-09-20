import streamlit as st
import tempfile
import sqlite3
import hashlib
from datetime import datetime
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import json
import difflib
from deep_translator import GoogleTranslator
import base64
import os


# CONFIGURATION

MODEL_PATH = r"C:\Users\thenn\OneDrive\thenn\OneDrive\Desktop\my_plant_disease_predictionapplication\plant_disease_model_transfer.pth"
JSON_PATH = r"C:\Users\thenn\OneDrive\thenn\OneDrive\Desktop\my_plant_disease_predictionapplication\recommondation.json"
DB_PATH = "farmers.db"
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COMMON_BG = r"C:\Users\thenn\OneDrive\thenn\OneDrive\Desktop\my_plant_disease_predictionapplication\detection_img.jpg"


# DATABASE FUNCTIONS

def get_db_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS farmers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password_hash TEXT,
            name TEXT,
            location TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            farmer_id INTEGER,
            disease TEXT,
            confidence REAL,
            timestamp TEXT,
            FOREIGN KEY (farmer_id) REFERENCES farmers(id)
        )
    """)
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_farmer(username, password, name, location):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO farmers (username, password_hash, name, location) VALUES (?, ?, ?, ?)",
            (username, hash_password(password), name.strip(), location.strip())
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def login_farmer(username, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, username, name, location FROM farmers WHERE username=? AND password_hash=?",
        (username, hash_password(password))
    )
    user = cursor.fetchone()
    conn.close()
    return user

def save_prediction(farmer_id, disease, confidence):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO predictions (farmer_id, disease, confidence, timestamp) VALUES (?, ?, ?, ?)",
        (farmer_id, disease, confidence, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()

def get_farmer_predictions(farmer_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT disease, confidence, timestamp FROM predictions WHERE farmer_id=? ORDER BY timestamp DESC",
        (farmer_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    return rows

init_db()


# LOAD MODEL & DATA

with open(JSON_PATH, "r") as f:
    disease_info = json.load(f)

class_names = [
    "Apple_Apple_scab","Apple_Black_rot","Apple_Cedar_apple_rust","Apple_healthy",
    "Blueberry__healthy","Cherry(including_sour)Powdery_mildew","Cherry(including_sour)_healthy",
    "Corn_(maize)Cercospora_leaf_spot Gray_leaf_spot","Corn(maize)Common_rust",
    "Corn_(maize)Northern_Leaf_Blight","Corn(maize)healthy","Grape__Black_rot",
    "Grape_Esca(Black_Measles)","GrapeLeaf_blight(Isariopsis_Leaf_Spot)","Grape__healthy",
    "Orange_Haunglongbing(Citrus_greening)","PeachBacterial_spot","Peach_healthy",
    "Pepper,bell_Bacterial_spot","Pepper,bell_healthy","Potato_Early_blight","Potato_Late_blight",
    "Potato_healthy","Raspberry_healthy","Soybean_healthy","Squash_Powdery_mildew",
    "Strawberry_Leaf_scorch","Strawberry_healthy","Tomato_Bacterial_spot","Tomato_Early_blight",
    "Tomato_Late_blight","Tomato_Leaf_Mold","Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites Two-spotted_spider_mite","Tomato_Target_Spot",
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus","Tomato_Tomato_mosaic_virus","Tomato_healthy"
]

@st.cache_resource(show_spinner=False)
def load_model():
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.classifier[1].in_features, len(class_names))
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

def predict_image(image_path):
    img = Image.open(image_path).convert("L")
    img = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        pred_class_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class_idx].item()
        return class_names[pred_class_idx], confidence

def translate_text(text, target_lang):
    if target_lang == "en":
        return text
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except:
        return text


# STREAMLIT CONFIG

st.set_page_config(page_title="Farmer Plant Disease Detection", layout="wide")


# LIGHT BACKGROUND + SHADOW CARD FILE UPLOADER

if os.path.exists(COMMON_BG):
    with open(COMMON_BG, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(255,255,255,0.35), rgba(255,255,255,0.35)),
                        url("data:image/png;base64,{b64}");
            background-size: cover;
            background-position: center;
        }}
        input, textarea {{
            background-color: rgba(255,255,255,0.85) !important;
            color: black !important;
        }}
        div.stButton > button {{
            background-color: white !important;
            color: black !important;
            border-radius: 10px;
            border: 1px solid #ccc;
            font-weight: 600;
            padding: 0.5em 1em;
            box-shadow: 1px 1px 3px rgba(0,0,0,0.2);
            transition: all 0.2s ease-in-out;
        }}
        div.stButton > button:hover {{
            background-color: #f2f2f2 !important;
            transform: scale(1.03);
            box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
        }}
        .stFileUploader {{
            border: 3px solid black !important;
            border-radius: 15px !important;
            box-shadow: 5px 5px 15px rgba(0,0,0,0.4) !important;
            padding: 10px !important;
            background-color: rgba(255,255,255,0.95) !important;
        }}
        .stImage img {{
            border-radius: 10px;
            border: 2px solid black;
        }}
        </style>
        """, unsafe_allow_html=True
    )


# SESSION STATE

if "farmer" not in st.session_state: st.session_state.farmer = None
if "page" not in st.session_state: st.session_state.page = "Login"
if "selected_lang_name" not in st.session_state: st.session_state.selected_lang_name = "English"


# TOP NAVIGATION

cols = st.columns([1,1,1,1,1])
nav_labels = ["Login","Register","Detection","History","Logout"]
for i, col in enumerate(cols):
    if col.button(nav_labels[i]):
        if nav_labels[i]=="Logout":
            if st.session_state.farmer:
                st.session_state.farmer = None
                st.session_state.page = "Login"
                st.success("✅ You have been logged out!")
            else:
                st.warning("⚠ You are not logged in.")
        else:
            st.session_state.page = nav_labels[i]

st.markdown("---")


# PAGE FUNCTIONS

def page_register():
    st.subheader("🆕 Farmer Registration")
    username = st.text_input("Username", key="reg_username")
    password = st.text_input("Password", type="password", key="reg_password")
    name = st.text_input("Farmer Name", key="reg_name")
    location = st.text_input("Farm Location", key="reg_location")
    if st.button("Register Farmer"):
        if register_farmer(username,password,name,location):
            st.success("✅ Registration successful! Please login.")
        else:
            st.error("❌ Username already exists.")

def page_login():
    st.subheader("🔑 Farmer Login")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    if st.button("Login Farmer"):
        user = login_farmer(username,password)
        if user:
            st.session_state.farmer = user
            st.success(f"👋 Welcome, {user[2]}! You are now logged in.")
            st.session_state.page = "Detection"
        else:
            st.error("❌ Invalid username or password.")

def page_detection():
    if not st.session_state.farmer:
        st.warning("⚠ Please login first to access detection.")
        return

    # Language selection
    lang_options = GoogleTranslator().get_supported_languages(as_dict=True)
    lang_keys = [k.title() for k in lang_options.keys()]
    default_lang = st.session_state.selected_lang_name
    if default_lang not in lang_keys: default_lang="English"
    selected_lang_name = st.selectbox("🌍 Choose Language", lang_keys, index=lang_keys.index(default_lang))
    st.session_state.selected_lang_name = selected_lang_name
    selected_lang_code = lang_options[selected_lang_name.lower()]

    uploaded_file = st.file_uploader("Upload Plant Leaf Image", type=["png","jpg","jpeg"])
    if uploaded_file:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        temp_file.write(uploaded_file.getbuffer())
        st.subheader("🖼 Selected Image")
        st.image(temp_file.name, caption="Uploaded Leaf", width=400, output_format="JPEG")

        predicted_class, confidence = predict_image(temp_file.name)
        save_prediction(st.session_state.farmer[0], predicted_class, confidence)

        st.subheader("Prediction Result")
        st.write(f"Detected Disease: {translate_text(predicted_class, selected_lang_code)}")
        st.write(f"Confidence: {confidence*100:.2f}%")
        if "healthy" in predicted_class.lower():
            st.success(translate_text("This plant is healthy! ✅", selected_lang_code))
        else:
            st.warning(translate_text(f"This plant may have {predicted_class} ⚠", selected_lang_code))

        # Recommendations
        info = disease_info.get(predicted_class)
        if not info:
            closest = difflib.get_close_matches(predicted_class, disease_info.keys(), n=1, cutoff=0.6)
            if closest: info = disease_info[closest[0]]
        if info:
            st.subheader(f"🌿 Disease Management Tips ({selected_lang_name})")
            for key, title in [("preventive","Preventive Measures"),("organic","Organic Treatments"),
                               ("chemical","Chemical Treatments"),("notes","Notes")]:
                if key in info and info[key]:
                    st.markdown(f"{translate_text(title, selected_lang_code)}:")
                    if isinstance(info[key], list):
                        for item in info[key]:
                            st.write(f"- {translate_text(item, selected_lang_code)}")
                    else:
                        st.write(translate_text(info[key], selected_lang_code))
        else:
            st.error(translate_text("No recommendations found.", selected_lang_code))

def page_history():
    if not st.session_state.farmer:
        st.warning("⚠ Please login to view history.")
        return
    st.subheader("📜 Your Prediction History")
    rows = get_farmer_predictions(st.session_state.farmer[0])
    if rows:
        for disease, confidence, timestamp in rows:
            st.write(f"- {timestamp} → {disease} ({confidence*100:.2f}%)")
    else:
        st.info("No predictions found yet.")


# PAGE ROUTER

pages = {
    "Register": page_register,
    "Login": page_login,
    "Detection": page_detection,
    "History": page_history
}

pages[st.session_state.page]()