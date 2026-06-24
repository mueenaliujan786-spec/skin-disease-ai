# ══════════════════════════════════════════════════════════════════════════════
# Intelligent Skin Disease Diagnosis System
# CS-3310 Artificial Intelligence | Assignment 3 | HAM10000 Dataset
# Model: ResNet50 Transfer Learning
# ══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt
from PIL import Image
import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm as rcm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.enums import TA_CENTER

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Skin Disease Diagnosis System",
    page_icon="🔬",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .risk-critical {
        background: linear-gradient(135deg, #ff0000, #cc0000);
        color: white; padding: 18px 20px; border-radius: 10px;
        font-size: 16px; font-weight: bold; text-align: center;
        margin: 10px 0; box-shadow: 0 4px 12px rgba(255,0,0,0.4);
        animation: pulse 1.5s infinite;
    }
    .risk-warning {
        background: linear-gradient(135deg, #ff6600, #cc4400);
        color: white; padding: 15px 20px; border-radius: 10px;
        font-size: 15px; font-weight: bold; text-align: center; margin: 10px 0;
    }
    .risk-uncertain {
        background: linear-gradient(135deg, #886600, #554400);
        color: #ffeeaa; padding: 15px 20px; border-radius: 10px;
        font-size: 15px; font-weight: bold; text-align: center; margin: 10px 0;
        border: 1px solid #bbaa00;
    }
    .risk-safe {
        background: linear-gradient(135deg, #00aa44, #007733);
        color: white; padding: 15px 20px; border-radius: 10px;
        font-size: 15px; font-weight: bold; text-align: center; margin: 10px 0;
    }
    .confidence-low  { color: #e53935; font-weight: bold; font-size: 18px; }
    .confidence-med  { color: #f9a825; font-weight: bold; font-size: 18px; }
    .confidence-high { color: #2e7d32; font-weight: bold; font-size: 18px; }
    .patient-card {
        background: #1e1e2e; border: 1px solid #444; border-radius: 10px;
        padding: 15px 20px; margin-bottom: 15px; color: #cdd6f4;
    }
    .patient-card h4 { color: #89b4fa; margin: 0 0 8px 0; }
    .patient-card p  { margin: 3px 0; font-size: 14px; }
    .abcde-box {
        background: #1e1e2e; border-left: 4px solid #89b4fa;
        border-radius: 8px; padding: 14px 18px; margin: 6px 0;
    }
    .abcde-box h5 { color: #cba6f7; margin: 0 0 4px 0; font-size: 15px; }
    .abcde-box p  { color: #bac2de; margin: 0; font-size: 13px; }
    .rec-box {
        border-radius: 10px; padding: 18px 20px; margin-top: 10px;
        font-size: 14px; line-height: 1.7;
    }
    .rec-urgent  { background: #3b0000; border: 1px solid #ff4444; color: #ffaaaa; }
    .rec-caution { background: #2b1a00; border: 1px solid #ff8800; color: #ffd080; }
    .rec-monitor { background: #002b10; border: 1px solid #00cc55; color: #80ffaa; }
    .legend-table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 8px; }
    .legend-table th { background: #313244; color: #cdd6f4; padding: 8px 10px; text-align: left; }
    .legend-table td { padding: 7px 10px; border-bottom: 1px solid #313244; color: #bac2de; }
    .legend-mal { color: #f38ba8; font-weight: bold; }
    .legend-pre { color: #fab387; font-weight: bold; }
    .legend-ben { color: #a6e3a1; font-weight: bold; }
    .section-header {
        font-size: 20px; font-weight: 700; color: #cdd6f4;
        border-bottom: 2px solid #89b4fa; padding-bottom: 6px; margin-bottom: 14px;
    }
    .risk-factor-box {
        background: #2b1500; border: 1px solid #ff8800; border-radius: 8px;
        padding: 10px 14px; margin: 6px 0; color: #ffd080; font-size: 13px;
    }
    @keyframes pulse {
        0%   { box-shadow: 0 0 0 0 rgba(255,0,0,0.5); }
        70%  { box-shadow: 0 0 0 10px rgba(255,0,0,0); }
        100% { box-shadow: 0 0 0 0 rgba(255,0,0,0); }
    }
</style>
""", unsafe_allow_html=True)

# ── Load model (cached so it only loads once) ──────────────────────────────────
@st.cache_resource
def load_resources():
    def categorical_focal_loss(alpha, gamma=2.0):
        alpha = tf.constant(alpha, dtype=tf.float32)
        def loss_fn(y_true, y_pred):
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
            cross_entropy = -y_true * tf.math.log(y_pred)
            weight = alpha * tf.math.pow(1 - y_pred, gamma)
            return tf.reduce_sum(weight * cross_entropy, axis=-1)
        return loss_fn
    dummy_alpha = tf.ones(7, dtype=tf.float32) / 7
    model = load_model(
        'resnet50_skin_model.keras',
        custom_objects={'loss_fn': categorical_focal_loss(dummy_alpha)},
        compile=False
    )
    with open('class_list.json', 'r') as f:
        class_list = json.load(f)
    return model, class_list

model, class_list = load_resources()

# ── Disease info dictionary ────────────────────────────────────────────────────
disease_info = {
    'akiec': {
        'name': 'Actinic Keratoses / Intraepithelial Carcinoma',
        'severity': '⚠️ Pre-malignant',
        'type': 'premalignant',
        'description': 'A rough, scaly patch caused by years of sun exposure. Can progress to squamous cell carcinoma.',
        'recommendation': '🔶 Schedule a dermatologist visit within 2–4 weeks. May require cryotherapy or topical treatment.',
        'urgency': 'caution'
    },
    'bcc': {
        'name': 'Basal Cell Carcinoma',
        'severity': '🔴 Malignant',
        'type': 'malignant',
        'description': 'Most common type of skin cancer. Rarely spreads but can cause local tissue damage if untreated.',
        'recommendation': '🚨 Seek medical attention within 1–2 weeks. Surgical removal or Mohs surgery is usually required.',
        'urgency': 'urgent'
    },
    'bkl': {
        'name': 'Benign Keratosis',
        'severity': '✅ Benign',
        'type': 'benign',
        'description': 'Non-cancerous skin growth including seborrheic keratoses and solar lentigines.',
        'recommendation': '✅ No urgent action needed. Monitor for changes in size, shape, or color every 6–12 months.',
        'urgency': 'monitor'
    },
    'df': {
        'name': 'Dermatofibroma',
        'severity': '✅ Benign',
        'type': 'benign',
        'description': 'Harmless fibrous nodule, usually on lower legs. Firm and may dimple when pinched.',
        'recommendation': '✅ No treatment necessary. Can be removed for cosmetic reasons if desired.',
        'urgency': 'monitor'
    },
    'mel': {
        'name': 'Melanoma',
        'severity': '🔴 MALIGNANT — Seek Immediate Medical Attention',
        'type': 'malignant',
        'description': 'Most dangerous form of skin cancer arising from melanocytes. Early detection is critical for survival.',
        'recommendation': '🚨 URGENT: See a dermatologist or oncologist IMMEDIATELY. Melanoma can be life-threatening if untreated.',
        'urgency': 'urgent'
    },
    'nv': {
        'name': 'Melanocytic Nevi (Common Mole)',
        'severity': '✅ Benign',
        'type': 'benign',
        'description': 'Ordinary moles caused by clusters of pigmented cells. Usually harmless but monitor for changes.',
        'recommendation': '✅ Generally harmless. Follow the ABCDE rule and consult a doctor if the mole changes.',
        'urgency': 'monitor'
    },
    'vasc': {
        'name': 'Vascular Lesion',
        'severity': '✅ Benign',
        'type': 'benign',
        'description': 'Includes angiomas, angiokeratomas, and pyogenic granulomas. Generally benign vascular growths.',
        'recommendation': '✅ Usually harmless. Consult a doctor if it bleeds frequently or grows rapidly.',
        'urgency': 'monitor'
    },
}

# ── Grad-CAM ───────────────────────────────────────────────────────────────────
def make_gradcam(img_array, model, layer_name='conv5_block3_out'):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_gradcam(original_img_array, heatmap, alpha=0.4):
    heatmap_uint8 = np.uint8(255 * heatmap)
    # Use plt.colormaps (new API, no deprecation warning)
    jet_colors = plt.colormaps['jet'](np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_uint8]
    jet_heatmap_img = Image.fromarray(np.uint8(jet_heatmap * 255)).resize((224, 224))
    jet_heatmap_arr = np.array(jet_heatmap_img).astype(float)
    superimposed = np.uint8(np.clip(jet_heatmap_arr * alpha + original_img_array, 0, 255))
    return superimposed

# ── Helper functions ───────────────────────────────────────────────────────────
def get_confidence_label(conf):
    """Returns label text and CSS class. Thresholds: 80=high, 65=moderate, else uncertain."""
    if conf >= 80:
        return "🟢 High Confidence", "confidence-high"
    elif conf >= 65:
        return "🟡 Moderate Confidence", "confidence-med"
    else:
        return "🔴 Low Confidence — Uncertain prediction", "confidence-low"

def get_confidence_level(conf):
    """Returns short level string for logic checks."""
    if conf >= 80:
        return "high"
    elif conf >= 65:
        return "moderate"
    else:
        return "uncertain"

def get_bar_color(cls):
    t = disease_info[cls]['type']
    if t == 'malignant':    return '#f38ba8'
    if t == 'premalignant': return '#fab387'
    return '#a6e3a1'

def get_risk_factors(age, localization, info):
    """Returns list of risk factor warning strings."""
    factors = []
    sun_exposed = ['face', 'scalp', 'ear', 'neck', 'hand', 'acral']
    if age > 60 and info['type'] in ['malignant', 'premalignant']:
        factors.append(f"🔴 Age risk: Patient is {age} years old — age over 60 increases malignancy risk significantly.")
    if localization in sun_exposed and info['type'] in ['malignant', 'premalignant']:
        factors.append(f"🔴 Location risk: '{localization.capitalize()}' is a sun-exposed area — increases risk of malignant progression.")
    return factors

# ── PDF Report Generator ───────────────────────────────────────────────────────
def generate_pdf_report(patient_data, prediction_data, img_array, overlay_array):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=2*rcm, leftMargin=2*rcm,
                            topMargin=2*rcm, bottomMargin=2*rcm)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle('title', parent=styles['Title'],
                                 fontSize=18, textColor=colors.HexColor('#1a1a2e'),
                                 spaceAfter=6, alignment=TA_CENTER)
    sub_style = ParagraphStyle('sub', parent=styles['Normal'],
                               fontSize=10, textColor=colors.grey,
                               spaceAfter=12, alignment=TA_CENTER)
    h2_style = ParagraphStyle('h2', parent=styles['Heading2'],
                              fontSize=13, textColor=colors.HexColor('#1565c0'),
                              spaceBefore=14, spaceAfter=6)
    body_style = ParagraphStyle('body', parent=styles['Normal'],
                                fontSize=10, leading=14, spaceAfter=6)

    # Title
    story.append(Paragraph("Intelligent Skin Disease Diagnosis Report", title_style))
    story.append(Paragraph("CS-3310 Artificial Intelligence | HAM10000 Dataset | ResNet50 Model", sub_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", sub_style))
    story.append(Spacer(1, 0.3*rcm))

    # Patient info
    story.append(Paragraph("Patient Information", h2_style))
    pt = patient_data
    p_data = [
        ['Field', 'Value'],
        ['Age', f"{pt['age']} years"],
        ['Sex', pt['sex'].capitalize()],
        ['Lesion Location', pt['location'].capitalize()],
        ['Analysis Date', datetime.now().strftime('%Y-%m-%d')],
    ]
    p_table = Table(p_data, colWidths=[5*rcm, 10*rcm])
    p_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1565c0')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f0f4ff'), colors.white]),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(p_table)
    story.append(Spacer(1, 0.4*rcm))

    # Diagnosis results
    story.append(Paragraph("Diagnosis Results", h2_style))
    pd_ = prediction_data
    info = disease_info[pd_['class']]
    sev_color = colors.HexColor('#c62828') if info['type'] == 'malignant' else \
                colors.HexColor('#e65100') if info['type'] == 'premalignant' else \
                colors.HexColor('#2e7d32')
    d_data = [
        ['Predicted Class', pd_['class'].upper()],
        ['Disease Name', info['name']],
        ['Severity', info['severity']],
        ['Confidence', f"{pd_['confidence']:.1f}%"],
        ['Confidence Level', pd_['conf_label']],
        ['2nd Most Likely', f"{pd_['second_class'].upper()} ({pd_['second_conf']:.1f}%)"],
        ['3rd Most Likely', f"{pd_['third_class'].upper()} ({pd_['third_conf']:.1f}%)"],
    ]
    d_table = Table(d_data, colWidths=[5*rcm, 10*rcm])
    d_table.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [colors.HexColor('#f0f4ff'), colors.white]),
        ('PADDING', (0,0), (-1,-1), 8),
        ('TEXTCOLOR', (1,2), (1,2), sev_color),
    ]))
    story.append(d_table)
    story.append(Spacer(1, 0.3*rcm))
    story.append(Paragraph(f"Description: {info['description']}", body_style))
    story.append(Paragraph(f"Recommendation: {info['recommendation']}", body_style))
    story.append(Spacer(1, 0.4*rcm))

    # Confidence per class table
    story.append(Paragraph("Confidence Per Class", h2_style))
    cls_data = [['Class', 'Disease', 'Confidence', 'Type']]
    type_labels = {'malignant': 'Malignant', 'premalignant': 'Pre-malignant', 'benign': 'Benign'}
    for i, cls in enumerate(pd_['class_list']):
        conf_val = f"{pd_['all_preds'][i] * 100:.1f}%"
        cls_data.append([cls.upper(), disease_info[cls]['name'], conf_val,
                         type_labels[disease_info[cls]['type']]])
    cls_table = Table(cls_data, colWidths=[2*rcm, 7*rcm, 3*rcm, 3*rcm])
    cls_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1565c0')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f0f4ff'), colors.white]),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(cls_table)
    story.append(Spacer(1, 0.4*rcm))

    # ABCDE rule
    story.append(Paragraph("ABCDE Rule Assessment", h2_style))
    abcde = [
        ('A — Asymmetry', 'One half of the lesion does not match the other half in shape or color.'),
        ('B — Border', 'Irregular, ragged, notched, or blurred edges are a warning sign.'),
        ('C — Color', 'Multiple colors (brown, black, red, white, blue) within one lesion.'),
        ('D — Diameter', 'Lesions larger than 6mm (size of a pencil eraser) warrant attention.'),
        ('E — Evolution', 'Any change in size, shape, color, or new symptoms like bleeding is concerning.'),
    ]
    for letter, desc in abcde:
        story.append(Paragraph(f"<b>{letter}:</b> {desc}", body_style))
    story.append(Spacer(1, 0.4*rcm))

    # Disclaimer
    story.append(Paragraph("Disclaimer", h2_style))
    story.append(Paragraph(
        "This report is generated by an AI system for educational and research purposes only. "
        "It is NOT a substitute for professional medical advice, diagnosis, or treatment. "
        "Always consult a qualified dermatologist or healthcare provider for any medical concerns.",
        ParagraphStyle('disc', parent=styles['Normal'], fontSize=9,
                       textColor=colors.HexColor('#b71c1c'), leading=14)
    ))

    doc.build(story)
    buf.seek(0)
    return buf

# ══════════════════════════════════════════════════════════════════════════════
# ── UI HEADER ──────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
st.title("🔬 Intelligent Skin Disease Diagnosis System")
st.markdown("**CS-3310 Artificial Intelligence | Assignment 3 | HAM10000 Dataset**")
st.markdown("Upload a dermoscopic skin lesion image along with patient details to get an AI-assisted diagnosis.")
st.info("ℹ️ This system uses ResNet50 transfer learning trained on HAM10000 (10,015 images, 7 disease categories).")
st.markdown("---")

# ── Input section ──────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Patient Information")
    uploaded_file = st.file_uploader("Upload skin lesion image (JPG/PNG)", type=['jpg', 'jpeg', 'png'])
    age = st.slider("Patient Age", min_value=0, max_value=85, value=45, step=1)
    sex = st.selectbox("Sex", options=['male', 'female', 'unknown'])
    localization = st.selectbox("Lesion Location", options=[
        'back', 'lower extremity', 'trunk', 'upper extremity',
        'abdomen', 'face', 'chest', 'foot', 'unknown',
        'neck', 'scalp', 'hand', 'ear', 'genital', 'acral'
    ])

with col2:
    if uploaded_file is not None:
        st.subheader("🖼️ Uploaded Image")
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, width=300)

    # Disease reference table — always visible
    st.subheader("📖 Disease Code Reference")
    st.markdown("""
    <table class="legend-table">
        <tr><th>Code</th><th>Disease Name</th><th>Type</th></tr>
        <tr><td><b>akiec</b></td><td>Actinic Keratoses / Intraepithelial Carcinoma</td><td class="legend-pre">⚠️ Pre-malignant</td></tr>
        <tr><td><b>bcc</b></td><td>Basal Cell Carcinoma</td><td class="legend-mal">🔴 Malignant</td></tr>
        <tr><td><b>bkl</b></td><td>Benign Keratosis</td><td class="legend-ben">✅ Benign</td></tr>
        <tr><td><b>df</b></td><td>Dermatofibroma</td><td class="legend-ben">✅ Benign</td></tr>
        <tr><td><b>mel</b></td><td>Melanoma</td><td class="legend-mal">🔴 Malignant</td></tr>
        <tr><td><b>nv</b></td><td>Melanocytic Nevi (Common Mole)</td><td class="legend-ben">✅ Benign</td></tr>
        <tr><td><b>vasc</b></td><td>Vascular Lesion</td><td class="legend-ben">✅ Benign</td></tr>
    </table>
    """, unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# ── ANALYSIS ───────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
if uploaded_file is not None:
    if st.button("🔍 Analyze Image", type="primary"):

        # Basic input validation
        if image.size[0] < 50 or image.size[1] < 50:
            st.error("❌ Image too small. Please upload a clear dermoscopic photo (minimum 50×50 pixels).")
            st.stop()

        with st.spinner("Analyzing image with ResNet50 + Grad-CAM... please wait"):

            # Preprocess
            img = image.resize((224, 224))
            img_array = np.array(img).astype(float)
            img_input = preprocess_input(np.expand_dims(img_array.copy(), axis=0))

            # Predict
            preds = model.predict(img_input, verbose=0)
            pred_idx        = int(np.argmax(preds[0]))
            confidence      = float(preds[0][pred_idx]) * 100
            predicted_class = class_list[pred_idx]

            # Top 3 predictions
            sorted_idx  = np.argsort(preds[0])[::-1]
            second_idx  = sorted_idx[1]
            third_idx   = sorted_idx[2]
            second_cls  = class_list[second_idx]
            second_conf = float(preds[0][second_idx]) * 100
            third_cls   = class_list[third_idx]
            third_conf  = float(preds[0][third_idx]) * 100

            # Gap between 1st and 2nd (close-call check)
            gap = confidence - second_conf
            is_close_call = gap < 20

            # Grad-CAM
            heatmap = make_gradcam(img_input, model)
            overlay = overlay_gradcam(img_array, heatmap)

            # Confidence label + level
            conf_label, conf_css = get_confidence_label(confidence)
            conf_level           = get_confidence_level(confidence)

            # Disease info for predicted class
            info = disease_info[predicted_class]

            # Melanoma secondary alert check
            mel_conf = float(preds[0][class_list.index('mel')]) * 100 if 'mel' in class_list else 0

            # Risk factors
            risk_factors = get_risk_factors(age, localization, info)

        st.markdown("---")

        # ── PATIENT SUMMARY CARD ────────────────────────────────────────────────
        st.markdown('<div class="section-header">📋 Patient Summary</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="patient-card">
            <h4>👤 Patient Profile</h4>
            <p>🎂 <b>Age:</b> {age} years &nbsp;|&nbsp;
               ⚧ <b>Sex:</b> {sex.capitalize()} &nbsp;|&nbsp;
               📍 <b>Lesion Location:</b> {localization.capitalize()}</p>
            <p>📅 <b>Analysis Date:</b> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
            <p>🤖 <b>Model Used:</b> ResNet50 (Transfer Learning) | HAM10000 Dataset</p>
        </div>
        """, unsafe_allow_html=True)

        # ── RISK ALERT BANNER ───────────────────────────────────────────────────
        if conf_level == "uncertain":
            # Low confidence — don't show a firm disease diagnosis in the banner
            st.markdown(f"""
            <div class="risk-uncertain">
                ⚠️ UNCERTAIN PREDICTION — Confidence: {confidence:.1f}%<br>
                <span style="font-size:13px; font-weight:normal;">
                The model is not confident. Top guess: <b>{predicted_class.upper()}</b>,
                2nd guess: <b>{second_cls.upper()}</b> ({second_conf:.1f}%).
                Do NOT rely on this result. Please consult a dermatologist.
                </span>
            </div>""", unsafe_allow_html=True)
        elif predicted_class in ['mel', 'bcc']:
            st.markdown(f"""
            <div class="risk-critical">
                🚨 CRITICAL RISK ALERT — {info['name'].upper()} DETECTED<br>
                <span style="font-size:13px; font-weight:normal;">
                Seek immediate medical attention. Do NOT delay consultation with a dermatologist.
                </span>
            </div>""", unsafe_allow_html=True)
        elif predicted_class == 'akiec':
            st.markdown(f"""
            <div class="risk-warning">
                ⚠️ PRE-MALIGNANT LESION DETECTED — {info['name']}<br>
                <span style="font-size:13px; font-weight:normal;">
                Schedule a dermatologist appointment within 2–4 weeks.
                </span>
            </div>""", unsafe_allow_html=True)
        elif mel_conf >= 35 and predicted_class != 'mel':
            st.markdown(f"""
            <div class="risk-warning">
                ⚠️ SECONDARY MELANOMA SUSPICION — Melanoma confidence: {mel_conf:.1f}%<br>
                <span style="font-size:13px; font-weight:normal;">
                Primary prediction is {predicted_class.upper()}, but elevated melanoma probability
                warrants a dermatologist review.
                </span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-safe">
                ✅ BENIGN LESION DETECTED — {info['name']}<br>
                <span style="font-size:13px; font-weight:normal;">
                No immediate risk detected. Continue routine monitoring.
                </span>
            </div>""", unsafe_allow_html=True)

        # ── CLOSE-CALL WARNING ──────────────────────────────────────────────────
        if is_close_call:
            st.warning(
                f"⚠️ **Close Call:** Model is torn between **{predicted_class.upper()}** ({confidence:.1f}%) "
                f"and **{second_cls.upper()}** ({second_conf:.1f}%). Gap is only {gap:.1f}%. "
                f"Please seek clinical verification."
            )

        # ── RISK FACTOR WARNINGS ────────────────────────────────────────────────
        if risk_factors:
            for rf in risk_factors:
                st.markdown(f'<div class="risk-factor-box">{rf}</div>', unsafe_allow_html=True)

        st.markdown("---")

        # ── RESULTS: 3 COLUMNS ──────────────────────────────────────────────────
        st.markdown('<div class="section-header">📊 Diagnosis Results</div>', unsafe_allow_html=True)
        res_col1, res_col2, res_col3 = st.columns([1.1, 1.1, 1.3])

        # ── COL 1: Prediction details ───────────────────────────────────────────
        with res_col1:
            st.markdown("##### 🎯 Primary Prediction")
            st.metric("Predicted Class", predicted_class.upper())
            st.markdown(f"**{info['name']}**")

            # Show severity differently based on confidence
            if conf_level == "uncertain":
                st.warning("⚠️ Confidence too low — treat this prediction with caution")
            elif info['type'] == 'malignant':
                st.error(f"{info['severity']}")
            elif info['type'] == 'premalignant':
                st.warning(f"{info['severity']}")
            else:
                st.success(f"{info['severity']}")

            st.markdown(f"*{info['description']}*")

            st.markdown("---")
            st.markdown("##### 🥈 2nd Most Likely")
            info2 = disease_info[second_cls]
            st.markdown(f"**{second_cls.upper()}** — {second_conf:.1f}%")
            st.markdown(f"{info2['name']}")
            st.markdown(f"{info2['severity']}")

            st.markdown("---")
            st.markdown("##### 🥉 3rd Most Likely")
            info3 = disease_info[third_cls]
            st.markdown(f"**{third_cls.upper()}** — {third_conf:.1f}%")
            st.markdown(f"{info3['name']}")
            st.markdown(f"{info3['severity']}")

            st.markdown("---")
            st.markdown("##### 📊 Confidence Level")
            st.markdown(f'<span class="{conf_css}">{conf_label}</span>', unsafe_allow_html=True)
            st.markdown(f"**Score: {confidence:.1f}%**")

        # ── COL 2: Confidence bar chart ─────────────────────────────────────────
        with res_col2:
            st.markdown("##### 📈 Confidence Per Class")

            cls_names  = [c.upper() for c in class_list]
            cls_values = [float(preds[0][i]) * 100 for i in range(len(class_list))]
            bar_colors = [get_bar_color(c) for c in class_list]

            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor('#1e1e2e')
            ax.set_facecolor('#1e1e2e')

            bars = ax.barh(cls_names, cls_values, color=bar_colors, edgecolor='none', height=0.6)
            for bar, val in zip(bars, cls_values):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                        f'{val:.1f}%', va='center', ha='left',
                        color='white', fontsize=9, fontweight='bold')

            ax.set_xlim(0, max(cls_values) * 1.25 if max(cls_values) > 0 else 100)
            ax.set_xlabel('Confidence (%)', color='#cdd6f4', fontsize=9)
            ax.tick_params(colors='#cdd6f4', labelsize=9)
            for spine in ax.spines.values():
                spine.set_visible(False)

            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#f38ba8', label='Malignant'),
                Patch(facecolor='#fab387', label='Pre-malignant'),
                Patch(facecolor='#a6e3a1', label='Benign'),
            ]
            ax.legend(handles=legend_elements, loc='lower right',
                      facecolor='#313244', labelcolor='#cdd6f4', fontsize=8,
                      framealpha=0.8, edgecolor='#444')

            plt.tight_layout()
            buf_chart = io.BytesIO()
            plt.savefig(buf_chart, format='png', dpi=100, bbox_inches='tight', facecolor='#1e1e2e')
            buf_chart.seek(0)
            st.image(buf_chart, use_container_width=True)
            plt.close()

        # ── COL 3: Grad-CAM ─────────────────────────────────────────────────────
        with res_col3:
            st.markdown("##### 🔬 Grad-CAM Explainability")
            st.markdown("*Red/yellow = high model attention | Blue = low attention*")

            fig2, axes = plt.subplots(1, 2, figsize=(7, 3.5))
            fig2.patch.set_facecolor('#1e1e2e')
            for ax in axes:
                ax.set_facecolor('#1e1e2e')

            axes[0].imshow(img_array.astype('uint8'))
            axes[0].set_title("Original Image", color='#cdd6f4', fontsize=10, pad=8)
            axes[0].axis('off')

            axes[1].imshow(overlay)
            axes[1].set_title("Grad-CAM Heatmap", color='#cdd6f4', fontsize=10, pad=8)
            axes[1].axis('off')

            sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            cbar = fig2.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)
            cbar.set_label('Attention Intensity', color='#cdd6f4', fontsize=8)
            cbar.ax.yaxis.set_tick_params(color='#cdd6f4', labelcolor='#cdd6f4', labelsize=7)

            plt.tight_layout()
            buf2 = io.BytesIO()
            plt.savefig(buf2, format='png', dpi=100, bbox_inches='tight', facecolor='#1e1e2e')
            buf2.seek(0)
            st.image(buf2, use_container_width=True)
            plt.close()

        st.markdown("---")

        # ── ABCDE RULE ──────────────────────────────────────────────────────────
        st.markdown('<div class="section-header">🔤 ABCDE Rule — Dermoscopy Checklist</div>',
                    unsafe_allow_html=True)
        st.markdown("*Standard dermatology rule used to evaluate suspicious skin lesions:*")

        abcde_cols = st.columns(5)
        abcde_data = [
            ("A", "Asymmetry", "One half doesn't match the other in shape or color."),
            ("B", "Border",    "Irregular, ragged, notched, or blurred edges."),
            ("C", "Color",     "Multiple colors (brown, black, red, white, blue) in one lesion."),
            ("D", "Diameter",  "Larger than 6mm (pencil eraser size) is concerning."),
            ("E", "Evolution", "Any recent change in size, shape, color, or new symptoms."),
        ]
        for col, (letter, title, desc) in zip(abcde_cols, abcde_data):
            with col:
                st.markdown(f"""
                <div class="abcde-box">
                    <h5>{letter} — {title}</h5>
                    <p>{desc}</p>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── CLINICAL RECOMMENDATION ─────────────────────────────────────────────
        st.markdown('<div class="section-header">📋 Clinical Recommendation</div>',
                    unsafe_allow_html=True)

        rec_class = 'rec-urgent'  if info['urgency'] == 'urgent'  else \
                    'rec-caution' if info['urgency'] == 'caution' else 'rec-monitor'

        timeline_map = {
            'urgent':  '🕐 Timeframe: See a doctor IMMEDIATELY (within 24–72 hours)',
            'caution': '🕐 Timeframe: Schedule an appointment within 2–4 weeks',
            'monitor': '🕐 Timeframe: Routine check-up every 6–12 months',
        }
        next_steps = {
            'urgent':  "• Contact a dermatologist or oncologist immediately<br>• Do not use home remedies or delay<br>• Bring this report to your appointment",
            'caution': "• Book a dermatology appointment<br>• Avoid sun exposure on the lesion<br>• Photograph the lesion weekly to track changes",
            'monitor': "• Monitor using the ABCDE rule monthly<br>• Photograph the lesion every 3 months<br>• See a doctor if any change is noticed",
        }
        st.markdown(f"""
        <div class="rec-box {rec_class}">
            <b>{info['recommendation']}</b><br><br>
            {timeline_map[info['urgency']]}<br><br>
            📌 <b>Next Steps:</b><br>
            {next_steps[info['urgency']]}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # ── PDF DOWNLOAD ────────────────────────────────────────────────────────
        st.markdown('<div class="section-header">📄 Download Report</div>', unsafe_allow_html=True)

        patient_data = {'age': age, 'sex': sex, 'location': localization}
        prediction_data = {
            'class':        predicted_class,
            'confidence':   confidence,
            'conf_label':   conf_label,
            'second_class': second_cls,
            'second_conf':  second_conf,
            'third_class':  third_cls,
            'third_conf':   third_conf,
            'all_preds':    preds[0],
            'class_list':   class_list,
        }

        try:
            pdf_buf = generate_pdf_report(patient_data, prediction_data, img_array, overlay)
            st.download_button(
                label="📥 Download Full PDF Report",
                data=pdf_buf,
                file_name=f"skin_diagnosis_{predicted_class}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                type="primary"
            )
            st.caption("PDF includes patient info, diagnosis, top-3 predictions, all confidence scores, ABCDE rule, and recommendations.")
        except Exception as e:
            st.info(f"PDF generation requires `reportlab`. Install with: `pip install reportlab`\n\nError: {e}")

        # ── FINAL DISCLAIMER ────────────────────────────────────────────────────
        st.markdown("---")
        if conf_level == "uncertain":
            st.error("🚨 This prediction has LOW CONFIDENCE. Do NOT use this result for any medical decision. Consult a qualified dermatologist.")
        else:
            st.warning("⚠️ **Disclaimer:** This system is for **educational and research purposes only**. "
                       "It is NOT a substitute for professional medical advice. "
                       "Always consult a qualified dermatologist for any skin concerns.")

else:
    st.info("👆 Please upload a dermoscopic skin lesion image to begin analysis.")
    st.markdown("""
    **How to use this system:**
    1. Upload a dermoscopic JPG/PNG image of the skin lesion
    2. Enter patient age, sex, and lesion location
    3. Click **Analyze Image**
    4. Review the AI diagnosis, confidence scores, Grad-CAM heatmap, and recommendations
    5. Download the full PDF report
    """)