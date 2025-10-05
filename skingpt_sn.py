import streamlit as st
from PIL import Image, ImageEnhance
import pandas as pd
import datetime
import os
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# --- Configuration de la page ---
st.set_page_config(
    page_title="SkinAI Sénégal - Diagnostic Réel",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS personnalisé ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .diagnostic-box {
        background-color: #f0f8f0;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        margin: 1rem 0;
    }
    .product-box {
        background-color: #fffaf0;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #FFA500;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    .analysis-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2196f3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Fonctions de détection RÉELLE ---
def validate_image_quality(image):
    """Valide la qualité de l'image pour l'analyse"""
    issues = []
    img_array = np.array(image)
    
    # Vérification de la netteté
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    if sharpness < 100:
        issues.append("floue")
    
    # Vérification de la luminosité
    brightness = np.mean(gray)
    if brightness < 50:
        issues.append("trop sombre")
    elif brightness > 200:
        issues.append("trop lumineuse")
    
    return issues, sharpness, brightness

def detect_hyperpigmentation(image):
    """
    Détection RÉELLE de l'hyperpigmentation par analyse de couleur
    """
    # Conversion en array numpy
    img_array = np.array(image)
    original = img_array.copy()
    
    # Conversion en différents espaces de couleur
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    
    # --- Détection des taches brunes (hyperpigmentation) ---
    
    # Méthode 1: Segmentation par couleur dans l'espace HSV
    # Plage pour les couleurs brunes (hyperpigmentation)
    lower_brown1 = np.array([0, 30, 20])
    upper_brown1 = np.array([30, 150, 180])
    
    lower_brown2 = np.array([0, 0, 0])
    upper_brown2 = np.array([180, 255, 100])
    
    mask1 = cv2.inRange(hsv, lower_brown1, upper_brown1)
    mask2 = cv2.inRange(hsv, lower_brown2, upper_brown2)
    brown_mask = cv2.bitwise_or(mask1, mask2)
    
    # Méthode 2: Analyse de la luminance dans l'espace LAB
    L_channel = lab[:,:,0]
    # Les zones plus sombres (faible luminance) peuvent indiquer une hyperpigmentation
    _, dark_mask = cv2.threshold(L_channel, 60, 255, cv2.THRESH_BINARY_INV)
    
    # Combinaison des masques
    combined_mask = cv2.bitwise_or(brown_mask, dark_mask)
    
    # Nettoyage du masque
    kernel = np.ones((3,3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    
    # Détection des contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrage des petits contours (bruit)
    min_contour_area = 50  # pixels
    significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    
    # Calcul des métriques
    total_pixels = img_array.shape[0] * img_array.shape[1]
    hyperpigmentation_pixels = np.count_nonzero(combined_mask)
    hyperpigmentation_ratio = hyperpigmentation_pixels / total_pixels
    
    # Nombre de taches détectées
    spot_count = len(significant_contours)
    
    # Création de l'image de visualisation
    visualization = original.copy()
    cv2.drawContours(visualization, significant_contours, -1, (255, 0, 0), 2)
    
    # Calcul de la sévérité moyenne
    if spot_count > 0:
        avg_spot_size = sum(cv2.contourArea(cnt) for cnt in significant_contours) / spot_count
    else:
        avg_spot_size = 0
    
    return {
        'ratio': hyperpigmentation_ratio,
        'spot_count': spot_count,
        'avg_spot_size': avg_spot_size,
        'mask': combined_mask,
        'visualization': visualization,
        'contours': significant_contours
    }

def detect_acne(image):
    """
    Détection RÉELLE de l'acné par analyse de texture et couleur
    """
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Détection des imperfections rouges/inflammées
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    # Plage pour les rougeurs (acné inflammée)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    redness_mask = cv2.bitwise_or(mask1, mask2)
    
    # Détection des contours circulaires (boutons)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20,
                              param1=50, param2=30, minRadius=5, maxRadius=30)
    
    acne_count = 0
    if circles is not None:
        acne_count = len(circles[0])
    
    return {
        'acne_count': acne_count,
        'redness_ratio': np.count_nonzero(redness_mask) / (img_array.shape[0] * img_array.shape[1]),
        'redness_mask': redness_mask
    }

def analyze_skin_conditions(image):
    """
    Analyse COMPLÈTE de toutes les conditions cutanées
    """
    hyperpigmentation_data = detect_hyperpigmentation(image)
    acne_data = detect_acne(image)
    
    return {
        'hyperpigmentation': hyperpigmentation_data,
        'acne': acne_data
    }

def get_diagnosis_and_recommendation(analysis_results, age, skin_type):
    """
    Génère le diagnostic et les recommandations basés sur l'analyse RÉELLE
    """
    hyper = analysis_results['hyperpigmentation']
    acne = analysis_results['acne']
    
    # --- Diagnostic Principal ---
    conditions = []
    
    # Diagnostic hyperpigmentation
    if hyper['ratio'] > 0.15:
        conditions.append("Hyperpigmentation sévère")
        hyper_severity = "sévère"
    elif hyper['ratio'] > 0.08:
        conditions.append("Hyperpigmentation modérée")
        hyper_severity = "modérée"
    elif hyper['ratio'] > 0.03:
        conditions.append("Hyperpigmentation légère")
        hyper_severity = "légère"
    
    # Diagnostic acné
    if acne['acne_count'] > 10:
        conditions.append("Acné sévère")
        acne_severity = "sévère"
    elif acne['acne_count'] > 5:
        conditions.append("Acné modérée")
        acne_severity = "modérée"
    elif acne['acne_count'] > 2:
        conditions.append("Acné légère")
        acne_severity = "légère"
    
    if not conditions:
        conditions.append("Peau saine avec imperfections mineures")
    
    diagnosis = " + ".join(conditions)
    
    # --- Produits Recommandés ---
    products = []
    
    # Produits pour hyperpigmentation
    if hyper['ratio'] > 0.03:
        if hyper_severity == "sévère":
            products.append("SÉRUM INTENSIF ANTI-TACHES + CRÈME ÉCLAT NUIT")
        elif hyper_severity == "modérée":
            products.append("SÉRUM ÉCLAT ANTI-TACHES")
        else:
            products.append("SOIN ÉQUILIBRANT PEAUX SENSIBLES")
    
    # Produits pour acné
    if acne['acne_count'] > 2:
        if acne_severity == "sévère":
            products.append("GEL PURIFIANT INTENSIF + MASQUE DÉTOX")
        elif acne_severity == "modérée":
            products.append("GEL PURIFIANT QUOTIDIEN")
        else:
            products.append("NETTOYANT DOUX ACNE-STOP")
    
    if not products:
        products.append("CRÈME HYDRATANTE QUOTIDIENNE")
    
    recommended_product = " + ".join(products)
    
    # --- Conseils Personnalisés ---
    advice = []
    
    if hyper['ratio'] > 0.03:
        advice.append(f"• **Hyperpigmentation détectée :** {hyper['ratio']:.1%} de la peau affectée ({hyper['spot_count']} taches)")
        advice.append("• Appliquer les soins éclaircissants matin et soir")
        advice.append("• PROTECTION SOLAIRE SPF 50+ obligatoire")
        advice.append("• Éviter l'exposition solaire entre 12h-16h")
    
    if acne['acne_count'] > 2:
        advice.append(f"• **Acné détectée :** {acne['acne_count']} boutons identifiés")
        advice.append("• Nettoyer la peau matin et soir")
        advice.append("• Ne pas percer les boutons")
        advice.append("• Changer les taies d'oreiller régulièrement")
    
    if not (hyper['ratio'] > 0.03 or acne['acne_count'] > 2):
        advice.append("• Maintenir une routine de soin équilibrée")
        advice.append("• Nettoyer quotidiennement")
        advice.append("• Hydrater matin et soir")
        advice.append("• Protection solaire préventive")
    
    advice_text = "\n".join(advice)
    
    # --- Recommandation Médicale ---
    needs_doctor = (hyper['ratio'] > 0.15 or acne['acne_count'] > 10 or age > 50)
    medical_advice = "🔔 **Consultation dermatologique recommandée**" if needs_doctor else ""
    
    return diagnosis, recommended_product, advice_text, medical_advice

# --- Interface principale ---
st.markdown('<h1 class="main-header">🌿 SkinAI Sénégal - Diagnostic Réel par IA</h1>', unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("https://via.placeholder.com/200x200/2E8B57/FFFFFF?text=SKINAI", width=150)
    st.title("📋 Votre Profil")
    
    age = st.slider("Âge", 15, 70, 25)
    skin_type = st.selectbox("Type de peau", 
                           ["Peau grasse", "Peau sèche", "Peau mixte", "Peau normale", "Je ne sais pas"])
    
    st.markdown("---")
    st.info("""
    **🎯 Notre technologie :**
    • Détection RÉELLE de l'hyperpigmentation
    • Analyse des taches brunes
    • Comptage des imperfections
    • Recommandations personnalisées
    """)

# --- Section upload et analyse ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📸 Analyse de Votre Peau")
    
    uploaded_file = st.file_uploader(
        "Téléchargez une photo claire de votre visage", 
        type=["jpg", "jpeg", "png"],
        help="Photo nette, bon éclairage, visage bien visible"
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Photo téléchargée", use_column_width=True)
        
        # Validation de la qualité
        with st.spinner("🔍 Analyse de la qualité de l'image..."):
            issues, sharpness, brightness = validate_image_quality(image)
            
            if issues:
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                st.warning(f"**Qualité d'image à améliorer :** Votre photo est {', '.join(issues)}.")
                st.write("""
                **Conseils pour une meilleure photo :**
                • Prenez la photo à la lumière du jour
                • Tenez le téléphone stable
                • Rapprochez-vous légèrement
                • Évitez les ombres sur le visage
                """)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.success("✅ Qualité d'image excellente pour l'analyse !")

with col2:
    if uploaded_file:
        issues, _, _ = validate_image_quality(image)
        
        if not issues:
            st.subheader("🎯 Diagnostic en Temps Réel")
            
            # Analyse RÉELLE avec indicateur de progression
            with st.spinner("🧠 SkinAI analyse votre peau en détail..."):
                # Analyse complète
                analysis_results = analyze_skin_conditions(image)
                
                # Diagnostic et recommandations
                diagnosis, product, advice, medical_advice = get_diagnosis_and_recommendation(
                    analysis_results, age, skin_type
                )
            
            # Affichage des résultats
            st.markdown('<div class="diagnostic-box">', unsafe_allow_html=True)
            st.write("**📋 DIAGNOSTIC IA :**")
            st.success(f"**{diagnosis}**")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Produit gratuit recommandé
            st.markdown('<div class="product-box">', unsafe_allow_html=True)
            st.write("**🎁 PRODUIT GRATUIT RECOMMANDÉ :**")
            st.info(f"**{product}**")
            st.write("""
            **Offert par SkinAI Sénégal :**
            • Formule adaptée à VOTRE problème détecté
            • Ingrédients naturels et locaux
            • Livraison gratuite au Sénégal
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Affichage de l'analyse détaillée
            with st.expander("🔍 **Détails de l'analyse technique**", expanded=True):
                col_anal1, col_anal2 = st.columns(2)
                
                with col_anal1:
                    st.write("**📊 Hyperpigmentation :**")
                    hyper = analysis_results['hyperpigmentation']
                    st.metric("Surface affectée", f"{hyper['ratio']:.1%}")
                    st.metric("Nombre de taches", hyper['spot_count'])
                    st.metric("Taille moyenne", f"{hyper['avg_spot_size']:.0f} px")
                
                with col_anal2:
                    st.write("**📊 Acné & Imperfections :**")
                    acne = analysis_results['acne']
                    st.metric("Boutons détectés", acne['acne_count'])
                    st.metric("Rougeurs", f"{acne['redness_ratio']:.1%}")
            
            # Conseils personnalisés
            st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
            st.write("**💡 CONSEILS PERSONNALISÉS :**")
            st.write(advice)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Recommandation médicale
            if medical_advice:
                st.warning(medical_advice)

# --- Visualisation de la détection ---
if uploaded_file and not issues:
    st.markdown("---")
    st.subheader("👁️ Visualisation de la Détection")
    
    col_vis1, col_vis2 = st.columns(2)
    
    with col_vis1:
        st.image(image, caption="📷 Photo originale", use_column_width=True)
    
    with col_vis2:
        hyper = analysis_results['hyperpigmentation']
        st.image(hyper['visualization'], caption="🔴 Zones d'hyperpigmentation détectées", use_column_width=True)

# --- Section enregistrement des données ---
if uploaded_file and not issues:
    st.markdown("---")
    st.subheader("💾 Enregistrement du Diagnostic")
    
    if st.button("💾 Enregistrer Mon Diagnostic & Recevoir Mon Produit Gratuit", type="primary", use_container_width=True):
        # Création des dossiers
        os.makedirs("skinai_data/images", exist_ok=True)
        os.makedirs("skinai_data/diagnostics", exist_ok=True)
        
        # Sauvegarde de l'image
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        image_filename = f"skinai_data/images/{timestamp}.jpg"
        image.save(image_filename)
        
        # Préparation des données pour l'entraînement
        hyper = analysis_results['hyperpigmentation']
        acne = analysis_results['acne']
        
        patient_data = {
            "timestamp": [timestamp],
            "age": [age],
            "skin_type": [skin_type],
            "hyperpigmentation_ratio": [hyper['ratio']],
            "hyperpigmentation_spots": [hyper['spot_count']],
            "acne_count": [acne['acne_count']],
            "redness_ratio": [acne['redness_ratio']],
            "diagnosis": [diagnosis],
            "recommended_product": [product],
            "image_path": [image_filename],
            "needs_medical_followup": [bool(medical_advice)]
        }
        
        df = pd.DataFrame(patient_data)
        
        # Sauvegarde dans CSV
        csv_path = "skinai_data/diagnostics/skinai_training_data.csv"
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode="a", header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)
        
        # Message de succès avec code produit
        coupon_code = f"SKINAI{timestamp[:8]}GARTUIT"
        
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.success(f"""
        ✅ **Diagnostic enregistré avec succès !**
        
        **🎁 Votre produit gratuit vous attend :**
        📦 **Produit :** {product}
        🔢 **Code :** `{coupon_code}`
        📍 **Retrait :** Pharmacie partenaire la plus proche
        
        **📊 Données enregistrées pour SkinGPT :**
        • Hyperpigmentation : {hyper['ratio']:.1%} de surface
        • Taches détectées : {hyper['spot_count']}
        • Imperfections : {acne['acne_count']}
        
        **Merci de contribuer à la recherche !** 🌿
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# --- Section statistiques en temps réel ---
st.markdown("---")
st.subheader("📈 Statistiques des Diagnostics")

try:
    if os.path.exists("skinai_data/diagnostics/skinai_training_data.csv"):
        df_stats = pd.read_csv("skinai_data/diagnostics/skinai_training_data.csv")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Diagnostics réalisés", len(df_stats))
        
        with col2:
            if 'hyperpigmentation_ratio' in df_stats.columns:
                avg_hyper = df_stats['hyperpigmentation_ratio'].mean()
                st.metric("Hyperpigmentation moyenne", f"{avg_hyper:.1%}")
        
        with col3:
            if 'acne_count' in df_stats.columns:
                total_acne = df_stats['acne_count'].sum()
                st.metric("Boutons analysés", total_acne)
        
        with col4:
            products_given = len(df_stats)
            st.metric("Produits offerts", products_given)

except Exception as e:
    st.info("📊 Les statistiques s'afficheront ici après les premiers diagnostics")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🌿 <strong>SkinAI Sénégal</strong> - Détection RÉELLE par Intelligence Artificielle</p>
    <p><small>🔬 Algorithmes de détection d'hyperpigmentation et d'acné en temps réel</small></p>
    <p><small>⚠️ Diagnostic IA informatif - Consultation médicale recommandée pour les cas sévères</small></p>
</div>
""", unsafe_allow_html=True)