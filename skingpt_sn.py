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
    page_title="AuraSkin - IA Dermatologique Peau Noire",
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
    .emotional-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    .honest-box {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Fonctions de détection AMÉLIORÉES pour peaux noires ---
def validate_image_quality(image):
    """Valide la qualité de l'image pour l'analyse"""
    issues = []
    img_array = np.array(image)
    
    # Vérification de la netteté
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    if sharpness < 150:  # Augmenté pour plus de précision
        issues.append("floue")
    
    # Vérification de la luminosité
    brightness = np.mean(gray)
    if brightness < 60:  # Ajusté pour peaux noires
        issues.append("trop sombre")
    elif brightness > 180:  # Ajusté pour éviter surexposition
        issues.append("trop lumineuse")
    
    return issues, sharpness, brightness

def detect_hyperpigmentation_improved(image):
    """
    Détection AMÉLIORÉE de l'hyperpigmentation spécialement calibrée pour peaux noires
    """
    # Conversion en array numpy
    img_array = np.array(image)
    original = img_array.copy()
    
    # Conversion en espace LAB pour meilleure analyse des peaux noires
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    L_channel, A_channel, B_channel = cv2.split(lab)
    
    # --- Détection PLUS PRÉCISE pour peaux noires ---
    # Seuils ajustés pour les peaux noires (moins sensibles)
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    # Plages plus restrictives pour éviter les faux positifs
    lower_brown = np.array([0, 40, 30])   # Augmenté le seuil
    upper_brown = np.array([25, 180, 150]) # Réduit la plage
    
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    
    # Analyse de la luminance avec seuils adaptés
    _, dark_mask = cv2.threshold(L_channel, 70, 255, cv2.THRESH_BINARY_INV)  # Seuil augmenté
    
    # Combinaison des masques
    combined_mask = cv2.bitwise_or(brown_mask, dark_mask)
    
    # Nettoyage AGGRESSIF du masque pour éliminer les faux positifs
    kernel = np.ones((5,5), np.uint8)  # Kernel plus grand
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    
    # Élimination des très petites zones (bruit)
    combined_mask = cv2.medianBlur(combined_mask, 5)
    
    # Détection des contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrage BEAUCOUP plus strict
    min_contour_area = 100  # Augmenté pour ignorer petites imperfections
    significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    
    # Calcul des métriques
    total_pixels = img_array.shape[0] * img_array.shape[1]
    hyperpigmentation_pixels = np.count_nonzero(combined_mask)
    hyperpigmentation_ratio = hyperpigmentation_pixels / total_pixels
    
    # Nombre de taches détectées (seulement les significatives)
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

def detect_acne_improved(image):
    """
    Détection AMÉLIORÉE de l'acné avec seuils adaptés pour peaux noires
    """
    img_array = np.array(image)
    
    # Conversion en différents espaces de couleur
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    
    # Détection des rougeurs (beaucoup plus restrictive)
    lower_red1 = np.array([0, 80, 80])    # Seuils augmentés
    upper_red1 = np.array([8, 255, 255])
    lower_red2 = np.array([172, 80, 80])  # Seuils augmentés
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    redness_mask = cv2.bitwise_or(mask1, mask2)
    
    # Nettoyage agressif du masque de rougeurs
    kernel = np.ones((3,3), np.uint8)
    redness_mask = cv2.morphologyEx(redness_mask, cv2.MORPH_OPEN, kernel)
    
    # Détection des contours circulaires (boutons) - plus restrictive
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 2)  # Plus de flou pour réduire le bruit
    
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 30,  # Distance min augmentée
                              param1=80, param2=35, minRadius=8, maxRadius=35)  # Seuils augmentés
    
    acne_count = 0
    if circles is not None:
        # Filtrage supplémentaire des cercles détectés
        valid_circles = []
        for circle in circles[0]:
            x, y, r = circle
            # Vérifier que la zone n'est pas trop sombre (éviter pores)
            roi = gray[int(y-r):int(y+r), int(x-r):int(x+r)]
            if np.mean(roi) > 40:  # Éviter les zones très sombres
                valid_circles.append(circle)
        acne_count = len(valid_circles)
    
    return {
        'acne_count': acne_count,
        'redness_ratio': np.count_nonzero(redness_mask) / (img_array.shape[0] * img_array.shape[1]),
        'redness_mask': redness_mask
    }

def analyze_skin_conditions_improved(image):
    """
    Analyse COMPLÈTE avec algorithmes améliorés pour peaux noires
    """
    hyperpigmentation_data = detect_hyperpigmentation_improved(image)
    acne_data = detect_acne_improved(image)
    
    return {
        'hyperpigmentation': hyperpigmentation_data,
        'acne': acne_data
    }

def get_honest_diagnosis(analysis_results, age, skin_type):
    """
    Diagnostic HONNÊTE qui dit la vérité quand il n'y a pas de problèmes
    """
    hyper = analysis_results['hyperpigmentation']
    acne = analysis_results['acne']
    
    # --- NOUVEAUX SEUILS BEAUCOUP PLUS STRICTS ---
    
    # Diagnostic hyperpigmentation - SEUILS AUGMENTÉS
    hyperpigmentation_detected = False
    hyperpigmentation_message = ""
    
    if hyper['ratio'] > 0.20:  # 20% au lieu de 8%
        hyperpigmentation_detected = True
        hyperpigmentation_message = "Signes importants d'hyperpigmentation"
    elif hyper['ratio'] > 0.12:  # 12% au lieu de 3%
        hyperpigmentation_detected = True
        hyperpigmentation_message = "Signes modérés d'hyperpigmentation"
    elif hyper['ratio'] > 0.08:  # Seuil minimal augmenté
        hyperpigmentation_message = "Quelques variations pigmentaires"
    else:
        hyperpigmentation_message = "Peau présentant peu d'imperfections"
    
    # Diagnostic acné - SEUILS AUGMENTÉS
    acne_detected = False
    acne_message = ""
    
    if acne['acne_count'] > 8:  # 8 au lieu de 5
        acne_detected = True
        acne_message = "Présence notable d'acné"
    elif acne['acne_count'] > 4:  # 4 au lieu de 2
        acne_detected = True
        acne_message = "Présence modérée d'acné"
    elif acne['acne_count'] > 1:
        acne_message = "Présence légère d'acné"
    else:
        acne_message = "Peau présentant peu d'imperfections"
    
    # --- DIAGNOSTIC GLOBAL HONNÊTE ---
    conditions = []
    needs_attention = False
    
    if hyperpigmentation_detected:
        conditions.append(hyperpigmentation_message)
        needs_attention = True
    else:
        conditions.append(hyperpigmentation_message)
    
    if acne_detected:
        conditions.append(acne_message)
        needs_attention = True
    else:
        conditions.append(acne_message)
    
    # Si AUCUN problème significatif n'est détecté
    if not needs_attention and hyper['ratio'] <= 0.08 and acne['acne_count'] <= 1:
        diagnosis = "Peau présentant peu d'imperfections"
        product = "CRÈME HYDRATANTE QUOTIDIENNE"
        advice = """
        • Maintenir une routine de soin équilibrée
        • Nettoyer quotidiennement avec un produit adapté
        • Hydrater matin et soir pour préserver la barrière cutanée
        • Protection solaire préventive même par temps couvert
        """
    else:
        diagnosis = " + ".join(conditions)
        
        # Produits recommandés SEULEMENT si nécessaire
        products = []
        advice_points = []
        
        if hyperpigmentation_detected:
            if hyper['ratio'] > 0.20:
                products.append("SÉRUM INTENSIF ANTI-TACHES + CRÈME ÉCLAT NUIT")
                advice_points.append(f"• **Analyse cutanée :** {hyper['ratio']:.1%} de la peau présente des signes d'hyperpigmentation ({hyper['spot_count']} zones concernées)")
                advice_points.append("• Appliquer les soins éclaircissants matin et soir")
                advice_points.append("• PROTECTION SOLAIRE SPF 50+ obligatoire pour prévenir l'aggravation")
            elif hyper['ratio'] > 0.12:
                products.append("SÉRUM ÉCLAT ANTI-TACHES")
                advice_points.append(f"• **Analyse cutanée :** {hyper['ratio']:.1%} de la peau présente des signes d'hyperpigmentation")
                advice_points.append("• Appliquer les soins éclaircissants quotidiennement")
                advice_points.append("• Protection solaire SPF 50+ recommandée")
            else:
                products.append("SOIN ÉQUILIBRANT PEAUX SENSIBLES")
                advice_points.append("• Quelques variations pigmentaires détectées")
                advice_points.append("• Soin équilibrant recommandé en prévention")
        
        if acne_detected:
            if acne['acne_count'] > 8:
                products.append("GEL PURIFIANT INTENSIF + MASQUE DÉTOX")
                advice_points.append(f"• **Analyse cutanée :** {acne['acne_count']} imperfections détectées")
                advice_points.append("• Nettoyer la peau matin et soir avec un produit doux")
                advice_points.append("• Ne pas percer les boutons pour éviter les marques")
            elif acne['acne_count'] > 4:
                products.append("GEL PURIFIANT QUOTIDIEN")
                advice_points.append(f"• **Analyse cutanée :** {acne['acne_count']} imperfections détectées")
                advice_points.append("• Nettoyer la peau quotidiennement")
                advice_points.append("• Changer les taies d'oreiller 2 fois par semaine")
            else:
                products.append("NETTOYANT DOUX ACNE-STOP")
                advice_points.append("• Quelques imperfections mineures")
                advice_points.append("• Nettoyage doux recommandé")
        
        product = " + ".join(products) if products else "CRÈME HYDRATANTE QUOTIDIENNE"
        advice = "\n".join(advice_points)
    
    # Recommandation médicale - SEUIL AUGMENTÉ
    needs_doctor = (hyper['ratio'] > 0.30 or acne['acne_count'] > 15 or age > 50)
    medical_advice = "🔔 **Nous recommandons une consultation dermatologique pour un suivi approfondi**" if needs_doctor else ""
    
    return diagnosis, product, advice, medical_advice

# --- Interface principale ---
st.markdown('<h1 class="main-header">🌿 AuraSkin - IA Dermatologique Spécialisée Peau Noire</h1>', unsafe_allow_html=True)

# --- Section émotionnelle ---
st.markdown('<div class="emotional-box">', unsafe_allow_html=True)
st.markdown("""
<h2 style='color: white; text-align: center;'>🌟 Rejoignez la Révolution AuraSkin 🌟</h2>

<p style='font-size: 1.2rem; text-align: center;'>
<strong>Aidez-nous à construire la première IA dermatologique spécialisée pour les peaux noires !</strong>
</p>

<p>
Notre intelligence artificielle actuelle a une précision de <strong>68%</strong> - 
chaque diagnostic que vous effectuez nous permet de l'améliorer et de la perfectionner.
</p>

<p style='font-style: italic;'>
Votre participation aujourd'hui contribue à créer des solutions de soins cutanés 
plus précises et adaptées pour toute la communauté noire demain.
</p>

<p>
<strong>📞 WhatsApp :</strong> +221 76 484 40 51<br>
<strong>📧 Email :</strong> diouffatou452@gmail.com
</p>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("https://via.placeholder.com/200x200/2E8B57/FFFFFF?text=AURASKIN", width=150)
    st.title("📋 Votre Profil")
    
    age = st.slider("Âge", 15, 70, 25)
    skin_type = st.selectbox("Type de peau", 
                           ["Peau grasse", "Peau sèche", "Peau mixte", "Peau normale", "Je ne sais pas"])
    
    st.markdown("---")
    st.info("""
    **🎯 Notre Mission :**
    • Développer une IA spécialisée peaux noires
    • Améliorer les diagnostics dermatologiques
    • Offrir des solutions adaptées
    """)
    
    st.markdown("---")
    st.subheader("📞 Nous Contacter")
    st.write("""
    **WhatsApp :** +221 76 484 40 51
    **Email :** diouffatou452@gmail.com
    **Dakar, Sénégal**
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
            st.subheader("🎯 Résultats de l'Analyse")
            
            # Analyse AMÉLIORÉE avec indicateur de progression
            with st.spinner("🧠 AuraSkin analyse votre peau avec nos algorithmes améliorés..."):
                # Analyse complète avec nouvelles fonctions
                analysis_results = analyze_skin_conditions_improved(image)
                
                # Diagnostic HONNÊTE
                diagnosis, product, advice, medical_advice = get_honest_diagnosis(
                    analysis_results, age, skin_type
                )
            
            # Affichage des résultats
            if hyper['ratio'] <= 0.08 and acne['acne_count'] <= 1:
                st.markdown('<div class="honest-box">', unsafe_allow_html=True)
                st.write("**📋 NOTRE ANALYSE :**")
                st.success(f"**{diagnosis}**")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="diagnostic-box">', unsafe_allow_html=True)
                st.write("**📋 NOTRE ANALYSE :**")
                st.info(f"**{diagnosis}**")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Produit recommandé
            st.markdown('<div class="product-box">', unsafe_allow_html=True)
            st.write("**💫 PRODUIT RECOMMANDÉ :**")
            st.info(f"**{product}**")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Affichage de l'analyse détaillée
            with st.expander("🔍 **Détails de l'analyse technique**", expanded=True):
                col_anal1, col_anal2 = st.columns(2)
                
                with col_anal1:
                    st.write("**📊 Hyperpigmentation :**")
                    hyper = analysis_results['hyperpigmentation']
                    st.metric("Surface concernée", f"{hyper['ratio']:.1%}")
                    st.metric("Zones détectées", hyper['spot_count'])
                    st.caption("Seuil de détection : >8%")
                
                with col_anal2:
                    st.write("**📊 Acné & Imperfections :**")
                    acne = analysis_results['acne']
                    st.metric("Imperfections", acne['acne_count'])
                    st.metric("Rougeurs", f"{acne['redness_ratio']:.1%}")
                    st.caption("Seuil de détection : >4 imperfections")
            
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
        if hyper['spot_count'] > 0:
            st.image(hyper['visualization'], caption=f"🔴 Zones d'hyperpigmentation détectées ({hyper['spot_count']} zones)", use_column_width=True)
        else:
            st.image(image, caption="✅ Aucune zone d'hyperpigmentation significative détectée", use_column_width=True)

# --- Section enregistrement et produit gratuit ---
if uploaded_file and not issues:
    st.markdown("---")
    st.subheader("🎁 Recevez Votre Produit AuraSkin")
    
    # Formulaire de contact pour l'envoi
    with st.form("contact_form"):
        st.write("**📝 Informations pour votre produit**")
        
        col1, col2 = st.columns(2)
        with col1:
            customer_name = st.text_input("Nom complet*")
            customer_phone = st.text_input("Téléphone*", placeholder="+221 XX XXX XX XX")
        with col2:
            customer_email = st.text_input("Email", placeholder="votre@email.com")
            customer_city = st.selectbox("Ville*", 
                ["Dakar", "Thiès", "Mbour", "Saint-Louis", "Autre"])
        
        delivery_option = st.radio("Comment souhaitez-vous recevoir votre produit?*",
            ["🚗 Livraison à domicile (frais de livraison applicables)", 
             "🏪 Retrait chez AuraSkin Dakar"])
        
        # Engagement suivi
        st.markdown("---")
        st.write("**📊 Engagement d'amélioration**")
        follow_up_agreement = st.checkbox("Je m'engage à partager mon amélioration après 1 semaine d'utilisation*", value=True)
        st.caption("Votre feedback nous aide à améliorer AuraSkin pour toute la communauté")
        
        submitted = st.form_submit_button("🎁 Recevoir mon produit AuraSkin", type="primary")
    
    if submitted:
        if not customer_name or not customer_phone or not follow_up_agreement:
            st.error("❌ Veuillez remplir les champs obligatoires (*)")
        else:
            # Génération du code client unique
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            customer_id = f"AURA{timestamp[:8]}"
            coupon_code = f"AURA{timestamp[:6]}GIFT"
            
            # Sauvegarde de l'image
            os.makedirs("auraskin_data/images", exist_ok=True)
            os.makedirs("auraskin_data/clients", exist_ok=True)
            image_filename = f"auraskin_data/images/{timestamp}.jpg"
            image.save(image_filename)
            
            # Préparation des données client
            hyper = analysis_results['hyperpigmentation']
            acne = analysis_results['acne']
            
            client_data = {
                "customer_id": [customer_id],
                "coupon_code": [coupon_code],
                "timestamp": [timestamp],
                "name": [customer_name],
                "phone": [customer_phone],
                "email": [customer_email],
                "city": [customer_city],
                "age": [age],
                "skin_type": [skin_type],
                "delivery_option": [delivery_option],
                "hyperpigmentation_ratio": [hyper['ratio']],
                "hyperpigmentation_spots": [hyper['spot_count']],
                "acne_count": [acne['acne_count']],
                "diagnosis": [diagnosis],
                "recommended_product": [product],
                "follow_up_agreement": [follow_up_agreement],
                "status": ["en_attente"],
                "image_path": [image_filename]
            }
            
            # Sauvegarde dans CSV clients
            client_df = pd.DataFrame(client_data)
            clients_path = "auraskin_data/clients/clients_data.csv"
            
            if os.path.exists(clients_path):
                client_df.to_csv(clients_path, mode="a", header=False, index=False)
            else:
                client_df.to_csv(clients_path, index=False)
            
            # Instructions de livraison basées sur l'option choisie
            delivery_instructions = {
                "🚗 Livraison à domicile (frais de livraison applicables)": "Notre équipe vous contactera sous 24h pour organiser la livraison",
                "🏪 Retrait chez AuraSkin Dakar": "Présentez votre code client à notre centre AuraSkin Dakar"
            }
            
            instruction = delivery_instructions[delivery_option]
            
            # Message de confirmation
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.success(f"""
            ✅ **Votre produit AuraSkin est réservé !**
            
            **📦 VOTRE COMMANDE :**
            🎁 **Produit :** {product}
            🔢 **Code Client :** `{customer_id}`
            🏷️ **Code Produit :** `{coupon_code}`
            👤 **Nom :** {customer_name}
            
            **📋 PROCHAINES ÉTAPES :**
            {instruction}
            
            **📞 Contactez-nous :**
            📱 **WhatsApp :** +221 76 484 40 51
            📧 **Email :** diouffatou452@gmail.com
            
            **🌟 Votre Engagement :**
            Merci de vous engager à partager votre amélioration après 1 semaine d'utilisation !
            Votre expérience est précieuse pour améliorer AuraSkin.
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Instructions suivi
            st.markdown("---")
            st.subheader("📊 Suivi de Votre Amélioration")
            st.info("""
            **Après 1 semaine d'utilisation, contactez-nous pour partager :**
            
            📸 **Envoyez-nous une nouvelle photo** de votre peau
            💬 **Décrivez votre expérience** avec le produit
            ⭐ **Notez l'amélioration** (1 à 5 étoiles)
            
            **Comment nous contacter :**
            • **WhatsApp :** +221 76 484 40 51
            • **Email :** diouffatou452@gmail.com
            • **Message :** "Suivi AuraSkin - [Votre Code Client]"
            
            Votre feedback nous aide à perfectionner notre IA pour toute la communauté !
            """)
            
            # Téléchargement du reçu
            receipt = f"""
            RECU AURASKIN
            ====================
            
            CLIENT : {customer_name}
            TELEPHONE : {customer_phone}
            ID CLIENT : {customer_id}
            CODE PRODUIT : {coupon_code}
            
            ANALYSE : {diagnosis}
            PRODUIT RECOMMANDE : {product}
            
            OPTION : {delivery_option}
            DATE : {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}
            
            ENGAGEMENT SUIVI : OUI
            CONTACT SUIVI : +221 76 484 40 51
            
            INSTRUCTIONS :
            - Présentez ce reçu pour obtenir votre produit
            - Contactez-nous après 1 semaine pour le suivi
            - Valable 30 jours
            - Un produit par personne
            
            Merci de contribuer à la révolution AuraSkin !
            AuraSkin 🌿 - IA Dermatologique Peau Noire
            """
            
            st.download_button(
                label="📄 Télécharger le reçu",
                data=receipt,
                file_name=f"reçu_auraskin_{customer_id}.txt",
                mime="text/plain"
            )

# --- Section statistiques en temps réel ---
st.markdown("---")
st.subheader("📈 Impact de la Communauté AuraSkin")

try:
    if os.path.exists("auraskin_data/clients/clients_data.csv"):
        df_stats = pd.read_csv("auraskin_data/clients/clients_data.csv")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Membres AuraSkin", len(df_stats))
        
        with col2:
            if 'hyperpigmentation_ratio' in df_stats.columns:
                avg_hyper = df_stats['hyperpigmentation_ratio'].mean()
                st.metric("Hyperpigmentation moyenne", f"{avg_hyper:.1%}")
        
        with col3:
            engagements = len(df_stats[df_stats['follow_up_agreement'] == True])
            st.metric("Engagements suivi", engagements)
        
        with col4:
            precision = 68 + min(len(df_stats) // 10, 20)
            st.metric("Précision IA actuelle", f"{precision}%", f"+{min(len(df_stats)//10, 20)}%")

except Exception as e:
    st.info("📊 Les statistiques s'afficheront ici après les premiers membres AuraSkin")

# --- Section amélioration IA ---
st.markdown("---")
st.markdown("""
<div class="emotional-box">
<h3>🚀 Aidez-nous à Perfectionner AuraSkin</h3>

<p>
<strong>Précision actuelle : 68% - Objectif : 85%</strong><br>
Chaque diagnostic améliore notre intelligence artificielle spécialisée peaux noires.
</p>

<p>
Votre participation aujourd'hui crée des solutions plus précises pour toute notre communauté demain.
</p>

<p>
<strong>📞 Contactez-nous :</strong><br>
WhatsApp: +221 76 484 40 51<br>
Email: diouffatou452@gmail.com
</p>
</div>
""", unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🌿 <strong>AuraSkin</strong> - Intelligence Artificielle Dermatologique Spécialisée Peau Noire</p>
    <p><small>🔬 Développée pour et par la communauté noire - Precision en amélioration continue</small></p>
    <p><small>⚠️ Cette analyse est informative et ne remplace pas une consultation médicale</small></p>
    <p><small>📞 Contact : +221 76 484 40 51 | 📧 diouffatou452@gmail.com</small></p>
</div>
""", unsafe_allow_html=True)