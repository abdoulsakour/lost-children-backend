from flask import Flask, request, jsonify
from pymongo import MongoClient
import numpy as np
from PIL import Image
import io
import os
import dlib  # ← On l'importe explicitement
# Clé secrète pour l'authentification admin
ADMIN_PASSWORD = "admin123"  # ← À changer en production !

# Créer l'app Flask
app = Flask(__name__)


# Augmenter la limite de taille des fichiers uploadés
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 Mo
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.jpeg']


# Se connecter à MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['lost_children_db']
children_collection = db['children']

# ⚙️ CHARGER LES MODÈLES À LA MAIN — SOLUTION ULTIME
print("🔍 Chargement manuel des modèles dlib...")

# Chemin vers le dossier models à côté de app.py
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

# Charger le détecteur de visage (HOG par défaut)
face_detector = dlib.get_frontal_face_detector()

# Charger le prédicteur de points faciaux (68 landmarks)
predictor_path = os.path.join(MODELS_DIR, "shape_predictor_68_face_landmarks.dat")
if not os.path.exists(predictor_path):
    raise FileNotFoundError(f"❌ Fichier introuvable : {predictor_path}")

face_predictor = dlib.shape_predictor(predictor_path)

# Charger le modèle de reconnaissance faciale (embedding)
recognition_model_path = os.path.join(MODELS_DIR, "dlib_face_recognition_resnet_model_v1.dat")
if not os.path.exists(recognition_model_path):
    raise FileNotFoundError(f"❌ Fichier introuvable : {recognition_model_path}")

face_encoder = dlib.face_recognition_model_v1(recognition_model_path)

print("✅ Modèles chargés avec succès !")
from functools import wraps

def require_admin_password(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        password = request.headers.get('X-Admin-Password')
        if not password or password != ADMIN_PASSWORD:
            return jsonify({"error": "Accès refusé. Mot de passe administrateur requis."}), 403
        return f(*args, **kwargs)
    return decorated_function

# Route de test
@app.route('/')
def home():
    return "✅ Serveur actif — Bienvenue sur l'API Enfants Perdus !"

# Route pour enregistrer un enfant AVEC photo
@app.route('/register', methods=['POST'])
@require_admin_password  # ← AJOUTÉ
def register_child():
    # Récupérer les champs du formulaire
    child_name = request.form.get('child_name')
    child_age = request.form.get('child_age')
    parent_phone = request.form.get('parent_phone')
    parent_phone_2 = request.form.get('parent_phone_2')  # Récupère le 2ème numéro
    print("📞 parent_phone_2 reçu :", parent_phone_2)     # ← LIGNE DE DEBUG — très importante
    description = request.form.get('description')
    location_found = request.form.get('location_found')

    # Vérifier les champs obligatoires
    if not all([child_name, child_age, parent_phone]):
        return jsonify({"error": "Les champs 'child_name', 'child_age', 'parent_phone' sont obligatoires."}), 400

    # Récupérer la photo
    if 'photo' not in request.files:
        return jsonify({"error": "Aucune photo fournie."}), 400

    photo = request.files['photo']

    if photo.filename == '':
        return jsonify({"error": "Aucun fichier sélectionné."}), 400

    try:
        # Lire l'image avec PIL
        image = Image.open(photo.stream)
        image_np = np.array(image)

        # Convertir en RGB si nécessaire (PIL peut charger en RGBA)
        if image_np.shape[2] == 4:
            image_np = image_np[:, :, :3]

        # Détecter les visages avec dlib
        gray_image = np.array(Image.fromarray(image_np).convert('L'))  # Conversion en niveaux de gris
        faces = face_detector(gray_image, 1)

        if len(faces) == 0:
            return jsonify({"error": "Aucun visage détecté dans la photo."}), 400

        if len(faces) > 1:
            return jsonify({"error": "Plusieurs visages détectés. Veuillez envoyer une photo avec un seul enfant."}), 400

        # Prendre le premier visage
        face = faces[0]

        # Prédire les 68 points du visage
        shape = face_predictor(gray_image, face)

        # Générer l'embedding (128D)
        embedding = face_encoder.compute_face_descriptor(image_np, shape)
        face_embedding = np.array(embedding).tolist()  # Convertir en liste pour MongoDB

        # Créer l'objet à sauvegarder
        child_data = {
            "child_name": child_name,
            "child_age": int(child_age),
            "parent_phone": parent_phone,
            "parent_phone_2": parent_phone_2 or "",
            "description": description,
            "location_found": location_found,
            "face_embedding": face_embedding  # ← Embedding stocké !
        }

        # Sauvegarder dans MongoDB
        result = children_collection.insert_one(child_data)

        return jsonify({
            "message": "Enfant enregistré avec succès !",
            "id": str(result.inserted_id)
        }), 201

    except Exception as e:
        return jsonify({"error": f"Erreur lors du traitement de l'image : {str(e)}"}), 500

# Route pour lister tous les enfants
@app.route('/children', methods=['GET'])
@require_admin_password  # ← AJOUTÉ
def get_children():
    children = list(children_collection.find({}, {'face_embedding': 0, '_id': 0}))
    return jsonify(children), 200
# Route pour identifier un enfant à partir d'une photo
@app.route('/identify', methods=['POST'])
def identify_child():
    if 'photo' not in request.files:
        return jsonify({"error": "Aucune photo fournie."}), 400

    photo = request.files['photo']
    if photo.filename == '':
        return jsonify({"error": "Aucun fichier sélectionné."}), 400

    try:
        # Lire l'image
        image = Image.open(photo.stream)
        image_np = np.array(image)
        if image_np.shape[2] == 4:
            image_np = image_np[:, :, :3]

        # Détecter le visage
        gray_image = np.array(Image.fromarray(image_np).convert('L'))
        faces = face_detector(gray_image, 1)

        if len(faces) == 0:
            return jsonify({"error": "Aucun visage détecté."}), 400
        if len(faces) > 1:
            return jsonify({"error": "Plusieurs visages détectés."}), 400

        face = faces[0]
        shape = face_predictor(gray_image, face)
        query_embedding = np.array(face_encoder.compute_face_descriptor(image_np, shape))

        # Comparer avec tous les enfants enregistrés
        best_match = None
        best_distance = 0.6  # Seuil de similarité (à ajuster)

        for child in children_collection.find():
            stored_embedding = np.array(child['face_embedding'])
            distance = np.linalg.norm(query_embedding - stored_embedding)

            if distance < best_distance:
                best_distance = distance
                best_match = child

        if best_match:
            # Ne pas renvoyer l'embedding ni l'_id
            best_match.pop('face_embedding', None)
            best_match.pop('_id', None)
            return jsonify({"match": best_match}), 200
        else:
            return jsonify({"error": "Aucun enfant correspondant trouvé."}), 404

    except Exception as e:
        return jsonify({"error": f"Erreur lors de l'identification : {str(e)}"}), 500

# Lancer le serveur
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
