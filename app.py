from flask import Flask, request, jsonify
from pymongo import MongoClient
import os

app = Flask(__name__)
client = MongoClient('mongodb://localhost:27017/')
db = client['lost_children_db']
children_collection = db['children']

ADMIN_PASSWORD = "admin123"

from functools import wraps
def require_admin_password(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        password = request.headers.get('X-Admin-Password')
        if not password or password != ADMIN_PASSWORD:
            return jsonify({"error": "Accès refusé."}), 403
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def home():
    return "✅ Serveur actif — Bienvenue sur l'API Enfants Perdus !"

@app.route('/register', methods=['POST'])
@require_admin_password
def register_child():
    data = request.get_json()
    required = ['child_name', 'child_age', 'parent_phone', 'face_embedding']
    if not all(k in data for k in required):
        return jsonify({"error": "Données manquantes"}), 400
    result = children_collection.insert_one(data)
    return jsonify({"message": "Enfant enregistré", "id": str(result.inserted_id)}), 201

@app.route('/identify', methods=['POST'])
def identify_child():
    data = request.get_json()
    query_embedding = data.get('face_embedding')
    if not query_embedding:
        return jsonify({"error": "Embedding manquant"}), 400

    best_match = None
    best_distance = 0.6
    for child in children_collection.find():
        stored_embedding = child.get('face_embedding', [])
        if len(stored_embedding) != len(query_embedding):
            continue
        distance = sum((a - b) ** 2 for a, b in zip(query_embedding, stored_embedding)) ** 0.5
        if distance < best_distance:
            best_distance = distance
            best_match = child

    if best_match:
        best_match.pop('face_embedding', None)
        best_match.pop('_id', None)
        return jsonify({"match": best_match}), 200
    else:
        return jsonify({"error": "Aucun enfant trouvé"}), 404

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)