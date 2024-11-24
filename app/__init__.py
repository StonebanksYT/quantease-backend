from flask import Flask
from flask_cors import CORS  # Import CORS for handling cross-origin requests
from routes import api  # Import your blueprint that contains routes

def create_app():
    app = Flask(__name__)
    
    # Enable CORS for the entire app. If you want to allow requests from any domain
    # CORS(app) allows all origins, or you can specify allowed origins like:
    CORS(app, origins="http://localhost:3000")  # Allow requests from your frontend

    # Register the Blueprint
    app.register_blueprint(api, url_prefix='/api')

    return app