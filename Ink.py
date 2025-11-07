#!/usr/bin/env python3
"""
INK - AI Chatbot with Gemini Integration
A portable AI assistant with SVG generation, code writing, and web interface
"""

import os
import sys
import json
import subprocess
import webbrowser
import threading
import time
from datetime import datetime
from pathlib import Path

REQUIRED_PACKAGES = [
    "flask",
    "google-generativeai",
    "svgwrite"
]

def install_dependencies():
    """Auto-install required dependencies on first run"""
    print("🔧 Checking dependencies...")
    for package in REQUIRED_PACKAGES:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            print(f"📦 Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
    print("✅ All dependencies installed!")

def setup_cache_folders():
    """Create necessary cache and data folders"""
    folders = ["cache", "cache/conversations", "cache/images", "cache/datasets"]
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
    print("📁 Cache folders created!")

install_dependencies()
setup_cache_folders()

from flask import Flask, render_template, request, jsonify, send_from_directory
import google.generativeai as genai
import svgwrite

app = Flask(__name__)
app.secret_key = os.environ.get('SESSION_SECRET', 'ink-secret-key')

class INKMemory:
    """JSON-based memory system for conversations"""
    
    def __init__(self):
        self.memory_file = "cache/conversations/memory.json"
        self.current_session = []
        self.thinking_log = []
        self.load_memory()
    
    def load_memory(self):
        """Load conversation history from JSON"""
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r') as f:
                data = json.load(f)
                self.current_session = data.get('current_session', [])
                self.thinking_log = data.get('thinking_log', [])
    
    def save_memory(self):
        """Save conversation history to JSON"""
        with open(self.memory_file, 'w') as f:
            json.dump({
                'current_session': self.current_session[-100:],
                'thinking_log': self.thinking_log[-100:],
                'last_updated': datetime.now().isoformat()
            }, f, indent=2)
    
    def add_message(self, role, content):
        """Add message to conversation history"""
        self.current_session.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
        self.save_memory()
    
    def add_thinking(self, thought):
        """Add thinking process to log"""
        self.thinking_log.append({
            'thought': thought,
            'timestamp': datetime.now().isoformat()
        })
        self.save_memory()
    
    def get_context(self, limit=10):
        """Get recent conversation context"""
        return self.current_session[-limit:]
    
    def clear_session(self):
        """Clear current session"""
        self.current_session = []
        self.thinking_log = []
        self.save_memory()

class INKAIEngine:
    """AI Engine powered by Gemini"""
    
    def __init__(self):
        self.api_key = os.environ.get('GEMINI_API_KEY', '')
        self.model = None
        self.memory = INKMemory()
        self.training_mode = False
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize Gemini model"""
        if not self.api_key:
            print("⚠️  GEMINI_API_KEY not found. Please set it in environment variables.")
            return
        
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            print("✅ Gemini AI initialized successfully!")
        except Exception as e:
            print(f"❌ Error initializing Gemini: {e}")
    
    def generate_response(self, user_input):
        """Generate AI response with thinking process"""
        if not self.model:
            return {
                'response': 'Please configure GEMINI_API_KEY to use INK AI.',
                'thinking': 'Model not initialized - API key missing'
            }
        
        try:
            self.memory.add_thinking(f"Processing user input: {user_input[:50]}...")
            
            context = self.memory.get_context()
            context_text = "\n".join([f"{m['role']}: {m['content']}" for m in context])
            
            prompt = f"""Previous conversation:
{context_text}

User: {user_input}

Respond as INK, an intelligent AI assistant. Be concise, accurate, and helpful. If asked to generate code, provide clean, working code. If asked to create an SVG, describe what you'd create."""
            
            self.memory.add_thinking("Generating response from Gemini...")
            response = self.model.generate_content(prompt)
            
            ai_response = response.text
            
            self.memory.add_message('user', user_input)
            self.memory.add_message('assistant', ai_response)
            self.memory.add_thinking("Response generated successfully")
            
            return {
                'response': ai_response,
                'thinking': '\n'.join([t['thought'] for t in self.memory.thinking_log[-5:]])
            }
        
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.memory.add_thinking(error_msg)
            return {
                'response': f"I encountered an error: {str(e)}",
                'thinking': error_msg
            }
    
    def generate_code(self, description):
        """Generate code based on description"""
        self.memory.add_thinking(f"Code generation requested: {description}")
        
        prompt = f"""Generate clean, working, production-ready code for: {description}

Provide only the code without explanations. Make it blazing fast and correct."""
        
        try:
            response = self.model.generate_content(prompt)
            code = response.text
            self.memory.add_thinking("Code generated successfully")
            return code
        except Exception as e:
            return f"Error generating code: {str(e)}"
    
    def train_mode(self, instruction):
        """Training mode for model customization"""
        self.training_mode = True
        self.memory.add_thinking(f"Training instruction received: {instruction}")
        return f"Training mode activated. Instruction: {instruction}"

def generate_svg(description, filename="output.svg"):
    """Generate SVG image based on description"""
    filepath = f"cache/images/{filename}"
    
    dwg = svgwrite.Drawing(filepath, size=('400px', '400px'))
    
    if "circle" in description.lower():
        dwg.add(dwg.circle(center=(200, 200), r=100, fill='blue'))
    elif "rect" in description.lower() or "square" in description.lower():
        dwg.add(dwg.rect(insert=(100, 100), size=(200, 200), fill='green'))
    elif "star" in description.lower():
        points = [(200, 50), (230, 150), (350, 150), (250, 220), (280, 320), (200, 250), (120, 320), (150, 220), (50, 150), (170, 150)]
        dwg.add(dwg.polygon(points=points, fill='gold'))
    else:
        dwg.add(dwg.rect(insert=(50, 50), size=(300, 300), fill='lightgray'))
        dwg.add(dwg.text(description[:20], insert=(200, 200), text_anchor='middle', font_size='20px'))
    
    dwg.save()
    return filepath

ink_engine = INKAIEngine()

@app.route('/')
def index():
    """Serve the main chatbot interface"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    data = request.json
    user_message = data.get('message', '')
    
    if user_message.startswith('/train'):
        instruction = user_message[7:].strip()
        response = ink_engine.train_mode(instruction)
        thinking = "Entering training mode"
    elif user_message.startswith('/code'):
        description = user_message[6:].strip()
        response = ink_engine.generate_code(description)
        thinking = "Generating code"
    elif user_message.startswith('/svg'):
        description = user_message[5:].strip()
        filename = f"svg_{int(time.time())}.svg"
        filepath = generate_svg(description, filename)
        response = f"SVG created! View it at: /images/{filename}"
        thinking = "Generating SVG image"
    elif user_message.startswith('/clear'):
        ink_engine.memory.clear_session()
        response = "Memory cleared!"
        thinking = "Session reset"
    else:
        result = ink_engine.generate_response(user_message)
        response = result['response']
        thinking = result['thinking']
    
    return jsonify({
        'response': response,
        'thinking': thinking
    })

@app.route('/api/thinking')
def get_thinking():
    """Get current thinking process"""
    return jsonify({
        'thinking': '\n'.join([t['thought'] for t in ink_engine.memory.thinking_log[-10:]])
    })

@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve generated images"""
    return send_from_directory('cache/images', filename)

def open_browser():
    """Open browser after server starts"""
    time.sleep(1.5)
    webbrowser.open('http://127.0.0.1:5000')

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🖋️  INK AI - Intelligent Neural Knowledge")
    print("="*50)
    print("📡 Starting web server...")
    print("🌐 Opening browser...")
    print("="*50 + "\n")
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    app.run(host='0.0.0.0', port=5000, debug=False)
