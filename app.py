from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import subprocess
import uuid
import logging

# Initialize the Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'E:/AI_AB2_final no finetuning/main_final_year_project_sayanthika/static/uploads'
app.config['GRAPH_FOLDER'] = 'E:/AI_AB2_final no finetuning/main_final_year_project_sayanthika/graphs'

# Ensure required directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)

@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/process-images', methods=['POST'])
def process_images():
    """Handles image uploads, processing, and returning results."""
    uploaded_files = request.files.getlist('files[]')
    logging.info(f"Number of files received: {len(uploaded_files)}")

    saved_files = []
    try:
        # Save uploaded images with unique filenames
        for file in uploaded_files:
            unique_filename = f"{uuid.uuid4()}_{file.filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            saved_files.append(file_path)
            logging.info(f"Saved file: {file_path}")

        # Run the external processing script
        result = subprocess.run(
            ['python', 'latest.py'],  # Command to run latest.py
            stdout=subprocess.PIPE,    # Capture standard output
            stderr=subprocess.PIPE,    # Capture errors
            text=True                  # Decode output as text
        )

        # Check if the script execution was successful
        if result.returncode != 0:
            raise Exception(result.stderr)

        # Paths for the output files in the graphs folder (outside static)
        captions_file = os.path.join(app.config['GRAPH_FOLDER'], 'captions_and_entities.txt')
        story_file = os.path.join(app.config['GRAPH_FOLDER'], 'generated_story.txt')
        graph_file = os.path.join(app.config['GRAPH_FOLDER'], 'knowledge_graph.jpg')

        # Read the content of the text files
        with open(captions_file, 'r') as f:
            captions_content = f.read()
        
        with open(story_file, 'r') as f:
            story_content = f.read()

        # Return JSON response with file content and URL for the graph
        return jsonify({
            'captions': captions_content,
            'story': story_content,
            'graph_url': f"/graphs/knowledge_graph.jpg"
        })

    except Exception as e:
        # Cleanup on error
        for file_path in saved_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        logging.error(f"Error processing images: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serves uploaded files (not used directly in the discussed workflow)."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/graphs/<filename>')
def serve_graphs(filename):
    """Serves the files from the custom graphs folder."""
    return send_from_directory(app.config['GRAPH_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

