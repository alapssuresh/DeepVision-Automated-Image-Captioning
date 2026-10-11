import os
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from PIL import Image
import nltk
import networkx as nx
import matplotlib.pyplot as plt
import cohere

# Initialize the Cohere API client
co = cohere.Client('p5w84ulMRkqgd6VJqJIIr3sg5gmfNGhA207dseGQ')

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('punkt_tab')

from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag, RegexpParser

# Load models, feature extractors, and tokenizers
# For captioning
caption_model_name = "nlpconnect/vit-gpt2-image-captioning"
caption_model = VisionEncoderDecoderModel.from_pretrained(caption_model_name)
caption_feature_extractor = ViTFeatureExtractor.from_pretrained(caption_model_name)
caption_tokenizer = AutoTokenizer.from_pretrained(caption_model_name)

# Function to classify individual words
def classify_word(word, pos_tag):
    if pos_tag.startswith("NN"):
        synsets = wordnet.synsets(word, pos=wordnet.NOUN)
        
    if pos_tag.startswith("VB"):
        return "Activity"
    return "Unknown"

# Function to classify the caption
def classify_caption(caption):
    tokens = nltk.word_tokenize(caption)
    pos_tags = nltk.pos_tag(tokens)
    classifications = {}
    for word, pos in pos_tags:
        classifications[word] = classify_word(word, pos)
    return classifications

# Function to extract noun and verb phrases
def extract_phrases(caption):
    tokens = word_tokenize(caption)
    pos_tags = pos_tag(tokens)
    grammar = r"""
        NP: {<NN.*>}             
        VP: {<VB.><VBN|VBG|VB.>*}         
        VP: {<VB.*>}                         
        PP: {<IN>}       
    """
    chunk_parser = RegexpParser(grammar)
    chunked_tree = chunk_parser.parse(pos_tags)

    def extract_chunks(tree, label):
        return [" ".join(word for word, pos in subtree.leaves()) for subtree in tree.subtrees(filter=lambda t: t.label() == label)]

    noun_phrases = extract_chunks(chunked_tree, "NP")
    verb_phrases = extract_chunks(chunked_tree, "VP")
    prep_phrases = extract_chunks(chunked_tree, "PP")

    return noun_phrases, verb_phrases, prep_phrases

# Function to generate a caption for an input image
def generate_caption(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        pixel_values = caption_feature_extractor(images=image, return_tensors="pt").pixel_values
        outputs = caption_model.generate(pixel_values)
        caption = caption_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        return f"Error processing image {image_path}: {e}"

# Function to process multiple images
def process_images(image_paths):
    results = {}
    for idx, path in enumerate(image_paths):
        caption = generate_caption(path)
        classifications = classify_caption(caption)
        noun_phrases, verb_phrases, prep_phrases = extract_phrases(caption)
        results[f"Image {idx + 1}"] = {
            "caption": caption,
            "classifications": classifications,
            "noun_phrases": noun_phrases,
            "verb_phrases": verb_phrases,
            "prep_phrases": prep_phrases
        }
    return results

def build_knowledge_graph_nx(results):
    graph = nx.DiGraph()

    for Image, data in results.items():
        noun_phrases = data["noun_phrases"]
        verb_phrases = data["verb_phrases"]
        prep_phrases = data["prep_phrases"]

        # Add nodes (noun phrases)
        for noun in noun_phrases:
            graph.add_node(noun, color="blue")

        # Add edges (verb phrases as relationships)
        if verb_phrases:
            for verb in verb_phrases:
                if len(noun_phrases) > 1:
                    for i in range(len(noun_phrases) - 1):
                        graph.add_edge(noun_phrases[i], noun_phrases[i + 1], relationship=verb)
        elif prep_phrases:
            for prep in prep_phrases:
                if len(noun_phrases) > 1:
                    for i in range(len(noun_phrases) - 1):
                        graph.add_edge(noun_phrases[i], noun_phrases[i + 1], relationship=prep)

    return graph

def visualize_knowledge_graph_nx(graph, output_path):
    plt.figure(figsize=(12, 10))

    # Extract edge labels for relationships
    edge_labels = nx.get_edge_attributes(graph, 'relationship')

    # Assign node colors
    node_colors = [data["color"] for _, data in graph.nodes(data=True)]

    # Draw the graph
    pos = nx.spring_layout(graph, seed=42, k=1.5)
    nx.draw_networkx_edges(graph, pos, edge_color="orange", arrows=True, width=2)
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=2500)
    nx.draw_networkx_labels(graph, pos, font_size=10, font_color="white")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color="red", font_size=8)

    plt.title("Knowledge Graph", fontsize=20, fontweight='bold', pad=20)

    # Save the graph
    plt.savefig(output_path)
    plt.close()

# Function to generate a story using Cohere
def generate_story(entities, story_output_path):
    entity_string = ', '.join(entities)  # Combine entities, noun phrases, and verb phrases
    print("entities - ", entity_string)
    
    # Provide a structured story prompt
    prompt = (f"Write a short story which has a meaningful plot with a good beginning, middle part and also a satisfactory ending. "
              f"The story must  and should incorporate the following: {entity_string}.")
    
    response = co.generate(
        model='command-xlarge-nightly',
        prompt=prompt,
        max_tokens=400,
        temperature=1.0,
        k=0,
        p=0.7,
        frequency_penalty=0.3,
        presence_penalty=0,
        stop_sequences=[],
        return_likelihoods='NONE'
    )

    story = response.generations[0].text

    # Save the story to a file
    with open(story_output_path, 'w') as story_file:
        story_file.write(story)

# Function to process story generation from file
def process_story_generation(file_path, output_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Initialize variables to store noun phrases and relations
        noun_phrases = []
        relations = []

        for line in lines:
            if line.startswith("Noun Phrases:"):
                # Extract noun phrases and add them to the list
                noun_phrases.extend(eval(line.split("Noun Phrases:")[1].strip()))
            elif line.startswith("Relation:"):
                # Extract relations and add them to the list
                relations.extend(eval(line.split("Relation:")[1].strip()))

        # Combine noun phrases and relations into a single entity string
        entity_string = ', '.join(noun_phrases + relations)

        # Print entity_string to check
        print("Entity String:", entity_string)

        # Generate story with the entity string
        generate_story(entity_string, output_path)

    except Exception as e:
        print(f"Error processing story generation: {e}")

# Main function to process all images in the `static/uploads` folder and clean up
def process_and_cleanup():
    upload_folder = 'static/uploads'

    # Get a list of all image paths in the 'static/uploads' folder
    image_paths = [os.path.join(upload_folder, f) for f in os.listdir(upload_folder) if os.path.isfile(os.path.join(upload_folder, f))]
    
    if not image_paths:
        print("No images found in the 'static/uploads' folder.")
        return

    print(f"Found {len(image_paths)} images to process.")

    # Process images and save results
    results = process_images(image_paths)

    # Create output directory if not exists
    os.makedirs("graphs", exist_ok=True)

    # Save captions and entities in a text file
    captions_entities_path = "graphs/captions_and_entities.txt"
    with open(captions_entities_path, "w") as file:
        for img, result in results.items():
            file.write(f"{img}:\n")
            file.write(f"Caption: {result['caption']}\n")
            file.write(f"Noun Phrases: {result['noun_phrases']}\n")
            
            # Determine the relationship
            if result['verb_phrases']:
                relation = result['verb_phrases']
            elif result['prep_phrases']:
                relation = result['prep_phrases']
            else:
                relation = "None"

            file.write(f"Relation: {relation}\n\n")

    # Build and save the knowledge graph
    knowledge_graph = build_knowledge_graph_nx(results)
    visualize_knowledge_graph_nx(knowledge_graph, "graphs/knowledge_graph.jpg")

    # Generate and save story
    story_output_path = "graphs/generated_story.txt"
    process_story_generation(captions_entities_path, story_output_path)

    print("Captions and entities saved in 'graphs/captions_and_entities.txt'.")
    print("Knowledge graph saved as 'graphs/knowledge_graph.jpg'.")
    print("Generated story saved in 'graphs/generated_story.txt'.")

    # Delete all images from the 'static/uploads' folder
    for img_path in image_paths:
        os.remove(img_path)
        print(f"Deleted {img_path} from 'static/uploads'.")

if __name__ == "__main__":
    process_and_cleanup()
