
import cohere
import os
import re

co = cohere.Client(os.environ["COHERE_API_KEY"])


def extract_noun_phrases(text):
    """
    Extracts noun phrases from the given text based on the 'Noun Phrases:' label.

    Args:
        text: The input text string.

    Returns:
        A list of noun phrases.
    """
    # Match the line starting with "Noun Phrases:" and extract the list
    match = re.search(r'Noun Phrases:\s*\[([^\]]+)\]', text)
    if match:
        # Extract phrases, split them, and strip any surrounding whitespace
        phrases = [phrase.strip().strip("'") for phrase in match.group(1).split(',')]
        return phrases
    return []

def generate_story_from_captions(captions_file):
    """
    Generates a story using noun phrases from the captions file.

    Args:
        captions_file: The path to the captions file.

    Returns:
        The generated story.
    """
    with open(captions_file, 'r') as f:
        captions_text = f.read()

    all_noun_phrases = []
    for image_caption in captions_text.split('\n\n'):  # Split by double newline for each image block
        noun_phrases = extract_noun_phrases(image_caption)
        all_noun_phrases.extend(noun_phrases)
        

    # Combine all noun phrases, removing duplicates
    entity_string = ' '.join(set(all_noun_phrases))
    # print(entity_string)

    # Provide a structured story prompt
    prompt = f"Craft a captivating narrative centered around the following key concepts: {entity_string}. The story should have a clear beginning, a rising action that builds tension, a climactic moment, and a satisfying resolution. Ensure that each of these concepts plays a significant role in driving the plot forward and shaping the characters' journeys. Feel free to add creative twists and turns to make the story engaging and memorable. Finish the story in less than 500 words"

    response = co.generate(
        model='command-light',
        prompt=prompt,
        max_tokens=500,  # Increased length
        temperature=1.0,  # Increased creativity
        k=0,
        p=0.7,
        frequency_penalty=0.3,
        presence_penalty=0,
        stop_sequences=[],
        return_likelihoods='NONE'
    )

    return response.generations[0].text

if __name__ == "__main__":
    captions_file = "C:\\Users\\krish\\OneDrive\\Desktop\\main_final_year_project\\graphs\\captions_and_entities.txt"
    story = generate_story_from_captions(captions_file)
    print(story)
