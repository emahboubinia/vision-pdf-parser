#!/usr/bin/env python3
"""
Advanced PDF to Text Converter with AI-Powered Image Description

This script converts PDF documents into structured text format, intelligently processing
both text and embedded images. It leverages an external Vision AI model to generate
descriptive text for visuals, creating a more accessible and informative output for LLMs.
"""
# LLM required Libraries
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
import urllib.parse

# Libraries to Saving Image form PDF file
from PIL import Image
import io

# PDF reading library
import pymupdf 

# Basic Libraries
import os
import sys
import pathlib
import logging
from typing import List, Dict


# Variables
pdf_path = ""                    # Path to the pdf you want to process
model_path = ""                  # Path to the vision Model
clip_model_path = ""             # Path to the clip model
temp_images_dir = "images_tmp"

if pdf_path == "" or model_path == "" or clip_model_path == "" or temp_images_dir == "":
    raise Exception(f"Variable(s) isn't (aren't) set(s) Please review the variable in the {__file__}")

image_describing_prompt = """
Goal
Provide extremely detailed, comprehensive descriptions of images with particular focus on charts, molecular pathways, diagrams, and information-dense visuals. Extract and describe every visible element, relationship, label, value, and structural component with maximum precision and completeness. Use highly technical and specific terminology appropriate to the subject matter.
Return Format
Structure your response comprehensively, prioritizing completeness over brevity. Aim for thorough detail (2048+ characters when content warrants) and include:

Image Type & Overview: Precise classification and general description
Main Components: Systematic breakdown of all major elements
Detailed Elements:

All text (exact transcription of labels, titles, legends, values, annotations, units)
All visual elements (specific shapes, colors, lines, symbols, markers)
All relationships and connections (arrows, lines, groupings)
All numerical data and measurements (exact values, ranges, scales)


Spatial Organization: Precise description of layout, positioning, and arrangement
Technical Details: Chemical formulas, mathematical equations, statistical parameters, molecular structures
Data Interpretation: Scientific meaning, relationships, and implications shown

Warning
Do not make assumptions about data not clearly visible
If text is unclear or partially obscured, state "text unclear" rather than guessing
Distinguish between what is explicitly shown versus what might be implied
For scientific content, use precise technical terminology without simplification

Context Dumps
Please analyze this image with the following considerations:
For Charts/Graphs (all types - bar, line, scatter, pie, heatmap, box plot, etc.):

Identify exact chart type, all axis labels, tick marks, scales, units of measurement
Transcribe all data series names, legend items, and their precise values
Describe specific trends, patterns, correlations, and statistical relationships
Note confidence intervals, error bars, regression lines, R-squared values, p-values
Include grid lines, annotations, callouts, and any mathematical functions displayed

For Molecular Pathways:
Identify all molecules, proteins, enzymes, genes, metabolites with exact names/symbols
Describe all reaction arrows (solid, dashed, curved), inhibition symbols (⊥, blunt), activation markers
List every pathway step, intermediate compounds, and cofactors
Note regulatory elements, feedback loops, allosteric sites, phosphorylation sites
Include enzyme commission numbers, molecular weights, concentrations if shown

For Technical Diagrams:
Describe all components and their labels
Explain connections, flows, and relationships
List any measurements, specifications, or parameters
Note any symbols, icons, or standardized notations

For Information Graphics:
Transcribe all text content verbatim
Describe visual hierarchy and organization
List all graphical elements and their purposes
Note color coding, sizing, or other visual encoding

General Instructions:
Be exhaustively thorough - no detail is too small if visible
Use maximum technical precision and domain-specific terminology
Provide exact measurements, coordinates, angles, and numerical values
Maintain systematic organization while ensuring comprehensive coverage
Cross-reference and interconnect all related elements and their relationships
"""

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pdf_converter.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def get_image_descriptions(
    vision_model_path: str,
    clip_model_path: str,
    image_paths: List[str],
    prompt: str = "Describe this image in detail.",
    max_token: int = 2048
) -> Dict[str, str]:
    """
    Get descriptions for a list of images using a Pixtral model.
    Models are loaded once, and each image is processed individually.

    Args:
        vision_model_path (str): Path to the Pixtral model (GGUF file).
        clip_model_path (str): Path to the CLIP model (mmproj file).
        image_paths (List[str]): A list of paths to the image files.
        prompt (str): The prompt to use for each image.
        max_token (int): Maximum number of tokens for each response.

    Returns:
        Dict[str, str]: A dictionary mapping each image's filename to its description.
    """
    # 1. Load the models once at the beginning for efficiency.
    print("Loading models... This may take a moment.")
    try:
        chat_handler = Llava15ChatHandler(clip_model_path=clip_model_path, verbose=False)
        llm = Llama(
            model_path=vision_model_path,
            chat_handler=chat_handler,
            n_ctx=4096,
            n_gpu_layers=-1,  # Use -1 to offload all possible layers to GPU
            verbose=False,
        )
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        return {} # Return an empty dictionary if models fail to load

    # 2. Create a dictionary to store the results.
    descriptions_dict = {}

    # 3. Loop through each image path provided in the list.
    for image_path in image_paths:
        filename = os.path.basename(image_path)
        print(f"\nProcessing image: {filename}...")

        if not os.path.exists(image_path):
            print(f"  -> Warning: File not found at '{image_path}'. Skipping.")
            descriptions_dict[filename] = "Error: File not found."
            continue

        try:
            # 4. Create a separate chat completion for each image.
            response = llm.create_chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"file://{urllib.parse.quote(image_path)}"}}
                        ],
                    }
                ],
                max_tokens=max_token,
            )
            description = response['choices'][0]['message']['content'].strip()
            
            # 5. Save the description in the dictionary with the filename as the key.
            descriptions_dict[filename] = description
            print(f"  -> Description generated for {filename}.")

        except Exception as e:
            print(f"  -> Error processing {filename}: {e}")
            descriptions_dict[filename] = f"Error: Could not process image. Details: {e}"

    # 6. Return the dictionary containing all descriptions.
    return descriptions_dict

def save_image(image_block, path:str, name:str):
    image = Image.open(io.BytesIO(image_block['image']))
    image.save(open(os.path.join(path,f"{name}.{image_block['ext']}"), "wb"), image_block['ext'])
    print(f"{name}.{image_block['ext']} were save at {path}")

def get_text(lines_object):
    '''
    This function extracts text from a list of lines in a structured format.
    Args:
        lines_object (list): A list of lines, where each line is a dictionary containing spans
                             or text.
    Returns:
        str: The extracted text as a single string.
    '''
    text = ''
    for l in lines_object:
        if 'spans' in l:
            for s in l['spans']:
                text += s['text'] + " "
        else:
            text += l['text']
    return text.strip()

def page_to_text(page_blocks,page_number:int):
    '''
    Extracts text from a page's blocks.
    Args:
        page_blocks (pymupdf.fitz.PageBlocks): The blocks of the page.
    Returns:
        str: The extracted text from the page.
    '''
    text = ''
    for block_n in range(len(page_blocks)):
        if page_blocks[block_n]['type'] == 0:  # Text block
            text += get_text(page_blocks[block_n]['lines'])
            text += '\n'
        elif page_blocks[block_n]['type'] == 1:  # Image block
            save_image(
                image_block=page_blocks[block_n], 
                path=os.path.join(temp_images_dir), 
                name=f"p{page_number}-b{page_blocks[block_n]['number']}"
            )
            text += f"[image: p{page_number}-b{page_blocks[block_n]['number']}.{page_blocks[block_n]['ext']}]\n"
    return text

def insert_image_descriptions(
    images_dir:str, 
    pdf_text:str, 
    vision_model_path: str,
    clip_model_path: str,
    prompt: str
    ):
    images_file = os.listdir(images_dir)
    images_path = [os.path.join(pathlib.Path(__file__).parent.resolve(),images_dir,i) for i in images_file]
    print(images_path)
    description_dict = get_image_descriptions(
        vision_model_path=vision_model_path, 
        clip_model_path=clip_model_path, 
        image_paths=images_path, 
        prompt=prompt
    )
    for file_name in description_dict:
        pdf_text = pdf_text.replace(file_name,description_dict[file_name])
    
    return pdf_text

def save_text(text:str, txt_name):
    with open(txt_name,"w+") as f:
        f.write(text)

    
def pdf_to_text(
        pdf_path:str,
        vision_model_path: str,
        clip_model_path: str,
        prompt: str
    ):
    doc = pymupdf.open(pdf_path) # open a document
    # Choose a page (e.g., first page is index 0)
    pages_num = len(doc)
    logger.info(f"Document has {pages_num} pages.")
    
    os.makedirs(os.path.join(pathlib.Path(__file__).parent.resolve(),temp_images_dir), exist_ok=True)
    raw_text = ""
    for i in range(pages_num):
        page = doc[i]
        page_dict = page.get_text("dict")
        raw_text += page_to_text(page_dict['blocks'],i+1)
        raw_text += "\n\n"

    process_text = insert_image_descriptions(
        images_dir=temp_images_dir,
        pdf_text=raw_text,
        vision_model_path=vision_model_path,
        clip_model_path=clip_model_path,
        prompt=prompt
    )
    txt_file_name = "".join(os.path.basename(pdf_path).split(".")[:-1])
    save_text(process_text,os.path.join(os.path.join(pathlib.Path(__file__).parent.resolve(),f"{txt_file_name}.txt")))

if __name__ == '__main__':
    pdf_to_text(pdf_path=pdf_path, vision_model_path=model_path, clip_model_path=clip_model_path,prompt= image_describing_prompt)
