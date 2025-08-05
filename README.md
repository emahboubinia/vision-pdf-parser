# Vision PDF Parser
This script transforms a complex, visual PDF into a single, comprehensive text file, making the entire document's content—including charts, diagrams, and figures—fully accessible for Large Language Models (LLMs), search indexing, or accessibility purposes.

> When you start working with Local LLM one of the main problems it that those models are mostly good with text and not PDF files or images or any other things. Although there are State-of-Art models with vision but they require a lot more resources.

> I works with a lot of scientific paper and in those kind of paper it's necessary to get the images that; and for futures project I have in mind, I need to make a fully detailed text file out of pdf so it would be easier for Local LLMs to work with it.

***I am currently learning how to work with LLMs and would really appreciate comments and suggestions.***

## Features
+ **Hybrid Extraction:** Accurately extracts both embedded text and images.

+ **AI-Powered Vision:** Leverages local VLMs to understand and describe complex visuals, not just perform OCR.

+ **Detailed Descriptions:** Uses a customizable, in-depth prompt to generate high-quality descriptions for charts, diagrams, and scientific figures.

+ **Local First:** Runs entirely on your local machine, ensuring data privacy. No cloud APIs needed.

+ **Structured Output:** Creates a clean, human-readable, and LLM-friendly text file.

## How does it work
It use [Pymupdf](https://pymupdf.readthedocs.io/en/latest/) to get structure of pdf files
1. It first extract text and save Images in a temporary directory and left the initial text with placeholder for image
2. For processing the image it use CLIP models and vision models which I will explain further.
3. Then the placeholder would be replace by description of the image
4. Then it save the text as `.txt` file

For Processing image I use `mistral-community_pixtral-12b-Q5_K_M.gguf`, you can download it from [HuggingFace](https://huggingface.co/bartowski/mistral-community_pixtral-12b-GGUF)

For CLIP model I use `mmproj-pixtral-12b-f16.gguf`, you can download it from [Hugging Face](https://huggingface.co/ggml-org/pixtral-12b-GGUF/blob/main/mmproj-pixtral-12b-f16.gguf)

I use [llama.cpp](https://github.com/ggml-org/llama.cpp) for running the models. 

## Prerequisites
1. **Python 3.8+**
2. **C compiler** (*to install Llama.cpp*)
    + Linux: gcc or clang
    + Windows: Visual Studio or MinGW
    + MacOS: Xcode
3. **llama.cpp** (Recommended):
   + although `pip install llama-cpp-python` would install it but I recommend you to install it yourself for better build and optimization and GPU acceleration. 
   + Here's the [llama.cpp repo](https://github.com/ggml-org/llama.cpp) for installation guide
   + If you are using Macbook with M1 or higher I recommend you to use [homebrew](https://brew.sh)
        
        To install Homebrew use this command:
        ```bash
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        ```
        Then you can install llama.cpp with this command:
        ```bash
        brew install llama.cpp
        ```
    
4. **Local Vision Model Files:** You must download and provide your own LLaVA-compatible GGUF model files. This script was tested with **Pixtral**.
   + A **Vision Model** (e.g., `pixtral-12b-Q5_K_M.gguf`)
   + A corresponding **CLIP Model** (e.g., `mmproj-pixtral-12b-f16.gguf`)

## Installation
1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/vision-pdf-parser.git
    ```
2. **(Recommended) Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
## Configuration
Before running, you must configure the script with the paths to your files. Open `PDF_to_text.py` and set the following variables at the top of the file:
```python
# --- USER CONFIGURATION ---
pdf_path = "path/to/your/document.pdf"       # Path to the PDF you want to process
model_path = "path/to/your/model.gguf"       # Path to the vision model GGUF file
clip_model_path = "path/to/your/clip.gguf"   # Path to the CLIP mmproj file
# --------------------------
```

## Usage 
Once configured, run the script from your terminal:
```bash
PDF_to_text.py [-h] [--vision_model_path VISION_MODEL_PATH] [--clip_model_path CLIP_MODEL_PATH] [--prompt PROMPT] pdf_path
```
```bash
python PDF_to_text.py -help
usage: PDF_to_text.py [-h] [--vision_model_path VISION_MODEL_PATH] [--clip_model_path CLIP_MODEL_PATH] [--prompt PROMPT] pdf_path

Advanced PDF to Text Converter with AI-Powered Image Description

positional arguments:
  pdf_path              Path to the PDF file

options:
  -h, --help            show this help message and exit
  --vision_model_path VISION_MODEL_PATH
                        Path to the Vision model (GGUF file)
  --clip_model_path CLIP_MODEL_PATH
                        Path to the CLIP model (GGUF file)
  --prompt PROMPT       Prompt to use for each image
```
The script will log its progress to the console and create a `pdf_converter.log` file. A temporary `images_tmp` directory will be created to store images during processing. The final output will be a `.txt` file with the same name as your input PDF, saved in the same directory.

## Customizing the AI Prompt
The quality of the image descriptions is heavily influenced by the prompt. You can modify the `image_describing_prompt` variable in the script to tailor the AI's output to your specific needs. The default prompt is optimized for detailed, technical descriptions of scientific figures.

Here's the defualt prompt:
<details>
<summary>Click to see the full default prompt</summary>

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
</details>

## TODO List
- [x] Make the CLI
- [ ] Add picture checking to save resources by not processing useless images
- [ ] Add API feature for getting picture description