import os
import tempfile
import re
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv, find_dotenv
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

# Load environment variables
load_dotenv(find_dotenv())
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
MODEL_NAME = "gemma2-9b-it"  

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Define author styles
AUTHOR_STYLES = {
    "Dr. Seuss": "Use rhyming couplets, made-up words, and a bouncy, playful rhythm. Include short, simple sentences with fun repetition.",
    "Roald Dahl": "Use witty language, dark humor, and quirky character descriptions. Include made-up words and a slightly mischievous tone.",
    "Eric Carle": "Use simple, descriptive language with short sentences and repetitive patterns. Focus on sensory details and nature.",
    "Maurice Sendak": "Use dreamlike, slightly mysterious language with a touch of the wild. Include emotional undertones while keeping it accessible.",
    "Beatrix Potter": "Use gentle, proper language with British sensibilities. Include detailed descriptions of settings and animals with anthropomorphic qualities.",
    "A.A. Milne": "Use whimsical, thoughtful language with a philosophical undertone that children can understand. Include gentle humor and friendship themes.",
    "Shel Silverstein": "Use playful, unconventional language with surprising twists. Include imaginative scenarios and subtle life lessons.",
    "Mark Twain": "Use folksy, colloquial language with a distinctive American voice. Include clever wit, satirical observations, and regional dialect when appropriate. Maintain a warm narrative tone with touches of humor and occasional social commentary simplified for children.",
    "L. Frank Baum": "Use imaginative, whimsical descriptions of fantastical places and characters. Include straightforward, accessible language with a sense of wonder and adventure. Balance lighthearted moments with gentle lessons about courage, friendship, and self-discovery.",
    "Enid Blyton": "Use bright, enthusiastic language with a distinctly British tone. Include cozy descriptions of food, nature, and friendship. Focus on adventure and mystery with clear distinctions between good and bad characters. Incorporate exclamations and a sense of excitement about ordinary pleasures.",

    # Western Authors
    "J.K. Rowling": "Use richly descriptive language with British English expressions. Include magical elements, clever wordplay, and a mix of wonder and everyday life. Balance humor and serious moments while building a sense of mystery.",
    "Rick Riordan": "Use fast-paced, contemporary dialogue with pop culture references. Include first-person narration with sarcastic humor, mythology-inspired adventures, and relatable teen perspectives on extraordinary events.",
    "Lemony Snicket": "Use sophisticated vocabulary with parenthetical definitions. Include a pessimistic, foreboding narrator who directly addresses the reader. Add dark humor and mysterious asides.",
    "Kate DiCamillo": "Use lyrical, emotionally resonant language. Include philosophical questions embedded in simple narratives. Focus on themes of hope, healing, and unexpected friendships with sparse but meaningful descriptions.",
    "E.B. White": "Use clear, elegant prose with precise descriptions of nature and animal behavior. Include gentle humor and thoughtful observations about life with a hint of melancholy.",
    "Beverly Cleary": "Use straightforward, accessible language that captures authentic childhood experiences. Include realistic dialogue, gentle humor, and everyday situations that highlight the importance of family relationships.",
    "Judy Blume": "Use conversational, honest language that addresses real issues children face. Include first-person narration that captures authentic thoughts and emotions of pre-teens and teens.",
    "Louis Sachar": "Use clever plot structures with interconnected storylines. Include deadpan humor, unexpected twists, and detailed backstories within a seemingly simple narrative.",
    "C.S. Lewis": "Use rich descriptive language with British expressions and biblical allegories. Include talking animals, moral lessons, and magical journeys between worlds.",
    "Lois Lowry": "Use precise language with emotional depth. Include thoughtful explorations of complex social issues through accessible narratives and memorable characters.",
    "Jacqueline Wilson": "Use first-person narration that captures authentic children's voices. Include difficult subjects addressed with honesty and warmth, focusing on family dynamics and friendship.",
    
    # Indian Authors
    "Ruskin Bond": "Use simple, evocative language with rich natural imagery of the Indian Himalayas. Include gentle humor, warm human connections, and a deep appreciation for small joys and everyday encounters.",
    "Sudha Murty": "Use straightforward, value-based storytelling with Indian cultural references. Include moral lessons through everyday situations, focusing on kindness, simplicity, and the wisdom of ordinary people.",
    "R.K. Narayan": "Use gentle humor and precise observations of small-town Indian life. Include cultural specificity of South India with universal emotional themes, creating memorable eccentric characters.",
    "Anushka Ravishankar": "Use playful, nonsensical rhymes with Indian cultural references. Include innovative wordplay, surprising imagery, and a sense of joyful absurdity.",
    "Paro Anand": "Use bold, contemporary language addressing real issues Indian children face. Include diverse characters from different social backgrounds and thoughtful exploration of challenging topics with sensitivity.",
    "Subhadra Sen Gupta": "Use engaging historical narratives that bring Indian history to life. Include vivid details of different time periods with a focus on everyday life alongside significant historical events.",
    "Anupa Lal": "Use simple language infused with Indian folktales and mythology. Include magical elements from diverse Indian traditions with moral lessons embedded in entertaining stories.",
    "Deepa Agarwal": "Use rich descriptions of diverse Indian landscapes and cultures. Include strong female characters, traditional folktales with modern sensibilities, and themes of courage and determination.",
    "Aabid Surti": "Use humorous, quirky storytelling with environmental themes. Include distinctive characters and social messages delivered through entertaining narratives.",
    "Jerry Pinto": "Use perceptive, emotionally nuanced language exploring family relationships. Include authentic dialogue reflecting Indian English and regional expressions with a blend of humor and poignancy.",
    
    # More International Authors
    "Antoine de Saint-Exupéry": "Use philosophical and poetic language with childlike wonder. Include metaphorical storytelling with deeper meanings for adults, focusing on themes of innocence, human connection, and what truly matters in life.",
    "Astrid Lindgren": "Use energetic, rebellious characters with a celebration of childhood freedom. Include humor, adventure, and a touch of the absurd while respecting children's independence and capabilities.",
    "Tove Jansson": "Use whimsical, philosophical language with Scandinavian sensibilities. Include quirky characters with distinctive personalities, cozy settings, and subtle life wisdom.",
    "Hans Christian Andersen": "Use lyrical, sometimes melancholic language with vivid imagery. Include moral teachings, emotional depth, and elements of both beauty and suffering.",
    "Michael Ende": "Use richly imaginative fantasy worlds with philosophical depth. Include themes of time, creativity, and the power of stories with both whimsical and serious elements.",
    "Khaled Hosseini (adapted for children)": "Use emotionally resonant language with cultural specificity of Afghanistan. Include themes of friendship, loyalty, and redemption through vivid sensory details and powerful relationships.",
    "Pablo Neruda (children's works)": "Use sensory-rich, poetic language celebrating ordinary objects. Include surprising metaphors, appreciation for nature, and a sense of wonder at the everyday world.",
    "Cornelia Funke": "Use atmospheric, detailed fantasy worlds with German folklore influences. Include brave characters, magical elements, and themes of courage, books, and imagination.",
    "Norton Juster": "Use clever wordplay, puns, and literal interpretations of figurative language. Include philosophical concepts presented through absurd adventures and memorable characters.",
    "Tetsuko Kuroyanagi": "Use gentle, observant storytelling focused on school experiences. Include Japanese cultural elements, the importance of individuality, and the impact of empathetic teaching.",
    "Cao Wenxuan": "Use lyrical descriptions of rural Chinese life with poetic language. Include themes of resilience, nature's beauty and harshness, and coming-of-age experiences in challenging circumstances."
}

def process_pdf(pdf_file):
    """Process uploaded PDF file and create vector store"""
    # Save uploaded file to temporary location
    temp_dir = tempfile.TemporaryDirectory()
    temp_path = os.path.join(temp_dir.name, "uploaded.pdf")
    
    # Handle the file based on whether it's a file path or file object
    if hasattr(pdf_file, 'name'):
        # This is a Gradio file object
        pdf_file_path = pdf_file.name
        # Just use the file that Gradio already saved
        temp_path = pdf_file_path
    else:
        # If it's bytes data
        with open(temp_path, "wb") as f:
            f.write(pdf_file)
    
    # Load PDF
    loader = PyPDFLoader(temp_path)
    documents = loader.load()
    
    # Combine all pages into a single text for initial chapter detection
    full_text = " ".join([doc.page_content for doc in documents])
    
    # Extract chapters/sections using improved detection methods
    chapters = detect_chapters(documents, full_text)
    
    # If no chapters were detected, try alternative methods
    if len(chapters) <= 1:
        chapters = fallback_chapter_detection(documents)
    
    # Create text chunks for vector storage
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)
    
    # Create vector store
    vector_store = FAISS.from_documents(text_chunks, embedding_model)
    
    temp_dir.cleanup()
    return vector_store, chapters

def detect_chapters(documents, full_text):
    """More robust chapter detection with multiple pattern recognition approaches"""
    chapters = {}
    
    # Common chapter heading patterns
    chapter_patterns = [
        r'(?:CHAPTER|Chapter)\s+([IVXLCDM0-9]+|\d+)(?:[:\.\s]+([^\n]+))?',  # Chapter I, Chapter 1, etc.
        r'(?:SECTION|Section)\s+([IVXLCDM0-9]+|\d+)(?:[:\.\s]+([^\n]+))?',  # Section I, Section 1, etc.
        r'(?:\n|\s|^)([IVXLCDM]+)(?:\.|:|\s)\s+([^\n]+)',  # Roman numerals like I. Title, II. Title
        r'\n\s*(\d+\.)\s+([^\n]+)',  # Numbered sections like "1. Title"
    ]
    
    # First pass: identify chapter boundaries
    chapter_positions = []
    
    for pattern in chapter_patterns:
        for match in re.finditer(pattern, full_text):
            # Get the position of the match in the full text
            start_pos = match.start()
            chapter_num = match.group(1)
            chapter_title = match.group(2) if len(match.groups()) > 1 and match.group(2) else f"Chapter {chapter_num}"
            chapter_title = chapter_title.strip()
            
            chapter_positions.append((start_pos, f"Chapter {chapter_num}: {chapter_title}"))
    
    # Sort by position in document
    chapter_positions.sort(key=lambda x: x[0])
    
    # If we found chapter markers, divide the content accordingly
    if chapter_positions:
        # Process text between chapter markers
        current_page_index = 0
        accumulated_text_length = 0
        current_chapter_index = 0
        chapter_content = []
        
        # Go through each page and assign it to the appropriate chapter
        for i, doc in enumerate(documents):
            text = doc.page_content
            text_length = len(text)
            
            # Add page content to accumulated text to track position
            current_text_position = accumulated_text_length
            accumulated_text_length += text_length + 1  # +1 for the space we added between pages
            
            # Check if we've crossed into a new chapter
            while (current_chapter_index < len(chapter_positions) - 1 and 
                   accumulated_text_length > chapter_positions[current_chapter_index + 1][0]):
                # Save current chapter content
                if chapter_content:
                    chapters[chapter_positions[current_chapter_index][1]] = "\n".join(chapter_content)
                
                # Move to next chapter
                current_chapter_index += 1
                chapter_content = []
            
            # Add current page to current chapter
            if current_chapter_index < len(chapter_positions):
                chapter_content.append(text)
            else:
                # We're past the last detected chapter, add to "Appendix"
                if "Appendix" not in chapters:
                    chapters["Appendix"] = text
                else:
                    chapters["Appendix"] += "\n" + text
        
        # Add the last chapter content
        if chapter_content and current_chapter_index < len(chapter_positions):
            chapters[chapter_positions[current_chapter_index][1]] = "\n".join(chapter_content)
        
        return chapters
    else:
        # Use the original method as fallback but with improved detection
        # This code runs if we didn't find explicit chapter markers
        current_chapter = None
        chapter_content = []
        chapter_count = 0
        
        for i, page in enumerate(documents):
            text = page.page_content
            
            # Check for chapter indicators at the beginning of pages
            chapter_indicator = False
            first_lines = text.strip().split("\n")[:3]  # Check first few lines
            
            for line in first_lines:
                line = line.strip()
                if (re.match(r'^(?:CHAPTER|Chapter|SECTION|Section)\s+\w+', line) or
                    re.match(r'^[IVXLCDM]+\.?\s+\w+', line) or  # Roman numerals
                    re.match(r'^\d+\.?\s+\w+', line)):  # Arabic numerals
                    chapter_indicator = True
                    chapter_title = line
                    break
            
            if chapter_indicator:
                # Save previous chapter if it exists
                if current_chapter is not None and chapter_content:
                    chapters[current_chapter] = "\n".join(chapter_content)
                
                # Start new chapter
                chapter_count += 1
                current_chapter = f"Chapter {chapter_count}: {chapter_title}"
                chapter_content = [text]
            else:
                if current_chapter is None:
                    # No chapter detected yet, start with Chapter 1
                    chapter_count += 1
                    current_chapter = f"Chapter {chapter_count}: Introduction"
                    chapter_content = [text]
                else:
                    # Continue with current chapter
                    chapter_content.append(text)
        
        # Add final chapter
        if current_chapter is not None and chapter_content:
            chapters[current_chapter] = "\n".join(chapter_content)
        
        return chapters

def fallback_chapter_detection(documents):
    """Fallback method when no clear chapter structure is identified"""
    chapters = {}
    
    # Try page-based chunking with intelligent merging
    current_section = []
    section_count = 0
    
    # Analyze text density to identify potential chapter breaks
    for i, page in enumerate(documents):
        text = page.page_content.strip()
        
        # Heuristics to detect new section/chapter:
        # 1. Less than 100 characters could be a title page or chapter header
        # 2. First few lines contain capitalized words like "CHAPTER"
        # 3. Significant blank space at the top
        is_new_section = False
        
        if len(text) < 100:
            is_new_section = True
        elif i > 0:
            first_lines = text.split("\n")[:3]
            first_line = first_lines[0] if first_lines else ""
            
            # Check if first line looks like a title
            if (first_line.isupper() or 
                first_line.istitle() or 
                re.search(r'(CHAPTER|Chapter|PART|Part|BOOK|Book|SECTION|Section)', first_line)):
                is_new_section = True
        
        # First page is always the start of a section
        if i == 0:
            is_new_section = True
        
        if is_new_section and current_section:
            # Save previous section
            section_count += 1
            section_title = f"Section {section_count}"
            
            # Try to extract a title from the first page of the section
            first_page_text = current_section[0].strip()
            first_lines = first_page_text.split("\n")[:3]
            
            for line in first_lines:
                line = line.strip()
                if len(line) > 0 and len(line) < 100:  # Reasonable title length
                    section_title = f"Section {section_count}: {line}"
                    break
            
            chapters[section_title] = "\n".join(current_section)
            current_section = [text]
        else:
            current_section.append(text)
    
    # Add final section
    if current_section:
        section_count += 1
        section_title = f"Section {section_count}"
        
        # Try to extract a title from the first page
        first_page_text = current_section[0].strip()
        first_lines = first_page_text.split("\n")[:3]
        
        for line in first_lines:
            line = line.strip()
            if len(line) > 0 and len(line) < 100:
                section_title = f"Section {section_count}: {line}"
                break
        
        chapters[section_title] = "\n".join(current_section)
    
    # If we still have too few chapters, divide by page groups
    if len(chapters) <= 1:
        page_chunks = 10  # Group pages in chunks of 10
        chapters = {}
        
        for i in range(0, len(documents), page_chunks):
            chunk_pages = documents[i:i+page_chunks]
            chunk_text = "\n".join([page.page_content for page in chunk_pages])
            chapters[f"Part {i//page_chunks + 1}: Pages {i+1}-{min(i+page_chunks, len(documents))}"] = chunk_text
    
    return chapters

def load_llm():
    """Create the LLM instance"""
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=MODEL_NAME,
        temperature=0.7,  # Slightly higher for more creative rewriting
        max_tokens=1024   # Increased for longer responses
    )
    return llm

def create_qa_chain(vector_store, author_style):
    """Create a QA chain with custom prompt for children's style adaptation"""
    custom_prompt = f"""
    You are an expert children's book adapter who rewrites text in the style of famous children's authors.
    
    Based on the provided context, identify the main events, characters, and ideas, then rewrite them in the style of the specified author.
    
    Style to use: {AUTHOR_STYLES.get(author_style, "simple and child-friendly language")}
    
    Context: {{context}}
    Query about: {{question}}
    
    Your rewrite should:
    1. Be appropriate for children aged 5-10
    2. Maintain the core story and concepts from the original text
    3. Remove any violence, scary elements, or adult themes
    4. Use simpler vocabulary while still being engaging
    5. Perfectly match the requested author's style
    6. Be creative and charming

    Rewritten text:
    """
    
    prompt = PromptTemplate(template=custom_prompt, input_variables=["context", "question"])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=load_llm(),
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={'k': 5}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    
    return qa_chain

def process_chapter(vector_store, chapter_text, author_style):
    """Process a specific chapter with the selected author style"""
    qa_chain = create_qa_chain(vector_store, author_style)
    
    # For very long chapters, we need to summarize or process in parts
    if len(chapter_text) > 5000:
        # Process the chapter in parts if it's very long
        parts = [chapter_text[i:i+5000] for i in range(0, len(chapter_text), 4000)]
        results = []
        
        for i, part in enumerate(parts):
            response = qa_chain.invoke({
                'query': f"This is part {i+1} of {len(parts)} of a chapter. Please rewrite this part in the style specified, maintaining continuity: {part}"
            })
            results.append(response["result"])
        
        return "\n\n".join(results)
    else:
        # Process the entire chapter at once if it's a reasonable length
        response = qa_chain.invoke({
            'query': f"Please rewrite this chapter content in the style specified: {chapter_text}"
        })
        
        return response["result"]

# Gradio UI Functions
def upload_pdf(pdf_file):
    """Handle PDF upload in Gradio"""
    if pdf_file is None:
        return None, {}, gr.Dropdown(choices=[], interactive=False), "Please upload a PDF file."
    
    try:
        vector_store, chapters = process_pdf(pdf_file)
        chapter_names = list(chapters.keys())
        return vector_store, chapters, gr.Dropdown(choices=chapter_names, interactive=True), f"PDF processed successfully! Found {len(chapter_names)} chapters/sections."
    except Exception as e:
        return None, {}, gr.Dropdown(choices=[], interactive=False), f"Error processing PDF: {str(e)}"

def generate_story(vector_store, chapter_name, author_style, chapters_data):
    """Generate the rewritten story based on user selections"""
    if vector_store is None:
        return "Please upload a PDF first."
    
    if not chapter_name:
        return "Please select a chapter to continue."
    
    if not author_style:
        return "Please select an author style to continue."
        
    try:
        chapter_content = chapters_data.get(chapter_name, "")
        if not chapter_content:
            return "Chapter content not found. Please try another chapter."
        
        result = process_chapter(vector_store, chapter_content, author_style)
        return result
    except Exception as e:
        return f"Error generating story: {str(e)}"

# Create Gradio Interface
def create_interface():
    with gr.Blocks(title="Children's Book Style Converter") as app:
        gr.Markdown("# 📚 Classic to Children's Book Style Converter")
        gr.Markdown("Upload a PDF book and convert chapters into children-friendly versions in the style of famous children's authors!")
        
        # Store state variables
        vector_store_state = gr.State(None)
        chapters_data_state = gr.State({})
        
        with gr.Row():
            with gr.Column(scale=1):
                pdf_input = gr.File(label="Upload PDF Book", file_types=[".pdf"])
                upload_button = gr.Button("Process PDF")
                status_output = gr.Textbox(label="Status", interactive=False)
                
                chapter_dropdown = gr.Dropdown(
                    label="Select Chapter", 
                    choices=[], 
                    interactive=False
                )
                
                author_dropdown = gr.Dropdown(
                    label="Select Author Style",
                    choices=list(AUTHOR_STYLES.keys()),
                    value="Dr. Seuss"
                )
                
                generate_button = gr.Button("Generate Story")
            
            with gr.Column(scale=2):
                story_output = gr.Textbox(
                    label="Children's Version", 
                    lines=15, 
                    interactive=False
                )
        
        # Set up event handlers
        upload_button.click(
            fn=upload_pdf,
            inputs=[pdf_input],
            outputs=[vector_store_state, chapters_data_state, chapter_dropdown, status_output]
        )
        
        generate_button.click(
            fn=generate_story,
            inputs=[vector_store_state, chapter_dropdown, author_dropdown, chapters_data_state],
            outputs=[story_output]
        )
    
    return app

# Main execution
if __name__ == "__main__":
    app = create_interface()
    app.launch(share=True)