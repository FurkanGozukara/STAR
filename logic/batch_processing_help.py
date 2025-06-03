"""
Batch Processing Help Content
Contains the informational content for the batch processing workflow.
"""

import gradio as gr

def create_batch_processing_help():
    """
    Creates the batch processing help content split into 3 columns.
    Returns a Gradio component that can be added to the interface.
    """
    
    # Column 1: Overview and File Discovery
    column1_content = """
### 📖 Batch Processing Workflow

**Batch processing** allows you to upscale multiple videos automatically using the same settings.

#### **Step 1: File Discovery**
- Scans the input folder for video files (mp4, avi, mov, mkv, wmv, flv, webm, m4v)
- Each video will be processed with the current Main tab settings
- Supports nested folder structures
- Automatically detects video formats

#### **💡 Getting Started:**
- Set your input folder path
- Configure output folder location
- Choose your processing settings in Main tab
- Click "Start Batch Upscaling"
"""

    # Column 2: Prompt Selection and Processing
    column2_content = """
#### **Step 2: Prompt Selection (Priority Order)**
1. **📄 Prompt Files** (if enabled): Looks for `filename.txt` next to each video
   - `video.mp4` → uses `video.txt` as prompt if it exists
   - **Highest Priority** - overrides user prompt and auto-caption

2. **🤖 Auto-Caption** (if enabled): Generates description using AI
   - Only used if no prompt file exists
   - Uses CogVLM2 model for intelligent descriptions

3. **👤 User Prompt**: Uses the prompt from Main tab
   - **Lowest Priority** - used as fallback
   - Applied when no other prompts available

#### **🔄 Processing Flow:**
- Videos processed sequentially
- Same settings applied to all videos
- Progress tracking for each video
"""

    # Column 3: Output and Management
    column3_content = """
#### **Step 3: Output Organization**
- Creates organized folder structure: `Output Folder/VideoName/VideoName.mp4`
- Optionally saves processing metadata, frames, and comparison videos
- Can skip existing outputs to resume interrupted batches
- Maintains original video names and structure

#### **Step 4: Caption Management**
- **Save Captions**: Auto-saves generated captions as `filename.txt` in input folder
- **Never Overwrites**: Existing prompt files are preserved
- **Reusable**: Saved captions become prompt files for future runs

#### **💡 Pro Tips:**
- Use **Skip Existing** to resume interrupted batch jobs
- **Save Captions** creates reusable prompts for consistent results
- **Prompt Files** allow per-video customization (e.g., "A red car" for car1.mp4)
- Enable **Auto-Caption** for videos without custom prompts
- Test settings on single video first
"""

    # Create the 3-column layout
    with gr.Accordion("How Batch Processing Works", open=False):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(column1_content)
            with gr.Column(scale=1):
                gr.Markdown(column2_content)
            with gr.Column(scale=1):
                gr.Markdown(column3_content) 