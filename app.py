import streamlit as st
from openai import OpenAI
from googleapiclient.discovery import build
import re
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube
import os
import whisper
import time
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from datetime import datetime
import tempfile
import requests


# Load environment variables
load_dotenv()

# Initialize OpenAI client with better error handling
def init_openai_client():
    """Initialize OpenAI client with proper error handling"""
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        if not api_key:
            raise ValueError("OpenAI API key not found in secrets")
        return OpenAI(api_key=api_key)
    except Exception as e:
        st.error("Error initializing OpenAI client. Please check your API key in Streamlit secrets.")
        st.error(f"Error details: {str(e)}")
        return None

# Initialize the OpenAI client
client = init_openai_client()


# Initialize the YouTube client
youtube = init_youtube_client()

def extract_video_id(url):
    """Extract YouTube video ID from URL"""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu.be\/)([\w-]+)',
        r'(?:youtube\.com\/embed\/)([\w-]+)',
        r'(?:youtube\.com\/v\/)([\w-]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def extract_frames_method1(video_url, num_frames=5):
    """Extract frames using pytube and opencv"""
    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download video
            yt = YouTube(video_url)
            
            # Get the highest quality stream that includes both video and audio
            stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
            
            if not stream:
                raise Exception("No suitable video stream found")
            
            # Download to temporary directory
            temp_video_path = os.path.join(temp_dir, 'temp_video.mp4')
            stream.download(output_path=temp_dir, filename='temp_video.mp4')
            
            # Open video file
            cap = cv2.VideoCapture(temp_video_path)
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            duration = total_frames / fps
            
            # Calculate frame intervals for even distribution
            interval = total_frames // (num_frames + 1)
            frames = []
            
            for i in range(num_frames):
                # Calculate position for evenly distributed frames
                frame_pos = interval * (i + 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ret, frame = cap.read()
                
                if ret:
                    # Convert to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Create PIL Image
                    pil_image = Image.fromarray(frame_rgb)
                    # Resize maintaining aspect ratio
                    pil_image.thumbnail((800, 450))
                    
                    # Convert to base64
                    buffered = BytesIO()
                    pil_image.save(buffered, format="JPEG", quality=85)
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    
                    # Convert time to timestamp
                    timestamp = frame_pos / fps
                    time_str = str(datetime.utcfromtimestamp(timestamp).strftime('%M:%S'))
                    
                    frames.append({
                        'image': img_str,
                        'timestamp': timestamp,
                        'time_str': time_str
                    })
            
            cap.release()
            return frames
            
    except Exception as e:
        st.error(f"Method 1 failed: {str(e)}")
        return None

def extract_video_thumbnails(video_url):
    """Extract video thumbnails as fallback"""
    try:
        video_id = extract_video_id(video_url)
        thumbnails = []
        
        # Get available thumbnail URLs
        thumb_urls = [
            f'https://img.youtube.com/vi/{video_id}/maxresdefault.jpg',
            f'https://img.youtube.com/vi/{video_id}/hqdefault.jpg',
            f'https://img.youtube.com/vi/{video_id}/mqdefault.jpg'
        ]
        
        for url in thumb_urls:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    image = Image.open(BytesIO(response.content))
                    
                    # Convert to base64
                    buffered = BytesIO()
                    image.save(buffered, format="JPEG", quality=85)
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    
                    thumbnails.append({
                        'image': img_str,
                        'timestamp': 0,
                        'time_str': '00:00'
                    })
                    break  # Stop after getting first successful thumbnail
            except:
                continue
                
        return thumbnails if thumbnails else None
        
    except Exception as e:
        st.error(f"Thumbnail extraction failed: {str(e)}")
        return None

def get_video_frames(video_url, num_frames=5):
    """Try multiple methods to extract frames"""
    
    # Try Method 1 (pytube + opencv)
    frames = extract_frames_method1(video_url, num_frames)
    if frames:
        return frames, "pytube"
        
    # Fallback to thumbnails
    frames = extract_video_thumbnails(video_url)
    if frames:
        return frames, "Thumbnails"
    
    return None, "Failed"

def get_video_transcript(video_id):
    """Get transcript from YouTube video with better error handling"""
    try:
        # First try getting transcript through YouTube API
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Try to get English transcript first
        try:
            transcript = transcript_list.find_transcript(['en'])
        except:
            # If no English transcript, try to get any transcript and translate it
            try:
                available_transcripts = transcript_list.find_generated_transcript()
                if available_transcripts:
                    transcript = available_transcripts.translate('en')
                else:
                    raise Exception("No transcripts available")
            except:
                st.warning("No automatic captions available. Attempting manual transcription...")
                return transcribe_audio(video_id)
        
        transcript_text = ' '.join([item['text'] for item in transcript.fetch()])
        return transcript_text
        
    except Exception as e:
        if "Video is private" in str(e):
            st.error("This video is private. Please try a public video.")
            return None
        elif "Video unavailable" in str(e):
            st.error("This video is unavailable. It might be deleted or restricted.")
            return None
        else:
            st.info("No transcript available. Attempting to transcribe audio...")
            return transcribe_audio(video_id)

def transcribe_audio(video_id):
    """Download and transcribe video audio with improved error handling"""
    try:
        # Get video URL
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        
        # Configure pytube with custom parameters and authentication
        yt = YouTube(video_url)
        
        # Add user agent to avoid 403 error
        yt.bypass_age_gate()
        
        # Add retry logic for audio download
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Get the audio stream with lowest bitrate to speed up download
                audio_stream = yt.streams.filter(only_audio=True).order_by('abr').first()
                
                if not audio_stream:
                    raise Exception("No audio stream available")
                
                # Create temp directory
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_file = os.path.join(temp_dir, f'temp_audio_{video_id}.mp4')
                    
                    # Download with timeout
                    audio_stream.download(filename=temp_file, timeout=30)
                    
                    # Check if file exists and has size
                    if not os.path.exists(temp_file) or os.path.getsize(temp_file) == 0:
                        raise Exception("Download failed or empty file")
                    
                    # Load Whisper model and transcribe
                    model = whisper.load_model("base")
                    result = model.transcribe(temp_file)
                    
                    return result["text"]
                
            except Exception as e:
                if "HTTP Error 403" in str(e):
                    st.error("Access to this video is restricted. Please try another video.")
                    return None
                elif attempt == max_retries - 1:
                    raise e
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
                
    except Exception as e:
        st.error(f"Error accessing video: {str(e)}")
        st.error("""
        Unable to process this video. This could be due to:
        - Video is private or restricted
        - Age-restricted content
        - Region-restricted content
        - Copyright restrictions
        
        Please try another public video.
        """)
        return None

def init_youtube_client():
    """Initialize YouTube API client with improved authentication"""
    try:
        api_key = st.secrets["YOUTUBE_API_KEY"]
        if not api_key:
            raise ValueError("YouTube API key not found in secrets")
            
        # Add retry mechanism for API client
        from googleapiclient.http import HttpRequest
        HttpRequest.MAX_RETRIES = 3
        
        return build('youtube', 'v3', 
                    developerKey=api_key,
                    cache_discovery=False)
    except Exception as e:
        st.error("Error initializing YouTube client. Please check your API key.")
        st.error(f"Error details: {str(e)}")
        return None

# Update the main function to include video accessibility check
def check_video_accessibility(video_id):
    """Check if video is accessible before processing"""
    if not youtube:
        return False
        
    try:
        request = youtube.videos().list(
            part="status",
            id=video_id
        )
        response = request.execute()
        
        if not response.get('items'):
            st.error("Video not found or inaccessible.")
            return False
            
        status = response['items'][0]['status']
        
        if status.get('privacyStatus') != 'public':
            st.error("This video is not public. Please try a public video.")
            return False
            
        return True
        
    except Exception as e:
        st.error(f"Error checking video accessibility: {str(e)}")
        return False
def get_video_details(video_id):
    """Get video title and description from YouTube API"""
    if youtube is None:
        st.error("YouTube API client not initialized")
        return None
        
    try:
        request = youtube.videos().list(
            part="snippet",
            id=video_id
        )
        response = request.execute()
        
        if response['items']:
            snippet = response['items'][0]['snippet']
            return {
                'title': snippet['title'],
                'description': snippet['description']
            }
        return None
    except Exception as e:
        st.error(f"Error fetching video details: {str(e)}")
        return None

# Define system instructions for different styles
SYSTEM_INSTRUCTION_DETAILED = """
You are converting the video transcript into a detailed blog post in the speaker's voice. Follow these guidelines:
1. Use the exact title provided in the title area.
2. Create a comprehensive, detailed blog post that expands on each point.
3. Include abundant examples and explanations.
4. Use "I", "my", and "we" to represent the speaker's direct thoughts and experiences.
5. Maintain the speaker's expertise and insights with detailed elaboration.
6. Organize content with detailed sections and subsections.
7. Use markdown formatting for headers (##) and emphasis (*).
8. Provide in-depth context for each major point.
9. Include relevant examples and case studies mentioned.
10. End with comprehensive concluding thoughts.
11. Keep the speaker's authentic voice throughout.
12. Reference images using [IMAGE_X] tags where appropriate, describing what each image shows.
"""

SYSTEM_INSTRUCTION_CONCISE = """
You are converting the video transcript into a concise blog post in the speaker's voice. Follow these guidelines:
1. Use the exact title provided in the title area.
2. Create a brief, focused blog post that captures key points succinctly.
3. Keep paragraphs short and focused.
4. Use "I", "my", and "we" to represent the speaker's direct thoughts and experiences.
5. Maintain the speaker's core message without excessive detail.
6. Organize content with minimal, essential sections.
7. Use markdown formatting for headers (##) and emphasis (*).
8. Focus on the most important insights only.
9. Include only the most impactful examples.
10. End with brief, actionable takeaways.
11. Keep the speaker's authentic voice throughout.
12. Reference images using [IMAGE_X] tags where appropriate, describing what each image shows.
"""

def clean_image_descriptions(text):
    """Convert image description tags into proper markdown"""
    # Pattern to match [IMAGE_X: description] format
    image_pattern = r'\[IMAGE_(\d+):\s*(.*?)\]'
    
    def replace_image(match):
        image_num = match.group(1)
        description = match.group(2)
        return f'*[Image {image_num}: {description}]*'
    
    # Replace image tags with markdown formatted text
    cleaned_text = re.sub(image_pattern, replace_image, text)
    return cleaned_text

def generate_article_from_transcript(title, transcript, video_details=None, style="detailed", frames=None):
    """Generate a blog post with images from the transcript and video details"""
    if client is None:
        st.error("OpenAI client not initialized. Cannot generate article.")
        return None
        
    # Select appropriate system instruction based on style
    system_instruction = SYSTEM_INSTRUCTION_DETAILED if style == "detailed" else SYSTEM_INSTRUCTION_CONCISE
    
    # Create a context-rich prompt using video details if available
    context = ""
    if video_details:
        context = f"""
        Video Title: {video_details['title']}
        Video Description: {video_details['description']}
        """
    
    # Add frame information to the prompt if available
    frame_context = ""
    if frames:
        frame_times = [f"Frame {i+1} at {frame['time_str']}" for i, frame in enumerate(frames)]
        frame_context = f"\nAvailable video frames: {', '.join(frame_times)}"
    
    content_prompt = f"""Write a {'detailed' if style == 'detailed' else 'concise'} blog post with the title: '{title}'
    
    Context: {context}
    {frame_context}
    
    Full transcript: {transcript}
    
    Convert this into a well-structured blog post while maintaining the speaker's voice and key insights.
    Make it {'comprehensive and detailed' if style == 'detailed' else 'concise and focused'}.
    
    When mentioning images, use the format [IMAGE_X: description] where X is the frame number and description 
    explains what the image shows.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": content_prompt}
            ],
            temperature=0.7
        )
        
        article_content = response.choices[0].message.content
        
        # Clean up the image descriptions
        article_content = clean_image_descriptions(article_content)
        
        # If we have frames, add them after their descriptions
        if frames:
            for i, frame in enumerate(frames, 1):
                img_pattern = f'*[Image {i}:'
                img_html = f'\n\n<img src="data:image/jpeg;base64,{frame["image"]}" alt="Frame at {frame["time_str"]}" style="max-width: 100%; height: auto; margin: 20px 0;">\n\n'
                article_content = article_content.replace(img_pattern, img_html + img_pattern)
        
        return article_content
        
    except Exception as e:
        st.error(f"Error generating article: {str(e)}")
        return None
def main():
    st.set_page_config(page_title="YouTube to Blog Post Generator", page_icon="📝", layout="wide")
    
    st.title("📝 YouTube Video to Blog Post Generator")
    st.markdown("""
    Transform any YouTube video into a well-structured blog post with images. 
    Choose between detailed and concise writing styles.
    """)
    
    # Create three columns for input
    col1, col2, col3 = st.columns(3)
    
    with col1:
        video_url = st.text_input("YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")
    
    with col2:
        title = st.text_input("Blog Post Title:", placeholder="Enter your desired blog post title...")
    
    with col3:
        style = st.selectbox(
            "Writing Style:",
            options=["detailed", "concise"],
            format_func=lambda x: "Detailed (Comprehensive)" if x == "detailed" else "Concise (To the point)",
            help="Choose between a detailed or concise writing style"
        )
        
        num_frames = st.slider(
            "Number of images to extract",
            min_value=1,
            max_value=10,
            value=5,
            help="Select how many images to extract from the video"
        )
    
    if st.button("Generate Article", type="primary"):
        if video_url and title:
            with st.spinner("Processing video... This may take a few minutes."):
                # Extract video ID and get details
                video_id = extract_video_id(video_url)
                if not video_id:
                    st.error("Invalid YouTube URL. Please check the URL and try again.")
                    return
                
                # Get video details
                video_details = get_video_details(video_id)
                
                # Extract frames
                with st.spinner("Extracting video frames..."):
                    frames, extraction_method = get_video_frames(video_url, num_frames)
                    if frames:
                        st.success(f"Successfully extracted frames using {extraction_method}")
                    else:
                        st.warning("Could not extract video frames. Generating article without images.")
                
                # Get transcript
                transcript = get_video_transcript(video_id)
                if not transcript:
                    return
                
                # Generate article
                article_content = generate_article_from_transcript(
                    title, 
                    transcript, 
                    video_details,
                    style,
                    frames
                )
                
                if article_content:
                    # Display results
                    st.success("✅ Article generated successfully!")
                    
                    # Show the article in a nice format
                    st.markdown("---")
                    st.markdown(f"## Generated Article ({style.capitalize()} Version)")
                    st.markdown(article_content, unsafe_allow_html=True)
                    
                    # Add download button
                    st.download_button(
                        label="Download Article as Markdown",
                        data=article_content,
                        file_name=f"generated_article_{style}.md",
                        mime="text/markdown"
                    )
        else:
            st.warning("Please enter both a YouTube URL and a title.")
    
    # Add footer with usage instructions
    st.markdown("---")
    st.markdown("""
    ### How to use:
    1. Paste a YouTube video URL in the input field
    2. Enter your desired blog post title
    3. Select your preferred writing style:
        - **Detailed**: Comprehensive coverage with examples and elaboration
        - **Concise**: Brief, focused version with key points only
    4. Choose the number of images to extract from the video
    5. Click "Generate Article" and wait for processing
    6. Download the generated article in Markdown format
    
    Note: Processing time may vary depending on video length and transcript availability.
    """)

if __name__ == "__main__":
    main()
