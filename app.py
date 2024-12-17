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

# Load environment variables
load_dotenv()

# Set up OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Set up YouTube API client
YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

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

def extract_video_frames(video_id, num_frames=5):
    """Extract representative frames from the video"""
    try:
        # Get video URL
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        
        # Download video
        yt = YouTube(video_url)
        
        # Get the highest resolution stream
        video_stream = yt.streams.filter(progressive=True).order_by('resolution').desc().first()
        
        if not video_stream:
            raise Exception("No suitable video stream available")
        
        # Download to temporary file
        temp_video = f"temp_video_{video_id}_{int(time.time())}.mp4"
        video_stream.download(filename=temp_video)
        
        # Open video file
        cap = cv2.VideoCapture(temp_video)
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        duration = total_frames / fps
        
        # Calculate frame intervals
        interval = total_frames // (num_frames + 1)
        frames = []
        
        for i in range(num_frames):
            frame_position = interval * (i + 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert to PIL Image
                pil_image = Image.fromarray(frame_rgb)
                # Resize to reasonable dimensions
                pil_image.thumbnail((800, 450), Image.Resampling.LANCZOS)
                # Convert to base64
                buffered = BytesIO()
                pil_image.save(buffered, format="JPEG", quality=85)
                img_str = base64.b64encode(buffered.getvalue()).decode()
                # Store frame with its timestamp
                timestamp = frame_position / fps
                frames.append({
                    'image': img_str,
                    'timestamp': timestamp,
                    'time_str': str(datetime.utcfromtimestamp(timestamp).strftime('%M:%S'))
                })
        
        cap.release()
        os.remove(temp_video)
        return frames
    
    except Exception as e:
        st.error(f"Error extracting frames: {str(e)}")
        return None
    finally:
        if 'temp_video' in locals() and os.path.exists(temp_video):
            os.remove(temp_video)

def transcribe_audio(video_id):
    """Download and transcribe video audio using Whisper"""
    try:
        # Get video URL
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        
        # Configure pytube with custom parameters
        yt = YouTube(video_url)
        
        # Add retry logic for audio download
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Get the audio stream with lowest bitrate to speed up download
                audio_stream = yt.streams.filter(only_audio=True).order_by('abr').first()
                
                if not audio_stream:
                    raise Exception("No audio stream available")
                
                # Use a unique temporary filename
                temp_file = f"temp_audio_{video_id}_{int(time.time())}.mp4"
                
                # Download with timeout
                audio_stream.download(filename=temp_file, timeout=30)
                
                # Check if file exists and has size
                if not os.path.exists(temp_file) or os.path.getsize(temp_file) == 0:
                    raise Exception("Download failed or empty file")
                
                # Load Whisper model and transcribe
                model = whisper.load_model("base")
                result = model.transcribe(temp_file)
                
                # Clean up
                os.remove(temp_file)
                return result["text"]
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
                
    except Exception as e:
        st.error(f"Error transcribing audio: {str(e)}")
        return None
    finally:
        # Ensure cleanup even if error occurs
        if 'temp_file' in locals() and os.path.exists(temp_file):
            os.remove(temp_file)

def get_video_transcript(video_id):
    """Get transcript from YouTube video with fallback to audio transcription"""
    try:
        # First try getting transcript through YouTube API
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Try to get English transcript first
        try:
            transcript = transcript_list.find_transcript(['en'])
        except:
            # If no English transcript, get the first available and translate it
            transcript = transcript_list.find_transcript(['en'])
            transcript = transcript.translate('en')
        
        transcript_text = ' '.join([item['text'] for item in transcript.fetch()])
        return transcript_text
        
    except Exception as e:
        st.info("No transcript available. Attempting to transcribe audio...")
        # Try the fallback method with additional error context
        transcript = transcribe_audio(video_id)
        if not transcript:
            st.error("""
            Unable to process video. This could be due to:
            - Video is private or restricted
            - No available transcripts
            - Audio transcription failed
            
            Please try another video or ensure the video is public.
            """)
        return transcript

def get_video_details(video_id):
    """Get video title and description from YouTube API"""
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
12. Elaborate on technical concepts when present.
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
12. Simplify technical concepts while maintaining accuracy.
"""

def generate_article_from_transcript(title, transcript, video_details=None, style="detailed", frames=None):
    """Generate a blog post with images from the transcript and video details"""
    
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
        frame_times = [f"Frame at {frame['time_str']}" for frame in frames]
        frame_context = f"\nKey video frames available at: {', '.join(frame_times)}"
    
    summary_prompt = f"""First, summarize the key points from this transcript and context:
    {context}
    {frame_context}
    Transcript excerpt: {transcript[:1500]}..."""
    
    # Get a summary first to help with context
    summary_response = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=[
            {"role": "system", "content": "Summarize the key points from this video transcript and context."},
            {"role": "user", "content": summary_prompt}
        ],
        temperature=0.7
    )
    summary = summary_response.choices[0].message.content
    
    # Now generate the full article with image placement suggestions
    content_prompt = f"""Write a {'detailed' if style == 'detailed' else 'concise'} blog post with the title: '{title}'
    
    Context: {context}
    {frame_context}
    
    Summary of content: {summary}
    
    Full transcript: {transcript}
    
    Convert this into a well-structured blog post while maintaining the speaker's voice and key insights.
    Make it {'comprehensive and detailed' if style == 'detailed' else 'concise and focused'}.
    
    If frame timestamps are provided, suggest appropriate places to insert images using the tag [IMAGE_FRAME_X] 
    where X is the frame number (1 to {len(frames) if frames else 0}).
    
    Use proper markdown formatting and create an engaging article."""
    
    response = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": content_prompt}
        ],
        temperature=0.7
    )
    
    article_content = response.choices[0].message.content
    
    # If we have frames, replace the image tags with actual images
    if frames:
        for i, frame in enumerate(frames, 1):
            img_tag = f'[IMAGE_FRAME_{i}]'
            img_html = f'<img src="data:image/jpeg;base64,{frame["image"]}" alt="Frame at {frame["time_str"]}" style="max-width: 100%; height: auto; margin: 20px 0;">'
            article_content = article_content.replace(img_tag, img_html)
    
    return article_content

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
                    frames = extract_video_frames(video_id)
                
                # Get transcript
                transcript = get_video_transcript(video_id)
                if not transcript:
                    return
                
                # Generate article with selected style and frames
                article_content = generate_article_from_transcript(
                    title, 
                    transcript, 
                    video_details,
                    style,
                    frames
                )
            
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
    4. Click "Generate Article" and wait for processing
    5. Download the generated article in Markdown format
    
    Note: Processing time may vary depending on video length and transcript availability.
    """)

if __name__ == "__main__":
    main()
