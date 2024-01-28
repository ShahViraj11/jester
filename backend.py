import assemblyai as aai
from transformers import pipeline
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, TextClip
import pysrt
from elevenlabs import Voice, VoiceSettings, generate, save
import PyPDF2

aai.settings.api_key = "936f6e2d85514e8a8874e65a7142d214"
# Hardcoded api key
elevenlabs_key = "832e0b7b8bed791f9024f446cb53852b"

# Ids for different voices:
# Morgan Freeman: LWF11sFDu95RpULuI2zh
# Obama: wylrcjVxMdzlgD60BJTD
# Trump: yWqXgpIk889VDcUHIK1X
# Lois: yjGDxIBhAV2uY16ATk7b

'''
Generate the audio file using text
'''


def generate_tts(text: str):
    audio = generate(
        text=text,  # Text to generate speech for
        api_key=elevenlabs_key,
        voice=Voice(
            voice_id='yWqXgpIk889VDcUHIK1X',
            settings=VoiceSettings(stability=0.71, similarity_boost=0.5, style=0.0, use_speaker_boost=True)
        ),
        # Either a voice name, voice_id, or Voice object (use voice object to control stability and similarity_boost)
        model="eleven_monolingual_v1",  # Either a model name or Model object
        stream=False,  # If True, returns a generator streaming bytes
        stream_chunk_size=2048,  # Size of each chunk when stream=True
        latency=1,  # [1-4] the higher the more optimized for streaming latency (only works with stream=True)
        output_format="mp3_44100_128"
        # The output format: mp3_44100_[64,96,128,192], pcm_[16000,22050,24000,44100], ulaw_8000
    )

    save(audio=audio, filename="celeb_voice.mp3")  # This may cause some integration issues


# PROCESS OUTLINED BELOW
# Convert video to text
# Summarize the text
# Create celebrity audio clip
# Combine audio clip and given visual template into video with subtitles

def convert_pdf_to_text(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
        return text


def video_to_text(vid_path):
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(vid_path)

    # srt = transcript.export_subtitles_srt(chars_per_caption=32)
    #
    # # Save it to a file
    # with open("subtitle_example.srt", "w") as f:
    #     f.write(srt)

    return transcript.text


def summarize_text(text, max_length=200, min_length=100, chunk_size=1024):
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    summaries = [summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text'] for chunk in chunks]
    return ' '.join(summaries)


def merge_video_audio(video_path, audio_path, output_path):
    # Load the video and the audio
    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)

    audio_length = get_audio_length("celeb_voice.mp3")
    video_clip = video_clip.subclip(0,audio_length)
    # Set the audio of the video clip as the audio clip
    final_clip = video_clip.set_audio(audio_clip)

    # Write the result to a file
    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")


def create_srt_file(video_path):
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(video_path)
    srt = transcript.export_subtitles_srt(chars_per_caption=32)

    # Save it to a file
    with open("subtitle_example.srt", "w") as f:
        f.write(srt)


def add_subtitles_from_srt(video_path, srt_path, output_path, txt_color='white', font_size=30, font='Arial-Bold'):
    # Load the video
    video = VideoFileClip(video_path)

    # Load and parse the SRT file
    subs = pysrt.open(srt_path)

    # Create a list to hold subtitle clips
    subtitles = []

    # Loop through the subtitles in the SRT file
    for sub in subs:
        # Create a TextClip for this subtitle
        txt_clip = TextClip(sub.text, fontsize=font_size, font=font, color=txt_color)

        # Set the position, start time, and duration of the text clip
        start_time = sub.start.hours * 3600 + sub.start.minutes * 60 + sub.start.seconds + sub.start.milliseconds / 1000.0
        end_time = sub.end.hours * 3600 + sub.end.minutes * 60 + sub.end.seconds + sub.end.milliseconds / 1000.0
        txt_clip = txt_clip.set_position('center').set_start(start_time).set_duration(end_time - start_time)

        subtitles.append(txt_clip)

    # Overlay the text clips on the video
    final_video = CompositeVideoClip([video, *subtitles])

    # Write the result to a file
    final_video.write_videofile(output_path, codec='libx264', audio_codec='aac')


def get_audio_length(audio_path):
    audio_clip = AudioFileClip(audio_path)
    return audio_clip.duration  # Duration in seconds


#pdf_text = convert_pdf_to_text('/Users/siddsatish/Desktop/1_GraphADT_Review_FS23.pdf')
# print(pdf_text)

vid_text = video_to_text("/Users/siddsatish/Desktop/owen_example.mov")

summarized_text = summarize_text(vid_text,500,50)

generate_tts(summarized_text)

merge_video_audio("/Users/siddsatish/Desktop/free_to_use_gameplay_7868943649_musicaldown.com.mp4","celeb_voice.mp3",
                  "vid_no_subtitles.mp4")

create_srt_file("vid_no_subtitles.mp4")

add_subtitles_from_srt("vid_no_subtitles.mp4","subtitle_example.srt","final_vid_example.mp4")

