import os
import streamlit as st
import whisper
import soundfile as sf
import librosa
import imageio_ffmpeg

# 파이썬 안에서 자동으로 ffmpeg 경로를 잡아주도록 설정합니다.
os.environ["PATH"] += os.pathsep + os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())

st.title("🎙️ MP4 음원 STT 텍스트 변환기")
st.write("MP4 또는 음원 파일을 올리면 인공지능이 텍스트로 변환해 줍니다!")

@st.cache_resource
def load_model():
    return whisper.load_model("tiny")

with st.spinner("AI 모델을 준비하는 중입니다. 잠시만 기다려주세요..."):
    model = load_model()

uploaded_file = st.file_uploader("MP4 또는 음원 파일을 선택하세요", type=["mp4", "mp3", "wav", "m4a", "aac"])

if uploaded_file is not None:
    st.video(uploaded_file)
    
    if st.button("텍스트로 변환하기"):
        with st.spinner("음성을 분석하고 글자로 변환하고 있어요... 잠시만 기다려주세요!"):
            temp_path = "temp_media.mp4"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                y, sr = librosa.load(temp_path, sr=16000)
                clean_audio_path = "clean_audio.wav"
                sf.write(clean_audio_path, y, sr)
                
                result = model.transcribe(clean_audio_path)
                transcribed_text = result["text"]
                
                if os.path.exists(clean_audio_path):
                    os.remove(clean_audio_path)
            except Exception as e:
                transcribed_text = f"변환 중 오류가 발생했습니다: {str(e)}"
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        st.success("변환 완료!")
        
        st.subheader("📝 변환된 텍스트 결과")
        st.text_area("내용을 복사해서 사용하세요", transcribed_text, height=250)
        
        st.download_button(
            label="텍스트 파일(.txt)로 다운로드",
            data=transcribed_text,
            file_name="stt_result.txt",
            mime="text/plain"
        )
