import os
import streamlit as st
import imageio_ffmpeg
import subprocess
import whisper
import librosa

# [핵심 해결책] 위스퍼 내부의 오디오 로더를 패치하여 시스템 ffmpeg 호출을 원천 차단합니다.
def custom_load_audio(file, sr=16000):
    audio, _ = librosa.load(file, sr=sr, mono=True)
    return audio

whisper.audio.load_audio = custom_load_audio

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
        with st.spinner("음성을 추출하고 텍스트로 변환하는 중입니다... 잠시만 기다려주세요!"):
            temp_path = "temp_media.mp4"
            clean_audio_path = "clean_audio.wav"
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                # 1. imageio_ffmpeg가 제공하는 내장 실행 파일 경로로 안전하게 오디오 추출
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                subprocess.run([
                    ffmpeg_exe, "-i", temp_path, 
                    "-ar", "16000", "-ac", "1", "-y", clean_audio_path
                ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # 2. Whisper 변환 실행 (패치된 로더가 작동하여 에러 발생 안 함)
                result = model.transcribe(clean_audio_path)
                transcribed_text = result["text"]
                
            except Exception as e:
                transcribed_text = f"변환 중 오류가 발생했습니다: {str(e)}"
            
            # 임시 파일 정리
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if os.path.exists(clean_audio_path):
                os.remove(clean_audio_path)
        
        st.success("변환 완료!")
        
        st.subheader("📝 변환된 텍스트 결과")
        st.text_area("내용을 복사해서 사용하세요", transcribed_text, height=250)
        
        st.download_button(
            label="텍스트 파일(.txt)로 다운로드",
            data=transcribed_text,
            file_name="stt_result.txt",
            mime="text/plain"
        )
