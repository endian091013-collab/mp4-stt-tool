import os
import streamlit as st
import imageio_ffmpeg
import subprocess
import whisper
import librosa

def custom_load_audio(file, sr=16000):
    audio, _ = librosa.load(file, sr=sr, mono=True)
    return audio

whisper.audio.load_audio = custom_load_audio

st.title("🎙️ MP4 음원 STT 텍스트 변환기")
st.write("MP4 또는 음원 파일을 올리면 인공지능이 텍스트로 변환해 줍니다!")

# [정확도 업그레이드 1] tiny 대신 한 단계 위인 'base' 모델을 사용합니다 (정확도 대폭 상승!)
@st.cache_resource
def load_model():
    return whisper.load_model("base")

with st.spinner("AI 모델을 준비하는 중입니다. (처음 실행 시 잠시 걸릴 수 있어요)..."):
    model = load_model()

uploaded_file = st.file_uploader("MP4 또는 음원 파일을 선택하세요", type=["mp4", "mp3", "wav", "m4a", "aac"])

if uploaded_file is not None:
    st.video(uploaded_file)
    
    if st.button("텍스트로 변환하기"):
        with st.spinner("음성을 분석하고 텍스트로 변환하는 중입니다... 잠시만 기다려주세요!"):
            temp_path = "temp_media.mp4"
            clean_audio_path = "clean_audio.wav"
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                # 오디오 추출
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                subprocess.run([
                    ffmpeg_exe, "-i", temp_path, 
                    "-ar", "16000", "-ac", "1", "-y", clean_audio_path
                ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # [정확도 업그레이드 2] language="ko"를 추가하여 한국어에만 집중하도록 강제합니다.
                result = model.transcribe(clean_audio_path, language="ko")
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
