import os
import streamlit as st
import whisper
import soundfile as sf
import librosa

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
            # 1. 업로드된 파일을 임시 저장
            temp_path = "temp_media.mp4"
            with open(temp_path, "wbf" if False else "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                # 2. librosa를 이용해 오디오 데이터로 바로 읽어들이기 (ffmpeg 오류 우회)
                y, sr = librosa.load(temp_path, sr=16000)
                clean_audio_path = "clean_audio.wav"
                sf.write(clean_audio_path, y, sr)
                
                # 3. Whisper로 변환 실행
                result = model.transcribe(clean_audio_path)
                transcribed_text = result["text"]
                
                # 4. 임시 파일들 정리
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
