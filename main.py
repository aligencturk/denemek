import os
import shutil
import logging
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI
from moviepy.editor import VideoFileClip
from PIL import Image
import io
import numpy as np
import cv2
from dotenv import load_dotenv

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log", encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

# Helper: Saniye cinsinden süreyi SRT formatına (HH:MM:SS,mmm) çevirir
def format_timestamp(seconds: float) -> str:
    milliseconds = int((seconds % 1) * 1000)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

# Helper: Whisper segmentlerini SRT formatına çevirir
def segments_to_srt(segments):
    srt_content = ""
    for i, segment in enumerate(segments, start=1):
        # Segment bir obje ise attribute, dict ise key olarak eriş
        start_time = getattr(segment, 'start', segment['start'] if isinstance(segment, dict) else 0)
        end_time = getattr(segment, 'end', segment['end'] if isinstance(segment, dict) else 0)
        text_content = getattr(segment, 'text', segment['text'] if isinstance(segment, dict) else "").strip()
        
        start = format_timestamp(start_time)
        end = format_timestamp(end_time)
        
        srt_content += f"{i}\n{start} --> {end}\n{text_content}\n\n"
    return srt_content

# Tarayıcının bu sunucuya erişmesine izin ver
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API key'i .env dosyasından al, yoksa fallback olarak hardcoded key kullan
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Video Altyazı API'si çalışıyor!",
        "docs": "/docs",
        "endpoints": {
            "upload_video": "POST /upload-video/",
            "remove_background": "POST /remove-background/",
            "detect_language": "POST /detect-language/",
            "translate_srt": "POST /translate-srt/"
        }
    }

# Request modelleri
class DetectLanguageRequest(BaseModel):
    srt_content: str

class TranslateSRTRequest(BaseModel):
    srt_content: str
    target_language: str

@app.post("/upload-video/")
async def create_subtitle(file: UploadFile = File(...)):
    logger.info(f"Videoyu aldım: {file.filename}")
    
    # Dosya isimleri
    video_filename = f"temp_{file.filename}"
    audio_filename = f"temp_{file.filename}.mp3"
    
    # 1. Videoyu bilgisayara kaydet
    with open(video_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 2. Videodan sesi ayıkla (MoviePy kullanarak)
        logger.info("Ses ayıklanıyor (FFmpeg)...")
        video = VideoFileClip(video_filename)
        video.audio.write_audiofile(audio_filename, codec='mp3', logger=None)
        video.close()

        # 3. Sesi OpenAI'ya gönder
        logger.info("OpenAI Whisper servisine gönderiliyor...")
        with open(audio_filename, "rb") as audio_file:
            # verbose_json formatı kullanarak dil bilgisini de alıyoruz
            transcript_response = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file, 
                response_format="verbose_json"
            )
        
        # Dil ve Segmentleri al
        # OpenAI v1.x dönüşü obje veya dict olabilir, güvenli erişim sağlayalım
        detected_language = getattr(transcript_response, 'language', None)
        if detected_language is None and isinstance(transcript_response, dict):
             detected_language = transcript_response.get('language', 'unknown')
        
        segments = getattr(transcript_response, 'segments', [])
        if not segments and isinstance(transcript_response, dict):
             segments = transcript_response.get('segments', [])

        logger.info(f"Whisper tarafından tespit edilen dil: {detected_language}")
        
        # SRT oluştur
        srt_content = segments_to_srt(segments)

        logger.info("Altyazı başarıyla oluşturuldu!")
        # detected_language bilgisini de dönüyoruz
        return {"status": "success", "srt_content": srt_content, "detected_language": detected_language}

    except Exception as e:
        logger.error(f"HATA OLUŞTU: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"status": "error", "message": str(e)}

    finally:
        # Temizlik: Geçici dosyaları sil (dosyalar kapatıldıktan sonra)
        try:
            if os.path.exists(video_filename):
                os.remove(video_filename)
        except Exception as e:
            logger.error(f"Video dosyası silinemedi: {e}")
        
        try:
            if os.path.exists(audio_filename):
                os.remove(audio_filename)
        except Exception as e:
            logger.error(f"Ses dosyası silinemedi: {e}")

@app.post("/remove-background/")
async def remove_background(file: UploadFile = File(...)):
    """Görsel arka planını kaldırır - GrabCut algoritması ile"""
    logger.info(f"Görsel alındı: {file.filename}")
    
    try:
        # Görseli oku
        image_data = await file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        logger.info("Arka plan kaldırılıyor (GrabCut)...")
        
        # GrabCut için maske oluştur
        mask = np.zeros(img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        
        # Görüntünün ortasındaki dikdörtgeni ön plan olarak işaretle
        h, w = img.shape[:2]
        rect = (int(w*0.05), int(h*0.05), int(w*0.9), int(h*0.9))
        
        # GrabCut uygula
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        
        # Maskeyi düzenle (0,2 = arka plan, 1,3 = ön plan)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
        # RGBA formatına çevir
        img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        img_rgba[:, :, 3] = mask2 * 255
        
        # PIL Image'a çevir
        output_image = Image.fromarray(img_rgba, 'RGBA')
        
        # PNG olarak kaydet
        img_byte_arr = io.BytesIO()
        output_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        logger.info("Arka plan başarıyla kaldırıldı!")
        return StreamingResponse(img_byte_arr, media_type="image/png")
        
    except Exception as e:
        logger.error(f"HATA OLUŞTU: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"status": "error", "message": str(e)}

@app.post("/detect-language/")
async def detect_language(request: DetectLanguageRequest):
    """SRT içeriğindeki dili otomatik algılar"""
    try:
        logger.info("Dil tespiti isteği alındı")
        # İlk birkaç altyazı metnini al
        lines = request.srt_content.strip().split('\n')
        sample_text = ' '.join([line for line in lines if line and not '-->' in line and not line.isdigit()])[:500]
        
        logger.info(f"Analiz edilecek örnek metin (ilk 100): {sample_text[:100]}")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Sen bir dil tespit uzmanısın. Verilen metni analiz et ve sadece dilin ISO 639-1 kodunu ver (örn: 'tr', 'en', 'de'). Sadece kod yaz, başka hiçbir şey yazma."},
                {"role": "user", "content": sample_text}
            ],
            temperature=0.3
        )
        
        detected_lang = response.choices[0].message.content.strip().lower()
        logger.info(f"Tespit edilen dil: {detected_lang}")
        return {"status": "success", "language": detected_lang}
        
    except Exception as e:
        logger.error(f"Dil tespiti hatası: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/translate-srt/")
async def translate_srt(request: TranslateSRTRequest):
    """SRT içeriğini hedef dile çevirir, zaman kodları korunur"""
    try:
        logger.info(f"SRT {request.target_language} diline çevriliyor...")
        
        # SRT bloklarını ayır
        blocks = request.srt_content.strip().split('\n\n')
        translated_blocks = []
        
        for block in blocks:
            if not block.strip():
                continue
                
            lines = block.strip().split('\n')
            if len(lines) < 3:
                continue
            
            # Numara ve zaman kodu
            number = lines[0]
            timecode = lines[1]
            # Metin (birden fazla satır olabilir)
            text = ' '.join(lines[2:])
            
            # Metni çevir
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"Sen profesyonel bir çevirmensin. Verilen altyazı metnini {request.target_language} diline çevir. Sadece çeviriyi yaz, başka hiçbir şey ekleme. Doğal ve akıcı olsun."},
                    {"role": "user", "content": text}
                ],
                temperature=0.7
            )
            
            translated_text = response.choices[0].message.content.strip()
            
            # SRT formatında birleştir
            translated_block = f"{number}\n{timecode}\n{translated_text}"
            translated_blocks.append(translated_block)
        
        translated_srt = '\n\n'.join(translated_blocks)
        logger.info(f"Çeviri tamamlandı!")
        return {"status": "success", "translated_srt": translated_srt}
        
    except Exception as e:
        logger.error(f"Çeviri hatası: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    # Sunucuyu başlat
    uvicorn.run(app, host="0.0.0.0", port=8002)
