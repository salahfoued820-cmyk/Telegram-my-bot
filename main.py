import os
import time
import logging
import requests
from flask import Flask, request, jsonify

# 1. إعداد نظام السجلات والمراقبة (Logging)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 2. استدعاء متغيرات البيئة الحساسة من خادم التستضيف (Render)
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")

# الروابط الرسمية للخدمات السحابية
TELEGRAM_API_URL = f"https://telegram.org{TELEGRAM_TOKEN}/"
REPLICATE_API_URL = "https://replicate.com"

# اسم موديل توليد الفيديو المعتمد على منصة Replicate
MODEL_VERSION = "luma/ray"


def _generate_video_url(prompt_text: str) -> str:
    """ترسل النص إلى Replicate وتنتظر حتى انتهاء السيرفر من معالجة وتوليد الفيديو"""
    if not REPLICATE_API_TOKEN:
        logger.error("خطأ: مفتاح REPLICATE_API_TOKEN غير معرف في متغيرات البيئة.")
        return None

    headers = {
        "Authorization": f"Token {REPLICATE_API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # المعايير البرمجية لطلب توليد الفيديو
    payload = {
        "version": MODEL_VERSION,
        "input": {
            "prompt": prompt_text,
            "aspect_ratio": "16:9",
            "duration": 5
        }
    }
    
    try:
        # خطوة أ: إنشاء الطلب الأولي (إنتاج الـ Prediction)
        response = requests.post(REPLICATE_API_URL, headers=headers, json=payload, timeout=20)
        if response.status_code != 201:
            logger.error(f"فشل Replicate في قبول الطلب: {response.text}")
            return None
            
        prediction_data = response.json()
        poll_url = prediction_data["urls"]["get"]
        
        # خطوة ب: فحص دوري للسيرفر (Polling) كل 3 ثوانٍ لمعرفة هل انتهى توليد الفيديو
        logger.info(f"بدء عملية التوليد السحابي للطلب: {prediction_data['id']}")
        for attempt in range(25):  # المحاولة لمدة 75 ثانية كحد أقصى لفيديوهات عريضة الجودة
            time.sleep(3)
            check_response = requests.get(poll_url, headers=headers, timeout=15)
            
            if check_response.status_code == 200:
                status_data = check_response.json()
                status = status_data["status"]
                
                if status == "succeeded":
                    logger.info("تم توليد ملف الفيديو بنجاح.")
                    # إرجاع رابط الـ MP4 المباشر الناتجة عن المعالجة
                    return status_data["output"]
                elif status == "failed":
                    logger.error(f"فشل الموديل في توليد الفيديو: {status_data.get('error')}")
                    return None
            else:
                logger.error(f"فشل الاتصال برابط الفحص الدوري: {check_response.text}")
                
        logger.warning("انتهت مهلة الانتظار ولم يكتمل الفيديو بعد.")
        return None

    except Exception as e:
        logger.critical(f"خطأ غير متوقع في دالة توليد الفيديو: {str(e)}")
        return None


def handle_telegram_message(chat_id: int, user_text: str):
    """تقوم بإدارة التفاعل مع المستخدم وإرسال التحديثات والفيديو النهائي"""
    # 1. طمأنة المستخدم برسالة فورية لمنع تكرار الضغط
    requests.post(f"{TELEGRAM_API_URL}sendMessage", json={
        "chat_id": chat_id,
        "text": "جاري تخيل وتوليد الفيديو الخاص بك بدقة عالية... 🎬\nقد تستغرق هذه العملية ما بين 15 إلى 30 ثانية."
    })
    
    # 2. استدعاء المعالجة السحابية
    video_url = _generate_video_url(user_text)
    
    if video_url:
        # 3. إرسال الفيديو كملف MP4 متحرك للمستخدم في حال النجاح
        payload = {
            "chat_id": chat_id,
            "video": video_url,
            "caption": f"🎬 تم توليد الفيديو الخاص بك بنجاح بناءً على الوصف:\n\"{user_text}\""
        }
        video_response = requests.post(f"{TELEGRAM_API_URL}sendVideo", json=payload)
        if video_response.status_code != 200:
            logger.error(f"فشل تليجرام في إرسال الفيديو: {video_response.text}")
    else:
        # 4. إخطار المستخدم بالاعتذار إذا فشل السيرفر
        requests.post(f"{TELEGRAM_API_URL}sendMessage", json={
            "chat_id": chat_id,
            "text": "عذراً، واجه السيرفر ضغطاً أو مشكلة في تخيل هذا الوصف. يرجى إعادة المحاولة بكلمات أخرى."
        })


@app.route("/", methods=["POST"])
def telegram_webhook():
    """المستقبل الرئيسي (Webhook) لرسائل مستخدمي البوت من خوادم تليجرام"""
    try:
        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({"status": "ignored"}), 200
            
        message = data["message"]
        chat_id = message["chat"]["id"]
        user_text = message.get("text", "").strip()
        
        if user_text:
            if user_text.startswith("/start"):
                requests.post(f"{TELEGRAM_API_URL}sendMessage", json={
                    "chat_id": chat_id,
                    "text": "مرحباً بك في بوت توليد الفيديوهات الذكي! 🚀\nاكتب لي أي وصف تتخيله باللغة الإنجليزية وسأقوم بتحويله إلى فيديو عالي الجودة فوراً."
                })
            else:
                # معالجة الطلب وتوليد الفيديو
                handle_telegram_message(chat_id, user_text)
                
        return jsonify({"status": "success"}), 200

    except Exception as e:
        logger.error(f"خطأ في استقبال الـ Webhook: {str(e)}")
        return jsonify({"status": "error"}), 500


if __name__ == "__main__":
    # تشغيل خادم Flask محلياً عند التطوير والامتداد
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
