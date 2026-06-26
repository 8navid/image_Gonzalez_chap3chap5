# -*- coding: utf-8 -*-
"""
کد کامل: آموزش انواع نویز در تصاویر (Noise Models)
بر اساس کتاب Digital Image Processing - Gonzalez & Woods, Chapter 5.2

این کد:
1. یک تصویر ورودی را می‌خواند (یا در صورت نبود، تصویر تصادفی می‌سازد).
2. چهار نوع نویز رایج را به آن اضافه می‌کند.
3. نتایج را در یک نمودار با عنوان‌های فارسی صحیح نمایش می‌دهد.
4. تمام متن‌های چاپی (پرینت) را به جای CMD در فایل output.txt ذخیره می‌کند.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display
import matplotlib.font_manager as fm
import os

# =========================== بخش ۱: تنظیم matplotlib برای فارسی ===========================
def persian_text(txt):
    """بازسازی و جهت‌دهی متن فارسی برای نمایش در matplotlib"""
    reshaped = arabic_reshaper.reshape(txt)
    return get_display(reshaped)

# تنظیم فونت فارسی برای matplotlib (در صورت موجود بودن Vazir)
font_path = 'Vazir-Regular.ttf'   # مسیر فایل فونت
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
else:
    print("هشدار: فایل فونت Vazir-Regular.ttf پیدا نشد. از فونت پیش‌فرض استفاده می‌شود.")

# =========================== بخش ۲: توابع کمکی نویز ===========================
def add_gaussian_noise(image, mean=0, sigma=25):
    noise = np.random.normal(mean, sigma, image.shape)
    noisy = image + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_salt_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    noisy = np.copy(image)
    total_pixels = image.size
    num_salt = int(total_pixels * salt_prob)
    num_pepper = int(total_pixels * pepper_prob)
    
    coords_salt = [np.random.randint(0, i-1, num_salt) for i in image.shape]
    noisy[coords_salt[0], coords_salt[1]] = 255
    
    coords_pepper = [np.random.randint(0, i-1, num_pepper) for i in image.shape]
    noisy[coords_pepper[0], coords_pepper[1]] = 0
    return noisy

def add_uniform_noise(image, low=-30, high=30):
    noise = np.random.uniform(low, high, image.shape)
    noisy = image + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_exponential_noise(image, scale=20):
    noise = np.random.exponential(scale, image.shape)
    noisy = image + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

# =========================== بخش ۳: نوشتن در فایل به جای CMD ===========================
# باز کردن فایل خروجی (با کدگذاری UTF-8)
output_file = open("noiseoutput.txt", "w", encoding="utf-8")

def log_print(text):
    """نوشتن متن در فایل output.txt (و همچنین نمایش در کنسول - اختیاری)"""
    output_file.write(text + "\n")
    output_file.flush()   # اطمینان از نوشتن سریع
    # اگر می‌خواهید همزمان در کنسول هم ببینید (اختیاری)، خط زیر را فعال کنید:
    # print(text)

# =========================== بخش ۴: بارگذاری تصویر ===========================
image_path = "noiseimage.jpg"  # نام فایل ورودی
try:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError
    log_print(f"تصویر با موفقیت بارگذاری شد. ابعاد: {img.shape}")
except:
    log_print("خطا: فایل تصویر پیدا نشد. یک تصویر تصادفی 256x256 ساخته شد.")
    img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)

img_float = img.astype(np.float64)

# =========================== بخش ۵: اعمال نویز ===========================
original_uint8 = img_float.astype(np.uint8)

noisy_gaussian = add_gaussian_noise(img_float, mean=0, sigma=25)
noisy_sp = add_salt_pepper_noise(original_uint8, salt_prob=0.02, pepper_prob=0.02)
noisy_uniform = add_uniform_noise(img_float, low=-30, high=30)
noisy_exponential = add_exponential_noise(img_float, scale=20)

# =========================== بخش ۶: نمایش نتایج با matplotlib ===========================
plt.figure(figsize=(15, 6))

plt.subplot(2, 3, 1)
plt.imshow(original_uint8, cmap='gray')
plt.title(persian_text('تصویر اصلی'), fontsize=12)
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(noisy_gaussian, cmap='gray')
plt.title(persian_text('نویز گاوسی (Gaussian)'), fontsize=12)
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(noisy_sp, cmap='gray')
plt.title(persian_text('نویز نمک و فلفل (Salt & Pepper)'), fontsize=12)
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(noisy_uniform, cmap='gray')
plt.title(persian_text('نویز یکنواخت (Uniform)'), fontsize=12)
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(noisy_exponential, cmap='gray')
plt.title(persian_text('نویز نمایی (Exponential)'), fontsize=12)
plt.axis('off')

plt.tight_layout()
plt.show()

# =========================== بخش ۷: ذخیره توضیحات ریاضی در فایل ===========================
log_print("\n--- توضیحات ریاضی نویزها (بر اساس فصل ۵ گونزالس) ---")
log_print("1. نویز گاوسی: PDF = (1/(σ√(2π))) * exp(-(z-μ)^2/(2σ^2))")
log_print("2. نویز نمک و فلفلی: مقادیر تصادفی 0 یا 255 با احتمالات مشخص")
log_print("3. نویز یکنواخت: PDF = 1/(b-a) برای a ≤ z ≤ b")
log_print("4. نویز نمایی: PDF = a*exp(-a*z) برای z≥0 که a = 1/scale")

# بستن فایل خروجی
output_file.close()
print("خروجی متنی با موفقیت در فایل output.txt ذخیره شد.")