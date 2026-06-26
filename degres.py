# -*- coding: utf-8 -*-
"""
کد کامل نهایی: مدل تخریب (گاوسی و حرکتی) و بازگردانی (معکوس، وینر، حداقل مربعات مقید)
بر اساس کتاب Digital Image Processing - Gonzalez & Woods, فصل 5

اصلاحات:
- Padding صحیح PSF با استفاده از ifftshift
- استفاده از fftconvolve برای تارشدگی پایدار
- تنظیم پارامترهای بهینه (epsilon, K, gamma)
- بهبود هسته تار حرکتی
- اضافه شدن پشتیبانی از نمایش فارسی در نمودارها و ذخیره خروجی در فایل
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import arabic_reshaper
from bidi.algorithm import get_display
import matplotlib.font_manager as fm
import os

# ======================= بخش اضافه شده برای رفع مشکل فارسی =======================
def persian_text(txt):
    """بازسازی و جهت‌دهی متن فارسی برای نمایش در matplotlib"""
    reshaped = arabic_reshaper.reshape(txt)
    return get_display(reshaped)

# تنظیم فونت فارسی (اختیاری)
font_path = 'Vazir-Regular.ttf'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()

# ======================= بخش ذخیره خروجی در فایل به جای CMD =======================
output_file = open("degresoutput.txt", "w", encoding="utf-8")

def log_print(text):
    """نوشتن متن در فایل output.txt"""
    output_file.write(text + "\n")
    output_file.flush()
# ============================================================================

# ----------------------------------------------------------------------
# مرحله 1: خواندن تصویر ورودی (خاکستری) و نرمالیزه به [0,1]
# ----------------------------------------------------------------------
image_path = "degresimage.jpg"  # نام فایل را در صورت نیاز تغییر دهید
try:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError
    log_print(f"تصویر اصلی با ابعاد {img.shape} بارگذاری شد.")
except:
    log_print("خطا: فایل تصویر یافت نشد. یک تصویر مصنوعی 256x256 ساخته می‌شود.")
    img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)

img_float = img.astype(np.float64) / 255.0  # نرمالیزه به [0,1]

# ----------------------------------------------------------------------
# مرحله 2: توابع ایجاد تارشدگی (هسته‌های تخریب) اصلاح‌شده
# ----------------------------------------------------------------------
def gaussian_kernel(size, sigma=1.0):
    """
    ایجاد هسته گاوسی دو بعدی (PSF) با ابعاد size×size و انحراف معیار sigma
    """
    kernel = np.zeros((size, size), dtype=np.float64)
    center = size // 2
    for i in range(size):
        for j in range(size):
            dx = i - center
            dy = j - center
            kernel[i, j] = (1/(2*np.pi*sigma**2)) * np.exp(-(dx**2 + dy**2)/(2*sigma**2))
    kernel = kernel / np.sum(kernel)
    return kernel

def motion_blur_kernel(length, angle):
    """
    ایجاد هسته تارشدگی حرکتی (Motion Blur) با طول مشخص (پیکسل) و زاویه (درجه)
    اصلاح شده: خطی با ضخامت 1 پیکسل و بدون شکستگی
    """
    kernel = np.zeros((length, length), dtype=np.float64)
    center = length // 2
    rad = np.deg2rad(angle)
    
    # اگر زاویه قائمه است، یک مقدار کوچک اضافه کن تا تقسیم بر صفر نشود
    if np.abs(np.cos(rad)) < 1e-6:
        rad = np.deg2rad(angle + 1e-6)
    
    tan_theta = np.tan(rad)
    for i in range(length):
        x = i - center
        y = x * tan_theta
        if abs(y) <= center:
            y_int = int(round(center + y))
            if 0 <= y_int < length:
                kernel[y_int, i] = 1.0
    # نرمالیزه کردن (اگر هیچ پیکسلی پر نشده بود، یک حرکت افقی فرض کن)
    if np.sum(kernel) < 1e-6:
        kernel[center, :] = 1.0
    kernel = kernel / np.sum(kernel)
    return kernel

# ----------------------------------------------------------------------
# مرحله 3: تابع اضافه کردن نویز گاوسی
# ----------------------------------------------------------------------
def add_gaussian_noise(image, sigma_noise=0.01):
    noise = np.random.normal(0, sigma_noise, image.shape)
    noisy = image + noise
    return np.clip(noisy, 0, 1)

# ----------------------------------------------------------------------
# مرحله 4: توابع بازگردانی اصلاح‌شده (با padding صحیح)
# ----------------------------------------------------------------------
def pad_psf(psf, target_shape):
    """
    PSF را به اندازه target_shape با قرارگیری مرکز PSF در مرکز تصویر، پد می‌کند.
    سپس ifftshift را اعمال می‌کند تا برای FFT آماده شود.
    """
    padded = np.zeros(target_shape, dtype=np.float64)
    h_center = psf.shape[0] // 2
    w_center = target_shape[0] // 2
    # اطمینان از قرارگیری کامل (در صورت فرد/زوج بودن ابعاد)
    rows, cols = psf.shape
    start_row = w_center - h_center
    start_col = w_center - h_center
    padded[start_row:start_row+rows, start_col:start_col+cols] = psf
    # اعمال ifftshift: منشأ PSF را به گوشه بالا‑چپ منتقل می‌کند (برای FFT)
    return ifftshift(padded)

def inverse_filter(degraded, psf, epsilon=1e-4):
    G = fft2(degraded)
    psf_padded = pad_psf(psf, degraded.shape)
    H = fft2(psf_padded)
    H_inv = np.zeros_like(H, dtype=np.complex128)
    mask = np.abs(H) > epsilon
    H_inv[mask] = 1.0 / H[mask]
    F_hat = H_inv * G
    restored = np.real(ifft2(F_hat))
    return np.clip(restored, 0, 1)

def wiener_filter(degraded, psf, K=0.001):
    G = fft2(degraded)
    psf_padded = pad_psf(psf, degraded.shape)
    H = fft2(psf_padded)
    H_conj = np.conj(H)
    wiener = H_conj / (np.abs(H)**2 + K)
    F_hat = wiener * G
    restored = np.real(ifft2(F_hat))
    return np.clip(restored, 0, 1)

def constrained_least_squares(degraded, psf, gamma=0.001):
    G = fft2(degraded)
    psf_padded = pad_psf(psf, degraded.shape)
    H = fft2(psf_padded)
    H_conj = np.conj(H)
    
    # عملگر لاپلاسین 3x3
    laplacian = np.array([[0, -1, 0],
                          [-1, 4, -1],
                          [0, -1, 0]], dtype=np.float64)
    lap_padded = pad_psf(laplacian, degraded.shape)
    P = fft2(lap_padded)
    
    denom = np.abs(H)**2 + gamma * (np.abs(P)**2)
    filt = H_conj / denom
    F_hat = filt * G
    restored = np.real(ifft2(F_hat))
    return np.clip(restored, 0, 1)

# ----------------------------------------------------------------------
# مرحله 5: انتخاب نوع تارشدگی و اعمال تخریب
# ----------------------------------------------------------------------
#print("\n--- انتخاب نوع تارشدگی ---")
#print("1. تارشدگی گاوسی (Gaussian Blur)")
#print("2. تارشدگی حرکتی (Motion Blur)")
choice = input("please select Gaussian (1) or Motion (2) Blur: ")

if choice == '1':
    sigma_blur = 2.0          # کاهش sigma=σ برای تارشدگی ملایم‌تر
    kernel_size = 11
    psf = gaussian_kernel(kernel_size, sigma_blur)
    blur_name = f"Gaussian (sigma={sigma_blur})"
    blur_name_fa = f"گاوسی (sigma={sigma_blur})"
elif choice == '2':
    length = 11
    angle = 30
    psf = motion_blur_kernel(length, angle)
    blur_name = f"Motion (len={length}, ang={angle})"
    blur_name_fa = f"حرکتی (طول={length}, زاویه={angle})"
else:
    log_print("ورودی نامعتبر. استفاده از گاوسی پیش‌فرض.")
    sigma_blur = 2.0
    kernel_size = 11
    psf = gaussian_kernel(kernel_size, sigma_blur)
    blur_name = "Gaussian (default)"
    blur_name_fa = "گاوسی (پیش‌فرض)"

# اعمال تارشدگی با fftconvolve (پایدارتر از convolve2d)
blurred = fftconvolve(img_float, psf, mode='same')

# اضافه کردن نویز گاوسی با شدت کم
sigma_noise = 0.01   # نویز خفیف
degraded = add_gaussian_noise(blurred, sigma_noise)

# ----------------------------------------------------------------------
# مرحله 6: اعمال روش‌های بازگردانی با پارامترهای بهینه
# ----------------------------------------------------------------------
restored_inv = inverse_filter(degraded, psf, epsilon=1e-4)
restored_wiener = wiener_filter(degraded, psf, K=0.001)
restored_cls = constrained_least_squares(degraded, psf, gamma=0.001)

# ----------------------------------------------------------------------
# مرحله 7: نمایش نتایج با عنوان‌های فارسی
# ----------------------------------------------------------------------
plt.figure(figsize=(15, 12))

plt.subplot(2, 3, 1)
plt.imshow(img_float, cmap='gray')
plt.title(persian_text('تصویر اصلی'), fontsize=12)
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(blurred, cmap='gray')
plt.title(persian_text(f'فقط تارشدگی ({blur_name_fa})'), fontsize=12)
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(degraded, cmap='gray')
plt.title(persian_text(f'تخریب‌شده: تار + نویز (sigma={sigma_noise})'), fontsize=12)
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(restored_inv, cmap='gray')
plt.title(persian_text('فیلتر معکوس'), fontsize=12)
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(restored_wiener, cmap='gray')
plt.title(persian_text('فیلتر وینر'), fontsize=12)
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(restored_cls, cmap='gray')
plt.title(persian_text('حداقل مربعات مقید'), fontsize=12)
plt.axis('off')

plt.suptitle(persian_text(' '), fontsize=14)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------
# ذخیره اطلاعات در فایل output.txt
# ----------------------------------------------------------------------
log_print("\n--- خلاصه اجرا ---")
log_print(f"نوع تارشدگی: {blur_name}")
log_print(f"شدت نویز گاوسی: sigma = {sigma_noise}")
log_print("\n--- فرمول‌های استفاده شده ---")
log_print("فیلتر معکوس: F_hat = G / H  (با آستانه ε)")
log_print("فیلتر وینر:   F_hat = (H* / (|H|^2 + K)) * G")
log_print("حداقل مربعات مقید: F_hat = (H* / (|H|^2 + γ|P|^2)) * G")

output_file.close()
print(".txt saved successfully")