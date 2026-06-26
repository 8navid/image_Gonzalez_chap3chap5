# -*- coding: utf-8 -*-
"""
کد سوم: بهبود تصویر (Image Enhancement) – بدون استفاده از مدل تخریب
بر اساس کتاب Digital Image Processing - Gonzalez & Woods, فصل 3 (مقایسه با فصل 5)

بهبود تصویر (Enhancement) برخلاف بازگردانی (Restoration):
- نیاز به دانستن مدل تخریب (h و η) ندارد.
- روش‌ها عمدتاً اکتشافی (Heuristic) و سلیقه‌ای هستند.
- هدف: قشنگ‌تر و قابل‌دیدن‌تر کردن تصویر، نه لزوماً بازگردانی فیزیکی آن.

در این کد:
1. تصویر ورودی را می‌خوانیم.
2. چند روش ساده بهبود را اعمال می‌کنیم:
   - بهبود کنتراست با کشش هیستوگرام (Contrast Stretching)
   - یکسان‌سازی هیستوگرام (Histogram Equalization)
   - فیلتر ساده میانگین‌گیر (برای کاهش نویز – بدون مدل نویز)
3. نتایج را با تصویر اصلی مقایسه می‌کنیم.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
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
output_file = open("output.txt", "w", encoding="utf-8")

def log_print(text):
    """نوشتن متن در فایل output.txt"""
    output_file.write(text + "\n")
    output_file.flush()
# ============================================================================

# ----------------------------------------------------------------------
# مرحله 1: خواندن تصویر ورودی (خاکستری)
# ----------------------------------------------------------------------
image_path = "enhanceimage.jpg"  # نام فایل را در صورت نیاز تغییر دهید
try:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError
    log_print(f"تصویر اصلی با ابعاد {img.shape} بارگذاری شد.")
except:
    log_print("خطا: فایل تصویر یافت نشد. یک تصویر مصنوعی تیره ساخته می‌شود.")
    # یک تصویر تیره مصنوعی برای نمایش اثر بهبود
    img = np.random.randint(50, 100, (256, 256), dtype=np.uint8)

# ----------------------------------------------------------------------
# مرحله 2: تعریف توابع بهبود (بدون مدل تخریب)
# ----------------------------------------------------------------------

def contrast_stretching(image, low_percent=2, high_percent=98):
    """
    کشش خطی هیستوگرام (Contrast Stretching)
    مقادیر پایین‌تر از صدک low_percent را به 0 و بالاتر از صدک high_percent را به 255 نگاشت می‌کند.
    روشی ساده برای افزایش کنتراست.
    """
    low_val = np.percentile(image, low_percent)
    high_val = np.percentile(image, high_percent)
    stretched = np.clip((image - low_val) * (255.0 / (high_val - low_val)), 0, 255)
    return stretched.astype(np.uint8)

def histogram_equalization(image):
    """
    یکسان‌سازی هیستوگرام (Histogram Equalization)
    تابع توزیع تجمعی (CDF) هیستوگرام را به یک خط صاف تبدیل می‌کند.
    باعث افزایش کنتراست به صورت خودکار می‌شود.
    """
    return cv2.equalizeHist(image)

def mean_filter(image, kernel_size=5):
    """
    فیلتر میانگین (میانگین‌گیر ساده) – یک فیلتر پایین‌گذر
    هر پیکسل با میانگین همسایگان خود جایگزین می‌شود.
    بدون دانستن نوع نویز انجام می‌شود (روش Enhancement).
    """
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
    filtered = cv2.filter2D(image, -1, kernel)
    return filtered.astype(np.uint8)

# ----------------------------------------------------------------------
# مرحله 3: اعمال روش‌های بهبود بر روی تصویر اصلی
# ----------------------------------------------------------------------
img_enhanced_stretch = contrast_stretching(img, low_percent=2, high_percent=98)
img_enhanced_histeq = histogram_equalization(img)
img_enhanced_mean = mean_filter(img, kernel_size=5)

# ----------------------------------------------------------------------
# مرحله 4: نمایش نتایج در کنار هم (با عنوان‌های فارسی)
# ----------------------------------------------------------------------
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(img, cmap='gray')
plt.title(persian_text('تصویر اصلی (کم‌کنتراست)'), fontsize=12)
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(img_enhanced_stretch, cmap='gray')
plt.title(persian_text('بهبود: کشش هیستوگرام'), fontsize=12)
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(img_enhanced_histeq, cmap='gray')
plt.title(persian_text('بهبود: یکسان‌سازی هیستوگرام'), fontsize=12)
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(img_enhanced_mean, cmap='gray')
plt.title(persian_text('بهبود: فیلتر میانگین (کاهش نویز)'), fontsize=12)
plt.axis('off')

# نمایش هیستوگرام تصویر اصلی و نسخه یکسان‌سازی شده
plt.subplot(2, 3, 5)
plt.hist(img.ravel(), bins=256, range=(0,255), alpha=0.7, label=persian_text('اصلی'))
plt.hist(img_enhanced_histeq.ravel(), bins=256, range=(0,255), alpha=0.7, label=persian_text('یکسان‌سازی'))
plt.legend()
plt.title(persian_text('مقایسه هیستوگرام'))

plt.subplot(2, 3, 6)
plt.text(0.1, 0.5, 
         persian_text('نکته: روش‌های Enhancement هیچ مدل ریاضی از تخریب (h و η) ندارند.\n'
         'آنها صرفاً تصویر را به صورت سلیقه‌ای بهبود می‌بخشند.\n'
         'در مقابل، Restoration نیاز به دانستن h و η دارد.\n'
         'مثال: فیلتر میانگین ساده (اینجا) بدون تشخیص نوع نویز استفاده شده است.\n'
         'در Restoration با دانستن مدل نویز، فیلتر بهینه (مثل وینر) بکار می‌رود.'),
         fontsize=10, verticalalignment='center')
plt.axis('off')

plt.suptitle(persian_text(' '), fontsize=14)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------
# ذخیره توضیحات مفهومی در فایل output.txt
# ----------------------------------------------------------------------
log_print("\n--- تفاوت Enhancement و Restoration (مطابق فصل 3 و 5 گونزالس) ---")
log_print("Enhancement (اینجا):")
log_print("  - مبتنی بر سلیقه (Heuristic)")
log_print("  - به مدل تخریب نیاز ندارد")
log_print("  - مثال: کشش هیستوگرام، یکسان‌سازی، فیلتر میانگین ساده")
log_print("Restoration (در کد بعدی):")
log_print("  - مبتنی بر مدل ریاضی تخریب (h و η)")
log_print("  - هدف: بازگردانی تصویر اصلی با حذف دقیق تخریب")
log_print("  - مثال: فیلتر وینر، فیلتر معکوس (Inverse Filter)")

output_file.close()
print("خروجی متنی با موفقیت در فایل output.txt ذخیره شد.")