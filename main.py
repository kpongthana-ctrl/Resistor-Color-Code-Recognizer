import cv2
import numpy as np

def show_image(title, img, width=800, height=600):
    im_show = cv2.resize(img, (width, height))
    cv2.imshow(title, im_show)

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect
def find_perspective_resistor(cnt, orig):
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype="float32")
    ordered_box = order_points(box)
    (tl, tr, br, bl) = ordered_box
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(ordered_box, dst)
    warp = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

    if warp.shape[1] > warp.shape[0]:
        warp = cv2.rotate(warp, cv2.ROTATE_90_CLOCKWISE)
    return M, warp
def fix_resistor_orientation(res_img):
    hsv = cv2.cvtColor(res_img, cv2.COLOR_BGR2HSV)
    lower_gold = np.array([15, 50, 80])
    upper_gold = np.array([35, 255, 255])
    mask_gold = cv2.inRange(hsv, lower_gold, upper_gold)
    lower_silver = np.array([0, 0, 120]) 
    upper_silver = np.array([180, 70, 255])
    mask_silver = cv2.inRange(hsv, lower_silver, upper_silver)

    mask_tolerance = cv2.bitwise_or(mask_gold, mask_silver)
    h, w = mask_tolerance.shape
    top = mask_tolerance[0:int(h*0.3), :]
    bottom = mask_tolerance[int(h*0.7):h, :]
    top_count = cv2.countNonZero(top)
    bottom_count = cv2.countNonZero(bottom)
    if top_count > bottom_count:
        res_img = cv2.flip(res_img, 0)

    return res_img
def remove_glare(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, None, iterations=2)
    result = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    return result

def normalize_illumination(img):
    # *** แก้ไข: กำหนด Target Luminance ให้สูงขึ้น (180-200) เพื่อยกโทนสว่าง ***
    TARGET_LUMINANCE = 110.0 # ลองปรับค่านี้ ถ้าภาพยังมืดไปให้เพิ่ม (สูงสุด 220)
    
    # คำนวณแผนที่เงาจากช่องสีเทา
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # สร้าง Shadow Map ด้วย Gaussian Blur
    ksize = int(img.shape[1] / 3)
    if ksize % 2 == 0: ksize += 1
    kernel = (ksize, ksize)
    bg_blur = cv2.GaussianBlur(gray, kernel, 0)
    
    # คำนวณ Factor: (Target / Shadow Map)
    bg_blur_f = bg_blur.astype(np.float32) + 1.0
    factor = TARGET_LUMINANCE / bg_blur_f # ใช้ค่าคงที่ที่กำหนด
    
    # ใช้ Factor ไปคูณกับภาพสีต้นฉบับทั้ง 3 ช่อง
    factor_color = cv2.merge([factor, factor, factor])
    
    # กำหนด dtype=cv2.CV_32F เพื่อแก้ error
    normalized_img = cv2.multiply(img.astype(np.float32), factor_color, dtype=cv2.CV_32F)
    
    # ตัดค่าที่เกิน 255 ทิ้ง
    normalized_img[normalized_img > 255] = 255
    return normalized_img.astype(np.uint8)

def adjust_gamma(image, gamma=1.2):
    # ใช้เพื่อดึงรายละเอียดในสีอ่อนอีกครั้งหลัง Normalization
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)
def boost_saturation(img, factor=1.5):
    """
    เพิ่มความอิ่มตัวของสี (Saturation) ในภาพ
    
    Parameters:
        factor (float): ตัวคูณความอิ่มตัว (แนะนำ 1.2 ถึง 1.8)
    """
    # 1. แปลงเป็น HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # 2. เพิ่ม Saturation (S channel)
    s = s.astype(np.float32) * factor
    
    # 3. จำกัดค่าสูงสุดไม่ให้เกิน 255
    s[s > 255] = 255
    
    # 4. รวมกลับเป็น BGR
    s = s.astype(np.uint8)
    final_hsv = cv2.merge([h, s, v])
    result = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return result
def apply_soft_clahe(img):
    # ใช้ CLAHE ที่ Clip Limit ต่ำมาก (0.8 ถึง 1.5) เพื่อเพิ่มความคมชัดของขอบ
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8,8)) # ClipLimit 1.2 คืออ่อนมาก
    cl = clahe.apply(l)
    
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final
def repad_resistor_body(img):
    h, w= img.shape[0:2]
    crop_left = int(w * 0.20)   
    crop_right = int(w * 0.80)  
    crop_top = int(h * 0.10)    
    crop_bottom = int(h * 0.90) 
    cropped_img = img[crop_top:crop_bottom, crop_left:crop_right]
    repad = np.zeros_like(img)
    repad[crop_top:crop_bottom, crop_left:crop_right] = cropped_img
    return repad
def zero_out_rows_with_zeroes(binary_image_2d):
    row_min = np.min(binary_image_2d, axis=1)
    rows_to_zero = row_min == 0
    result_image = binary_image_2d.copy()
    result_image[rows_to_zero, :] = 0  
    return result_image
def filter_by_connected_components(binary_image, min_area_threshold=100):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    filtered_image = np.zeros_like(binary_image)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area_threshold:
            component_mask = (labels == i).astype(np.uint8) * 255
            filtered_image = cv2.bitwise_or(filtered_image, component_mask)            
    return filtered_image
def get_band_mask(resister_img):
    blur = cv2.GaussianBlur(cv2.cvtColor(resister_img, cv2.COLOR_BGR2GRAY), (5,5),0)
    canny = cv2.Canny(blur, 50, 150) 
    band_mask = repad_resistor_body(canny)
    band_mask = cv2.morphologyEx(band_mask, cv2.MORPH_CLOSE,  cv2.getStructuringElement(cv2.MORPH_RECT, (canny.shape[1]//2,5)), iterations=1)
    band_mask = filter_by_connected_components(band_mask, min_area_threshold=50)
    band_mask = cv2.morphologyEx(band_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (int(canny.shape[1]*1.2),canny.shape[0]//15)), iterations=1) #assume full width bands
    band_mask = zero_out_rows_with_zeroes(band_mask)
    band_mask = cv2.morphologyEx(band_mask, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)), iterations=2)
    return band_mask
def get_band_locations(eroded_bands_image):
    if len(eroded_bands_image.shape) == 3:
        gray_image = cv2.cvtColor(eroded_bands_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = eroded_bands_image
    _, binary_mask = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    band_locations = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        band_locations.append({'x': x, 'y': y, 'w': w, 'h': h, 'contour': cnt})
    band_locations.sort(key=lambda item: item['y'])    
    return band_locations
def classify_color(h_value, s_value, v_value):   
    # 1. ตรวจสอบ Gold/Silver/White (Tolerance Colors)
    # **GOLD / SILVER:** V ต้องสูงกว่า V เฉลี่ยของภาพ (ลดเกณฑ์ V จาก 120 เป็น 110)
    if v_value >= 110: # ลดเกณฑ์ V ลงเพื่อให้ Gold ที่มืดกว่าถูก Detect ได้
        if s_value < 50:
            # S ต่ำมาก, V สูง -> Silver หรือ White
            if v_value >= 200:
                return "White"
            return "Silver"
        # Gold: H อยู่ในช่วงส้ม/เหลือง (15-35)
        if 15 <= h_value <= 35 and s_value >= 50: 
            return "Gold"
    # 2. ตรวจสอบ Black/Gray (V ต่ำ)
    if v_value < 50:
        if s_value < 30:
            return "Gray"
        return "Black"
    # 3. ตรวจสอบ Green ก่อน
    if 36 <= h_value <= 80:
        return "Green"
    # 4. ตรวจสอบ Brown/Orange/Red (H-based)
    # **BROWN:** H อยู่ในช่วง 0-28 และ V ต้องต่ำกว่า Orange/Gold (V <= 130)
    # Vmax ของ Brown (130) ยังคาบเกี่ยวกับ Vmin ของ Gold (110)
    if 0 <= h_value <= 28: 
        if v_value <= 130: 
            if s_value >= 50:
                # Logic พิเศษ: ถ้า H/V/S ใกล้เคียง Brown/Gold (H=15-28, V=110-130) 
                # และ H/V/S ไม่ผ่านเงื่อนไข Gold แสดงว่าเป็น Brown
                return "Brown"
    # **RED:** H ใกล้ 0 หรือ 180 
    if h_value <= 15 or h_value >= 165: 
        return "Red"
    # **ORANGE:** H อยู่ในช่วง 16-35 
    if 16 <= h_value <= 35:
        if v_value >= 100:
            return "Orange"
    # **BLUE / VIOLET / YELLOW:**
    if 90 <= h_value <= 130:
        return "Blue"
    if 135 <= h_value <= 155:
        return "Violet"
    if 29 <= h_value <= 35: 
        return "Yellow"
    return "Unknown"
def analyze_resistor_colors(original_color_img, bands_mask, band_locations):
    hsv_img = cv2.cvtColor(original_color_img, cv2.COLOR_BGR2HSV)
    band_colors = []
    for i, band in enumerate(band_locations):
        x, y, w, h = band['x'], band['y'], band['w'], band['h']
        band_mask_roi = bands_mask[y:y+h, x:x+w]
        hsv_roi = hsv_img[y:y+h, x:x+w]
        is_band_pixel = band_mask_roi > 0 
        if np.any(is_band_pixel):
            h_values = hsv_roi[:,:,0][is_band_pixel] 
            s_values = hsv_roi[:,:,1][is_band_pixel] 
            v_values = hsv_roi[:,:,2][is_band_pixel] 
            median_h = np.median(h_values)
            median_s = np.median(s_values)
            median_v = np.median(v_values)
            color_name = classify_color(median_h, median_s, median_v)
            band_colors.append(color_name)
        else:
            band_colors.append("Unknown")
            
    return band_colors
COLOR_CODES = {
    # Code (Band 1, 2)
    'Black': {'code': 0, 'multiplier': 1, 'tolerance': None},
    'Brown': {'code': 1, 'multiplier': 10, 'tolerance': 1.0},
    'Red':   {'code': 2, 'multiplier': 100, 'tolerance': 2.0},
    'Orange': {'code': 3, 'multiplier': 1000, 'tolerance': None},
    'Yellow': {'code': 4, 'multiplier': 10000, 'tolerance': None},
    'Green': {'code': 5, 'multiplier': 100000, 'tolerance': 0.5},
    'Blue':  {'code': 6, 'multiplier': 1000000, 'tolerance': 0.25},
    'Violet': {'code': 7, 'multiplier': 10000000, 'tolerance': 0.1},
    'Gray':  {'code': 8, 'multiplier': None, 'tolerance': None},
    'White': {'code': 9, 'multiplier': None, 'tolerance': None},
    # Tolerance (Band 4)
    'Gold':  {'code': None, 'multiplier': 0.1, 'tolerance': 5.0},
    'Silver': {'code': None, 'multiplier': 0.01, 'tolerance': 10.0},
}

def calculate_resistance_value(color_bands):
    if len(color_bands) != 4:
        return {'error': 'ต้องมี 4 แถบสี'}
    try:
        band1_color = color_bands[0]
        band2_color = color_bands[1]
        band3_color = color_bands[2] # Multiplier
        band4_color = color_bands[3] # Tolerance
        
        # 1. รหัสตัวเลข (Band 1 และ Band 2)
        code1 = COLOR_CODES[band1_color]['code']
        code2 = COLOR_CODES[band2_color]['code']
        
        # 2. ตัวคูณ (Band 3)
        multiplier = COLOR_CODES[band3_color]['multiplier']
        
        # 3. ค่าความเผื่อ (Band 4)
        tolerance = COLOR_CODES[band4_color]['tolerance']
        
        # สูตร: Value = (Code1 * 10 + Code2) * Multiplier
        # รวมหลักที่ 1 และ 2 เป็นตัวเลขหลักคู่
        two_digit_code = code1 * 10 + code2
        
        resistance_ohm = two_digit_code * multiplier
        
        # แปลงเป็นหน่วยที่เหมาะสม (kOhm, MOhm) เพื่อให้อ่านง่าย
        if resistance_ohm >= 1_000_000:
            resistance_unit = f"{resistance_ohm / 1_000_000:.2f} MOhm"
        elif resistance_ohm >= 1_000:
            resistance_unit = f"{resistance_ohm / 1_000:.2f} kOhm"
        else:
            resistance_unit = f"{resistance_ohm:.2f} Ohm"
            
        return {
            'resistance_ohm': resistance_ohm,
            'resistance_unit': resistance_unit,
            'tolerance_percent': tolerance
        }
        
    except KeyError as e:
        return {'error': f"ไม่พบรหัสสี: {e}. ตรวจสอบชื่อสีที่ส่งเข้ามา"}
    except TypeError:
        return {'error': "ค่าตัวคูณหรือรหัสสีไม่สมบูรณ์ (เช่น Brown เป็น Tolerance)"}
def find_resistor_roi(img):
    b,g,r = cv2.split(img)
    diiff = cv2.absdiff(b, g)
    diiff2 = cv2.absdiff(b, r)
    diiff3 = cv2.absdiff(g, r)
    bgr_mask =  ((diiff>10) & (diiff2>10) & (diiff3>10)).astype(np.uint8)*255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    bgr_mask = cv2.morphologyEx(bgr_mask, cv2.MORPH_OPEN, kernel, iterations=3)
    bgr_mask = cv2.morphologyEx(bgr_mask, cv2.MORPH_CLOSE, kernel, iterations=9)
    return bgr_mask
def main():
    # name = "serie"
    # name = "paralell"
    name = "mix"
    path = f"images/{name}.jfif"
    img = cv2.imread(path)
    if img is None:
        print(f"Error: Cannot load image at {path}")
        return
    # Resize ให้ประมวลผลเร็วและดูง่าย
    if img.shape[1] > 1000:
        scale = 1000 / img.shape[1]
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    orig = img.copy()
    bgr_mask = find_resistor_roi(img)
    contours, _ = cv2.findContours(bgr_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # aligned_resistors = list()
    for i,cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < 1000 :  # filter noise
            continue
        _,aligned_resistor = find_perspective_resistor(cnt, cv2.bitwise_and(img, img, mask=bgr_mask))
        aligned_resistor = fix_resistor_orientation(aligned_resistor)
        # Enhancement resistor image
        gamma = adjust_gamma(aligned_resistor, gamma=1.2)
        img_no_glare = remove_glare(gamma)
        norm_illu = normalize_illumination(img_no_glare)
        clahe = apply_soft_clahe(norm_illu)
        boosted_sat = boost_saturation(clahe, factor=1.4)
        # Get band resistor mask
        band_mask = get_band_mask(boosted_sat)
        bands = cv2.bitwise_and(boosted_sat, boosted_sat, mask=band_mask)
        band_location = get_band_locations(bands)
        band_colors = analyze_resistor_colors(boosted_sat, band_mask, band_location)
        print(f"Resistor {i+1} bands: {band_colors}")
        resistance_info = calculate_resistance_value(band_colors)
        text = ""
        if 'error' in resistance_info:
            text = f"Error calculating resistance: {resistance_info['error']}"
        else:
            text = (f"#{i} {resistance_info['resistance_unit']} +- {resistance_info['tolerance_percent']}%")
        print(text)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, text, (x-50, y-5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    cv2.imshow(f"aligned_resistor_fixed {i}", img)
    cv2.imwrite(f"out/{name}_out.jpg", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()