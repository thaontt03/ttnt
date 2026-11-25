"""
Module: codebook.py
Mô tả: Bảng mã tiếng Việt cho các biến phân loại của dataset UCI
       "Predict Students' Dropout and Academic Success" (Realinho et al., 2022)
       và tên hiển thị tiếng Việt cho 36 đặc trưng.
"""

from __future__ import annotations

FEATURE_VI = {
    "Marital status": "Tình trạng hôn nhân",
    "Application mode": "Hình thức xét tuyển",
    "Application order": "Thứ tự nguyện vọng",
    "Course": "Ngành học",
    "Daytime/evening attendance": "Ca học",
    "Previous qualification": "Trình độ trước khi nhập học",
    "Previous qualification (grade)": "Điểm trình độ trước",
    "Nacionality": "Quốc tịch",
    "Mother's qualification": "Trình độ của mẹ",
    "Father's qualification": "Trình độ của bố",
    "Mother's occupation": "Nghề nghiệp của mẹ",
    "Father's occupation": "Nghề nghiệp của bố",
    "Admission grade": "Điểm xét tuyển",
    "Displaced": "Học xa nhà",
    "Educational special needs": "Nhu cầu giáo dục đặc biệt",
    "Debtor": "Đang nợ học phí",
    "Tuition fees up to date": "Đóng học phí đúng hạn",
    "Gender": "Giới tính",
    "Scholarship holder": "Có học bổng",
    "Age at enrollment": "Tuổi nhập học",
    "International": "Sinh viên quốc tế",
    "Curricular units 1st sem (credited)": "HK1 · số môn được công nhận",
    "Curricular units 1st sem (enrolled)": "HK1 · số môn đăng ký",
    "Curricular units 1st sem (evaluations)": "HK1 · số lượt dự thi",
    "Curricular units 1st sem (approved)": "HK1 · số môn đạt",
    "Curricular units 1st sem (grade)": "HK1 · điểm trung bình",
    "Curricular units 1st sem (without evaluations)": "HK1 · số môn không dự thi",
    "Curricular units 2nd sem (credited)": "HK2 · số môn được công nhận",
    "Curricular units 2nd sem (enrolled)": "HK2 · số môn đăng ký",
    "Curricular units 2nd sem (evaluations)": "HK2 · số lượt dự thi",
    "Curricular units 2nd sem (approved)": "HK2 · số môn đạt",
    "Curricular units 2nd sem (grade)": "HK2 · điểm trung bình",
    "Curricular units 2nd sem (without evaluations)": "HK2 · số môn không dự thi",
    "Unemployment rate": "Tỷ lệ thất nghiệp (%)",
    "Inflation rate": "Tỷ lệ lạm phát (%)",
    "GDP": "Tăng trưởng GDP (%)",
}

MARITAL = {1: "Độc thân", 2: "Đã kết hôn", 3: "Goá", 4: "Ly hôn",
           5: "Sống chung không đăng ký", 6: "Ly thân"}

GENDER = {0: "Nữ", 1: "Nam"}
ATTENDANCE = {1: "Ban ngày", 0: "Buổi tối"}
YES_NO = {0: "Không", 1: "Có"}

COURSE = {
    33: "Công nghệ sản xuất nhiên liệu sinh học",
    171: "Thiết kế hoạt hình & đa phương tiện",
    8014: "Công tác xã hội (buổi tối)",
    9003: "Nông học",
    9070: "Thiết kế truyền thông",
    9085: "Điều dưỡng thú y",
    9119: "Kỹ thuật tin học",
    9130: "Chăn nuôi ngựa",
    9147: "Quản trị",
    9238: "Công tác xã hội",
    9254: "Du lịch",
    9500: "Điều dưỡng",
    9556: "Vệ sinh răng miệng",
    9670: "Quản trị quảng cáo & marketing",
    9773: "Báo chí & truyền thông",
    9853: "Giáo dục cơ bản",
    9991: "Quản trị (buổi tối)",
}

APPLICATION_MODE = {
    1: "Đợt 1 – chỉ tiêu chung",
    2: "Theo Nghị định 612/93",
    5: "Đợt 1 – chỉ tiêu riêng (đảo Azores)",
    7: "Đã có bằng đại học khác",
    10: "Theo Nghị định 854-B/99",
    15: "Sinh viên quốc tế (cử nhân)",
    16: "Đợt 1 – chỉ tiêu riêng (đảo Madeira)",
    17: "Đợt 2 – chỉ tiêu chung",
    18: "Đợt 3 – chỉ tiêu chung",
    26: "Nghị định 533-A/99 b2 (kế hoạch khác)",
    27: "Nghị định 533-A/99 b3 (trường khác)",
    39: "Trên 23 tuổi",
    42: "Chuyển trường",
    43: "Chuyển ngành",
    44: "Có bằng chuyên ngành công nghệ",
    51: "Đổi trường/ngành",
    53: "Có bằng chu kỳ ngắn",
    57: "Đổi trường/ngành (quốc tế)",
}

PREV_QUALIFICATION = {
    1: "Tốt nghiệp THPT",
    2: "Cử nhân",
    3: "Đại học",
    4: "Thạc sĩ",
    5: "Tiến sĩ",
    6: "Đã từng học đại học",
    9: "Chưa hoàn thành lớp 12",
    10: "Chưa hoàn thành lớp 11",
    12: "Khác – lớp 11",
    14: "Lớp 10",
    15: "Chưa hoàn thành lớp 10",
    19: "Giáo dục cơ bản chu kỳ 3 (lớp 9–11)",
    38: "Giáo dục cơ bản chu kỳ 2 (lớp 6–8)",
    39: "Khoá chuyên ngành công nghệ",
    40: "Đại học (chu kỳ 1)",
    42: "Cao đẳng kỹ thuật chuyên nghiệp",
    43: "Thạc sĩ (chu kỳ 2)",
}

PARENT_QUALIFICATION = {
    1: "THPT (lớp 12)",
    2: "Cử nhân",
    3: "Đại học",
    4: "Thạc sĩ",
    5: "Tiến sĩ",
    6: "Đã từng học đại học",
    9: "Chưa hoàn thành lớp 12",
    10: "Chưa hoàn thành lớp 11",
    11: "Lớp 7 (hệ cũ)",
    12: "Khác – lớp 11",
    13: "Năm 2 khoá bổ túc THPT",
    14: "Lớp 10",
    18: "Khoá thương mại tổng quát",
    19: "Giáo dục cơ bản chu kỳ 3 (lớp 9–11)",
    20: "Khoá bổ túc THPT",
    22: "Khoá kỹ thuật – nghề",
    25: "Khoá bổ túc THPT (chưa xong)",
    26: "Lớp 7",
    27: "Chu kỳ 2 THPT tổng quát",
    29: "Chưa hoàn thành lớp 9",
    30: "Lớp 8",
    31: "Khoá hành chính & thương mại",
    33: "Kế toán & hành chính bổ túc",
    34: "Không rõ",
    35: "Không biết đọc viết",
    36: "Biết đọc nhưng chưa hết lớp 4",
    37: "Giáo dục cơ bản chu kỳ 1 (lớp 4–5)",
    38: "Giáo dục cơ bản chu kỳ 2 (lớp 6–8)",
    39: "Khoá chuyên ngành công nghệ",
    40: "Đại học (chu kỳ 1)",
    41: "Khoá nghiên cứu chuyên sâu",
    42: "Cao đẳng kỹ thuật chuyên nghiệp",
    43: "Thạc sĩ (chu kỳ 2)",
    44: "Tiến sĩ (chu kỳ 3)",
}

OCCUPATION = {
    0: "Sinh viên",
    1: "Lãnh đạo, quản lý cấp cao",
    2: "Chuyên gia khoa học – trí thức",
    3: "Kỹ thuật viên trung cấp",
    4: "Nhân viên hành chính",
    5: "Dịch vụ cá nhân, bảo vệ, bán hàng",
    6: "Nông – lâm – ngư nghiệp",
    7: "Thợ lành nghề công nghiệp, xây dựng",
    8: "Vận hành máy móc, lắp ráp",
    9: "Lao động phổ thông",
    10: "Lực lượng vũ trang",
    90: "Tình trạng khác",
    99: "(không ghi)",
    101: "Sĩ quan quân đội",
    102: "Hạ sĩ quan quân đội",
    103: "Quân nhân khác",
    112: "Giám đốc hành chính – thương mại",
    114: "Quản lý khách sạn, nhà hàng, bán lẻ",
    121: "Chuyên gia khoa học tự nhiên, kỹ thuật",
    122: "Chuyên gia y tế",
    123: "Giáo viên",
    124: "Chuyên gia tài chính, kế toán",
    125: "Chuyên gia CNTT – truyền thông",
    131: "Kỹ thuật viên khoa học, kỹ thuật",
    132: "Kỹ thuật viên y tế",
    134: "Kỹ thuật viên pháp lý, xã hội, văn hoá",
    135: "Kỹ thuật viên CNTT – truyền thông",
    141: "Nhân viên văn phòng, thư ký",
    143: "Nhân viên kế toán, thống kê",
    144: "Nhân viên hỗ trợ hành chính khác",
    151: "Nhân viên dịch vụ cá nhân",
    152: "Nhân viên bán hàng",
    153: "Nhân viên chăm sóc cá nhân",
    154: "Bảo vệ, an ninh",
    161: "Nông dân sản xuất hàng hoá",
    163: "Nông dân tự cung tự cấp",
    171: "Thợ xây dựng",
    172: "Thợ cơ khí, luyện kim",
    173: "Thợ in, chế tác chính xác, thủ công",
    174: "Thợ điện, điện tử",
    175: "Thợ chế biến thực phẩm, gỗ, may mặc",
    181: "Vận hành máy cố định",
    182: "Công nhân lắp ráp",
    183: "Lái xe, vận hành thiết bị di động",
    191: "Nhân viên vệ sinh",
    192: "Lao động phổ thông nông nghiệp",
    193: "Lao động phổ thông công nghiệp, vận tải",
    194: "Phụ bếp",
    195: "Bán hàng rong",
}

NATIONALITY = {
    1: "Bồ Đào Nha", 2: "Đức", 6: "Tây Ban Nha", 11: "Ý", 13: "Hà Lan", 14: "Anh",
    17: "Litva", 21: "Angola", 22: "Cabo Verde", 24: "Guinea", 25: "Mozambique",
    26: "São Tomé", 32: "Thổ Nhĩ Kỳ", 41: "Brazil", 62: "Romania", 100: "Moldova",
    101: "Mexico", 103: "Ukraina", 105: "Nga", 108: "Cuba", 109: "Colombia",
}

CODEBOOK = {
    "Marital status": MARITAL,
    "Application mode": APPLICATION_MODE,
    "Course": COURSE,
    "Daytime/evening attendance": ATTENDANCE,
    "Previous qualification": PREV_QUALIFICATION,
    "Nacionality": NATIONALITY,
    "Mother's qualification": PARENT_QUALIFICATION,
    "Father's qualification": PARENT_QUALIFICATION,
    "Mother's occupation": OCCUPATION,
    "Father's occupation": OCCUPATION,
    "Gender": GENDER,
    "Displaced": YES_NO,
    "Educational special needs": YES_NO,
    "Debtor": YES_NO,
    "Tuition fees up to date": YES_NO,
    "Scholarship holder": YES_NO,
    "International": YES_NO,
}


def label(feature: str, code) -> str:
    """Nhãn tiếng Việt của một mã; trả về 'Mã x' nếu chưa có trong bảng."""
    try:
        code = int(code)
    except (TypeError, ValueError):
        return str(code)
    return CODEBOOK.get(feature, {}).get(code, f"Mã {code}")


def feature_vi(feature: str) -> str:
    return FEATURE_VI.get(feature, feature)
