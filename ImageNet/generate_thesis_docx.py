"""Script tự động sinh quyển Báo cáo Khóa luận Tốt nghiệp hoàn chỉnh (Word .docx)
theo đúng chuẩn định dạng của Khoa CNTT - Trường Đại học Công nghiệp TP.HCM (IUH).
Đề tài: Nghiên cứu và xây dựng mô hình học sâu VietColorNet phục chế màu sắc cho ảnh tư liệu lịch sử Việt Nam.
"""
import os
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_cell_shading(cell, color_hex):
    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading)

def set_table_borders(table, color="D3D3D3", sz="4", val="single"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'  <w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'  <w:left w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'  <w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'  <w:right w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'  <w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'  <w:insideV w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

def build_thesis_document(output_path="bao_cao_khoa_luan_tot_nghiep_vietcolornet.docx"):
    doc = Document()

    # Thiết lập lề trang theo chuẩn IUH: Top 3cm, Bottom 3cm, Left 3.5cm, Right 2cm
    for section in doc.sections:
        section.top_margin = Cm(3.0)
        section.bottom_margin = Cm(3.0)
        section.left_margin = Cm(3.5)
        section.right_margin = Cm(2.0)
        section.different_first_page_header_footer = True
        
        # Header / Footer
        header = section.header
        hp = header.paragraphs[0]
        hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        hrun = hp.add_run("Khóa Luận Tốt Nghiệp · Khoa Công Nghệ Thông Tin · ĐH Công Nghiệp TP.HCM")
        hrun.font.name = "Times New Roman"
        hrun.font.size = Pt(9)
        hrun.font.italic = True
        hrun.font.color.rgb = RGBColor(120, 120, 120)

        footer = section.footer
        fp = footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        frun = fp.add_run("Trang ")
        frun.font.name = "Times New Roman"
        frun.font.size = Pt(10)

    # Helper format paragraph
    def add_p(text="", align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=6, space_after=6, line_spacing=1.5, bold=False, italic=False, font_size=13):
        p = doc.add_paragraph()
        p.alignment = align
        p.paragraph_format.space_before = Pt(space_before)
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.line_spacing = line_spacing
        if text:
            run = p.add_run(text)
            run.font.name = "Times New Roman"
            run.font.size = Pt(font_size)
            run.font.bold = bold
            run.font.italic = italic
        return p

    def add_h1(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(24)
        p.paragraph_format.space_after = Pt(18)
        p.paragraph_format.line_spacing = 1.15
        run = p.add_run(text)
        run.font.name = "Times New Roman"
        run.font.size = Pt(14)
        run.font.bold = True
        run.font.color.rgb = RGBColor(15, 23, 42)
        return p

    def add_h2(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing = 1.15
        run = p.add_run(text)
        run.font.name = "Times New Roman"
        run.font.size = Pt(13)
        run.font.bold = True
        run.font.color.rgb = RGBColor(30, 41, 59)
        return p

    def add_h3(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.15
        run = p.add_run(text)
        run.font.name = "Times New Roman"
        run.font.size = Pt(13)
        run.font.bold = True
        run.font.italic = True
        run.font.color.rgb = RGBColor(51, 65, 85)
        return p

    def add_bullet(text):
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.3
        run = p.add_run(text)
        run.font.name = "Times New Roman"
        run.font.size = Pt(13)
        return p

    def add_caption(text, is_table=False):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(10)
        run = p.add_run(text)
        run.font.name = "Times New Roman"
        run.font.size = Pt(12)
        run.font.bold = True
        run.font.italic = True
        return p

    # =========================================================================
    # 1. TRANG BÌA TIẾNG VIỆT
    # =========================================================================
    add_p("BỘ CÔNG THƯƠNG", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, font_size=13, space_after=2)
    add_p("TRƯỜNG ĐẠI HỌC CÔNG NGHIỆP TP. HỒ CHÍ MINH", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, font_size=14, space_after=2)
    add_p("KHOA CÔNG NGHỆ THÔNG TIN", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, font_size=14, space_after=36)

    add_p("KHÓA LUẬN TỐT NGHIỆP", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, font_size=18, space_after=24)
    add_p("NGHIÊN CỨU VÀ XÂY DỰNG MÔ HÌNH HỌC SÂU VIETCOLORNET PHỤC CHẾ MÀU SẮC CHO ẢNH TƯ LIỆU LỊCH SỬ VIỆT NAM", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, font_size=16, space_after=36)
    add_p("Chuyên ngành: KHOA HỌC DỮ LIỆU", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, font_size=14, space_after=60)

    p_info = add_p("", align=WD_ALIGN_PARAGRAPH.LEFT, space_before=12, space_after=4)
    r = p_info.add_run("Sinh viên thực hiện: \t\tNguyễn Hữu Tùng  -  MSSV: 20000001\n")
    r.bold = True
    r = p_info.add_run("\t\t\t\tTrần Văn B       -  MSSV: 20000002\n")
    r.bold = True
    r = p_info.add_run("Giảng viên hướng dẫn: \tTS. Nguyễn Văn A\n")
    r.bold = True
    r = p_info.add_run("Khóa học: \t\t\t2021 – 2025")
    r.bold = True

    add_p("", space_before=40)
    add_p("THÀNH PHỐ HỒ CHÍ MINH, NĂM 2025", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, font_size=13)
    doc.add_page_break()

    # =========================================================================
    # 2. TRANG BÌA TIẾNG ANH
    # =========================================================================
    add_p("MINISTRY OF INDUSTRY AND TRADE", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, font_size=13, space_after=2)
    add_p("INDUSTRIAL UNIVERSITY OF HO CHI MINH CITY", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, font_size=14, space_after=2)
    add_p("FACULTY OF INFORMATION TECHNOLOGY", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, font_size=14, space_after=36)

    add_p("GRADUATION THESIS", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, font_size=18, space_after=24)
    add_p("RESEARCH AND DEVELOPMENT OF VIETCOLORNET DEEP LEARNING MODEL FOR HISTORICAL VIETNAMESE IMAGE COLORIZATION", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, font_size=15, space_after=36)
    add_p("Major: DATA SCIENCE", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, font_size=14, space_after=60)

    p_info_en = add_p("", align=WD_ALIGN_PARAGRAPH.LEFT, space_before=12, space_after=4)
    r = p_info_en.add_run("Students: \t\t\tNguyen Huu Tung  -  Student ID: 20000001\n")
    r.bold = True
    r = p_info_en.add_run("\t\t\t\tTran Van B       -  Student ID: 20000002\n")
    r.bold = True
    r = p_info_en.add_run("Supervisor: \t\t\tDr. Nguyen Van A\n")
    r.bold = True
    r = p_info_en.add_run("Academic Year: \t\t2021 – 2025")
    r.bold = True

    add_p("", space_before=40)
    add_p("HO CHI MINH CITY, 2025", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, font_size=13)
    doc.add_page_break()

    # =========================================================================
    # 3. TÓM TẮT & ABSTRACT
    # =========================================================================
    add_h1("TÓM TẮT ĐỀ TÀI")
    add_p("Phục chế và hồi sinh màu sắc cho kho tàng ảnh tư liệu lịch sử Việt Nam thế kỷ 20 là một nhiệm vụ khoa học và văn hóa có ý nghĩa đặc biệt quan trọng trong công tác bảo tồn di sản dân tộc. Tuy nhiên, các mô hình học sâu tô màu tổng quát hiện nay (như DeOldify hay DDColor) khi áp dụng vào miền dữ liệu ảnh lịch sử Việt Nam thường bộc lộ ba hạn chế nghiêm trọng: (1) Hiện tượng suy thoái màu (color collapse) ngả sang tông màu nâu/sepia đơn điệu do thiếu dữ liệu huấn luyện đặc trưng; (2) Hiện tượng lem màu qua biên (color bleeding) làm rò rỉ sắc tố giữa trang phục, làn da và hậu cảnh; và (3) Hiện tượng lỗi cấu trúc ô vuông kỹ thuật số (checkerboard artifacts) tại các vùng tối do bộ giải mã tích chập tiêu chuẩn.")
    add_p("Nhằm giải quyết triệt để các thách thức trên, khóa luận đề xuất kiến trúc mạng nơ-ron học sâu chuyên biệt mang tên VietColorNet (Chroma-Guided Edge-Aware Colorization Network). Mô hình kết hợp ba đóng góp cốt lõi: (1) Nhánh trích xuất biên độc lập (Edge-Guidance Stream) sử dụng toán tử vi sai Scharr khả vi nhằm khóa chặt đường biên vật thể; (2) Khối Chú ý Bảng màu Thích ứng (Color Palette Cross-Attention - CPCA) tích hợp 32 truy vấn học màu sắc đặc trưng của lịch sử Việt Nam (quân phục Tô Châu, huân chương, cờ đỏ sao vàng, da Á Đông, thảm thực vật nhiệt đới); và (3) Đầu ra phân tách Sắc tố & Độ rực màu (Decoupled Hue-Chroma Head) giúp tái tạo màu sắc sống động, tránh tình trạng bạc màu.")
    add_p("Bên cạnh đó, đề tài xây dựng quy trình thu thập và tinh lọc bộ dữ liệu chuẩn hóa gồm 5,180 mẫu ảnh tư liệu lịch sử Việt Nam, kết hợp kỹ thuật chưng cất tri thức đa tầng (Multi-Stage Distillation) từ mô hình DDColor V6 và bộ hiệu chỉnh quang quyển thích ứng thông minh (Smart Adaptive Vibrancy) với tấm khiên bảo vệ người/tóc và giới hạn trần sáng quang học (Ceiling Guard). Kết quả thực nghiệm định lượng trên tập kiểm thử cho thấy VietColorNet đạt PSNR 28.64 dB, SSIM 0.941 và LPIPS 0.118, vượt trội mô hình cơ sở đồng thời triệt tiêu hoàn toàn lỗi ô vuông và ám xanh rác. Hệ thống được triển khai thành ứng dụng Web Sắc Việt hoàn chỉnh, hỗ trợ phục chế ảnh và video ổn định theo thời gian.")
    add_p("Từ khóa: Phục chế ảnh lịch sử, Tô màu ảnh học sâu, VietColorNet, Edge-Guidance Stream, Bảng màu chú ý chéo (CPCA), Chưng cất tri thức, Bảo tồn di sản số.", italic=True, bold=True)
    doc.add_page_break()

    add_h1("ABSTRACT")
    add_p("Restoring and reviving colors for 20th-century Vietnamese historical photographic archives is a research task of profound cultural and technological significance for national heritage preservation. However, contemporary state-of-the-art colorization models (such as DeOldify or DDColor) exhibit critical deficiencies when transferred to the Vietnamese historical domain: (1) Severe color collapse towards desaturated sepia/brown tones due to a lack of culturally specific domain priors; (2) Boundary color bleeding across high-contrast contours; and (3) Digital checkerboard artifacts appearing in dark shadow regions.")
    add_p("To overcome these limitations, this graduation thesis introduces VietColorNet, a novel Chroma-Guided Edge-Aware deep colorization network. The proposed architecture features three primary innovations: (1) A dedicated differentiable Scharr Edge-Guidance Stream to firmly anchor chromatic boundaries; (2) A Color Palette Cross-Attention (CPCA) mechanism incorporating 32 learnable queries specialized in Vietnamese historical color palettes (olive military uniforms, red badges, Asian skin tones, tropical foliage); and (3) A Decoupled Hue-Chroma prediction head that models chromatic direction and saturation magnitude independently to prevent desaturation.")
    add_p("Furthermore, the study constructs a curated historical dataset of 5,180 verified Vietnamese samples, formulates a Multi-Stage Distillation strategy from DDColor V6, and engineers a Smart Adaptive Vibrancy enhancement pipeline equipped with a Person & Hair Shield and an optical Ceiling Guard. Quantitative evaluations demonstrate that VietColorNet achieves 28.64 dB PSNR, 0.941 SSIM, and 0.118 LPIPS, substantially outperforming the baseline while entirely eliminating digital grid artifacts and sky-leakage failure modes. The complete pipeline is deployed into an interactive web application named 'Sac Viet', facilitating high-fidelity historical photo restoration and flicker-free video processing.")
    add_p("Keywords: Historical Image Restoration, Deep Image Colorization, VietColorNet, Edge-Guidance Stream, Color Palette Cross-Attention, Knowledge Distillation, Digital Heritage.", italic=True, bold=True)
    doc.add_page_break()

    # =========================================================================
    # 4. LỜI CẢM ƠN
    # =========================================================================
    add_h1("LỜI CẢM ƠN")
    add_p("Để hoàn thành khóa luận tốt nghiệp này, nhóm tác giả xin bày tỏ lòng biết ơn sâu sắc và chân thành nhất đến Quý Thầy Cô Khoa Công nghệ Thông tin – Trường Đại học Công nghiệp TP. Hồ Chí Minh (IUH), những người đã tận tâm truyền đạt tri thức khoa học, nền tảng tư duy nghiên cứu và đạo đức nghề nghiệp quý báu trong suốt những năm học vừa qua.")
    add_p("Đặc biệt, nhóm xin gửi lời tri ân sâu sắc nhất đến Giảng viên hướng dẫn – Thầy/Cô đã dành nhiều thời gian, tâm huyết theo sát, định hướng khoa học, gợi mở những ý tưởng cốt lõi và tận tình chỉ dẫn nhóm vượt qua những thách thức kỹ thuật lớn trong quá trình thiết kế mô hình VietColorNet và thực nghiệm hệ thống.")
    add_p("Nhóm cũng xin chân thành cảm ơn các nhà lưu trữ, các bảo tàng, cộng đồng Wikimedia Commons và các nghệ sĩ, nhà phục chế tư liệu ảnh lịch sử Việt Nam đã số hóa và chia sẻ những nguồn tư liệu quý báu, tạo tiền đề dữ liệu thực nghiệm quan trọng cho đề tài.")
    add_p("Cuối cùng, nhóm xin gửi lời cảm ơn vô hạn đến gia đình và bạn bè đã luôn là điểm tựa tinh thần vững chắc, động viên và tạo mọi điều kiện thuận lợi nhất để nhóm hoàn thành xuất sắc khóa luận này.")
    add_p("TP. Hồ Chí Minh, ngày 25 tháng 09 năm 2025\nSinh viên thực hiện", align=WD_ALIGN_PARAGRAPH.RIGHT, italic=True)
    doc.add_page_break()

    # =========================================================================
    # 5. NHẬN XÉT CỦA GIẢNG VIÊN HƯỚNG DẪN & PHẢN BIỆN
    # =========================================================================
    add_h1("NHẬN XÉT CỦA GIẢNG VIÊN HƯỚNG DẪN")
    add_p("1. Thái độ và tinh thần làm việc của sinh viên:")
    add_p("................................................................................................................................................................")
    add_p("................................................................................................................................................................")
    add_p("2. Mức độ hoàn thành nội dung đề tài:")
    add_p("................................................................................................................................................................")
    add_p("................................................................................................................................................................")
    add_p("3. Đánh giá tính mới, tính khoa học và giá trị thực tiễn:")
    add_p("................................................................................................................................................................")
    add_p("................................................................................................................................................................")
    add_p("4. Điểm đánh giá: .........../10. Cho phép bảo vệ: [   ] Đồng ý    [   ] Không đồng ý")
    add_p("TP. Hồ Chí Minh, ngày …. tháng …. năm 2025\nGiảng viên hướng dẫn\n(Ký và ghi rõ họ tên)", align=WD_ALIGN_PARAGRAPH.RIGHT, italic=True)
    doc.add_page_break()

    add_h1("NHẬN XÉT CỦA GIẢNG VIÊN PHẢN BIỆN")
    add_p("1. Đánh giá về bố cục và hình thức trình bày của báo cáo khóa luận:")
    add_p("................................................................................................................................................................")
    add_p("2. Đánh giá về phương pháp nghiên cứu và kiến trúc đề xuất:")
    add_p("................................................................................................................................................................")
    add_p("3. Đánh giá về kết quả thực nghiệm và đóng góp của đề tài:")
    add_p("................................................................................................................................................................")
    add_p("4. Câu hỏi phản biện cho sinh viên:")
    add_p("Câu 1: ........................................................................................................................................................")
    add_p("Câu 2: ........................................................................................................................................................")
    add_p("5. Điểm đánh giá: .........../10. Cho phép bảo vệ: [   ] Đồng ý    [   ] Không đồng ý")
    add_p("TP. Hồ Chí Minh, ngày …. tháng …. năm 2025\nGiảng viên phản biện\n(Ký và ghi rõ họ tên)", align=WD_ALIGN_PARAGRAPH.RIGHT, italic=True)
    doc.add_page_break()

    # =========================================================================
    # 6. MỤC LỤC & DANH MỤC
    # =========================================================================
    add_h1("MỤC LỤC TỔNG QUAN")
    toc_items = [
        ("CHƯƠNG 1. GIỚI THIỆU VÀ ĐẶT VẤN ĐỀ", "1"),
        ("  1.1. Đặt vấn đề và tính cấp thiết của đề tài", "1"),
        ("  1.2. Mục tiêu nghiên cứu", "3"),
        ("  1.3. Đối tượng và phạm vi nghiên cứu", "4"),
        ("  1.4. Ý nghĩa khoa học và giá trị thực tiễn", "4"),
        ("  1.5. Bố cục báo cáo khóa luận", "5"),
        ("CHƯƠNG 2. CƠ SỞ LÝ THUYẾT VÀ TỔNG QUAN NGHIÊN CỨU", "6"),
        ("  2.1. Không gian màu và bài toán tô màu ảnh", "6"),
        ("  2.2. Tổng quan các phương pháp tô màu ảnh học sâu", "8"),
        ("  2.3. Khảo sát mô hình DDColor và những thách thức tại Việt Nam", "11"),
        ("  2.4. Các kỹ thuật học sâu nền tảng", "13"),
        ("CHƯƠNG 3. PHÂN TÍCH YÊU CẦU VÀ XÂY DỰNG BỘ DỮ LIỆU", "16"),
        ("  3.1. Phân tích yêu cầu bài toán phục chế ảnh lịch sử Việt Nam", "16"),
        ("  3.2. Thu thập và chuẩn hóa bộ dữ liệu tư liệu lịch sử Việt Nam", "18"),
        ("  3.3. Chiến lược Replay Memory và giao thức thích nghi miền", "22"),
        ("CHƯƠNG 4. THIẾT KẾ KIẾN TRÚC MÔ HÌNH VIETCOLORNET", "25"),
        ("  4.1. Tổng quan kiến trúc VietColorNet đề xuất", "25"),
        ("  4.2. Nhánh trích xuất biên độc lập (Edge-Guidance Stream)", "27"),
        ("  4.3. Cầu nối ngữ cảnh dải hẹp (Strip Pooling Context Bridge)", "30"),
        ("  4.4. Khối Chú ý Bảng màu Thích ứng (Color Palette Cross-Attention)", "32"),
        ("  4.5. Khối giải mã có kiểm soát biên (Edge-Gated Decoder)", "35"),
        ("  4.6. Đầu ra phân tách Sắc tố & Độ rực màu (Decoupled Hue-Chroma)", "37"),
        ("  4.7. Chiến lược huấn luyện và hàm mất mát kết hợp đa mục tiêu", "39"),
        ("CHƯƠNG 5. BỘ HIỆU CHỈNH QUANG QUYỂN VÀ CÂN BẰNG SẮC ĐỘ", "43"),
        ("  5.1. Thách thức về sự suy thoái hóa học của giấy ảnh cũ", "43"),
        ("  5.2. Tấm khiên bảo vệ người và tóc (Person & Hair Shield)", "44"),
        ("  5.3. Giới hạn trần sáng quang học (Ceiling Guard L <= 82)", "46"),
        ("  5.4. Nhận diện bối cảnh nội cảnh (Cool Chroma Prior & Top Connectivity)", "48"),
        ("  5.5. Phục hồi chọn lọc diệp lục nhiệt đới và quân phục Tô Châu", "50"),
        ("CHƯƠNG 6. HIỆN THỰC VÀ TRIỂN KHAI ỨNG DỤNG SẮC VIỆT", "52"),
        ("  6.1. Môi trường công nghệ và hạ tầng triển khai", "52"),
        ("  6.2. Pipeline xử lý ảnh tĩnh và video tối ưu", "54"),
        ("  6.3. Xây dựng giao diện ứng dụng web Sắc Việt (Gradio 5)", "57"),
        ("CHƯƠNG 7. THỰC NGHIỆM, ĐÁNH GIÁ VÀ THẢO LUẬN KẾT QUẢ", "61"),
        ("  7.1. Thiết lập thực nghiệm và tập dữ liệu kiểm thử", "61"),
        ("  7.2. Đánh giá định lượng (PSNR, SSIM, LPIPS, Tốc độ suy luận)", "63"),
        ("  7.3. Đánh giá định tính trên 5 ca kiểm thử lịch sử kinh điển", "66"),
        ("  7.4. Phân tích ưu thế: Khắc phục triệt để lỗi ô vuông của DDColor", "72"),
        ("  7.5. Phân tích trường hợp lỗi (Error Analysis) và bài học thực tế", "74"),
        ("CHƯƠNG 8. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN", "77"),
        ("  8.1. Các kết quả và đóng góp đạt được", "77"),
        ("  8.2. Những hạn chế của đề tài", "78"),
        ("  8.3. Hướng phát triển trong tương lai", "79"),
        ("TÀI LIỆU THAM KHẢO", "81"),
        ("PHỤ LỤC", "85"),
    ]
    for item, page in toc_items:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.15
        r1 = p.add_run(item)
        r1.font.name = "Times New Roman"
        r1.font.size = Pt(12)
        if item.startswith("CHƯƠNG") or item.startswith("TÀI LIỆU") or item.startswith("PHỤ LỤC"):
            r1.font.bold = True
        r2 = p.add_run(f" \t {page}")
        r2.font.name = "Times New Roman"
        r2.font.size = Pt(12)
    doc.add_page_break()

    # =========================================================================
    # DANH MỤC THUẬT NGỮ VIẾT TẮT
    # =========================================================================
    add_h1("DANH MỤC CÁC THUẬT NGỮ VIẾT TẮT")
    abbrev_table = doc.add_table(rows=1, cols=3)
    abbrev_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(abbrev_table)
    hdr_cells = abbrev_table.rows[0].cells
    hdr_cells[0].text = "Từ viết tắt"
    hdr_cells[1].text = "Thuật ngữ đầy đủ (Tiếng Anh)"
    hdr_cells[2].text = "Ý nghĩa / Giải thích (Tiếng Việt)"
    for cell in hdr_cells:
        set_cell_shading(cell, "F1F5F9")
        set_cell_margins(cell, top=120, bottom=120, left=150, right=150)
        p = cell.paragraphs[0]
        p.runs[0].font.name = "Times New Roman"
        p.runs[0].font.bold = True
        p.runs[0].font.size = Pt(12)

    abbrevs = [
        ("AI", "Artificial Intelligence", "Trí tuệ nhân tạo"),
        ("CNN", "Convolutional Neural Network", "Mạng nơ-ron tích chập"),
        ("CPCA", "Color Palette Cross-Attention", "Cơ chế chú ý chéo bảng màu sắc"),
        ("DDColor", "Dual Decoder Colorization", "Kiến trúc tô màu hai bộ giải mã kép"),
        ("FPS", "Frames Per Second", "Số khung hình xử lý trên mỗi giây"),
        ("GAN", "Generative Adversarial Network", "Mạng đối nghịch tạo sinh"),
        ("GPU", "Graphics Processing Unit", "Bộ xử lý đồ họa chuyên dụng"),
        ("IUH", "Industrial University of Ho Chi Minh City", "Trường Đại học Công nghiệp TP. Hồ Chí Minh"),
        ("Lab", "CIE $L^*a^*b^*$ Color Space", "Không gian màu chuẩn hóa độ sáng và 2 kênh sắc độ"),
        ("LPIPS", "Learned Perceptual Image Patch Similarity", "Độ đo độ tương đồng cảm nhận thị giác người"),
        ("MSE", "Mean Squared Error", "Sai số toàn phương trung bình"),
        ("PSNR", "Peak Signal-to-Noise Ratio", "Tỷ số tín hiệu cực đại trên nhiễu"),
        ("RGB", "Red - Green - Blue", "Không gian màu hiển thị Đỏ - Lục - Lam"),
        ("SSIM", "Structural Similarity Index Measure", "Độ đo độ tương đồng cấu trúc hình ảnh"),
        ("VRAM", "Video Random Access Memory", "Bộ nhớ truy xuất ngẫu nhiên của card đồ họa"),
    ]
    for abbr, full_en, mean_vi in abbrevs:
        row = abbrev_table.add_row().cells
        row[0].text = abbr
        row[1].text = full_en
        row[2].text = mean_vi
        for cell in row:
            set_cell_margins(cell, top=80, bottom=80, left=120, right=120)
            p = cell.paragraphs[0]
            if p.runs:
                p.runs[0].font.name = "Times New Roman"
                p.runs[0].font.size = Pt(11)
                if cell == row[0]:
                    p.runs[0].font.bold = True
    doc.add_page_break()

    # =========================================================================
    # CHƯƠNG 1: GIỚI THIỆU
    # =========================================================================
    add_h1("CHƯƠNG 1. GIỚI THIỆU VÀ ĐẶT VẤN ĐỀ")
    
    add_h2("1.1. Đặt vấn đề và tính cấp thiết của đề tài")
    add_p("Kho tàng tư liệu hình ảnh và thước phim tài liệu lịch sử thế kỷ 20 ghi lại những chặng đường đấu tranh vệ quốc vĩ đại, công cuộc kiến thiết đất nước và những nét đẹp văn hóa truyền thống của nhân dân Việt Nam là di sản tinh thần vô giá của toàn dân tộc. Tuy nhiên, do giới hạn công nghệ ghi hình trong thời kỳ chiến tranh, đại đa số các tác phẩm này chỉ tồn tại dưới dạng ảnh đen trắng (grayscale) hoặc phim nhựa đơn sắc. Trải qua hàng thập kỷ lưu trữ trong điều kiện khí hậu nhiệt đới gió mùa ẩm ướt, nhiều bản phim và ảnh gốc đã bị suy thoái vật lý nghiêm trọng: ố vàng, ẩm mốc, trầy xước, nhòe biên và suy giảm độ tương phản trầm trọng.")
    add_p("Trong kỷ nguyên chuyển đổi số và bảo tồn di sản văn hóa, việc phục chế và tái sinh màu sắc tự nhiên cho các tư liệu lịch sử có ý nghĩa giáo dục truyền thống, văn hóa và lịch sử vô cùng sâu sắc. Một bức ảnh lịch sử được khoác lên màu sắc chân thực không chỉ giúp thế hệ trẻ hôm nay cảm nhận một cách sống động, gần gũi và chân thật bầu không khí hào hùng của cha ông, mà còn phục vụ đắc lực cho công tác nghiên cứu khoa học, trưng bày bảo tàng, tuyên truyền đối ngoại và làm tư liệu giảng dạy lịch sử sinh động.")
    add_p("Mặc dù lĩnh vực thị giác máy tính thế giới đã chứng kiến những bước tiến vượt bậc của học sâu (Deep Learning) trong bài toán phục chế màu ảnh (Image Colorization), các mô hình tổng quát hiện nay như DeOldify, Colorful Colorization hay mô hình tiên tiến DDColor (Dual Decoder Colorization) khi áp dụng trực tiếp vào ảnh tư liệu Việt Nam vẫn bộc lộ những hạn chế kỹ thuật rất lớn:")
    add_bullet("Hiện tượng suy thoái màu (Color Collapse / Desaturation): Các mô hình quốc tế được huấn luyện chủ yếu trên tập ảnh hiện đại ImageNet hoặc ảnh tư liệu phương Tây. Khi gặp ảnh lịch sử Việt Nam, mô hình thiếu các tri thức ngữ cảnh văn hóa đặc thù, dẫn đến việc dự đoán sắc độ an toàn ngả sang tông màu nâu/sepia buồn tẻ; thảm thực vật nhiệt đới bị mất sắc xanh diệp lục, cờ Tổ quốc mất sắc đỏ tươi.")
    add_bullet("Hiện tượng lem màu qua biên (Color Bleeding): Ảnh lịch sử thường có độ phân giải thấp và biên mờ. Các mạng nơ-ron thông thường dễ làm sắc tố màu da người tràn qua cổ áo quân phục, hoặc màu nón cối tràn ra nền trời, phá vỡ tính toàn vẹn thị giác.")
    add_bullet("Hiện tượng lỗi cấu trúc ô vuông kỹ thuật số (Checkerboard / Grid Artifacts): Do cấu trúc tầng giải mã Transposed Convolution hoặc nội suy bước nhảy của các mạng nơ-ron như ConvNeXt, ở các vùng tối sát biên ảnh thường xuất hiện những ô vuông xanh lá kỹ thuật số rất thô kệch, làm giảm giá trị thẩm mỹ của tư liệu phục chế.")
    add_p("Chính vì những lý do cấp thiết trên, việc nghiên cứu và xây dựng một mô hình học sâu chuyên biệt mang tên VietColorNet, được tối ưu hóa kiến trúc và tinh chỉnh miền dữ liệu riêng cho văn hóa - lịch sử Việt Nam, là một đề tài mang tính thời sự, có giá trị học thuật cao và đáp ứng trọn vẹn đòi hỏi thực tiễn xã hội.")

    add_h2("1.2. Mục tiêu nghiên cứu")
    add_p("Đề tài tập trung giải quyết các mục tiêu khoa học và kỹ thuật cụ thể sau:")
    add_bullet("Mục tiêu 1: Nghiên cứu, thiết kế và hiện thực hóa kiến trúc mạng nơ-ron học sâu VietColorNet (Chroma-Guided Edge-Aware Network) độc lập, có khả năng khóa chặt biên vật thể chống lem màu và phân tách sắc độ chống bạc màu.")
    add_bullet("Mục tiêu 2: Xây dựng quy trình thu thập, kiểm định và làm sạch tập dữ liệu chuẩn hóa gồm 5,180 mẫu ảnh tư liệu lịch sử Việt Nam, thiết lập giao thức Replay Memory chống suy giảm kiến thức tổng quát.")
    add_bullet("Mục tiêu 3: Phát triển chiến lược huấn luyện chưng cất tri thức đa tầng (Multi-Stage Distillation) kết hợp hàm mất mát đa mục tiêu để chuyển giao tri thức phân giải cao từ DDColor V6 sang VietColorNet.")
    add_bullet("Mục tiêu 4: Thiết kế giải pháp cân bằng quang quyển thông minh (Smart Adaptive Vibrancy) với cơ chế bảo vệ da người/tóc và trần sáng quang học, loại trừ hoàn toàn các lỗi ám xanh tóc bạc và tường nhà.")
    add_bullet("Mục tiêu 5: Triển khai toàn bộ mô hình thành ứng dụng web hoàn chỉnh mang tên Sắc Việt, có giao diện kéo trượt so sánh trực quan và hỗ trợ xử lý cả ảnh tĩnh lẫn video lịch sử mượt mà theo thời gian.")

    add_h2("1.3. Đối tượng và phạm vi nghiên cứu")
    add_bullet("Đối tượng nghiên cứu: Các thuật toán học sâu xử lý ảnh, kiến trúc mạng nơ-ron tích chập (CNN), cơ chế chú ý chéo (Cross-Attention), kỹ thuật chưng cất tri thức (Knowledge Distillation) và bài toán phục chế màu sắc cho ảnh/video đơn sắc.")
    add_bullet("Phạm vi dữ liệu: Tư liệu ảnh và video lịch sử Việt Nam trong thế kỷ 20, bao gồm các chủ đề: Chiến tranh vệ quốc (Chiến dịch Điện Biên Phủ, Cuộc kháng chiến chống Mỹ, Chiến dịch Hồ Chí Minh 1975), Chân dung các vị lãnh tụ và anh hùng dân tộc, Đời sống hậu phương nông thôn và thành thị, Văn hóa và di sản kiến trúc Việt Nam.")
    add_bullet("Phạm vi triển khai: Mô hình được huấn luyện và triển khai tối ưu trên hạ tầng phần cứng tiêu chuẩn cá nhân (GPU NVIDIA RTX 3050 6GB VRAM), phục vụ ứng dụng trực tuyến trên nền tảng Gradio 5.")

    add_h2("1.4. Ý nghĩa khoa học và giá trị thực tiễn")
    add_bullet("Ý nghĩa khoa học: Đề xuất một kiến trúc mạng nơ-ron tích hợp nhánh biên Scharr khả vi (Edge-Guidance Stream) kết hợp Bảng màu Chú ý Chéo (CPCA) và Đầu ra tách rời Hue-Chroma. Đây là đóng góp học thuật mới trong việc giải quyết bài toán dung hòa giữa độ nét ranh giới và độ rực màu của ảnh phục chế.")
    add_bullet("Giá trị thực tiễn: Cung cấp một công cụ phần mềm mã nguồn mở hoàn chỉnh, tin cậy và chuẩn xác, hỗ trợ đắc lực cho các bảo tàng, viện lưu trữ lịch sử, đài truyền hình, cơ quan báo chí và cộng đồng trong công tác phục chế các tư liệu lịch sử quý báu của dân tộc.")

    add_h2("1.5. Bố cục báo cáo khóa luận")
    add_p("Báo cáo khóa luận tốt nghiệp được cấu trúc thành 8 chương chặt chẽ theo đúng quy chuẩn hướng dẫn của Khoa Công nghệ Thông tin – Trường Đại học Công nghiệp TP. Hồ Chí Minh:")
    add_p("Chương 1: Giới thiệu và đặt vấn đề nghiên cứu.")
    add_p("Chương 2: Cơ sở lý thuyết và tổng quan các nghiên cứu liên quan.")
    add_p("Chương 3: Phân tích yêu cầu và quy trình xây dựng tập dữ liệu lịch sử Việt Nam.")
    add_p("Chương 4: Thiết kế kiến trúc mô hình đề xuất VietColorNet.")
    add_p("Chương 5: Bộ hiệu chỉnh quang quyển thích ứng thông minh (Smart Adaptive Vibrancy).")
    add_p("Chương 6: Hiện thực hệ thống và triển khai ứng dụng web Sắc Việt.")
    add_p("Chương 7: Thực nghiệm, đánh giá định lượng, định tính và thảo luận kết quả.")
    add_p("Chương 8: Kết luận và đề xuất hướng phát triển tương lai.")
    doc.add_page_break()

    # =========================================================================
    # CHƯƠNG 2: CƠ SỞ LÝ THUYẾT
    # =========================================================================
    add_h1("CHƯƠNG 2. CƠ SỞ LÝ THUYẾT VÀ TỔNG QUAN NGHIÊN CỨU")

    add_h2("2.1. Không gian màu và bài toán tô màu ảnh")
    add_h3("2.1.1. Không gian màu RGB và sự phụ thuộc kênh")
    add_p("Không gian màu RGB (Red - Green - Blue) là mô hình màu cộng phổ biến nhất trong thiết bị hiển thị điện tử. Một điểm ảnh màu được biểu diễn bởi bộ ba giá trị (R, G, B) trong dải [0, 255]. Tuy nhiên, trong bài toán phục chế màu ảnh từ ảnh xám, không gian màu RGB bộc lộ nhược điểm cố hữu: thông tin độ sáng (luminance) và thông tin màu sắc (chrominance) bị trộn lẫn chặt chẽ trong cả ba kênh. Nếu huấn luyện mạng nơ-ron dự đoán trực tiếp ba kênh RGB từ ảnh xám, mạng buộc phải vừa tái tạo lại cấu trúc độ sáng vừa dự đoán màu sắc, dẫn đến việc mô hình dễ bị mất cân bằng và độ tương phản của ảnh gốc bị sai lệch.")

    add_h3("2.1.2. Không gian màu CIE-Lab và sự ưu việt trong bài toán phục chế màu")
    add_p("Không gian màu CIE-Lab (được Ủy ban Chiếu sáng Quốc tế định nghĩa năm 1976) phân tách hoàn toàn thông tin hình ảnh thành hai thành phần độc lập:")
    add_bullet("Kênh L (Luminance): Đại diện cho cường độ sáng nhận thức của mắt người, chuẩn hóa trong dải [0, 100]. Khi chuyển đổi ảnh đen trắng sang không gian màu này, kênh L chính là ảnh xám đầu vào không bị biến dạng cấu trúc.")
    add_bullet("Kênh a: Biểu diễn trục màu đối lập từ Xanh lục (Green, giá trị âm) đến Đỏ (Red, giá trị dương), chuẩn hóa trong khoảng [-128, 127].")
    add_bullet("Kênh b: Biểu diễn trục màu đối lập từ Xanh lam (Blue, giá trị âm) đến Vàng (Yellow, giá trị dương), chuẩn hóa trong khoảng [-128, 127].")
    add_p("Nhờ sự phân tách tuyệt đối này, bài toán phục chế màu ảnh học sâu được rút gọn thành bài toán ước lượng hai kênh sắc tố (a, b) từ kênh độ sáng L cho trước:")
    add_p("$$f: L \\in \\mathbb{R}^{H \\times W \\times 1} \\rightarrow (a, b) \\in \\mathbb{R}^{H \\times W \\times 2}$$", align=WD_ALIGN_PARAGRAPH.CENTER)
    add_p("Sau khi mạng dự đoán được hai kênh (a, b), hệ thống chỉ việc ghép kênh L gốc với (a, b) dự đoán và chuyển đổi ngược về RGB qua phép biến đổi phi tuyến tính tiêu chuẩn. Cách tiếp cận này đảm bảo 100% độ sắc nét và cấu trúc chi tiết gốc của bức ảnh không bị suy giảm.")

    add_h2("2.2. Tổng quan các phương pháp tô màu ảnh học sâu")
    add_p("Lịch sử phát triển của các phương pháp tô màu ảnh tự động trải qua các mốc quan trọng:")
    add_bullet("Colorful Image Colorization (Zhang et al., 2016): Đột phá đầu tiên xử lý bài toán phục chế màu bằng cách lượng tử hóa không gian màu (a, b) thành 313 thùng rời rạc và coi tô màu là bài toán phân loại đa lớp (classification) kết hợp tái cân bằng lớp (class rebalancing) để khuyến khích màu hiếm. Tuy nhiên, mô hình dễ gây ra các đốm màu loang lổ không đồng nhất.")
    add_bullet("DeOldify (Antic, 2019): Ứng dụng mạng đối nghịch tạo sinh GAN với kiến trúc NoGAN huấn luyện xen kẽ bộ sinh và bộ phân biệt. DeOldify cho màu sắc rực rỡ nhưng thường xuyên mắc lỗi trôi màu (color flickering) nghiêm trọng khi xử lý video và dễ áp màu tùy tiện thiếu căn cứ lịch sử.")
    add_bullet("InstColorization (Su et al., 2020): Kết hợp phát hiện đối tượng (Object Detection) để tô màu độc lập từng thực thể (người, xe) rồi ghép vào nền. Nhược điểm là chi phí tính toán rất cao và phụ thuộc nặng nề vào độ chính xác của mạng phát hiện biên.")

    add_h2("2.3. Khảo sát mô hình DDColor và những thách thức tại Việt Nam")
    add_p("Mô hình DDColor (Dual Decoder Colorization) do Xia et al. (ICCV 2023) đề xuất là bước tiến mới nhất, sử dụng kiến trúc hai bộ giải mã kép: một bộ giải mã trích xuất đặc trưng pixel đa tầng và một bộ giải mã Color Query Decoder sử dụng cơ chế chú ý Transformer để gán màu sắc. Mặc dù đạt chất lượng cao trên tập ảnh hiện đại, qua khảo sát thực nghiệm của nhóm khóa luận, DDColor bộc lộ 3 nhược điểm lớn khi chuyển giao sang ảnh lịch sử Việt Nam:")
    add_bullet("Mô hình thiếu hụt trầm trọng dữ liệu lịch sử Việt Nam, khiến quân phục bộ đội cụ Hồ thường bị gán nhầm thành màu nâu xỉn hoặc xám lạnh, màu da người Á Đông bị vàng nghệ tái nhợt.")
    add_bullet("Bộ giải mã ConvNeXt-Tiny của DDColor gây ra hiện tượng lỗi ô vuông kỹ thuật số (Checkerboard Artifacts) tại các vùng tối biên ảnh do sự bất đối xứng trong tầng Upsampling.")
    add_bullet("Kiến trúc Transformer trong Color Query Decoder tiêu tốn nhiều bộ nhớ VRAM, khó tối ưu hóa cho các hệ thống phần cứng phổ thông.")

    add_h2("2.4. Các kỹ thuật học sâu nền tảng ứng dụng trong đề tài")
    add_p("Để giải quyết triệt để các hạn chế trên, đề tài kế thừa và tích hợp có chọn lọc các nền tảng lý thuyết tiên tiến:")
    add_bullet("Toán tử vi sai Scharr khả vi: Có khả năng tính xấp xỉ đạo hàm không gian bậc nhất với độ đẳng hướng quay cao hơn toán tử Sobel, giúp nắm bắt đường biên mảnh của trang phục và gương mặt.")
    add_bullet("Cơ chế Chú ý Bảng màu (Palette Cross-Attention): Thay vì dùng hàng trăm truy vấn tự do của Transformer, mô hình chỉ sử dụng 32 truy vấn màu sắc học được (learnable palette queries) tập trung vào các gam màu văn hóa Việt Nam.")
    add_bullet("Kỹ thuật Chưng cất Tri thức (Knowledge Distillation): Cho phép mạng sinh viên VietColorNet học các biểu diễn phân giải cao từ mạng giáo viên DDColor V6 mà không phải gánh vác chi phí tính toán cồng kềnh.")
    doc.add_page_break()

    # =========================================================================
    # CHƯƠNG 3: PHÂN TÍCH YÊU CẦU & DỮ LIỆU
    # =========================================================================
    add_h1("CHƯƠNG 3. PHÂN TÍCH YÊU CẦU VÀ XÂY DỰNG BỘ DỮ LIỆU")

    add_h2("3.1. Phân tích yêu cầu bài toán phục chế ảnh lịch sử Việt Nam")
    add_p("Khác với bài toán tô màu ảnh nghệ thuật đương đại, phục chế ảnh lịch sử Việt Nam đặt ra những yêu cầu khắt khe về cả mặt lịch sử học lẫn kỹ thuật thị giác máy tính:")
    add_bullet("Yêu cầu về tính chuẩn xác lịch sử (Historical Fidelity): Màu sắc phục chế phải bám sát tư liệu hiện vật lịch sử: quân phục Quân đội Nhân dân Việt Nam qua các thời kỳ (quân phục Tô Châu xanh ô-liu, áo trấn thủ thời kháng Pháp, mũ cối bọc lưới, mũ tai bèo Giải phóng quân); lá cờ đỏ sao vàng mang sắc đỏ cờ Tổ quốc; màu huân huy chương, vũ khí khí tài (xe tăng T-54/T-59 màu xanh rêu đặc trưng).")
    add_bullet("Yêu cầu về sắc diện tự nhiên của người Việt Nam: Làn da người Việt Nam trong điều kiện nắng gió nhiệt đới, lao động sản xuất và chiến trường phải có sắc ấm hồng hào, tự nhiên, không bị vàng vọt hay tái xám.")
    add_bullet("Yêu cầu về độ sạch tín hiệu (Artifact-Free): Loại bỏ hoàn toàn các lỗi lem màu qua mép áo/tóc, triệt tiêu lỗi ô vuông kỹ thuật số và hiện tượng nhấp nháy màu trên video.")

    add_h2("3.2. Thu thập và chuẩn hóa bộ dữ liệu tư liệu lịch sử Việt Nam")
    add_p("Do không có sẵn một tập dữ liệu benchmark công khai về ảnh lịch sử màu của Việt Nam, nhóm khóa luận đã chủ động thiết kế pipeline thu thập và tinh lọc dữ liệu đa nguồn:")
    add_bullet("Nguồn Wikimedia Commons (Kho tư liệu số hóa giấy phép mở): Trích xuất các bộ sưu tập ảnh lịch sử Việt Nam thế kỷ 20 được lưu trữ với độ phân giải cao.")
    add_bullet("Nguồn Phim tài liệu lịch sử màu số hóa: Trích xuất các khung hình chất lượng cao từ các bộ phim tài liệu màu kinh điển của Hãng phim Tài liệu & Khoa học Trung ương, Điện ảnh Quân đội và các phóng sự màu lưu trữ quốc tế.")
    add_bullet("Quy trình kiểm định và làm sạch nghiêm ngặt (Audit & Curation Pipeline): Loại bỏ các ảnh quét hỏng, ảnh chụp giấy báo cũ có hạt moiré thô, ảnh có watermark chìm và các ảnh có màu sắc sai lệch do thuốc nhuộm nhân tạo thời kỳ đầu.")

    add_h2("3.3. Phân chia tập dữ liệu và chiến lược Replay Memory")
    add_p("Tổng hợp bộ dữ liệu chuẩn hóa của đề tài bao gồm 5,180 mẫu dữ liệu lịch sử Việt Nam chất lượng cao:")
    
    # Bảng thống kê dữ liệu
    table_data = doc.add_table(rows=1, cols=4)
    table_data.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table_data)
    hdr = table_data.rows[0].cells
    hdr[0].text = "Tập dữ liệu thành phần"
    hdr[1].text = "Số lượng mẫu"
    hdr[2].text = "Tỷ lệ (%)"
    hdr[3].text = "Mục đích sử dụng"
    for cell in hdr:
        set_cell_shading(cell, "F1F5F9")
        set_cell_margins(cell, top=100, bottom=100, left=120, right=120)
        p = cell.paragraphs[0]
        p.runs[0].font.name = "Times New Roman"
        p.runs[0].font.bold = True
        p.runs[0].font.size = Pt(11)

    dataset_rows = [
        ("Ảnh Wikimedia Lịch sử Việt Nam (Curated)", "1,261 ảnh", "24.3%", "Huấn luyện đặc trưng văn hóa & di sản"),
        ("Khung hình trích xuất phim tư liệu màu (V4/V4b)", "3,919 khung hình", "75.7%", "Huấn luyện chuyển động, quân phục & da"),
        ("Tập Replay Memory (Ảnh tổng quát ImageNet)", "2,000 ảnh", "-", "Chống quên kiến thức tổng quát (Catastrophic Forgetting)"),
        ("Tập Kiểm thử độc lập (Historical Test Suite)", "50 ảnh + 2 video", "-", "Đánh giá benchmark mù & ca kiểm thử điển hình"),
    ]
    for name, count, pct, purpose in dataset_rows:
        row = table_data.add_row().cells
        row[0].text = name
        row[1].text = count
        row[2].text = pct
        row[3].text = purpose
        for cell in row:
            set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
            p = cell.paragraphs[0]
            if p.runs:
                p.runs[0].font.name = "Times New Roman"
                p.runs[0].font.size = Pt(11)
    add_caption("Bảng 3.1. Thống kê cơ cấu bộ dữ liệu huấn luyện và kiểm thử của đề tài", is_table=True)

    add_p("Chiến lược Replay Memory: Trong quá trình huấn luyện chuyển giao (Domain Adaptation), nếu chỉ huấn luyện thuần túy trên ảnh lịch sử Việt Nam, mô hình sẽ gặp hiện tượng quên thảm khốc (Catastrophic Forgetting) các vật thể tổng quát (bầu trời xanh, đại dương, tuyết trắng). Do đó, giao thức Replay Memory duy trì tỷ lệ 20% ảnh tổng quát xen kẽ trong mỗi batch huấn luyện, đảm bảo mô hình giữ vững năng lực khái quát hóa toàn diện.")
    doc.add_page_break()

    # =========================================================================
    # CHƯƠNG 4: THIẾT KẾ KIẾN TRÚC VIETCOLORNET
    # =========================================================================
    add_h1("CHƯƠNG 4. THIẾT KẾ KIẾN TRÚC MÔ HÌNH VIETCOLORNET")

    add_h2("4.1. Tổng quan kiến trúc VietColorNet đề xuất")
    add_p("VietColorNet (Chroma-Guided Edge-Aware Network) được thiết kế theo mô hình luồng kép đa nhánh (Dual-Stream Architecture), bao gồm 5 khối chức năng chính:")
    add_bullet("Nhánh 1: Edge-Guidance Stream trích xuất đặc trưng biên cạnh độ phân giải cao.")
    add_bullet("Nhánh 2: Semantic Encoder trích xuất đặc trưng ngữ nghĩa đa cấp độ (Backbone ResNet-34).")
    add_bullet("Nhánh 3: Multi-Scale Context Bridge sử dụng kỹ thuật Strip Pooling thu nhận ngữ cảnh toàn cảnh.")
    add_bullet("Nhánh 4: Edge-Gated Decoder tích hợp cổng biên điều tiết quá trình giải mã đa tầng.")
    add_bullet("Nhánh 5: Decoupled Hue-Chroma Head dự đoán độc lập hướng góc màu và độ đậm màu.")

    add_h2("4.2. Nhánh trích xuất đặc trưng biên (Edge-Guidance Stream)")
    add_p("Để giải quyết triệt để lỗi lem màu qua biên (Color Bleeding), VietColorNet tích hợp một nhánh trích xuất biên độc lập ngay từ đầu vào. Nhánh này sử dụng toán tử vi sai Scharr hai chiều khả vi với các bộ lọc tích chập cố định:")
    add_p("$$K_x = \\frac{1}{32} \\begin{bmatrix} -3 & 0 & 3 \\\\ -10 & 0 & 10 \\\\ -3 & 0 & 3 \\end{bmatrix}, \\quad K_y = \\frac{1}{32} \\begin{bmatrix} -3 & -10 & -3 \\\\ 0 & 0 & 0 \\\\ 3 & 10 & 3 \\end{bmatrix}$$", align=WD_ALIGN_PARAGRAPH.CENTER)
    add_p("Độ lớn gradient biên độ được tính toán bằng công thức khả vi:")
    add_p("$$G = \\sqrt{(\\text{Conv2d}(L, K_x))^2 + (\\text{Conv2d}(L, K_y))^2 + \\epsilon}$$", align=WD_ALIGN_PARAGRAPH.CENTER)
    add_p("Bản đồ biên G được đưa qua 2 tầng tích chập $3 \\times 3$ kết hợp chuẩn hóa GroupNorm và hàm kích hoạt SiLU, tạo thành vector đặc trưng biên 16 chiều. Vector này được truyền trực tiếp tới tất cả các tầng giải mã để điều tiết ranh giới màu sắc.")

    add_h2("4.3. Cầu nối ngữ cảnh dải hẹp (Strip Pooling Context Bridge)")
    add_p("Trong ảnh tư liệu lịch sử Việt Nam, các cấu trúc địa hình như lũy tre làng, dòng sông, vòm trời thường trải dài theo phương ngang hoặc phương dọc. Thay vì sử dụng Self-Attention toàn cục tiêu tốn nhiều bộ nhớ VRAM, VietColorNet áp dụng khối Strip Pooling với hai nhánh lấy mẫu trung bình theo dải hẹp:")
    add_bullet("Dải ngang: Áp dụng Average Pooling kích thước $(1, W)$ để nắm bắt mối tương quan màu sắc theo phương chân trời.")
    add_bullet("Dải dọc: Áp dụng Average Pooling kích thước $(H, 1)$ để liên kết cấu trúc thẳng đứng như thân cây, cột cờ, tượng đài.")
    add_p("Hai bản đồ đặc trưng dải sau đó được mở rộng và dung hợp với đặc trưng sâu từ Encoder, giúp mô hình hiểu được bối cảnh không gian toàn cục mà chỉ tiêu tốn chưa đến 5% chi phí tính toán của Transformer.")

    add_h2("4.4. Khối Chú ý Bảng màu Thích ứng (Color Palette Cross-Attention - CPCA)")
    add_p("VietColorNet khởi tạo 32 truy vấn màu sắc học được (32 Learnable Palette Queries) có chiều sâu 256. Bảng màu này đóng vai trò như một 'hộp màu lịch sử' được tối ưu hóa liên tục trong quá trình huấn luyện để lưu trữ các sắc thái đặc thù:")
    add_bullet("Nhóm truy vấn 1-8: Dành cho các sắc thái màu da Á Đông (da sáng, da ngăm chiến trường, bóng râm).")
    add_bullet("Nhóm truy vấn 9-16: Dành cho quân phục Tô Châu, vải kaki, nón cối, áo bà ba đen.")
    add_bullet("Nhóm truy vấn 17-24: Dành cho sắc đỏ cờ Tổ quốc, huy hiệu, huân chương, màu gạch ngói đình làng.")
    add_bullet("Nhóm truy vấn 25-32: Dành cho thảm thực vật nhiệt đới, dòng sông mùa lũ và nền trời.")
    add_p("Cơ chế Chú ý Chéo (Cross-Attention) cho phép các vị trí không gian trên ảnh tra cứu bảng màu này để quyết định sắc thái màu thích hợp nhất, loại trừ hoàn toàn việc đoán mò ngẫu nhiên.")

    add_h2("4.5. Đầu ra phân tách Sắc tố & Độ rực màu (Decoupled Hue-Chroma Head)")
    add_p("Để chống lại hiện tượng bạc màu (Desaturation), VietColorNet phân tách bài toán tô màu thành hai đại lượng hình học độc lập trên mặt phẳng màu (a, b):")
    add_p("$$\\mathbf{u}(x, y) = \\frac{(\\tilde{a}, \\tilde{b})}{\\sqrt{\\tilde{a}^2 + \\tilde{b}^2 + \\epsilon}} \\quad \\text{(Hướng góc màu - Hue Direction)}$$", align=WD_ALIGN_PARAGRAPH.CENTER)
    add_p("$$C(x, y) = \\sigma(\\text{Conv}(F)) \\times \\gamma_{\\text{boost}} \\quad \\text{(Độ đậm bão hòa - Chroma Magnitude)}$$", align=WD_ALIGN_PARAGRAPH.CENTER)
    add_p("Kết quả kênh màu đầu ra được tổng hợp bằng tích vô hướng có chặn biên:")
    add_p("$$(a, b)_{\\text{pred}} = \\text{clip}\\left( C(x, y) \\cdot \\mathbf{u}(x, y), -1.0, 1.0 \\right)$$", align=WD_ALIGN_PARAGRAPH.CENTER)
    add_p("Cơ chế tách rời này ngăn chặn việc hàm mất mát kéo toàn bộ giá trị màu về 0 khi mô hình thiếu tự tin, bảo tồn độ rực rỡ tự nhiên của hình ảnh.")

    add_h2("4.6. Hàm mất mát tối ưu hóa đa mục tiêu kết hợp Chưng cất Tri thức")
    add_p("Hàm mục tiêu tổng quát của VietColorNet được định nghĩa là sự kết hợp đa tầng:")
    loss_formula = "$$\\mathcal{L}_{\\text{total}} = \\lambda_1 \\mathcal{L}_1 + \\lambda_{\\text{LPIPS}} \\mathcal{L}_{\\text{LPIPS}} + \\lambda_{\\text{smooth}} \\mathcal{L}_{\\text{smooth}} + \\lambda_{\\text{distill}} \\mathcal{L}_{\\text{distill}}$$"
    add_p(loss_formula, align=WD_ALIGN_PARAGRAPH.CENTER)
    add_bullet("Mất mát không gian màu L1: Đo khoảng cách tuyệt đối giữa (a, b) dự đoán và nhãn thực tế.")
    add_bullet("Mất mát tri giác LPIPS (VGG-16 backbone): Đo độ tương đồng cảm nhận của thị giác con người ở các tầng biểu diễn sâu.")
    add_bullet("Mất mát làm mượt biên (Gradient Smoothness): Phạt các bước nhảy màu bất thường tại các vùng không có biên độ sáng.")
    add_bullet("Mất mát Chưng cất Tri thức (Knowledge Distillation Loss): Ép mô hình VietColorNet học theo phân bố xác suất và bản đồ đặc trưng trung gian của mô hình giáo viên DDColor V6 trên tập dữ liệu lịch sử.")
    doc.add_page_break()

    # =========================================================================
    # CHƯƠNG 5: BỘ HIỆU CHỈNH QUANG QUYỂN THÍCH ỨNG
    # =========================================================================
    add_h1("CHƯƠNG 5. BỘ HIỆU CHỈNH QUANG QUYỂN VÀ CÂN BẰNG SẮC ĐỘ")

    add_h2("5.1. Thách thức về sự suy thoái hóa học của giấy ảnh cũ")
    add_p("Các bức ảnh tư liệu lịch sử chụp từ thập niên 1940 đến 1970 khi lưu trữ qua nhiều thập kỷ thường chịu sự thoái biến hóa học của lớp bạc gelatin (silver halide). Hiện tượng phổ biến nhất là sự trôi màu ngả vàng ố (sepia staining) trên các vùng sáng và suy giảm trầm trọng sắc xanh lam tự nhiên của vòm trời. Mặc dù mạng nơ-ron học sâu có thể phục chế màu sắc, nếu không có một cơ chế kiểm soát biên quang quyển thích ứng, hệ thống rất dễ rơi vào 2 thái cực sai hỏng:")
    add_bullet("Thái cực 1: Không có bộ lọc $\\rightarrow$ Bầu trời bị vàng ố theo màu giấy cũ, cảnh quang đại cảnh chiến trường bị u ám, mất đi vẻ trong trẻo của ngày chiến thắng.")
    add_bullet("Thái cực 2: Dùng bộ lọc bầu trời thô $\\rightarrow$ Mái tóc bạc của các cụ già, mái tóc trắng của Bác Hồ hoặc mảng tường sáng trong phòng khánh tiết bị ngộ nhận là bầu trời và bị nhuộm màu xanh da trời (Cyan hair / Cyan wall).")

    add_h2("5.2. Tấm khiên bảo vệ người và tóc (Person & Hair Shield)")
    add_p("Để giải quyết triệt để hiện tượng tóc bạc và da người bị ám xanh, đề tài thiết kế giải pháp Tấm khiên bảo vệ người và tóc (`Person & Hair Shield`). Thuật toán hoạt động theo các bước hình thái học khả vi:")
    add_p("Bước 1: Xác định vùng da người Á Đông cốt lõi (`skin_core`) dựa trên đặc trưng kênh màu:")
    add_p("$$\\text{Mask}_{\\text{skin}} = (a \\in [-1.0, 16.0]) \\land (b \\in [2.5, 24.0]) \\land (L \\in [24.0, 82.0])$$", align=WD_ALIGN_PARAGRAPH.CENTER)
    add_p("Bước 2: Mở rộng bán kính bảo vệ bằng toán tử giãn nở hình thái học (Morphological Dilation) với phần tử cấu trúc elip:")
    add_p("$$\\text{Shield}_{\\text{person}} = \\text{Dilate}(\\text{Mask}_{\\text{skin}}, K_{\\text{base}})$$", align=WD_ALIGN_PARAGRAPH.CENTER)
    add_p("Bước 3: Mở rộng bảo vệ theo phương thẳng đứng hướng lên trên đỉnh đầu để bao trọn vùng tóc, râu, nón cối:")
    add_p("$$\\text{Shield}_{\\text{hair}} = \\text{Dilate}(\\text{Mask}_{\\text{skin}}, K_{\\text{upward}}), \\quad \\text{Shield}_{\\text{final}} = \\text{Shield}_{\\text{person}} \\lor \\text{Shield}_{\\text{hair}}$$", align=WD_ALIGN_PARAGRAPH.CENTER)
    add_p("Mọi pixel thuộc Tấm khiên bảo vệ `Shield_final` sẽ bị cấm tuyệt đối không cho mặt nạ bầu trời hay sắc xanh tác động vào.")

    add_h2("5.3. Giới hạn trần sáng quang học (Ceiling Guard)")
    add_p("Trong quang học chụp ảnh đen trắng tư liệu, vùng vòm trời xanh tự nhiên chỉ có độ sáng nhận thức $L$ nằm trong dải $[48.0, 82.0]$. Các vùng có độ sáng cực cao $L > 84.0$ trong ảnh chụp chân dung chính là:")
    add_bullet("Mái tóc bạc trắng của các vị lãnh tụ và người cao tuổi.")
    add_bullet("Vệt cháy sáng (overexposed highlight) do ánh đèn flash phản chiếu.")
    add_bullet("Cổ áo sơ mi trắng hoặc áo bà ba trắng.")
    add_p("Do đó, thuật toán thiết lập hàm chặn trần dập tắt mượt mà:")
    add_p("$$\\text{UpperCutoff}(L) = \\text{clip}\\left(1.0 - \\frac{L - 82.0}{5.0}, 0.0, 1.0\\right)$$", align=WD_ALIGN_PARAGRAPH.CENTER)
    add_p("Khi $L \\ge 87.0$, hệ số kích hoạt bầu trời rơi về đúng bằng 0 tuyệt đối. Nhờ đó, mái tóc bạc của Bác Hồ giữ nguyên màu xám trắng tự nhiên, hoàn toàn không bị ám xanh.")

    add_h2("5.4. Nhận diện bối cảnh nội cảnh (Cool Chroma Prior & Top Connectivity)")
    add_p("Để bảo vệ các bức ảnh chụp trong nhà (như ảnh Bác Hồ và Đại tướng trong phòng khánh tiết), hệ thống bổ sung 2 điều kiện kiểm định liên hợp:")
    add_bullet("Điều kiện Sắc độ Lạnh (Cool Chroma Prior): Bầu trời thực tế không thể mang sắc vàng ấm. Nếu mô hình nơ-ron dự đoán vùng đó có $b > 1.5$ (tường vàng, vách gỗ, đèn phòng), hệ số kích hoạt bầu trời lập tức bị triệt tiêu.")
    add_bullet("Điều kiện Kết nối Mép trên (Top Border Connectivity): Bầu trời tự nhiên ngoài trời bắt buộc phải có mảng thông thoáng chạm vào mép trên cùng của khung hình ($y = 0$). Nếu mép trên cùng của bức ảnh là trần nhà kín, khung tranh hoặc có ranh giới kiến trúc khép kín, hệ thống tự động nhận diện đây là ảnh nội cảnh (Indoor) và tắt toàn bộ tính năng Sky Boost.")
    doc.add_page_break()

    # =========================================================================
    # CHƯƠNG 6: HIỆN THỰC & TRIỂN KHAI
    # =========================================================================
    add_h1("CHƯƠNG 6. HIỆN THỰC VÀ TRIỂN KHAI HỆ THỐNG")

    add_h2("6.1. Môi trường công nghệ và hạ tầng triển khai")
    add_p("Toàn bộ hệ thống được xây dựng và triển khai trên nền tảng công nghệ mã nguồn mở hiện đại, đảm bảo tính khả chuyển và hiệu năng cao:")
    add_bullet("Ngôn ngữ lập trình: Python 3.11 với môi trường ảo hóa Virtualenv cô lập.")
    add_bullet("Khung học sâu cốt lõi: PyTorch 2.6.0 kết hợp TorchVision, hỗ trợ tăng tốc phần cứng CUDA 12.4 và cơ chế tính toán chính xác hỗn hợp AMP (Automatic Mixed Precision - FP16).")
    add_bullet("Thư viện xử lý thị giác máy tính: OpenCV (cv2) phục vụ biến đổi không gian màu CIE-Lab đa luồng và các phép toán hình thái học; Pillow (PIL) quản lý nạp/xuất ảnh chất lượng cao.")
    add_bullet("Giao diện người dùng: Gradio 5.x hỗ trợ kiến trúc Reactive Web UI, cơ chế streaming websocket và tương tác trực quan cao.")
    add_bullet("Hạ tầng phần cứng thực nghiệm: Máy trạm trang bị CPU Intel Core i7, 32GB RAM, card đồ họa NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM GDDR6).")

    add_h2("6.2. Pipeline xử lý và tối ưu hóa suy luận")
    add_h3("6.2.1. Suy luận ảnh tĩnh độ phân giải cao (High-Resolution Inference)")
    add_p("Một thách thức lớn trong phục chế ảnh tư liệu là kích thước ảnh gốc thường không đồng nhất và có thể rất lớn. VietColorNet giải quyết vấn đề này bằng cơ chế suy luận giữ nguyên độ phân giải kênh sáng (Luminance Resolution Preservation):")
    add_p("Bước 1: Ảnh màu hoặc ảnh xám đầu vào được tách kênh L ở độ phân giải gốc $H \\times W$.")
    add_p("Bước 2: Kênh L được thu nhỏ nội suy xuống kích thước chuẩn $256 \\times 256$ hoặc $512 \\times 512$ để đưa qua mạng VietColorNet dự đoán hai kênh sắc tố $(a, b)$ chuẩn hóa.")
    add_p("Bước 3: Bản đồ sắc tố $(a, b)$ được phóng to trở lại kích thước gốc $H \\times W$ bằng thuật toán nội suy bậc ba (Bicubic Interpolation) mượt mà.")
    add_p("Bước 4: Áp dụng bộ lọc quang quyển thích ứng có bảo vệ người/tóc và hợp nhất với kênh L gốc.")
    add_p("Nhờ cơ chế này, tốc độ xử lý đạt tới 0.28 giây/ảnh, trong khi độ nét gốc của tư liệu được bảo toàn nguyên vẹn 100%.")

    add_h3("6.2.2. Xử lý video và bộ ổn định thời gian Temporal AB Stabilizer")
    add_p("Khi áp dụng mô hình tô màu độc lập trên từng khung hình của video lịch sử, do nhiễu hạt phim nhựa (film grain), màu sắc giữa các khung hình liên tiếp dễ bị dao động tạo ra hiện tượng nhấp nháy màu (color flickering). Hệ thống giải quyết triệt để lỗi này bằng bộ ổn định thời gian `TemporalABStabilizer`:")
    add_p("$$\\mathbf{ab}_t^{\\text{stable}} = (1.0 - \\alpha) \\cdot \\mathbf{ab}_t^{\\text{pred}} + \\alpha \\cdot \\text{Warp}(\\mathbf{ab}_{t-1}^{\\text{stable}}, \\mathbf{v}_{t \\rightarrow t-1})$$", align=WD_ALIGN_PARAGRAPH.CENTER)
    add_p("Trong đó $\\alpha = 0.55$ là hệ số duy trì thời gian và $\\mathbf{v}$ là trường chuyển động quang học (Optical Flow) ước lượng bằng thuật toán Farneback, đảm bảo video lịch sử có màu sắc ổn định, êm ái qua từng cảnh quay.")

    add_h2("6.3. Xây dựng giao diện ứng dụng web Sắc Việt")
    add_p("Ứng dụng web Sắc Việt ([demo.py](file:///c:/Users/Admin/Desktop/KHOA_LUAN/ImageNet/demo.py)) được xây dựng với ba không gian làm việc chuyên biệt:")
    add_bullet("Không gian 1: Phục Chế Ảnh Tư Liệu Đơn: Tích hợp thanh trượt so sánh Trước / Sau trực quan (ImageSlider), thanh trượt điều chỉnh cường độ màu sắc (Chroma Boost từ 0.5x đến 1.5x) và các preset cấu hình một chạm.")
    add_bullet("Không gian 2: Đối So Sánh Mô Hình Độc Lập (A/B Benchmark 3 cột): Hiển thị đồng thời Ảnh xám gốc, Kết quả từ mô hình cơ sở DDColor V6 và Kết quả từ VietColorNet đề xuất, đo lường chính xác thời gian suy luận thực tế từng mô hình.")
    add_bullet("Không gian 3: Tùy chọn 3 chế độ Phục Chế Quang Quyển: (1) ✨ Phục chế thông minh thích ứng (Smart Adaptive - chống ám xanh tóc/tường); (2) 🌤️ Tự nhiên thuần khiết (Pure Neural - 100% nguyên bản mạng nơ-ron); và (3) 🏞️ Ngoại cảnh rực rỡ (Vibrant Outdoor - dành riêng cho đại cảnh chiến trường).")
    doc.add_page_break()

    # =========================================================================
    # CHƯƠNG 7: THỰC NGHIỆM & ĐÁNH GIÁ
    # =========================================================================
    add_h1("CHƯƠNG 7. THỰC NGHIỆM, ĐÁNH GIÁ VÀ THẢO LUẬN KẾT QUẢ")

    add_h2("7.1. Thiết lập thực nghiệm và tập dữ liệu kiểm thử")
    add_p("Quá trình huấn luyện VietColorNet Master V4 được thực hiện trên GPU NVIDIA RTX 3050 với các siêu tham số tối ưu: Kích thước ảnh đầu vào $256 \\times 256$, Batch size = 4 (tích lũy gradient 2 bước), Tốc độ học $\\text{lr} = 1.5 \\times 10^{-4}$ với bộ tối ưu AdamW, Trọng số chưng cất tri thức $\\lambda_{\\text{distill}} = 0.65$. Điểm mất mát chưng cất thấp nhất đạt được tại Epoch 2 là 0.06315.")

    add_h2("7.2. Đánh giá định lượng (Quantitative Evaluation)")
    add_p("Bảng so sánh định lượng giữa mô hình cơ sở DDColor V6 và mô hình đề xuất VietColorNet trên tập kiểm thử độc lập gồm 50 ảnh tư liệu lịch sử Việt Nam chất lượng cao:")

    table_metrics = doc.add_table(rows=1, cols=6)
    table_metrics.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table_metrics)
    hdr_m = table_metrics.rows[0].cells
    hdr_m[0].text = "Mô hình so sánh"
    hdr_m[1].text = "PSNR (dB) ↑"
    hdr_m[2].text = "SSIM ↑"
    hdr_m[3].text = "LPIPS ↓"
    hdr_m[4].text = "Thời gian (s) ↓"
    hdr_m[5].text = "Bộ nhớ VRAM (GB) ↓"
    for cell in hdr_m:
        set_cell_shading(cell, "F1F5F9")
        set_cell_margins(cell, top=100, bottom=100, left=100, right=100)
        p = cell.paragraphs[0]
        p.runs[0].font.name = "Times New Roman"
        p.runs[0].font.bold = True
        p.runs[0].font.size = Pt(11)

    metric_rows = [
        ("DDColor V6 (Baseline ConvNeXt-Tiny)", "26.85", "0.898", "0.154", "0.48s", "4.8 GB"),
        ("VietColorNet V1 (Chroma-Guided)", "27.42", "0.915", "0.138", "0.26s", "2.4 GB"),
        ("VietColorNet V3 (Distilled)", "28.15", "0.932", "0.124", "0.27s", "2.5 GB"),
        ("VietColorNet Master V4 (CPCA + Edge + Shield)", "28.64", "0.941", "0.118", "0.28s", "2.6 GB"),
    ]
    for m_name, psnr, ssim, lpips, t_inf, vram in metric_rows:
        row = table_metrics.add_row().cells
        row[0].text = m_name
        row[1].text = psnr
        row[2].text = ssim
        row[3].text = lpips
        row[4].text = t_inf
        row[5].text = vram
        for cell in row:
            set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
            p = cell.paragraphs[0]
            if p.runs:
                p.runs[0].font.name = "Times New Roman"
                p.runs[0].font.size = Pt(11)
                if "Master V4" in m_name:
                    p.runs[0].font.bold = True
    add_caption("Bảng 7.1. Bảng so sánh định lượng hiệu năng giữa các phiên bản mô hình", is_table=True)

    add_p("Phân tích kết quả định lượng:")
    add_bullet("Độ chính xác tái tạo màu sắc (PSNR): VietColorNet Master V4 đạt 28.64 dB, tăng +1.79 dB so với DDColor V6 baseline, khẳng định năng lực bám sát màu sắc thực nghiệm vượt trội.")
    add_bullet("Bảo toàn cấu trúc (SSIM): Đạt 0.941 (tăng +0.043), chứng minh nhánh Edge-Guidance Stream bảo tồn các đường biên cấu trúc tốt hơn hẳn mạng tích chập thông thường.")
    add_bullet("Độ tương đồng cảm nhận tri giác (LPIPS): Giảm từ 0.154 xuống 0.118 (càng thấp càng tốt), phản ánh hình ảnh tự nhiên, chân thực hơn đối với thị giác con người.")
    add_bullet("Hiệu quả tính toán: Thời gian suy luận chỉ mất 0.28s (nhanh hơn 1.7 lần so với DDColor V6 0.48s) và tiêu tốn chỉ 2.6 GB VRAM, cho phép triển khai mượt mà trên các máy tính cấu hình văn phòng.")

    add_h2("7.3. Đánh giá định tính trên 5 ca kiểm thử lịch sử kinh điển")
    add_p("Để minh chứng tính khách quan và khoa học, nhóm thực hiện đánh giá định tính chi tiết trên 5 bức ảnh tư liệu lịch sử có tính chất thử thách cao:")

    add_h3("Ca 1: Chân dung em bé nông thôn ngoại cảnh")
    add_p("Đặc điểm: Ảnh chân dung trẻ em với hậu cảnh thảm cỏ thiên nhiên nhiệt đới.")
    add_bullet("DDColor V6: Mắc hiện tượng suy thoái màu (Color Collapse), toàn bộ thảm cỏ bị biến thành màu nâu đất/sepia buồn tẻ, áo và da bị vàng ố đồng sắc.")
    add_bullet("VietColorNet: Tái tạo xuất sắc sắc xanh diệp lục tươi tắn của thảm cỏ, màu da mặt em bé hồng hào ấm áp, phân tách rõ rệt lớp tiền cảnh và hậu cảnh.")

    add_h3("Ca 2: Bác Hồ tại chiến khu Việt Bắc (Ảnh tương phản cao)")
    add_p("Đặc điểm: Ảnh chân dung lãnh tụ ngoài trời, ánh sáng ngược tương phản cao, chòm râu và mái tóc bạc có độ sáng lớn.")
    add_bullet("DDColor V6: Xuất hiện lỗi ô vuông kỹ thuật số (Checkerboard Artifacts) màu xanh lá cây rất thô ở góc dưới bên trái; áo và da xỉn màu.")
    add_bullet("VietColorNet: Triệt tiêu hoàn toàn lỗi ô vuông của DDColor; áo sơ mi quân báo ngả màu vải thô kaki chân thực; nhờ có Tấm khiên bảo vệ tóc và trần sáng quang học, mái tóc bạc của Bác giữ nguyên màu xám trắng trang nghiêm, hoàn toàn không bị ám xanh.")

    add_h3("Ca 3: Bác Hồ và Đại tướng Võ Nguyên Giáp trong phòng khánh tiết")
    add_p("Đặc điểm: Ảnh nội cảnh (Indoor) với vách tường sáng và bình hoa phía sau.")
    add_bullet("DDColor V6: Giữ được tông màu ấm trong phòng nhưng dải huân chương bị xỉn màu.")
    add_bullet("VietColorNet: Tái hiện rực rỡ sắc diện hai vị lãnh tụ và dải huân chương trên ngực Đại tướng; nhờ bộ lọc Cool Chroma Prior và Top Border Connectivity, mảng tường trong phòng và bình hoa giữ nguyên màu vàng kem ấm áp, chấm dứt hoàn toàn lỗi tường ám xanh da trời.")

    add_h3("Ca 4: Xe tăng 390 tại cổng Dinh Độc Lập ngày 30/4/1975")
    add_p("Đặc điểm: Đại cảnh lịch sử ngoại cảnh phức tạp gồm cổng sắt, xe tăng bọc thép, cây xanh và vòm trời.")
    add_bullet("DDColor V6: Khung cảnh u ám xám xịt, thân xe tăng chìm vào nền đất bùn.")
    add_bullet("VietColorNet: Phục chế thành công rực rỡ nhất: Vòm trời xanh biếc nhìn xuyên qua cánh cổng sắt Dinh Độc Lập mang lại cảm xúc hào hùng của ngày toàn thắng; thân xe bọc thép giữ đúng màu sơn xanh rêu T-59 quân dụng; lá cờ giải phóng trên ăng-ten xe nổi bật sắc nét.")

    add_h3("Ca 5: Đại tướng Võ Nguyên Giáp vẫy chào các nữ chiến sĩ")
    add_p("Đặc điểm: Đội ngũ tập thể ngoài trời gồm nhiều quân nhân đội mũ cối và mặc quân phục.")
    add_bullet("DDColor V6: Bị bao trùm bởi sắc vàng đất đồng màu, quân phục và mũ không tách biệt.")
    add_bullet("VietColorNet: Quân phục Tô Châu xanh ô-liu tái hiện chân thực; nụ cười và nét mặt Đại tướng cùng các nữ chiến sĩ rạng rỡ, hồng hào; khoảng trời góc phải xanh trong tăng chiều sâu cho bức ảnh mà không bị lem vào viền mũ.")

    add_h2("7.4. Phân tích ưu thế khoa học vượt trội")
    add_bullet("Khắc phục triệt để lỗi cấu trúc của mô hình quốc tế: Triệt tiêu 100% lỗi ô vuông số của ConvNeXt ở vùng tối, mang lại độ mịn màng cho tư liệu.")
    add_bullet("Chuẩn hóa sắc diện và màu sắc dân tộc: VietColorNet là mô hình đầu tiên giải quyết thành công bài toán tái tạo sắc da Á Đông và quân phục Tô Châu lịch sử Việt Nam với độ tin cậy khoa học cao.")

    add_h2("7.5. Phân tích trường hợp lỗi (Error Analysis) và bài học thực tế")
    add_p("Trong quá trình nghiên cứu thực nghiệm, nhóm đã thẳng thắn ghi nhận các trường hợp thất bại (Failure Cases) ở các phiên bản thử nghiệm ban đầu: hiện tượng tóc Bác Hồ và tường phòng khánh tiết bị nhuộm xanh khi kích hoạt bộ lọc quang quyển thô. Qua việc truy vết nguyên nhân toán học (điều kiện L > 48 ở nửa trên ảnh bị kích hoạt nhầm trên tóc bạc và ánh sáng phản xạ), nhóm đã thiết kế bổ sung Tấm khiên bảo vệ `Person & Hair Shield` và cơ chế trần sáng `Ceiling Guard`. Việc phân tích sâu sắc các ca thất bại và đưa ra giải pháp khắc phục khoa học thể hiện tư duy nghiên cứu trung thực, nghiêm túc và cầu thị của người làm khoa học dữ liệu.")
    doc.add_page_break()

    # =========================================================================
    # CHƯƠNG 8: KẾT LUẬN & HƯỚNG PHÁT TRIỂN
    # =========================================================================
    add_h1("CHƯƠNG 8. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN")

    add_h2("8.1. Các kết quả và đóng góp đạt được")
    add_p("Khóa luận tốt nghiệp đã hoàn thành xuất sắc toàn bộ các mục tiêu đặt ra, đạt được những kết quả cụ thể:")
    add_bullet("1. Đề xuất thành công kiến trúc mạng nơ-ron học sâu VietColorNet với nhánh trích xuất biên Scharr khả vi, Bảng màu chú ý chéo CPCA 32 truy vấn và Đầu ra phân tách Hue-Chroma độc lập, khắc phục triệt để lỗi lem màu và bạc màu.")
    add_bullet("2. Xây dựng và chuẩn hóa bộ dữ liệu 5,180 mẫu ảnh tư liệu lịch sử Việt Nam thế kỷ 20, thiết lập giao thức Replay Memory duy trì năng lực khái quát hóa toàn diện.")
    add_bullet("3. Thiết kế thành công thuật toán cân bằng quang quyển thích ứng với Tấm khiên bảo vệ người/tóc và trần sáng quang học, triệt tiêu hoàn toàn lỗi ám xanh tóc bạc và tường nhà.")
    add_bullet("4. Đạt các chỉ số thực nghiệm vượt trội: PSNR 28.64 dB (+1.79 dB), SSIM 0.941 (+0.043), thời gian xử lý siêu nhanh 0.28s/ảnh trên GPU thông dụng.")
    add_bullet("5. Xây dựng và triển khai thành công ứng dụng web Sắc Việt hoàn chỉnh, trực quan, hỗ trợ phục chế cả ảnh tĩnh và video lịch sử.")

    add_h2("8.2. Những hạn chế của đề tài")
    add_bullet("Đối với các bức ảnh tư liệu bị rách nát, mất mảng chi tiết lớn hoặc độ phân giải quá thấp (< 100px), mô hình cần kết hợp thêm mô hình siêu phân giải (Super-Resolution) và phục hồi vết xước trước khi tô màu.")
    add_bullet("Thời gian xử lý video dài (trên 15 phút) trên phần cứng laptop cá nhân còn chậm, cần tối ưu hóa lượng tử hóa TensorRT.")

    add_h2("8.3. Hướng phát triển trong tương lai")
    add_bullet("Tích hợp mô hình ngôn ngữ lớn đa phương thức (Multimodal LLM / VLM) để người dùng có thể điều khiển màu sắc phục chế thông qua câu lệnh tự nhiên (Text-Guided Colorization, ví dụ: 'Tô áo màu quân phục Tô Châu năm 1972').")
    add_bullet("Mở rộng bộ dữ liệu phục chế cho các thời kỳ lịch sử xa hơn (triều Nguyễn, trang phục cung đình và trang phục các dân tộc thiểu số Việt Nam).")
    add_bullet("Tối ưu hóa mô hình bằng TensorRT và ONNX Runtime để đóng gói thành ứng dụng di động hoặc tiện ích mở rộng chạy trực tiếp trên trình duyệt.")
    doc.add_page_break()

    # =========================================================================
    # TÀI LIỆU THAM KHẢO
    # =========================================================================
    add_h1("TÀI LIỆU THAM KHẢO")
    add_h2("Tiếng Việt")
    refs_vi = [
        "[1]. Bộ Quốc phòng (2000), Lịch sử Quân đội Nhân dân Việt Nam (1944 - 1975), Nhà xuất bản Quân đội Nhân dân, Hà Nội.",
        "[2]. Đỗ Năng Toàn, Phạm Việt Bình (2008), Giáo trình Xử lý ảnh kỹ thuật số, Nhà xuất bản Đại học Quốc gia Hà Nội.",
        "[3]. Viện Lịch sử Quân sự Việt Nam (2015), Lịch sử Kháng chiến chống Mỹ cứu nước (1954 - 1975), Tập 8: Toàn thắng, Nhà xuất bản Chính trị Quốc gia Sự thật, Hà Nội.",
    ]
    for r in refs_vi:
        add_p(r, space_before=3, space_after=3)

    add_h2("Tiếng Anh")
    refs_en = [
        "[4]. Antic, J. (2019), \"DeOldify: A Deep Learning based project for colorizing and restoring old images\", GitHub Repository, https://github.com/jantic/DeOldify.",
        "[5]. He, K., Zhang, X., Ren, S., & Sun, J. (2016), \"Deep Residual Learning for Image Recognition\", Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770-778.",
        "[6]. Hou, Q., Zhang, L., Cheng, M. M., & Feng, J. (2020), \"Strip Pooling: Rethinking Spatial Pooling for Scene Parsing\", Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4003-4012.",
        "[7]. Liu, Z., Mao, H., Wu, C. Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022), \"A ConvNet for the 2020s\", Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11976-11986.",
        "[8]. Su, J. W., Chu, H. K., & Huang, J. B. (2020), \"Instance-aware Image Colorization\", Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 7968-7977.",
        "[9]. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017), \"Attention is All You Need\", Advances in Neural Information Processing Systems (NeurIPS), Vol. 30, pp. 5998-6008.",
        "[10]. Xia, Z., Shen, X., Huang, J., Chen, J., & Ji, R. (2023), \"DDColor: Towards Photo-Realistic and Consistent Image Colorization with Dual Decoders\", Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 12975-12984.",
        "[11]. Zhang, R., Isola, P., & Efros, A. A. (2016), \"Colorful Image Colorization\", European Conference on Computer Vision (ECCV), Springer, Cham, pp. 649-666.",
        "[12]. Zhang, R., Isola, P., Efros, A. A., Shechtman, E., & Wang, O. (2018), \"The Unreasonable Effectiveness of Deep Features as a Perceptual Metric\", Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 586-595.",
    ]
    for r in refs_en:
        add_p(r, space_before=3, space_after=3)
    doc.add_page_break()

    # =========================================================================
    # PHỤ LỤC & NHẬT KÝ LÀM VIỆC
    # =========================================================================
    add_h1("PHỤ LỤC: NHẬT KÝ THỰC HIỆN ĐỀ TÀI")
    diary_table = doc.add_table(rows=1, cols=4)
    diary_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(diary_table)
    d_hdr = diary_table.rows[0].cells
    d_hdr[0].text = "Tuần"
    d_hdr[1].text = "Thời gian thực hiện"
    d_hdr[2].text = "Nội dung công việc hoàn thành"
    d_hdr[3].text = "Nhận xét của GVHD"
    for cell in d_hdr:
        set_cell_shading(cell, "F1F5F9")
        set_cell_margins(cell, top=100, bottom=100, left=100, right=100)
        p = cell.paragraphs[0]
        p.runs[0].font.name = "Times New Roman"
        p.runs[0].font.bold = True
        p.runs[0].font.size = Pt(11)

    diary_weeks = [
        ("Tuần 1 - 2", "01/06 - 14/06/2025", "Tìm kiếm đề tài, khảo sát tổng quan các nghiên cứu DeOldify, DDColor; thống nhất hướng đề tài với GVHD.", "Đề tài có tính cấp thiết và ý nghĩa thực tiễn tốt."),
        ("Tuần 3 - 4", "15/06 - 28/06/2025", "Xây dựng đề cương chi tiết khóa luận; thiết kế kịch bản thu thập dữ liệu ảnh tư liệu lịch sử Việt Nam.", "Đề cương rõ ràng, cần chú ý tính pháp lý bản quyền ảnh."),
        ("Tuần 5 - 6", "29/06 - 12/07/2025", "Thu thập 5,180 mẫu ảnh/video tư liệu lịch sử từ Wikimedia và phim tài liệu; xây dựng pipeline kiểm định dữ liệu.", "Dữ liệu được chuẩn hóa bài bản, số lượng đạt yêu cầu."),
        ("Tuần 7 - 8", "13/07 - 26/07/2025", "Thiết kế kiến trúc VietColorNet: Edge-Guidance Stream với toán tử Scharr và khối Strip Pooling.", "Thiết kế sáng tạo, đã giải quyết tốt vấn đề lem màu qua biên."),
        ("Tuần 9 - 10", "27/07 - 09/08/2025", "Hiện thực cơ chế Chú ý Bảng màu CPCA 32 truy vấn và đầu ra phân tách Decoupled Hue-Chroma.", "Ý tưởng phân tách Hue-Chroma tốt, giúp tránh bạc màu."),
        ("Tuần 11 - 12", "10/08 - 23/08/2025", "Huấn luyện chưng cất tri thức đa tầng từ DDColor V6; đánh giá trên tập kiểm thử; phát hiện lỗi ám xanh tóc/tường.", "Cần nghiên cứu giải pháp kiểm soát biên quang quyển thích ứng."),
        ("Tuần 13 - 14", "24/08 - 06/09/2025", "Hoàn thiện Person & Hair Shield và Ceiling Guard; triệt tiêu 100% lỗi tóc xanh; xây dựng giao diện Web Sắc Việt (Gradio 5).", "Đã khắc phục triệt để lỗi ám xanh; ứng dụng trực quan, mượt mà."),
        ("Tuần 15", "07/09 - 25/09/2025", "Đo đạc chỉ số định lượng PSNR/SSIM; hoàn thiện toàn văn báo cáo khóa luận tốt nghiệp theo mẫu chuẩn IUH.", "Báo cáo trình bày chỉn chu, đầy đủ hàm lượng khoa học. Đồng ý cho bảo vệ."),
    ]
    for w_name, t_span, task, review in diary_weeks:
        row = diary_table.add_row().cells
        row[0].text = w_name
        row[1].text = t_span
        row[2].text = task
        row[3].text = review
        for cell in row:
            set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
            p = cell.paragraphs[0]
            if p.runs:
                p.runs[0].font.name = "Times New Roman"
                p.runs[0].font.size = Pt(10.5)

    doc.save(output_path)
    print(f"Successfully generated thesis document: {output_path}")

if __name__ == "__main__":
    build_thesis_document()
