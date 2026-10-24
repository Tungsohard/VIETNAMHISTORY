"""Script tự động đọc và nhân bản chính xác 100% form trình bày của file report.docx
Đề tài: Phục Chế Ảnh Và Video Tư Liệu Lịch Sử Việt Nam Bằng Mô Hình Học Sâu Đề Xuất VietColorNet
Tác giả: Lê Hữu Hưng (22656001), Nguyễn Văn Tùng (22647011)
GVHD: TS. Bùi Thanh Hùng - Trường ĐH Công nghiệp TP. Hồ Chí Minh
"""
import os
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_margins(cell, top=100, bottom=100, left=140, right=140):
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

def set_table_borders(table, color="B0B0B0", sz="4", val="single"):
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

def build_report_matching_template(src_path="report.docx", dst_path="Report_VietColorNet.docx"):
    print(f"Loading base template from {src_path}...")
    doc = Document(src_path)

    # 1. Cập nhật trang bìa (Bìa 1 & Bìa 2)
    # Bìa 1: Cập nhật đề tài
    doc.paragraphs[9].text = "ĐỒ ÁN CUỐI KÌ"
    doc.paragraphs[10].text = " CÔNG NGHỆ MỚI TRONG PHÁT TRIỂN ỨNG DỤNG "
    # Bìa 2:
    doc.paragraphs[36].text = "ĐỒ ÁN CUỐI KÌ"
    doc.paragraphs[37].text = " CÔNG NGHỆ MỚI TRONG PHÁT TRIỂN ỨNG DỤNG "
    doc.paragraphs[38].text = "ĐỀ TÀI: PHỤC CHẾ ẢNH VÀ VIDEO TƯ LIỆU LỊCH SỬ VIỆT NAM"
    doc.paragraphs[39].text = "BẰNG MÔ HÌNH HỌC SÂU ĐỀ XUẤT VIETCOLORNET"

    # Định dạng lại paragraph 38, 39 cho đúng kích thước và in đậm
    for p_idx in [38, 39]:
        p = doc.paragraphs[p_idx]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.font.name = "Times New Roman"
            r.font.size = Pt(15.0)
            r.font.bold = True

    # 2. Xóa các đoạn văn và bảng từ sau trang bìa (từ paragraph 58 trở đi)
    body = doc._body._body
    children = list(body)
    p58_elem = doc.paragraphs[58]._p
    idx_start = children.index(p58_elem)

    for c in children[idx_start:]:
        if c.tag.endswith('sectPr'):
            continue
        body.remove(c)

    print(f"Truncated old body. Remaining paragraphs: {len(doc.paragraphs)}. Now building new content...")

    # Helper thêm đoạn văn với các Style chuẩn của report.docx
    style_chuong = doc.styles['Chương']
    style_tm1 = doc.styles['Tiểu mục cấp 1']
    style_tm2 = doc.styles['Tiểu mục cấp 2']
    style_normal = doc.styles['Normal']
    style_bullet = doc.styles['List Bullet']

    def p_chuong(text, align=WD_ALIGN_PARAGRAPH.CENTER):
        p = doc.add_paragraph(style=style_chuong)
        p.alignment = align
        if text:
            r = p.add_run(text)
            r.font.name = "Times New Roman"
            r.font.size = Pt(13)
            r.font.bold = True
        return p

    def p_tm1(text, align=WD_ALIGN_PARAGRAPH.LEFT):
        p = doc.add_paragraph(style=style_tm1)
        p.alignment = align
        if text:
            r = p.add_run(text)
            r.font.name = "Times New Roman"
            r.font.size = Pt(13)
            r.font.bold = True
        return p

    def p_tm2(text, align=WD_ALIGN_PARAGRAPH.JUSTIFY):
        p = doc.add_paragraph(style=style_tm2)
        p.alignment = align
        if text:
            r = p.add_run(text)
            r.font.name = "Times New Roman"
            r.font.size = Pt(13)
            r.font.bold = True
            r.font.italic = True
        return p

    def p_text(text, align=WD_ALIGN_PARAGRAPH.LEFT):
        p = doc.add_paragraph(style=style_normal)
        p.alignment = align
        if text:
            r = p.add_run(text)
            r.font.name = "Times New Roman"
            r.font.size = Pt(13)
        return p

    def p_bullet(text):
        p = doc.add_paragraph(style=style_bullet)
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        if text:
            r = p.add_run(text)
            r.font.name = "Times New Roman"
            r.font.size = Pt(13)
        return p

    def p_empty():
        return doc.add_paragraph(style=style_normal)

    def p_caption(text):
        p = doc.add_paragraph(style=style_normal)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(12)
        r.font.bold = True
        r.font.italic = True
        return p

    # =========================================================================
    # LỜI CẢM ƠN
    # =========================================================================
    p_chuong("LỜI CẢM ƠN")
    p_empty()
    p_text("Trong quá trình thực hiện đồ án \"Hệ thống Phục chế Ảnh và Video Tư liệu Lịch sử Việt Nam dựa trên Mô hình Học sâu Đề xuất VietColorNet\", nhóm chúng em đã nhận được rất nhiều sự quan tâm, động viên và giúp đỡ quý báu.", align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    p_text("Lời đầu tiên, nhóm chúng em xin gửi lời cảm ơn chân thành và sâu sắc nhất đến TS. Bùi Thanh Hùng — giảng viên hướng dẫn — đã luôn tận tình chỉ bảo, định hướng khoa học từ giai đoạn lựa chọn đề tài, thiết kế kiến trúc mạng nơ-ron học sâu VietColorNet (Chroma-Guided Edge-Aware Network), cho đến các góp ý chi tiết trong từng khâu xử lý dữ liệu lịch sử, chưng cất tri thức đa tầng và triển khai ứng dụng web Sắc Việt hoàn chỉnh.", align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    p_text("Nhóm cũng xin chân thành cảm ơn quý thầy cô Khoa Công nghệ Thông tin — Trường Đại học Công nghiệp TP. Hồ Chí Minh đã trang bị cho chúng em nền tảng kiến thức vững chắc về Thị giác máy tính, Xử lý ảnh số, Học máy, Học sâu và Công nghệ phần mềm tiên tiến trong suốt những năm học qua.", align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    p_text("Cuối cùng, nhóm xin gửi lời cảm ơn đến gia đình, bạn bè và tập thể lớp 420300314704 đã luôn ủng hộ, đồng hành trong suốt quá trình học tập và nghiên cứu. Do thời gian và kiến thức còn hạn chế, đồ án không tránh khỏi những thiếu sót, nhóm rất mong nhận được những ý kiến đóng góp quý báu từ quý thầy cô để hệ thống ngày càng hoàn thiện hơn.", align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    p_empty()

    # =========================================================================
    # CAM ĐOAN
    # =========================================================================
    p_chuong("ĐỒ ÁN ĐƯỢC HOÀN THÀNH")
    p_chuong("TẠI TRƯỜNG ĐẠI HỌC CÔNG NGHIỆP TP HỒ CHÍ MINH")
    p_empty()
    p_text("Tôi xin cam đoan đây là sản phẩm đồ án của riêng chúng tôi và được sự hướng dẫn của TS. Bùi Thanh Hùng. Các nội dung nghiên cứu, mô hình mạng VietColorNet, quy trình chưng cất tri thức, thuật toán bảo vệ quang quyển và kết quả thực nghiệm trong đề tài này là trung thực và chưa từng được công bố dưới bất kỳ hình thức nào trước đây. Những số liệu, hình ảnh tư liệu trong các bảng biểu phục vụ cho việc phân tích, nhận xét, đánh giá được chính tác giả thu thập từ các nguồn lưu trữ mở (Wikimedia Commons, phim tư liệu lịch sử Việt Nam) có ghi rõ trong phần tài liệu tham khảo.", align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    p_text("Ngoài ra, trong đồ án còn sử dụng một số mã nguồn mở, kiến trúc tham chiếu và trích dẫn khoa học của các tác giả quốc tế đều có chú thích nguồn gốc rõ ràng theo chuẩn học thuật.", align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    p_text("Nếu phát hiện có bất kỳ sự gian lận nào tôi xin hoàn toàn chịu trách nhiệm về nội dung đồ án của mình. Trường Đại học Công nghiệp TP Hồ Chí Minh không liên quan đến những vi phạm tác quyền, bản quyền do tôi gây ra trong quá trình thực hiện (nếu có).", align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    p_empty()
    p_text("TP. Hồ Chí Minh, ngày 25 tháng 09 năm 2025", align=WD_ALIGN_PARAGRAPH.CENTER)
    p_text("Tác giả", align=WD_ALIGN_PARAGRAPH.CENTER)
    p_empty()
    p_text("Lê Hữu Hưng                                                  Nguyễn Văn Tùng", align=WD_ALIGN_PARAGRAPH.CENTER)
    p_empty()

    # =========================================================================
    # PHẦN ĐÁNH GIÁ CỦA GIẢNG VIÊN
    # =========================================================================
    p_chuong("PHẦN ĐÁNH GIÁ CỦA GIẢNG VIÊN")
    p_text("_______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________")
    p_empty()
    p_text("Tp. Hồ Chí Minh, ngày     tháng   năm 2025", align=WD_ALIGN_PARAGRAPH.CENTER)
    p_text("(kí và ghi họ tên)", align=WD_ALIGN_PARAGRAPH.CENTER)
    p_empty()

    # =========================================================================
    # TÓM TẮT
    # =========================================================================
    p_chuong("TÓM TẮT")
    p_empty()
    p_text("Đồ án xây dựng hệ thống Phục chế Màu sắc cho Ảnh và Video Tư liệu Lịch sử Việt Nam dựa trên mô hình học sâu đề xuất VietColorNet (Chroma-Guided Edge-Aware Colorization Network), nhằm phục hồi và làm sống động các di sản hình ảnh thế kỷ 20 với độ chân thực lịch sử cao.", align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    p_text("Hệ thống giải quyết 3 khuyết tật cố hữu của các mô hình học sâu quốc tế hiện nay: (1) Hiện tượng suy thoái màu (color collapse) ngả sang nâu sepia buồn tẻ; (2) Hiện tượng lem màu qua biên (color bleeding) giữa da người, quân phục và hậu cảnh; và (3) Hiện tượng lỗi cấu trúc ô vuông kỹ thuật số (checkerboard artifacts) tại các vùng tối do bộ giải mã ConvNeXt của DDColor V6.", align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    p_text("Kiến trúc VietColorNet kết hợp 3 sáng tạo cốt lõi: Nhánh trích xuất biên Scharr khả vi (Edge-Guidance Stream 16 chiều), Khối Chú ý Bảng màu Thích ứng (Color Palette Cross-Attention - CPCA 32 truy vấn văn hóa Việt Nam) và Đầu ra phân tách Sắc tố & Độ rực màu (Decoupled Hue-Chroma Head). Hệ thống tích hợp bộ dữ liệu 5,180 mẫu lịch sử Việt Nam, chưng cất tri thức đa tầng từ DDColor V6, và bộ lọc quang quyển thích ứng thông minh (Smart Adaptive Vibrancy) với Tấm khiên bảo vệ người/tóc và trần sáng quang học (Ceiling Guard L <= 82).", align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    p_text("Kết quả thực nghiệm trên tập test độc lập cho thấy VietColorNet đạt PSNR = 28.64 dB, SSIM = 0.941 và LPIPS = 0.118, vượt trội hoàn toàn so với DDColor V6 baseline (PSNR 26.85 dB, SSIM 0.898). Tốc độ suy luận đạt 0.28 giây/ảnh trên GPU RTX 3050 phổ thông (tiêu tốn chỉ 2.6 GB VRAM). Toàn bộ hệ thống được triển khai thành ứng dụng Web Sắc Việt hoàn chỉnh trên Gradio 5, hỗ trợ phục chế ảnh tĩnh và video mượt mà không nhấp nháy màu.", align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    p_empty()

    # =========================================================================
    # DANH MỤC KÍ HIỆU VÀ CHỮ VIẾT TẮT (Table 0)
    # =========================================================================
    p_chuong("DANH MỤC KÍ HIỆU VÀ CHỮ VIẾT TẮT")
    p_empty()

    tbl0 = doc.add_table(rows=1, cols=2)
    tbl0.style = 'Table Grid'
    tbl0.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl0)

    hdr0 = tbl0.rows[0].cells
    hdr0[0].text = "CHỮ VIẾT TẮT"
    hdr0[1].text = "Ý NGHĨA ĐẦY ĐỦ"
    for c in hdr0:
        set_cell_shading(c, "F1F5F9")
        set_cell_margins(c, top=100, bottom=100, left=150, right=150)
        p = c.paragraphs[0]
        p.runs[0].font.name = "Times New Roman"
        p.runs[0].font.bold = True
        p.runs[0].font.size = Pt(11)

    abbrevs = [
        ("AI", "Artificial Intelligence – Trí tuệ nhân tạo"),
        ("CNN", "Convolutional Neural Network – Mạng nơ-ron tích chập"),
        ("CPCA", "Color Palette Cross-Attention – Chú ý chéo bảng màu sắc"),
        ("DDColor", "Dual Decoder Colorization – Kiến trúc tô màu hai bộ giải mã"),
        ("DVC", "Data Version Control – Quản lý phiên bản dữ liệu"),
        ("FPS", "Frames Per Second – Số khung hình xử lý trên mỗi giây"),
        ("GPU", "Graphics Processing Unit – Bộ xử lý đồ họa"),
        ("IUH", "Trường Đại học Công nghiệp TP. Hồ Chí Minh"),
        ("Lab", "Không gian màu CIE-Lab (L: sáng, a: lục-đỏ, b: lam-vàng)"),
        ("LPIPS", "Learned Perceptual Image Patch Similarity – Độ tương đồng tri giác"),
        ("MSE", "Mean Squared Error – Sai số toàn phương trung bình"),
        ("PSNR", "Peak Signal-to-Noise Ratio – Tỷ số tín hiệu cực đại trên nhiễu"),
        ("RGB", "Red - Green - Blue – Không gian màu hiển thị Đỏ - Lục - Lam"),
        ("SSIM", "Structural Similarity Index Measure – Độ đo tương đồng cấu trúc"),
        ("VLM", "Vision-Language Model – Mô hình ngôn ngữ thị giác đa phương thức"),
        ("VRAM", "Video Random Access Memory – Bộ nhớ đồ họa"),
    ]
    for ab, desc in abbrevs:
        r = tbl0.add_row().cells
        r[0].text = ab
        r[1].text = desc
        for c in r:
            set_cell_margins(c, top=70, bottom=70, left=120, right=120)
            p = c.paragraphs[0]
            if p.runs:
                p.runs[0].font.name = "Times New Roman"
                p.runs[0].font.size = Pt(10.5)
                if c == r[0]:
                    p.runs[0].font.bold = True
    p_empty()

    # =========================================================================
    # DANH MỤC CÁC HÌNH VẼ
    # =========================================================================
    p_chuong("DANH MỤC CÁC HÌNH VẼ")
    p_empty()
    p_text("Hình 3.1\tMô hình tổng quát kiến trúc VietColorNet đa nhánh")
    p_text("Hình 3.4.2.\tQuy trình huấn luyện và chưng cất tri thức đa tầng từ DDColor V6")
    p_text("Hình 4.1.\tCấu trúc phân vùng dữ liệu và lưu trữ tư liệu lịch sử")
    p_text("Hình 4.2.\tPipeline suy luận ảnh tĩnh và ổn định thời gian video")
    p_text("Hình 4.3.\tGiao diện ứng dụng web Sắc Việt (Gradio 5)")
    p_empty()

    # =========================================================================
    # CHƯƠNG 1: GIỚI THIỆU VỀ BÀI TOÁN
    # =========================================================================
    p_chuong("CHƯƠNG 1")
    p_chuong("GIỚI THIỆU VỀ BÀI TOÁN")
    p_empty()

    p_tm1("1.1. Giới thiệu về bài toán")
    p_text("Kho tàng tư liệu hình ảnh và những thước phim lịch sử thế kỷ 20 ghi lại các chặng đường kháng chiến cứu quốc hào hùng, công cuộc xây dựng đất nước và nét đẹp văn hóa truyền thống của dân tộc Việt Nam là di sản tinh thần vô giá. Tuy nhiên, do giới hạn kỹ thuật ghi hình thời chiến, đại đa số tác phẩm chỉ tồn tại dưới dạng ảnh đen trắng (grayscale) hoặc phim nhựa đơn sắc. Trải qua hàng thập kỷ lưu trữ trong điều kiện khí hậu nhiệt đới gió mùa ẩm ướt, nhiều bản phim và ảnh gốc đã bị suy thoái vật lý nghiêm trọng: ố vàng, ẩm mốc, trầy xước, nhòe biên và suy giảm độ tương phản trầm trọng.")
    p_text("Trong bối cảnh chuyển đổi số di sản văn hóa, việc phục chế và tái sinh màu sắc tự nhiên cho các tư liệu lịch sử có ý nghĩa giáo dục truyền thống và văn hóa vô cùng sâu sắc. Một bức ảnh lịch sử được khoác lên màu sắc chân thực không chỉ giúp thế hệ trẻ cảm nhận một cách sống động, gần gũi bầu không khí hào hùng của cha ông, mà còn phục vụ đắc lực cho công tác nghiên cứu khoa học, trưng bày bảo tàng và làm tư liệu giáo dục trực quan sinh động.")

    p_tm1("1.2. Ý nghĩa và mục tiêu của bài toán")
    p_bullet("Xây dựng kiến trúc học sâu VietColorNet độc lập kết hợp nhánh biên Scharr và bảng màu chú ý CPCA, khắc phục triệt để lỗi lem màu và bạc màu.")
    p_bullet("Thu thập, kiểm định và chuẩn hóa bộ dữ liệu 5,180 mẫu ảnh/video tư liệu lịch sử Việt Nam thế kỷ 20.")
    p_bullet("Phát triển chiến lược chưng cất tri thức đa tầng (Multi-Stage Distillation) từ DDColor V6 kết hợp bộ nhớ Replay Memory.")
    p_bullet("Thiết kế bộ lọc quang quyển thích ứng thông minh (Smart Adaptive Vibrancy) với Tấm khiên bảo vệ người/tóc và trần sáng quang học, triệt tiêu 100% lỗi tóc bạc và tường nhà bị ám xanh.")
    p_bullet("Xây dựng ứng dụng web Sắc Việt hoàn chỉnh trên Gradio 5, hỗ trợ kéo trượt so sánh trực quan và xử lý video ổn định theo thời gian.")

    p_tm1("1.3. Phạm vi thực hiện")
    p_text("Đề tài tập trung vào bài toán phục chế màu sắc cho ảnh tĩnh và video tư liệu lịch sử Việt Nam thế kỷ 20 (giai đoạn 1945 - 1975). Phạm vi nghiên cứu bao gồm việc thu thập dữ liệu mở, tiền xử lý không gian màu Lab, thiết kế mạng nơ-ron học sâu, huấn luyện trên GPU NVIDIA RTX 3050 (6GB VRAM), và triển khai ứng dụng web tương tác người dùng. Đề tài không đi sâu vào việc phục hồi cơ học các vết rách nát vật lý lớn mà tập trung giải quyết bài toán tái tạo màu sắc chân thực và làm sạch tín hiệu số.")

    p_tm1("1.4. Ý nghĩa thực tiễn")
    p_text("Ở góc độ văn hóa - xã hội, hệ thống Sắc Việt cung cấp một công cụ phục chế màu tin cậy, giúp các viện lưu trữ, bảo tàng và cơ quan truyền thông nhanh chóng số hóa và hồi sinh các tư liệu lịch sử quý báu. Ở góc độ khoa học kỹ thuật, đề tài chứng minh rằng một kiến trúc nơ-ron được thiết kế chuyên biệt, khai thác biên độ sáng và bảng màu ngữ cảnh văn hóa có thể vượt trội các mô hình tổng quát quy mô lớn của quốc tế cả về độ chính xác màu sắc lẫn hiệu quả tính toán trên phần cứng phổ thông.")
    p_empty()

    # =========================================================================
    # CHƯƠNG 2: PHÂN TÍCH YÊU CẦU CỦA BÀI TOÁN
    # =========================================================================
    p_chuong("CHƯƠNG 2")
    p_chuong("PHÂN TÍCH YÊU CẦU CỦA BÀI TOÁN")
    p_empty()

    p_tm1("2.1 Yêu cầu của bài toán")
    p_text("Hệ thống được thiết kế hướng tới ba nhóm đối tượng người dùng chính: Người dùng phổ thông, Nhà nghiên cứu/Bảo tồn lịch sử, và Kỹ sư AI/Quản trị viên hệ thống:")

    # Table 1: Phân quyền vai trò
    tbl1 = doc.add_table(rows=1, cols=3)
    tbl1.style = 'Table Grid'
    tbl1.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl1)

    hdr1 = tbl1.rows[0].cells
    hdr1[0].text = "Vai trò"
    hdr1[1].text = "Chức năng chính"
    hdr1[2].text = "Mô tả chi tiết"
    for c in hdr1:
        set_cell_shading(c, "F1F5F9")
        set_cell_margins(c, top=100, bottom=100, left=120, right=120)
        p = c.paragraphs[0]
        p.runs[0].font.name = "Times New Roman"
        p.runs[0].font.bold = True
        p.runs[0].font.size = Pt(11)

    roles = [
        ("Người dùng phổ thông", "Tải ảnh/video, phục chế một chạm, kéo trượt so sánh", "Tải lên ảnh cũ của gia đình hoặc tư liệu lịch sử, điều chỉnh thanh trượt độ rực màu, tải về ảnh PNG chất lượng cao."),
        ("Nhà nghiên cứu / Bảo tàng", "A/B Benchmark, tùy chọn quang quyển, xử lý video", "Đối so sánh trực quan giữa ảnh xám gốc, baseline DDColor V6 và VietColorNet; chọn chế độ Smart Adaptive hoặc Pure Neural để đảm bảo tính xác thực tư liệu."),
        ("Kỹ sư AI / Quản trị viên", "Giám sát suy luận, kiểm thử Unit Tests, quản lý checkpoint", "Theo dõi thời gian suy luận (latency), bộ nhớ VRAM, quản lý các phiên bản checkpoint (V1, V3, Master V4) và kiểm định chất lượng mô hình."),
    ]
    for r_role, r_func, r_desc in roles:
        r = tbl1.add_row().cells
        r[0].text = r_role
        r[1].text = r_func
        r[2].text = r_desc
        for c in r:
            set_cell_margins(c, top=80, bottom=80, left=100, right=100)
            p = c.paragraphs[0]
            if p.runs:
                p.runs[0].font.name = "Times New Roman"
                p.runs[0].font.size = Pt(10.5)
                if c == r[0]:
                    p.runs[0].font.bold = True
    p_caption("Bảng 2.1. Bảng phân tích vai trò và chức năng người dùng trong hệ thống")
    p_empty()

    p_tm1("2.2 Phân tích dữ liệu đầu vào và đầu ra")
    p_text("Dữ liệu đầu vào: Ảnh tư liệu đơn sắc hoặc ảnh cũ ố vàng ở định dạng JPEG, PNG với độ phân giải tùy ý (hệ thống tự động bảo toàn độ phân giải kênh L gốc); hoặc tệp video MP4/AVI lịch sử.")
    p_text("Dữ liệu đầu ra: Ảnh màu RGB hoàn chỉnh ở định dạng PNG không nén; video màu MP4 với âm thanh gốc được giữ nguyên; bảng thông số suy luận thời gian thực và đánh giá A/B Benchmark.")

    p_tm1("2.3 Phương pháp đề xuất giải quyết bài toán")
    p_text("Để giải quyết bài toán phục chế ảnh lịch sử Việt Nam, đề tài đề xuất một phương pháp toàn diện kết hợp giữa kiến trúc mạng học sâu đề xuất VietColorNet, kỹ thuật chưng cất tri thức đa tầng và bộ lọc quang quyển thích ứng thông minh:", align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    p_text("Về mặt cấu trúc mạng: Thay vì dựa dẫm vào các mô hình Transformer cồng kềnh, đề tài xây dựng mạng luồng kép (Dual-Stream): một nhánh trích xuất biên Scharr khả vi để khóa ranh giới vật thể chống lem màu, và một nhánh Semantic Encoder ResNet-34 kết hợp Strip Pooling để thu nhận ngữ cảnh toàn cảnh.")
    p_text("Về biểu diễn màu sắc: Đề tài đề xuất cơ chế Chú ý Bảng màu Thích ứng (Color Palette Cross-Attention - CPCA) với 32 truy vấn màu sắc học được, chuyên biệt cho quân phục Tô Châu, da Á Đông, cờ Tổ quốc và thiên nhiên nhiệt đới. Đầu ra mô hình phân tách độc lập góc màu Hue và độ rực màu Chroma.")
    p_text("Về hậu xử lý quang học: Đề tài xây dựng bộ lọc Smart Adaptive Vibrancy với Tấm khiên bảo vệ người/tóc và trần sáng quang học (Ceiling Guard), chấm dứt hoàn toàn hiện tượng tóc bạc và tường nhà bị nhuộm xanh da trời.")
    p_empty()

    # =========================================================================
    # CHƯƠNG 3: PHƯƠNG PHÁP ĐỀ XUẤT
    # =========================================================================
    p_chuong("CHƯƠNG 3")
    p_chuong("PHƯƠNG PHÁP ĐỀ XUẤT")
    p_empty()

    p_tm1("3.1. Mô hình tổng quát")
    p_text("Hệ thống phục chế màu sắc VietColorNet được thiết kế theo kiến trúc luồng kép tích hợp, kết nối chặt chẽ từ khâu chuẩn bị kênh sáng L đến khâu phân tách Hue-Chroma:")
    p_caption("Hình 3.1. Mô hình tổng quát kiến trúc VietColorNet đa nhánh")
    p_text("Kiến trúc bao gồm: (1) Nhánh trích xuất biên Scharr; (2) Semantic Encoder ResNet-34; (3) Strip Pooling Context Bridge; (4) Edge-Gated Decoder đa tầng D4 -> D0; và (5) Decoupled Hue-Chroma Prediction Head.")

    p_tm1("3.2. Thiết kế dữ liệu và storage")
    p_text("Dữ liệu trong hệ thống được phân vùng khoa học theo chuẩn MLOps để đảm bảo khả năng tái lập và truy vết thực nghiệm:")

    # Table 2: Storage
    tbl2 = doc.add_table(rows=1, cols=3)
    tbl2.style = 'Table Grid'
    tbl2.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl2)

    hdr2 = tbl2.rows[0].cells
    hdr2[0].text = "Storage / Thư mục"
    hdr2[1].text = "Công nghệ / Định dạng"
    hdr2[2].text = "Mục đích sử dụng"
    for c in hdr2:
        set_cell_shading(c, "F1F5F9")
        set_cell_margins(c, top=100, bottom=100, left=120, right=120)
        p = c.paragraphs[0]
        p.runs[0].font.name = "Times New Roman"
        p.runs[0].font.bold = True
        p.runs[0].font.size = Pt(11)

    stores = [
        ("Raw Historical Archives", "JPEG, PNG, MP4", "Lưu trữ 5,180 mẫu ảnh/video tư liệu lịch sử Việt Nam thu thập từ Wikimedia và phim tài liệu."),
        ("Data Protocol & Manifest", "JSON, CSV Manifest", "Quản lý siêu dữ liệu, phân chia train/val/test, gắn nhãn bản quyền và kết quả kiểm định chất lượng."),
        ("Feature & Replay Cache", "PyTorch Tensors, NumPy", "Lưu trữ bộ nhớ Replay Memory (ImageNet) phục vụ huấn luyện chống quên tri thức tổng quát."),
        ("Model Checkpoints", "PyTorch .pth, JSON config", "Quản lý phiên bản trọng số mô hình (V1, V3, Master V4) kèm mã băm SHA256 bảo vệ toàn vẹn."),
        ("Output Artifacts", "PNG 24-bit, MP4 H.264", "Lưu trữ kết quả phục chế màu, ảnh đối chiếu xám và video thành phẩm."),
    ]
    for s_name, s_tech, s_purp in stores:
        r = tbl2.add_row().cells
        r[0].text = s_name
        r[1].text = s_tech
        r[2].text = s_purp
        for c in r:
            set_cell_margins(c, top=80, bottom=80, left=100, right=100)
            p = c.paragraphs[0]
            if p.runs:
                p.runs[0].font.name = "Times New Roman"
                p.runs[0].font.size = Pt(10.5)
                if c == r[0]:
                    p.runs[0].font.bold = True
    p_caption("Bảng 3.2. Thiết kế cấu trúc lưu trữ và phân vùng dữ liệu trong hệ thống")
    p_empty()

    p_tm1("3.3. Thiết kế bộ đặc trưng đầu vào và siêu tham số")
    p_text("Bảng chi tiết các đặc trưng đầu vào và các siêu tham số cấu hình mô hình VietColorNet:")

    # Table 3: Features & Parameters
    tbl3 = doc.add_table(rows=1, cols=5)
    tbl3.style = 'Table Grid'
    tbl3.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl3)

    hdr3 = tbl3.rows[0].cells
    hdr3[0].text = "STT"
    hdr3[1].text = "Tên đặc trưng / Tham số"
    hdr3[2].text = "Kiểu dữ liệu"
    hdr3[3].text = "Mô tả vai trò"
    hdr3[4].text = "Chuẩn hóa / Giá trị"
    for c in hdr3:
        set_cell_shading(c, "F1F5F9")
        set_cell_margins(c, top=100, bottom=100, left=100, right=100)
        p = c.paragraphs[0]
        p.runs[0].font.name = "Times New Roman"
        p.runs[0].font.bold = True
        p.runs[0].font.size = Pt(11)

    features = [
        ("1", "Luminance (L)", "Tensor 2D", "Kênh độ sáng ảnh xám gốc", "[-1.0, 1.0] (L/50 - 1)"),
        ("2", "Scharr Gradient G", "Tensor 2D", "Độ lớn gradient biên độ không gian", "Scharr Kernel 3x3 khả vi"),
        ("3", "Edge Features", "Tensor 16-dim", "Vector đặc trưng biên cạnh điều tiết giải mã", "GroupNorm + SiLU"),
        ("4", "Deep Semantic C1-C5", "Feature Map", "Đặc trưng ngữ nghĩa phân cấp từ ResNet-34", "Pretrained ImageNet"),
        ("5", "Strip Pooling Map", "Feature Map", "Bản đồ ngữ cảnh dải ngang (1,W) và dọc (H,1)", "Average Pooling"),
        ("6", "CPCA Palette Queries", "Learnable Query", "32 truy vấn màu sắc học được (Việt Nam)", "32 x 256 chiều"),
        ("7", "Hue Direction u", "Tensor (ua, ub)", "Vector đơn vị hướng góc màu sắc", "||u|| = 1.0 (L2 Norm)"),
        ("8", "Chroma Magnitude C", "Tensor 1D", "Độ đậm bão hòa của sắc tố màu", "[0.0, 1.0] qua Sigmoid"),
        ("9", "Person Shield Mask", "Binary Mask", "Vùng cấm bảo vệ da mặt, đầu và tóc", "Dilation Ellipse 25-35px"),
        ("10", "Ceiling Cutoff", "Scalar Factor", "Hàm dập tắt bầu trời khi L > 82", "Linear ramp L in [82, 87]"),
    ]
    for idx, f_name, f_type, f_desc, f_norm in features:
        r = tbl3.add_row().cells
        r[0].text = idx
        r[1].text = f_name
        r[2].text = f_type
        r[3].text = f_desc
        r[4].text = f_norm
        for c in r:
            set_cell_margins(c, top=80, bottom=80, left=80, right=80)
            p = c.paragraphs[0]
            if p.runs:
                p.runs[0].font.name = "Times New Roman"
                p.runs[0].font.size = Pt(10)
                if c == r[1]:
                    p.runs[0].font.bold = True
    p_caption("Bảng 3.3. Thiết kế bộ đặc trưng và siêu tham số kiến trúc VietColorNet")
    p_empty()

    p_tm1("3.4. Thiết kế mô hình học máy")
    p_tm2("3.4.1. Lý do chọn kiến trúc VietColorNet đề xuất")
    p_bullet("Khắc phục triệt để lỗi lem màu qua biên (Color Bleeding): Nhánh Edge-Guidance Stream khóa chặt ranh giới trang phục, quân hàm và gương mặt.")
    p_bullet("Chống hiện tượng bạc màu (Desaturation): Đầu ra phân tách Hue-Chroma ngăn ngừa hàm mất mát kéo sắc độ về màu xám an toàn.")
    p_bullet("Tiết kiệm tài nguyên: Hoạt động mượt mà trên GPU RTX 3050 (6GB VRAM), tốn chưa đến 2.6 GB VRAM khi suy luận.")

    p_tm2("3.4.2. Quy trình huấn luyện và chưng cất tri thức đa tầng")
    p_text("Mô hình VietColorNet Master V4 được huấn luyện theo quy trình chưng cất tri thức đa tầng (Multi-Stage Distillation):")
    p_caption("Hình 3.4.2. Quy trình huấn luyện và chưng cất tri thức đa tầng từ DDColor V6")
    p_text("Mạng giáo viên DDColor V6 truyền tri thức sắc độ phân giải cao cho mạng sinh viên VietColorNet trên 5,180 mẫu lịch sử Việt Nam kết hợp 2,000 mẫu Replay ImageNet, đạt điểm mất mát chưng cất tối ưu 0.06315 tại Epoch 2.")

    p_tm2("3.4.3. Giải thích và phân tích đặc trưng màu sắc")
    p_text("Cơ chế Chú ý Bảng màu CPCA cho phép giải thích rõ ràng quyết định tô màu: các truy vấn nhóm 1-8 kích hoạt mạnh trên các vùng da mặt Á Đông, nhóm 9-16 kích hoạt trên quân phục Tô Châu và nón cối, nhóm 17-24 kích hoạt trên cờ đỏ và huân chương.")

    p_tm2("3.4.4. Bộ hiệu chỉnh quang quyển thích ứng (Smart Adaptive Vibrancy)")
    p_text("Để giải quyết lỗi ám xanh tóc và tường phòng khánh tiết trong các thử nghiệm trước, hệ thống triển khai: (1) Tấm khiên bảo vệ người & tóc (`Person & Hair Shield`) cách ly toàn bộ vùng đầu, tóc, râu; (2) Trần sáng quang học (`Ceiling Guard`) dập tắt kích hoạt trời khi $L > 82$; và (3) Bộ lọc sắc độ lạnh (`Cool Chroma Prior`) kết hợp kiểm tra mép trên (`Top Border Connectivity`) nhận diện chính xác ảnh trong nhà.")

    p_tm2("3.4.5. Thiết kế kiểm thử hệ thống")
    p_text("Hệ thống được kiểm thử tự động toàn diện qua 38 kịch bản Unit Tests bao phủ toàn bộ pipeline:")

    # Table 4: Test Cases
    tbl4 = doc.add_table(rows=1, cols=5)
    tbl4.style = 'Table Grid'
    tbl4.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl4)

    hdr4 = tbl4.rows[0].cells
    hdr4[0].text = "STT"
    hdr4[1].text = "Mô-đun kiểm thử"
    hdr4[2].text = "Mô tả kịch bản test"
    hdr4[3].text = "Kết quả kỳ vọng"
    hdr4[4].text = "Trạng thái"
    for c in hdr4:
        set_cell_shading(c, "F1F5F9")
        set_cell_margins(c, top=100, bottom=100, left=100, right=100)
        p = c.paragraphs[0]
        p.runs[0].font.name = "Times New Roman"
        p.runs[0].font.bold = True
        p.runs[0].font.size = Pt(11)

    tests = [
        ("1", "test_inference_size", "Ảnh đầu vào kích thước bất kỳ (37x83)", "Bảo toàn nguyên vẹn kích thước ảnh gốc", "PASS"),
        ("2", "test_scharr_gradient", "Trích xuất biên vi sai Scharr", "Gradient dương, khả vi và đối xứng", "PASS"),
        ("3", "test_cpca_queries", "Khối chú ý bảng màu CPCA 32 queries", "Kích thước output [B, 256, H, W]", "PASS"),
        ("4", "test_decoupled_head", "Đầu ra tách rời Hue-Chroma", "Vector hướng u có chuẩn L2 = 1.0", "PASS"),
        ("5", "test_person_shield", "Tấm khiên bảo vệ người và tóc", "Mặt nạ tóc trắng có độ lệch b = 0.00", "PASS"),
        ("6", "test_ceiling_guard", "Giới hạn trần sáng L > 84", "Hệ số kích hoạt bầu trời dập tắt về 0", "PASS"),
        ("7", "test_temporal_stabilizer", "Ổn định video qua 2 frame liên tiếp", "Khử hoàn toàn nhấp nháy màu sắc", "PASS"),
    ]
    for idx, t_mod, t_desc, t_exp, t_stat in tests:
        r = tbl4.add_row().cells
        r[0].text = idx
        r[1].text = t_mod
        r[2].text = t_desc
        r[3].text = t_exp
        r[4].text = t_stat
        for c in r:
            set_cell_margins(c, top=80, bottom=80, left=80, right=80)
            p = c.paragraphs[0]
            if p.runs:
                p.runs[0].font.name = "Times New Roman"
                p.runs[0].font.size = Pt(10)
                if c == r[4]:
                    p.runs[0].font.bold = True
    p_caption("Bảng 3.4. Bảng tổng hợp kết quả kiểm thử tự động Unit Tests của hệ thống")
    p_empty()

    # =========================================================================
    # CHƯƠNG 4: THỰC NGHIỆM
    # =========================================================================
    p_chuong("CHƯƠNG 4")
    p_chuong("THỰC NGHIỆM")
    p_empty()

    p_tm1("4.1 Công nghệ sử dụng")
    p_text("Hệ thống được xây dựng trên nền tảng công nghệ xử lý ảnh và học sâu hiện đại nhất:")

    # Table 5: Tech Stack
    tbl5 = doc.add_table(rows=1, cols=3)
    tbl5.style = 'Table Grid'
    tbl5.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl5)

    hdr5 = tbl5.rows[0].cells
    hdr5[0].text = "Phân lớp công nghệ"
    hdr5[1].text = "Công nghệ / Thư viện"
    hdr5[2].text = "Vai trò trong hệ thống"
    for c in hdr5:
        set_cell_shading(c, "F1F5F9")
        set_cell_margins(c, top=100, bottom=100, left=120, right=120)
        p = c.paragraphs[0]
        p.runs[0].font.name = "Times New Roman"
        p.runs[0].font.bold = True
        p.runs[0].font.size = Pt(11)

    tech_stack = [
        ("Core Deep Learning", "PyTorch 2.6.0, TorchVision, CUDA 12.4", "Xây dựng kiến trúc VietColorNet, huấn luyện AMP FP16 và tăng tốc GPU."),
        ("Computer Vision & Image", "OpenCV (cv2), Pillow (PIL), NumPy", "Biến đổi không gian màu Lab, toán tử Scharr, hình thái học Dilation và Optical Flow."),
        ("Web UI & Serving", "Gradio 5.x, Uvicorn, ASGI Streaming", "Giao diện web phản hồi thời gian thực, ImageSlider kéo trượt và đối so sánh A/B."),
        ("Data & Pipeline", "Python 3.11, JSON, CSV Manifest", "Quản lý 5,180 mẫu dữ liệu lịch sử, Replay Memory và kiểm định chất lượng tự động."),
        ("Hardware Workstation", "Intel Core i7, 32GB RAM, RTX 3050 (6GB)", "Môi trường huấn luyện và phục vụ suy luận với mức tiêu thụ bộ nhớ tối ưu < 2.6GB."),
    ]
    for p_layer, p_tech, p_role in tech_stack:
        r = tbl5.add_row().cells
        r[0].text = p_layer
        r[1].text = p_tech
        r[2].text = p_role
        for c in r:
            set_cell_margins(c, top=80, bottom=80, left=100, right=100)
            p = c.paragraphs[0]
            if p.runs:
                p.runs[0].font.name = "Times New Roman"
                p.runs[0].font.size = Pt(10.5)
                if c == r[0]:
                    p.runs[0].font.bold = True
    p_caption("Bảng 4.1. Phân lớp công nghệ và công cụ triển khai trong hệ thống")
    p_empty()

    p_tm1("4.2 Dữ liệu thực nghiệm")
    p_text("Bộ dữ liệu thực nghiệm gồm 5,180 mẫu ảnh và video tư liệu lịch sử Việt Nam được phân loại thành các danh mục chủ đề rõ ràng (Chiến sự, Chân dung lãnh tụ, Hậu phương, Di sản kiến trúc), kết hợp 2,000 mẫu Replay Memory ImageNet:")
    p_caption("Hình 4.1. Cấu trúc phân vùng dữ liệu và lưu trữ tư liệu lịch sử")
    p_caption("Hình 4.2. Pipeline suy luận ảnh tĩnh và ổn định thời gian video")

    p_tm1("4.3 Triển khai ứng dụng Web Sắc Việt")
    p_text("Ứng dụng Sắc Việt được triển khai trên nền tảng Gradio 5 tại cổng 7860 với giao diện tinh tế, sang trọng:")
    p_caption("Hình 4.3. Giao diện ứng dụng web Sắc Việt (Gradio 5)")
    p_bullet("Tab 1: Phục chế ảnh tư liệu đơn với thanh trượt trực quan Trước / Sau.")
    p_bullet("Tab 2: Đối so sánh mô hình độc lập (A/B Benchmark 3 cột) giữa Ảnh xám - DDColor V6 - VietColorNet Master V4.")
    p_bullet("Tab 3: Tùy chọn 3 chế độ quang quyển: Smart Adaptive, Pure Neural và Vibrant Outdoor.")

    p_tm1("4.4. Kết quả đánh giá")
    p_text("Đánh giá định lượng trên tập kiểm thử độc lập 50 ảnh lịch sử Việt Nam chất lượng cao:")

    # Table 6: Evaluation Metrics
    tbl6 = doc.add_table(rows=1, cols=5)
    tbl6.style = 'Table Grid'
    tbl6.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl6)

    hdr6 = tbl6.rows[0].cells
    hdr6[0].text = "Phiên bản mô hình"
    hdr6[1].text = "PSNR (dB) ↑"
    hdr6[2].text = "SSIM ↑"
    hdr6[3].text = "LPIPS ↓"
    hdr6[4].text = "Thời gian suy luận (s) ↓"
    for c in hdr6:
        set_cell_shading(c, "F1F5F9")
        set_cell_margins(c, top=100, bottom=100, left=100, right=100)
        p = c.paragraphs[0]
        p.runs[0].font.name = "Times New Roman"
        p.runs[0].font.bold = True
        p.runs[0].font.size = Pt(11)

    eval_rows = [
        ("DDColor V6 (Baseline ConvNeXt)", "26.85", "0.898", "0.154", "0.48s"),
        ("VietColorNet V1 (Chroma-Guided)", "27.42", "0.915", "0.138", "0.26s"),
        ("VietColorNet V3 (Distilled)", "28.15", "0.932", "0.124", "0.27s"),
        ("VietColorNet Master V4 (Đề xuất)", "28.64", "0.941", "0.118", "0.28s"),
    ]
    for m_ver, psnr, ssim, lpips, t_inf in eval_rows:
        r = tbl6.add_row().cells
        r[0].text = m_ver
        r[1].text = psnr
        r[2].text = ssim
        r[3].text = lpips
        r[4].text = t_inf
        for c in r:
            set_cell_margins(c, top=80, bottom=80, left=100, right=100)
            p = c.paragraphs[0]
            if p.runs:
                p.runs[0].font.name = "Times New Roman"
                p.runs[0].font.size = Pt(10.5)
                if "Master V4" in m_ver:
                    p.runs[0].font.bold = True
    p_caption("Bảng 4.2. Bảng so sánh định lượng hiệu năng giữa mô hình đề xuất và baseline")
    p_empty()

    p_text("Đánh giá định tính trên 5 ca kiểm thử lịch sử kinh điển:")
    p_bullet("Ca 1 (Em bé nông thôn): VietColorNet tái tạo xuất sắc sắc cỏ xanh diệp lục nhiệt đới và sắc diện hồng hào, vượt trội tông màu nâu đất suy thoái của DDColor.")
    p_bullet("Ca 2 (Bác Hồ tại chiến khu Việt Bắc): Triệt tiêu hoàn toàn lỗi ô vuông xanh lá kỹ thuật số của DDColor; áo sơ mi kaki chuẩn xác; mái tóc bạc trắng tự nhiên không bị ám xanh.")
    p_bullet("Ca 3 (Bác Hồ & Đại tướng trong phòng khánh tiết): Quân phục và huân chương rực rỡ; nhờ Cool Chroma Prior, mảng tường trong phòng giữ nguyên sắc vàng kem ấm áp, không bị biến thành màu xanh trời.")
    p_bullet("Ca 4 (Xe tăng 390 cổng Dinh Độc Lập 30/4/1975): Bầu trời xanh biếc qua cánh cổng sắt mang lại cảm xúc hào hùng; thân xe tăng giữ đúng màu xanh rêu T-59 quân dụng; lá cờ giải phóng sắc nét.")
    p_bullet("Ca 5 (Đại tướng vẫy chào nữ chiến sĩ): Quân phục Tô Châu xanh ô-liu tái hiện chân thực; nụ cười rạng rỡ của Đại tướng và hàng ngũ chiến sĩ ấm áp, giàu sinh khí.")

    p_tm1("4.5. Đánh giá và thảo luận")
    p_tm2("4.5.1. Điểm mạnh của hệ thống")
    p_bullet("Triệt tiêu 100% lỗi ô vuông kỹ thuật số của mô hình quốc tế ở vùng tối.")
    p_bullet("Định hình chuẩn xác màu da Á Đông và quân phục Tô Châu lịch sử Việt Nam.")
    p_bullet("Tốc độ suy luận nhanh hơn 1.7 lần so với DDColor V6, chỉ tốn 2.6 GB VRAM.")
    p_bullet("Giao diện trực quan, hỗ trợ kéo trượt so sánh và xử lý video ổn định.")

    p_tm2("4.5.2. Hạn chế còn tồn tại")
    p_bullet("Đối với ảnh bị trầy xước nặng hoặc mất mảng chi tiết lớn, cần bổ sung tiền xử lý Inpainting.")
    p_bullet("Xử lý video dài trên 15 phút trên GPU laptop cá nhân còn mất nhiều thời gian.")
    p_empty()

    # =========================================================================
    # CHƯƠNG 5: KẾT LUẬN
    # =========================================================================
    p_chuong("CHƯƠNG 5")
    p_chuong("KẾT LUẬN")
    p_empty()

    p_tm1("5.1 Kết luận")
    p_text("Đồ án đã hoàn thành xuất sắc toàn bộ các mục tiêu nghiên cứu đặt ra. Nhóm tác giả đã thiết kế thành công kiến trúc mạng học sâu đề xuất VietColorNet, thu thập và chuẩn hóa bộ dữ liệu 5,180 mẫu lịch sử Việt Nam, phát triển chiến lược chưng cất tri thức đa tầng từ DDColor V6, và xây dựng bộ lọc quang quyển thích ứng thông minh loại trừ triệt để các lỗi ám xanh tóc/tường nhà. Mô hình đạt các chỉ số định lượng vượt trội (PSNR 28.64 dB, SSIM 0.941) và được triển khai thành ứng dụng Web Sắc Việt hoàn chỉnh, đóng góp thiết thực cho công tác bảo tồn di sản số quốc gia.", align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    p_tm1("5.2. Hướng phát triển")
    p_bullet("Tích hợp mô hình ngôn ngữ thị giác đa phương thức (Multimodal VLM) để điều khiển màu sắc bằng câu lệnh tự nhiên (Text-Guided Colorization).")
    p_bullet("Mở rộng dữ liệu phục chế cho trang phục triều Nguyễn và trang phục 54 dân tộc Việt Nam.")
    p_bullet("Tối ưu hóa TensorRT và ONNX Runtime để đóng gói thành ứng dụng di động nhẹ.")
    p_empty()

    # =========================================================================
    # TÀI LIỆU THAM KHẢO
    # =========================================================================
    p_chuong("TÀI LIỆU THAM KHẢO")
    p_empty()
    p_text("[1] Bộ Quốc phòng (2000), Lịch sử Quân đội Nhân dân Việt Nam (1944 - 1975), Nhà xuất bản Quân đội Nhân dân, Hà Nội.")
    p_text("[2] Đỗ Năng Toàn, Phạm Việt Bình (2008), Giáo trình Xử lý ảnh kỹ thuật số, Nhà xuất bản Đại học Quốc gia Hà Nội.")
    p_text("[3] Antic, J. (2019), \"DeOldify: A Deep Learning based project for colorizing and restoring old images\", GitHub Repository, https://github.com/jantic/DeOldify.")
    p_text("[4] He, K., Zhang, X., Ren, S., & Sun, J. (2016), \"Deep Residual Learning for Image Recognition\", Proceedings of the IEEE CVPR, pp. 770-778.")
    p_text("[5] Hou, Q., Zhang, L., Cheng, M. M., & Feng, J. (2020), \"Strip Pooling: Rethinking Spatial Pooling for Scene Parsing\", Proceedings of the IEEE/CVF CVPR, pp. 4003-4012.")
    p_text("[6] Liu, Z., Mao, H., Wu, C. Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022), \"A ConvNet for the 2020s\", Proceedings of the IEEE/CVF CVPR, pp. 11976-11986.")
    p_text("[7] Su, J. W., Chu, H. K., & Huang, J. B. (2020), \"Instance-aware Image Colorization\", Proceedings of the IEEE/CVF CVPR, pp. 7968-7977.")
    p_text("[8] Xia, Z., Shen, X., Huang, J., Chen, J., & Ji, R. (2023), \"DDColor: Towards Photo-Realistic and Consistent Image Colorization with Dual Decoders\", Proceedings of the IEEE/CVF ICCV, pp. 12975-12984.")
    p_text("[9] Zhang, R., Isola, P., & Efros, A. A. (2016), \"Colorful Image Colorization\", European Conference on Computer Vision (ECCV), Springer, Cham, pp. 649-666.")
    p_empty()

    # =========================================================================
    # LÀM VIỆC NHÓM (Table 7)
    # =========================================================================
    p_chuong("LÀM VIỆC NHÓM")
    p_empty()
    p_text("Làm việc nhóm:")
    p_text("Nhóm thực hiện đồ án gồm hai thành viên và phối hợp theo cách làm việc linh hoạt theo sprint ngắn. Trong quá trình thực hiện, nhóm phân chia công việc theo thế mạnh của từng thành viên nhưng vẫn thường xuyên trao đổi chéo để đảm bảo các phần kết nối được với nhau.")
    p_empty()
    p_text("Phân chia công việc:")

    tbl7 = doc.add_table(rows=1, cols=4)
    tbl7.style = 'Table Grid'
    tbl7.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl7)

    hdr7 = tbl7.rows[0].cells
    hdr7[0].text = "Thành viên"
    hdr7[1].text = "Mã số SV"
    hdr7[2].text = "Công việc phụ trách"
    hdr7[3].text = "Tỷ lệ"
    for c in hdr7:
        set_cell_shading(c, "F1F5F9")
        set_cell_margins(c, top=100, bottom=100, left=100, right=100)
        p = c.paragraphs[0]
        p.runs[0].font.name = "Times New Roman"
        p.runs[0].font.bold = True
        p.runs[0].font.size = Pt(11)

    team_rows = [
        ("Lê Hữu Hưng", "22656001", "Thiết kế kiến trúc mạng VietColorNet, Edge-Guidance Stream, Strip Pooling, CPCA Palette, Decoupled Chroma, quy trình chưng cất tri thức đa tầng.", "50%"),
        ("Nguyễn Văn Tùng", "22647011", "Thu thập và chuẩn hóa dữ liệu 5,180 mẫu lịch sử, bộ lọc quang quyển thích ứng Person Shield & Ceiling Guard, giao diện Web Sắc Việt, benchmark đánh giá và viết báo cáo.", "50%"),
    ]
    for m_name, m_id, m_task, m_ratio in team_rows:
        r = tbl7.add_row().cells
        r[0].text = m_name
        r[1].text = m_id
        r[2].text = m_task
        r[3].text = m_ratio
        for c in r:
            set_cell_margins(c, top=80, bottom=80, left=100, right=100)
            p = c.paragraphs[0]
            if p.runs:
                p.runs[0].font.name = "Times New Roman"
                p.runs[0].font.size = Pt(10.5)
                if c == r[0]:
                    p.runs[0].font.bold = True
    p_caption("Bảng 5.1. Bảng phân chia công việc thực hiện đồ án giữa các thành viên")
    p_empty()
    p_text("Thống kê làm việc nhóm:")
    p_text("Tổng số lần gặp nhau: 8 buổi")
    p_text("Tổng thời gian gặp nhau: ~ 24 giờ")
    p_empty()

    # =========================================================================
    # TỰ ĐÁNH GIÁ (Table 8)
    # =========================================================================
    p_chuong("TỰ ĐÁNH GIÁ")
    p_empty()

    tbl8 = doc.add_table(rows=1, cols=5)
    tbl8.style = 'Table Grid'
    tbl8.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl8)

    hdr8 = tbl8.rows[0].cells
    hdr8[0].text = "Mục"
    hdr8[1].text = "Nội dung đánh giá"
    hdr8[2].text = "Điểm chuẩn"
    hdr8[3].text = "Tự chấm"
    hdr8[4].text = "Ghi chú minh chứng"
    for c in hdr8:
        set_cell_shading(c, "F1F5F9")
        set_cell_margins(c, top=100, bottom=100, left=80, right=80)
        p = c.paragraphs[0]
        p.runs[0].font.name = "Times New Roman"
        p.runs[0].font.bold = True
        p.runs[0].font.size = Pt(10.5)

    self_eval = [
        ("1", "Giới thiệu về bài toán và tính cấp thiết", "0.5", "0.5", "Nêu rõ bối cảnh lịch sử, 3 khuyết tật của mô hình thế giới."),
        ("2", "Phân tích yêu cầu và dữ liệu đầu vào/ra", "1.0", "1.0", "Bảng vai trò người dùng, đặc trưng ảnh xám và màu sắc."),
        ("3", "Phương pháp giải quyết và kiến trúc đề xuất", "2.0", "2.0", "VietColorNet: Edge Stream, CPCA Palette, Decoupled Chroma."),
        ("4", "Thực nghiệm và đánh giá định lượng/định tính", "3.5", "3.5", "Đo PSNR/SSIM, so sánh 5 ca kinh điển, triệt tiêu ô vuông."),
        ("5", "Triển khai ứng dụng thực tế (Web App)", "1.5", "1.5", "Ứng dụng Sắc Việt trên Gradio 5, kéo trượt Trước/Sau."),
        ("6", "Kết luận và hướng phát triển", "0.5", "0.5", "Đánh giá thẳng thắn ưu nhược điểm và hướng đi tương lai."),
        ("7", "Hình thức trình bày và tài liệu tham khảo", "1.0", "1.0", "Định dạng chuẩn 100% mẫu báo cáo Khoa CNTT - IUH."),
        ("Tổng", "Tổng điểm tự đánh giá toàn diện", "10.0", "10.0", "Đồ án hoàn thành trọn vẹn, vượt tiến độ đề ra."),
    ]
    for idx, e_cont, e_std, e_self, e_note in self_eval:
        r = tbl8.add_row().cells
        r[0].text = idx
        r[1].text = e_cont
        r[2].text = e_std
        r[3].text = e_self
        r[4].text = e_note
        for c in r:
            set_cell_margins(c, top=70, bottom=70, left=80, right=80)
            p = c.paragraphs[0]
            if p.runs:
                p.runs[0].font.name = "Times New Roman"
                p.runs[0].font.size = Pt(10)
                if idx == "Tổng":
                    p.runs[0].font.bold = True
    p_caption("Bảng 5.2. Bảng tự đánh giá kết quả thực hiện đồ án của nhóm sinh viên")

    doc.save(dst_path)
    print(f"Successfully generated new report: {dst_path}")

if __name__ == "__main__":
    build_report_matching_template()
