"""
batch_pdf_extractor.py

Script xử lý hàng loạt PDF:
- Trích xuất văn bản (Markdown, giữ cấu trúc cột, bảng)
- Tách ảnh ra thư mục images/
- Phân tích ảnh bằng Gemini (mô tả + OCR text)
- Chèn ảnh + mô tả + OCR trực tiếp vào file .md

CÁCH CHẠY TRONG VS CODE:
- Mở file này trong VS Code
- Nhấn ▶️ Run Python File hoặc Ctrl+F5
"""

import base64
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Optional, Tuple

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser
from openai import OpenAI
from tqdm import tqdm

# ============================================================
# CẤU HÌNH
# ============================================================

# Gemini API key is read from GOOGLE_API_KEY in the environment.
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

# 📁 Thư mục input / output
INPUT_DIR = r"D:\fall26\DSP_PROJECT\data_unprocessing"
OUTPUT_DIR = r"D:\fall26\DSP_PROJECT\data_processed"

# 🤖 Có dùng VLM để phân tích ảnh không?
USE_VLM = True

# Model Gemini (từ danh sách bạn liệt kê)
VLM_MODEL = "gemini-3.5-flash"

# Endpoint OpenAI-compatible của Gemini
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Kích thước ảnh tối thiểu (pixel) để đưa vào VLM
MIN_IMAGE_SIZE = 150

# Bỏ qua file đã xử lý
SKIP_IF_EXISTS = False

# Retry khi gặp rate limit
MAX_RETRIES = 3
RETRY_DELAY = 5  # giây

# ============================================================
# PHẦN CODE
# ============================================================


class PDFBatchExtractor:
    """Lớp xử lý hàng loạt PDF."""

    def __init__(self, input_dir: str, output_dir: str, use_vlm: bool = True):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.use_vlm = use_vlm

        if not self.input_dir.exists():
            raise FileNotFoundError(f"❌ Thư mục input không tồn tại: {self.input_dir}")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Khởi tạo Gemini client
        self.client = None
        if self.use_vlm:
            if not GEMINI_API_KEY or "xxxx" in GEMINI_API_KEY or not GEMINI_API_KEY.startswith("AIza"):
                print("⚠️  GEMINI_API_KEY chưa được đặt hoặc không hợp lệ → tắt VLM.")
                self.use_vlm = False
            else:
                self.client = OpenAI(
                    api_key=GEMINI_API_KEY,
                    base_url=GEMINI_BASE_URL,
                )
                print(f"✅ Đã khởi tạo Gemini client (model: {VLM_MODEL}).")
        else:
            print("ℹ️  Chế độ không dùng VLM: chỉ trích xuất văn bản và ảnh.")

        # Khởi tạo Marker converter (đã tắt OCR)
        print("🔧 Đang khởi tạo Marker converter (đã tắt OCR)...")
        try:
            config_parser = ConfigParser({
                'disable_ocr': True,
                'output_format': 'markdown',
            })
            self.converter = PdfConverter(
                config=config_parser.generate_config_dict(),
                artifact_dict=create_model_dict(),
                processor_list=config_parser.get_processors(),
                renderer=config_parser.get_renderer(),
                llm_service=config_parser.get_llm_service(),
            )
            print("✅ Marker converter đã sẵn sàng.\n")
        except Exception as e:
            print(f"❌ Lỗi khởi tạo Marker: {e}")
            traceback.print_exc()
            sys.exit(1)

    # ----------------------------------------------------------
    # Vòng lặp chính
    # ----------------------------------------------------------
    def process_all(self):
        pdf_files = sorted(
            list(self.input_dir.glob("*.pdf")) + list(self.input_dir.glob("*.PDF"))
        )

        # Loại bỏ trùng (Windows không phân biệt hoa/thường)
        seen, unique = set(), []
        for f in pdf_files:
            k = str(f).lower()
            if k not in seen:
                seen.add(k)
                unique.append(f)
        pdf_files = unique

        if not pdf_files:
            print(f"⚠️  Không tìm thấy file PDF nào trong: {self.input_dir}")
            return

        print(f"📚 Tìm thấy {len(pdf_files)} file PDF.\n")

        ok = skip = fail = 0
        for pdf_path in tqdm(pdf_files, desc="Đang xử lý", unit="file"):
            try:
                r = self._process_single_pdf(pdf_path)
                if r == "skipped":
                    skip += 1
                else:
                    ok += 1
            except Exception as e:
                fail += 1
                print(f"\n❌ Lỗi xử lý {pdf_path.name}: {e}")
                traceback.print_exc()

        print(f"\n{'='*60}")
        print(f"🎉 HOÀN TẤT")
        print(f"   ✅ Thành công : {ok}")
        print(f"   ⏭️  Bỏ qua    : {skip}")
        print(f"   ❌ Thất bại   : {fail}")
        print(f"📂 Kết quả lưu tại: {self.output_dir}")
        print(f"{'='*60}")

    # ----------------------------------------------------------
    # Xử lý 1 PDF
    # ----------------------------------------------------------
    def _process_single_pdf(self, pdf_path: Path):
        pdf_name = pdf_path.stem
        file_output_dir = self.output_dir / pdf_name
        md_file_path = file_output_dir / f"{pdf_name}.md"

        # Skip nếu đã có .md
        if SKIP_IF_EXISTS and md_file_path.exists():
            print(f"⏭️  Bỏ qua {pdf_path.name} (đã có .md)")
            return "skipped"

        file_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n📄 [{pdf_path.name}]")

        # --- Bước 1: Chuyển PDF sang Markdown ---
        rendered = self.converter(str(pdf_path))
        markdown_text, metadata, images = text_from_rendered(rendered)

        # --- Bước 2: Lưu ảnh + phân tích bằng Gemini ---
        # Key: tên gốc của ảnh mà marker trả về (dùng để map trong markdown)
        # Value: {safe_filename, description, ocr}
        image_analyses = {}

        if images:
            images_dir = file_output_dir / "images"
            images_dir.mkdir(exist_ok=True)

            for img_filename, img_obj in images.items():
                safe_filename = self._sanitize_filename(img_filename)
                img_path = images_dir / safe_filename

                try:
                    # Chuẩn hóa định dạng
                    if img_path.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
                        img_path = img_path.with_suffix(".png")
                    img_format = "PNG"
                    if img_path.suffix.lower() in [".jpg", ".jpeg"]:
                        img_format = "JPEG"

                    # Lưu ảnh
                    if hasattr(img_obj, "save"):
                        img_obj.save(img_path, format=img_format)
                    else:
                        with open(img_path, "wb") as f:
                            f.write(img_obj)

                    # Phân tích bằng Gemini
                    desc, ocr = "", ""
                    if self.use_vlm and self.client and self._is_image_big_enough(img_path):
                        desc, ocr = self._analyze_image_with_vlm(img_path, pdf_name)
                        if desc or ocr:
                            print(f"      🖼️  Đã phân tích: {safe_filename}")

                    image_analyses[img_filename] = {
                        "safe_filename": safe_filename,
                        "description": desc,
                        "ocr": ocr,
                    }

                except Exception as e:
                    print(f"      ⚠️  Lỗi ảnh {img_filename}: {e}")

            print(f"   ✅ Đã lưu {len(images)} ảnh vào images/")

        # --- Bước 3: Chèn ảnh + mô tả + OCR vào markdown ---
        markdown_final = self._inject_image_info(markdown_text, image_analyses)

        # --- Bước 4: Lưu markdown ---
        with open(md_file_path, "w", encoding="utf-8") as f:
            f.write(markdown_final)
        print(f"   ✅ Markdown: {md_file_path.name}")

        # --- Bước 5: Lưu JSON phân tích ảnh (tùy chọn, để tiện tra cứu) ---
        if image_analyses:
            desc_file = file_output_dir / f"{pdf_name}_image_analyses.json"
            with open(desc_file, "w", encoding="utf-8") as f:
                json.dump(image_analyses, f, ensure_ascii=False, indent=2)

        # --- Bước 6: Lưu metadata ---
        if metadata:
            meta_file = file_output_dir / f"{pdf_name}_metadata.json"
            try:
                meta_dict = metadata if isinstance(metadata, dict) else metadata.__dict__
                with open(meta_file, "w", encoding="utf-8") as f:
                    json.dump(meta_dict, f, ensure_ascii=False, indent=2, default=str)
            except Exception:
                pass

        return "success"

    # ----------------------------------------------------------
    # Chèn thông tin ảnh vào markdown
    # ----------------------------------------------------------
    def _inject_image_info(self, markdown_text: str, image_analyses: dict) -> str:
        """
        Tìm mỗi dòng ảnh trong markdown, thay bằng block:
            ![alt](images/xxx.png)

            > 📝 Mô tả: ...
            >
            > 🔍 OCR: ...
        """
        if not image_analyses:
            return markdown_text

        # Lookup: cho phép match nhiều kiểu tên
        lookup = {}
        for orig_name, info in image_analyses.items():
            lookup[orig_name] = info
            lookup[Path(orig_name).name] = info
            lookup[info["safe_filename"]] = info

        def replace(match: re.Match) -> str:
            alt = match.group(1)
            path = match.group(2).strip()

            info = lookup.get(path) or lookup.get(Path(path).name)
            if not info:
                # Thử match một phần
                for k, v in lookup.items():
                    if k and (k in path or path in k):
                        info = v
                        break
            if not info:
                return match.group(0)  # giữ nguyên

            # Đường dẫn ảnh mới: images/xxx (tương đối từ file .md)
            new_path = f"images/{info['safe_filename']}"
            out = f"![{alt}]({new_path})"

            # Chèn mô tả dạng blockquote
            if info.get("description"):
                desc_text = info["description"].strip()
                quoted = "\n".join(
                    ("> " + line) if line.strip() else ">"
                    for line in desc_text.split("\n")
                )
                out += f"\n\n> **📝 Mô tả hình ảnh:**\n>\n{quoted}"

            # Chèn OCR dạng blockquote
            if info.get("ocr"):
                ocr_text = info["ocr"].strip()
                quoted = "\n".join(
                    ("> " + line) if line.strip() else ">"
                    for line in ocr_text.split("\n")
                )
                out += f"\n\n> **🔍 OCR Text:**\n>\n{quoted}"

            # Dấu phân cách với nội dung tiếp theo
            out += "\n"

            return out

        pattern = r"!\[([^\]]*)\]\(([^)]+)\)"
        return re.sub(pattern, replace, markdown_text)

    # ----------------------------------------------------------
    # Gọi Gemini 1 lần: mô tả + OCR
    # ----------------------------------------------------------
    def _analyze_image_with_vlm(self, image_path: Path, pdf_name: str) -> Tuple[str, str]:
        """Trả về (description, ocr_text)."""
        try:
            with open(image_path, "rb") as f:
                image_data = f.read()
            base64_image = base64.b64encode(image_data).decode("utf-8")

            suffix = image_path.suffix.lower()
            mime_type = "image/jpeg" if suffix in [".jpg", ".jpeg"] else "image/png"

            prompt = (
                f"Phân tích hình ảnh này (trích từ tài liệu '{pdf_name}') và trả lời "
                f"ĐÚNG theo format sau, KHÔNG thêm gì khác:\n\n"
                f"===MÔ TẢ===\n"
                f"[Mô tả chi tiết nội dung ảnh. Nếu là sơ đồ/pipeline/flowchart: "
                f"mô tả các thành phần, mũi tên, luồng dữ liệu, mối quan hệ giữa các phần. "
                f"Nếu là hình thường: mô tả nội dung chính.]\n\n"
                f"===OCR===\n"
                f"[Trích xuất TOÀN BỘ văn bản có trong ảnh. Giữ nguyên ngôn ngữ gốc. "
                f"Nếu có bảng, trình bày lại dạng Markdown table. "
                f"Nếu có công thức toán, dùng LaTeX. "
                f"Nếu ảnh KHÔNG có text nào, ghi đúng: KHONG_CO_TEXT]"
            )

            text = ""
            for attempt in range(MAX_RETRIES):
                try:
                    response = self.client.chat.completions.create(
                        model=VLM_MODEL,
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{base64_image}",
                                    },
                                },
                            ],
                        }],
                        max_tokens=2000,
                        temperature=0.2,
                    )
                    text = response.choices[0].message.content or ""
                    break
                except Exception as e:
                    err = str(e)
                    if "429" in err or "quota" in err.lower():
                        wait = RETRY_DELAY * (attempt + 1) * 2
                        print(f"      ⏳ Rate limit, đợi {wait}s...")
                        time.sleep(wait)
                    elif attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY)
                    else:
                        print(f"      ⚠️  Lỗi VLM: {err[:150]}")
                        return "", ""

            # Parse
            desc, ocr = "", ""
            if "===OCR===" in text:
                parts = text.split("===OCR===")
                desc = parts[0].replace("===MÔ TẢ===", "").strip()
                ocr = parts[1].strip() if len(parts) > 1 else ""
            elif "===MÔ TẢ===" in text:
                desc = text.replace("===MÔ TẢ===", "").strip()
            else:
                desc = text.strip()

            # Nếu Gemini báo không có text → để trống
            if "KHONG_CO_TEXT" in ocr:
                ocr = ""

            return desc, ocr

        except Exception as e:
            print(f"      ⚠️  Lỗi phân tích ảnh {image_path.name}: {e}")
            return "", ""

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------
    @staticmethod
    def _is_image_big_enough(img_path: Path) -> bool:
        try:
            from PIL import Image
            with Image.open(img_path) as img:
                w, h = img.size
                return w >= MIN_IMAGE_SIZE and h >= MIN_IMAGE_SIZE
        except Exception:
            return True

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        sanitized = re.sub(r"[^\w\-_.]", "_", filename)
        if len(sanitized) > 200:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[:190] + ext
        return sanitized


def main():
    print("=" * 60)
    print("🚀 BATCH PDF EXTRACTOR (OCR OFF + GEMINI VLM)")
    print("=" * 60)
    print(f"📥 Input : {INPUT_DIR}")
    print(f"📤 Output: {OUTPUT_DIR}")
    print(f"🤖 VLM   : {'BẬT (' + VLM_MODEL + ')' if USE_VLM else 'TẮT'}")
    print(f"⏭️  Skip  : {'BẬT' if SKIP_IF_EXISTS else 'TẮT'}")
    print("=" * 60 + "\n")

    extractor = PDFBatchExtractor(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        use_vlm=USE_VLM,
    )
    extractor.process_all()


if __name__ == "__main__":
    main()