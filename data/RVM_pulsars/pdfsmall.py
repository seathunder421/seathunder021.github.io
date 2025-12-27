import os
import fitz  # PyMuPDF
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def compress_pdf_raster(input_path, dpi=200):
    """PDF 光栅化压缩，输出覆盖原文件但保证不增大"""
    doc = fitz.open(input_path)
    new_pdf = fitz.open()
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        rect = fitz.Rect(0, 0, pix.width, pix.height)
        new_page = new_pdf.new_page(width=pix.width, height=pix.height)
        new_page.insert_image(rect, pixmap=pix)
    
    # 保存到临时文件
    tmp_path = input_path + ".tmp"
    new_pdf.save(tmp_path, deflate=True)
    doc.close()
    new_pdf.close()
    
    # 比较大小，如果更小则覆盖原文件，否则删除临时文件
    orig_size = os.path.getsize(input_path)
    new_size = os.path.getsize(tmp_path)
    if new_size < orig_size:
        os.replace(tmp_path, input_path)
        return input_path, orig_size, new_size
    else:
        os.remove(tmp_path)
        return input_path, orig_size, orig_size

def find_pdfs(folder):
    pdfs = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".pdf"):
                pdfs.append(os.path.join(root, f))
    return pdfs

def compress_worker(pdf_dpi_tuple):
    pdf, dpi = pdf_dpi_tuple
    try:
        return compress_pdf_raster(pdf, dpi)
    except Exception as e:
        return pdf, 0, 0, str(e)

if __name__ == "__main__":
    folder = "."
    DPI = 200
    pdf_files = find_pdfs(folder)
    print(f"✅ 共找到 {len(pdf_files)} 个 PDF 文件，开始压缩...\n")

    total_orig = 0
    total_new = 0

    # 多进程压缩
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(compress_pdf_raster, pdf, DPI) for pdf in pdf_files]
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if len(result) == 3:
                pdf, orig_size, new_size = result
                if new_size < orig_size:
                    print(f"{os.path.basename(pdf)}\n  原始: {orig_size/1024/1024:.2f} MB → 压缩后: {new_size/1024/1024:.2f} MB ({new_size/orig_size*100:.1f}%)")
                else:
                    print(f"{os.path.basename(pdf)}\n  压缩后的文件比原文件大或相等，保持原文件 ({orig_size/1024/1024:.2f} MB)")
                total_orig += orig_size / 1024 / 1024
                total_new += min(new_size, orig_size) / 1024 / 1024

    print("\n📊 总计压缩结果：")
    print(f"  原始体积: {total_orig:.2f} MB")
    print(f"  压缩后:   {total_new:.2f} MB")
    if total_orig > 0:
        print(f"  压缩率:   {total_new/total_orig*100:.1f}%")
