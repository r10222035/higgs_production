import os
import re
import zlib
import numpy as np
import pandas as pd

def parse_pdf(pdf_path):
    with open(pdf_path, 'rb') as f:
        content = f.read()
    
    streams = []
    stream_matches = list(re.finditer(b'stream\r?\n(.*?)\r?\nendstream', content, re.DOTALL))
    for match in stream_matches:
        stream_data = match.group(1)
        start_pos = match.start()
        dict_start = content.rfind(b'<<', 0, start_pos)
        dict_end = content.find(b'>>', dict_start, start_pos)
        dict_data = content[dict_start:dict_end+2]
        if b'FlateDecode' in dict_data:
            try:
                decompressed = zlib.decompress(stream_data)
                streams.append(decompressed.decode('ascii', errors='ignore'))
            except:
                pass
        else:
            streams.append(stream_data.decode('ascii', errors='ignore'))
            
    full_text = "\n".join(streams)
    
    # 提取文字刻度與其 PDF 坐標
    texts = []
    tm_matches = re.finditer(r'1\s+0\s+0\s+1\s+([\d\.-]+)\s+([\d\.-]+)\s+Tm\s*\r?\n?\s*\((.*?)\)\s*Tj', full_text)
    for m in tm_matches:
        x, y, txt = float(m.group(1)), float(m.group(2)), m.group(3).strip()
        texts.append((x, y, txt))
        
    td_matches = re.finditer(r'([\d\.-]+)\s+([\d\.-]+)\s+Td\s*\r?\n?\s*\((.*?)\)\s*Tj', full_text)
    for m in td_matches:
        x, y, txt = float(m.group(1)), float(m.group(2)), m.group(3).strip()
        texts.append((x, y, txt))
        
    # 提取折線路徑 (paths)
    # Matplotlib 畫折線時，會輸出 m 運算子，後續跟著 l 運算子
    # 為了使正則匹配更健壯，我們匹配以數字和 m 開頭，後面緊跟 4 個數字和 l 的結構
    paths = []
    path_pattern = re.compile(
        r'([\d\.-]+)\s+([\d\.-]+)\s+m\s*\n?'
        r'\s*([\d\.-]+)\s+([\d\.-]+)\s+l\s*\n?'
        r'\s*([\d\.-]+)\s+([\d\.-]+)\s+l\s*\n?'
        r'\s*([\d\.-]+)\s+([\d\.-]+)\s+l\s*\n?'
        r'\s*([\d\.-]+)\s+([\d\.-]+)\s+l'
    )
    for m in path_pattern.finditer(full_text):
        coords = [float(x) for x in m.groups()]
        path_pts = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
        paths.append(path_pts)
        
    return texts, paths

def reconstruct_data(pdf_path):
    texts, paths = parse_pdf(pdf_path)
    
    x_ticks = [] # (x_pdf, val_real)
    y_ticks = [] # (y_pdf, val_real)
    
    for x, y, txt in texts:
        clean_txt = re.sub(r'\\mathdefault|\{|\}|\$', '', txt)
        try:
            val = float(clean_txt)
            if val >= 100:
                x_ticks.append((x, np.log10(val)))
            elif val > 0.0 and val < 1.0:
                y_ticks.append((y, val))
        except ValueError:
            if '10^' in clean_txt or '10^{' in clean_txt:
                match = re.search(r'10\^?(\d+)', clean_txt)
                if match:
                    val = 10**int(match.group(1))
                    x_ticks.append((x, np.log10(val)))

    if len(x_ticks) < 2 or len(y_ticks) < 2:
        print(f"Error calibrating {os.path.basename(pdf_path)}. X-ticks found: {len(x_ticks)}, Y-ticks found: {len(y_ticks)}")
        return None
        
    x_pdf_pts = [pt[0] for pt in x_ticks]
    x_real_pts = [pt[1] for pt in x_ticks]
    a, b = np.polyfit(x_real_pts, x_pdf_pts, 1)
    
    y_pdf_pts = [pt[0] for pt in y_ticks]
    y_real_pts = [pt[1] for pt in y_ticks]
    c, d = np.polyfit(y_real_pts, y_pdf_pts, 1)
    
    lines_data = []
    # 我們只留下有 5 個點的 path，因為這對應的是我們需要的 5 個 luminosity 數據點。
    valid_paths = [p for p in paths if len(p) == 5]
    
    # 根據 y 坐標的平均值排序，以區分圖表中的不同曲線 (例如 Original vs Augmentation)
    valid_paths.sort(key=lambda p: np.mean([pt[1] for pt in p]))
    
    for i, path in enumerate(valid_paths):
        points = []
        for x_pdf, y_pdf in path:
            log10_L = (x_pdf - b) / a
            L = 10**log10_L
            auc = (y_pdf - d) / c
            # 亮度做簡單四捨五入到最近的整數 (100, 300, 900, 1800, 3000)
            L_rounded = int(round(L))
            points.append((L_rounded, auc))
        lines_data.append(points)
        
    return lines_data

def main():
    pdf_dir = './figures/paper'
    pdf_files = [
        'AUC_CWoLa-Aug_ex-zz4l-CNN.pdf',
        'AUC_CWoLa-Aug_ex-zz4l-ParT.pdf',
        'AUC_transfer_za2l_CNN.pdf',
        'AUC_transfer_za2l_ParT.pdf',
        'AUC_transfer_zz4l_CNN.pdf',
        'AUC_transfer_zz4l_ParT.pdf'
    ]
    
    all_data = []
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        if not os.path.exists(pdf_path):
            print(f"File not found: {pdf_path}")
            continue
            
        lines = reconstruct_data(pdf_path)
        if lines is None:
            continue
            
        # 決定模型名稱和曲線類別
        model_type = 'CNN' if 'CNN' in pdf_file else 'ParT'
        channel = 'za2l' if 'za2l' in pdf_file else ('zz4l' if 'zz4l' in pdf_file else 'ex-zz4l')
        
        # 依照圖中的 Legend 標籤，決定每條線的 Line Index / Legend
        # 一般有 Original, phi shifting +5, phi shifting +10 三條曲線 (或只有少數曲線)
        # 我們將它們分別標記
        for idx, line in enumerate(lines):
            # 根據 valid_paths 排序，y 值越小排在越前面。
            # AUC 越低通常代表表現越差，通常 Original 比較高？或者 Augmentation 比較高？
            # 我們可以在之後畫圖時用 exact value 來與 GGF_VBF_CWoLa_summary.csv 進行交叉比對
            for L, auc in line:
                all_data.append({
                    'File Name': pdf_file,
                    'Model Type': model_type,
                    'Channel': channel,
                    'Line Index': idx,
                    'Luminosity (fb^-1)': L,
                    'AUC': auc
                })
                
    df = pd.DataFrame(all_data)
    output_path = './figures/paper/extracted_data.csv'
    df.to_csv(output_path, index=False)
    print(f"\nSuccessfully extracted data and saved to {output_path}")
    print(df.head(20))

if __name__ == '__main__':
    main()
