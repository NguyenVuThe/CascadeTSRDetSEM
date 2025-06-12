import os
import cv2
import json
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def LoadTabStruct(xml_file, img_dir, ann_dir, visualize = True):
    filename = os.path.basename(xml_file)
    tab_id = int(filename.split('_')[-1].split('.')[0])  # table_0 → 0

    # Ví dụ: A_2007_page_46_table_0.xml → A_2007_page_46_table_0.jpg và A_2007_page_46_tables.json
    img_file = os.path.join(img_dir, filename.replace('.xml', '.jpg'))

    # Tên json = bỏ `_table_0`, đổi `table_0.xml` → `tables.json`
    json_filename = filename.split('_table_')[0] + '_tables.json'
    json_file = os.path.join(ann_dir, json_filename)

    # Parse XML để lấy bbox bảng
    tree = ET.parse(xml_file)
    page = tree.getroot()

    for obj in page.findall("object"):
        name = obj.find("name").text
        if name == 'table':
            bndbox = obj.find("bndbox")
            xmin = int(float(bndbox.find("xmin").text))
            ymin = int(float(bndbox.find("ymin").text))
            xmax = int(float(bndbox.find("xmax").text) + 0.5)
            ymax = int(float(bndbox.find("ymax").text) + 0.5)

            # Load image
            img = cv2.imread(img_file)
            if img is None:
                print(f"❌ Image not found: {img_file}")
                return None, None, None, None, None, None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img[ymin:ymax, xmin:xmax, :]

            if os.path.isfile(json_file):
                with open(json_file, encoding='utf-8') as f:
                    tables = json.load(f)
                    for tab in tables:
                        if tab.get('document_table_index') == tab_id:
                            tab_bb = tab.get('pdf_table_bbox')
                            scale_x = (xmax - xmin) / (tab_bb[2] - tab_bb[0])
                            scale_y = (ymax - ymin) / (tab_bb[3] - tab_bb[1])
                            scale = min(scale_x, scale_y)

                            cols = tab.get('columns')
                            for k in cols:
                                bb = cols[k].get('pdf_column_bbox')
                                bb = [int(scale * (bb[0]-tab_bb[0])),
                                      int(scale * (bb[1]-tab_bb[1])),
                                      int(scale * (bb[2]-tab_bb[0])),
                                      int(scale * (bb[3]-tab_bb[1]))]
                                cols[k]['pdf_column_bbox'] = bb

                            rows = tab.get('rows')
                            for k in rows:
                                bb = rows[k].get('pdf_row_bbox')
                                bb = [int(scale * (bb[0]-tab_bb[0])),
                                      int(scale * (bb[1]-tab_bb[1])),
                                      int(scale * (bb[2]-tab_bb[0])),
                                      int(scale * (bb[3]-tab_bb[1]))]
                                rows[k]['pdf_row_bbox'] = bb

                            cells = tab.get('cells')
                            cell_bboxes = []
                            cell_texts = []
                            for i, cell in enumerate(cells):
                                bb = cell.get('pdf_bbox')
                                bb = [int(scale * (bb[0]-tab_bb[0])),
                                      int(scale * (bb[1]-tab_bb[1])),
                                      int(scale * (bb[2]-tab_bb[0])),
                                      int(scale * (bb[3]-tab_bb[1]))]
                                cells[i]['pdf_bbox'] = bb
                                cell_bboxes.append(bb)
                                cell_texts.append(cell.get('json_text_content', '')[:100])
                                print(f"[Cell {i}] BBox: {bb} | Text: \"{cell.get('json_text_content', '')[:100]}\"")

                            # Hiển thị ảnh với bbox nếu visualize
                        
                            if visualize:
                                fig, ax = plt.subplots(figsize=(12, 10))
                                ax.imshow(img)
                                for bb, text in zip(cell_bboxes, cell_texts):
                                    x0, y0, x1, y1 = bb
                                    w, h = x1 - x0, y1 - y0
                                    rect = patches.Rectangle((x0, y0), w, h, linewidth=1, edgecolor='red', facecolor='none')
                                    ax.add_patch(rect)
                                    ax.text(x0, y0 - 2, text[:15], fontsize=8, color='blue', backgroundcolor='white')
                                ax.set_title("Cell BBoxes and Texts")
                                plt.axis('off')
                                plt.tight_layout()
                                plt.show()

                            return img, cols, rows, cells, cell_bboxes, cell_texts
            else:
                print(f"❌ JSON not found: {json_file}")
    return None, None, None, None, None, None


train_dir = r'D:\MyWorking\dataset\mini_FinTabNet.c\train'
val_dir   = 'FinTab/Structure/val'
img_dir   = r'D:\MyWorking\dataset\mini_FinTabNet.c\images'
ann_dir   = r'D:\MyWorking\dataset\FinTabNet.c\FinTabNet.c-PDF_Annotations'

xml_path = os.path.join(train_dir, 'ZION_2016_page_70_table_0.xml')
img, cols, rows, cells, cell_bboxes, cell_texts = LoadTabStruct(xml_path, img_dir, ann_dir)

if img is not None:
    print(f"✔ Loaded table with {len(cells)} cells.")
else:
    print("❌ Failed to load table structure.")
