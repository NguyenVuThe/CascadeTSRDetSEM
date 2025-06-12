import os
import xml.etree.ElementTree as ET
import json

voc_folder = r"D:\MyWorking\dataset\mini_FinTabNet.c\val"
ann_dir   = r'D:\MyWorking\dataset\FinTabNet.c\FinTabNet.c-PDF_Annotations'
output_json = "dataset/ftbc_mini_6classes_val.json"

categories_list = [
    "table",
    "table column",
    "table row",
    "table spanning cell",
    "table projected row header",
    "table column header"
]

category_id_map = {name: i + 1 for i, name in enumerate(categories_list)}

images = []
annotations = []
image_id = 0
annotation_id = 0

for file in sorted(os.listdir(voc_folder)):
    if not file.endswith(".xml"):
        continue

    xml_path = os.path.join(voc_folder, file)

    filename = os.path.basename(xml_path)
    tab_id = int(filename.split('_')[-1].split('.')[0])
    json_file = os.path.join(ann_dir, filename.split('_table_')[0] + '_tables.json')
    
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find("filename").text.strip()
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    images.append({
        "id": image_id,
        "file_name": filename,
        "width": width,
        "height": height
    })

    for obj in root.findall("object"):
        class_name = obj.find("name").text.strip()
        if class_name not in category_id_map:
            continue  # skip unknown or "no object"

        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)
        w = xmax - xmin
        h = ymax - ymin
        area = w * h

        # segmentation polygon: rectangular bbox
        segmentation = [
            [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]
        ]

        with open(json_file, encoding='utf-8') as f:
            tables = json.load(f)
            for tab in tables:
                if tab.get('document_table_index') != tab_id:
                    continue
                for cell in tab.get('cells', []):
                    cell_bb = cell['pdf_bbox']

                    annotations.append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id_map[class_name],
                        "bbox": [xmin, ymin, w, h],
                        "text": cell.get('json_text_content', '')[:200],
                        "area": area,
                        "iscrowd": 0,
                        "segmentation": segmentation
                    })

                    annotation_id += 1

    image_id += 1

categories = [
    {"id": i + 1, "name": name, "supercategory": "Table"}
    for i, name in enumerate(categories_list)
]

output = {
    "info": { "author": "Bin Xiao" },
    "images": images,
    "annotations": annotations,
    "licenses": { "author": "Bin Xiao" },
    "categories": categories
}

with open(output_json, "w") as f:
    json.dump(output, f, indent=2)

print(f"✅ COCO-style JSON saved as {output_json}")