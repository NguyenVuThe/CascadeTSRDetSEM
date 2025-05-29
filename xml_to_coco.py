import os
import json
import xml.etree.ElementTree as ET
from glob import glob

# Define COCO structure
coco_format = {
    "images": [],
    "annotations": [],
    "categories": [
        {"id": 1, "name": "table"},
        {"id": 2, "name": "table column"},
        {"id": 3, "name": "table row"},
        {"id": 4, "name": "table spanning cell"},
        {"id": 5, "name": "table projected row header"},
        {"id": 6, "name": "table column header"},
    ]
}

# Mapping element names to category IDs
category_mapping = {
    "table": 1,
    "table column": 2,
    "table row": 3,
    "table spanning cell": 4,
    "table projected row header": 5,
    "table column header": 6
}

image_id = 1
annotation_id = 1

def parse_xml(xml_file):
    global image_id, annotation_id
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Extract filename
    filename_elem = root.find("filename")
    filename = filename_elem.text if filename_elem is not None else os.path.basename(xml_file).replace(".xml", ".jpg")
    
    # Extract image dimensions
    size_elem = root.find("size")
    if size_elem is not None:
        width_elem = size_elem.find("width")
        height_elem = size_elem.find("height")
        image_width = int(width_elem.text) if width_elem is not None else 0
        image_height = int(height_elem.text) if height_elem is not None else 0
    else:
        image_width, image_height = 0, 0
    
    # Create image entry
    image_entry = {
        "id": image_id,
        "file_name": filename,
        "width": image_width,
        "height": image_height
    }
    coco_format["images"].append(image_entry)
    
    # Parse all object elements
    for obj in root.findall("object"):
        name_elem = obj.find("name")
        if name_elem is not None:
            category_name = name_elem.text.lower()
            if category_name in category_mapping:
                # Extract bounding box
                bbox_elem = obj.find("bndbox")
                if bbox_elem is not None:
                    xmin = float(bbox_elem.find("xmin").text)
                    ymin = float(bbox_elem.find("ymin").text)
                    xmax = float(bbox_elem.find("xmax").text)
                    ymax = float(bbox_elem.find("ymax").text)
                    
                    # COCO format uses [x, y, width, height]
                    width = xmax - xmin
                    height = ymax - ymin
                    
                    annotation_entry = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_mapping[category_name],
                        "bbox": [xmin, ymin, width, height],
                        "segmentation": [],
                        "area": width * height,
                        "iscrowd": 0
                    }
                    coco_format["annotations"].append(annotation_entry)
                    annotation_id += 1
    
    image_id += 1

# Set input directory
input_dir = r"D:\MyWorking\dataset\mini_FinTabNet.c\train"  # Change this to the folder where your XML files are stored
xml_files = glob(os.path.join(input_dir, "*.xml"))

# If no files found, try parsing the current file directly
if not xml_files and os.path.exists("paste.txt"):
    with open("paste.txt", "r") as f:
        xml_content = f.read()
        if "<annotation>" in xml_content:
            # Create a temporary XML file
            with open("temp.xml", "w") as temp_f:
                temp_f.write(xml_content)
            parse_xml("temp.xml")
            os.remove("temp.xml")  # Clean up
else:
    for xml_file in xml_files:
        parse_xml(xml_file)

# Save to JSON
output_file = r"dataset\mini_fintabnet_coco.json"
with open(output_file, "w") as f:
    json.dump(coco_format, f, indent=4)

print(f"COCO JSON file saved as {output_file}")