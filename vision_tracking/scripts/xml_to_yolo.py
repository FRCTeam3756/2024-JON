import os
import xml.etree.ElementTree as ET

################################################

def convert_xml_to_yolo(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

    for xml_file in os.listdir(folder):
        if xml_file.endswith('.xml'):
            tree = ET.parse(os.path.join(folder, xml_file))
            root = tree.getroot()
            
            image_name = root.find('filename').text
            image_name_without_ext = os.path.splitext(image_name)[0]

            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)

            yolo_data = []
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                xmin = int(obj.find('bndbox/xmin').text)
                ymin = int(obj.find('bndbox/ymin').text)
                xmax = int(obj.find('bndbox/xmax').text)
                ymax = int(obj.find('bndbox/ymax').text)

                class_id = 0 if class_name == 'note' else (1 if class_name == 'red_robot' else 2)  # Adjust class IDs as needed
                x_center = (xmin + xmax) / 2 / width
                y_center = (ymin + ymax) / 2 / height
                box_width = (xmax - xmin) / width
                box_height = (ymax - ymin) / height

                yolo_data.append(f"{class_id} {x_center} {y_center} {box_width} {box_height}")

            with open(os.path.join(folder, f"{image_name_without_ext}.txt"), 'w') as f:
                f.write("\n".join(yolo_data))

################################################

convert_xml_to_yolo(
    folder=r'yolov11n\dataset\labels\train',
)
