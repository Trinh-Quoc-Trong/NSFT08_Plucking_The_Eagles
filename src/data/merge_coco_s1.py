import json 
import os 
import shutil
from tqdm import tqdm 

def merge_and_filter_coco(input_json_path, output_dir, class_names_to_keep):
    """hàm này dùng để lọc và tìm ra những file dữ liệu bị lỗi, sau đó chuyển tất cả những class có trong class_names_to_keep  
       vào 1 folder chung
    Args:
        input_json_path (list): tên thư mục 
        output_dir (text): folder sẽ lưu sau khi sử lý xong
        class_names_to_keep (list): những class được chọn để lấy ra 
    """
    # chuyển dang sach ten class thanh chu thuong để dễ xữ lý
    class_names_to_keep_lower = [name.lower() for name in class_names_to_keep]

    # tạo thư mục đầu ra nếu chưa tồn tại 
    output_images_dir = os.path.join(output_dir, 'images')
    os.makedirs(output_images_dir, exist_ok = True)
    
    merged_coco = {
        'info': {'description': f'Bộ dữ liệu được gộp và lọc cho các lớp: {", ".join(class_names_to_keep)}'}
        'licenses': [],
        'images': [],
        'annotations': [],
        'categories': []
    }
    
    final_categories_map = {}
    next_final_cat_id = 1 
    
    current_max_image_id = 0
    current_max_annotation_id = 0
    
    print('start quá trình merge, lọc and copy images...')
    for json_path in tqdm(input_json_path, desc = 'dang su cac file'):
        print(f'\nDang xu ly {json_path}')

        with open(json_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f'can not read file json tu {json_path}, skiping')
                continue
    
        # -- bước 1: tìm và ánh xá các category ID mục tiêu trong file hiện tại -- 
        # ánh xạ từ source_cat_id -> {final_id, name}
        sour_cat_map = {}
        for categories in data.get('categories', []):
            cat_name_lower = category['name'].lower()
            if cat_name_lower in class_names_to_keep_lower:
                # nếu lớp chưa có trong bộ dữ liệu cuối cùng thì thêm vào 
                if cat_name_lower not in final_categories_map:
                    final_categories_map[cat_name_lower] = {
                        'id' : next_final_cat_id,
                        'name': category['name'],
                        'supercategory': category.get('supercategory', 'none')
                    }
                    next_final_cat_id += 1
                
                final_cat_id = final_categories_map[cat_name_lower]['id']
                source_cat_map[category['id']] = {'final_id': final_cat_id, 'name': cat_name_lower}
                
        if not source_cat_map:
            print(f"Thông tin: Không tìm thấy lớp nào trong danh sách cần giữ lại tại {json_path}. Đang bỏ qua.")
            continue
        
        # --- Bước 2: Lọc các chú thích thuộc về các lớp đối tượng mục tiêu ---
        source_ids_to_keep = set(source_cat_map.keys())
        relevant_annotations = [
            ann for ann in data.get('annotation', []) if ann['category_id'] in source_ids_to_keep
        ]
        
        if not relevant_annotations:
            print(f"Thông tin: Không có chú thích nào cho các lớp mục tiêu trong {json_path}. Đang bỏ qua.")
            continue
        
        relevant_image_ids = {ann['image_id'] for ann in relevant_annotations}
        relevant_images = [img for img in data.get('images', []) if img['id'] in relevant_image_ids]

        image_id_mapping = {}

        for image in tqdm(relevant_images, desc = 'dang sao chep anh'):
            source_image_path = os.path.join()

        
    

    



















