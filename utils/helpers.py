
import re

def get_split(file_path):
    if 'training' in file_path:
        return 'training'
    else:
        return 'test'
    

def get_doc_info(header):
    doc_id = re.findall(r'id="(.*?)"', header)[0]
    genre = re.findall(r'genre="(.*?)"', header)[0]
    gender = re.findall(r'gender="(.*?)"', header)[0]
    return doc_id, genre, gender