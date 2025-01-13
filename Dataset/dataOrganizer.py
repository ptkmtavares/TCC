import os
import shutil
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def setup_directories(data_path: str, index_path: str) -> None:
    """Creates necessary directories if they don't exist"""
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(os.path.dirname(index_path), exist_ok=True)

def process_email_file(file_path: str, encoding: str = 'latin_1') -> str:
    """Process an individual email file"""
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            return file.read()
    except UnicodeDecodeError:
        logging.error(f'Unicode error reading file: {file_path}')
        return ''

def get_phishing_email_list(folder_path: str) -> List[str]:
    """Gets list of phishing emails with parallel processing"""
    email_list = []
    
    def process_file(filename: str) -> List[str]:
        if not filename.endswith('.txt'):
            return []
        
        file_path = os.path.join(folder_path, filename)
        content = process_email_file(file_path)
        if not content:
            return []
            
        splits = content.split('\nFrom jose@monkey.org')
        return [f'From jose@monkey.org{split}' if i > 0 else split 
                for i, split in enumerate(splits)]

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, filename) 
                  for filename in os.listdir(folder_path)]
        for future in futures:
            email_list.extend(future.result())
    
    return email_list

def copy_spam_ham_files(source_path: str, dest_path: str) -> None:
    """Copies spam/ham files using multithreading"""
    logging.info(f'Copying files from {source_path} to {dest_path}')
    
    def copy_file(filename: str) -> None:
        src = os.path.join(source_path, filename)
        dst = os.path.join(dest_path, filename)
        shutil.copy(src, dst)
        
    with ThreadPoolExecutor() as executor:
        list(executor.map(copy_file, os.listdir(source_path)))

def update_index_file(index_path: str, data_folder_path: str, 
                     spam_ham_index_path: str) -> None:
    """Updates index file"""
    logging.info('Updating index file')
    
    with open(spam_ham_index_path, 'r', encoding='latin_1') as file:
        index_text = file.read()
    
    index_text = index_text.replace('../data', data_folder_path)
    
    with open(index_path, 'w', encoding='latin_1') as index:
        index.write(index_text)

def write_phishing_emails(emails: List[str], data_path: str, 
                         index_path: str, start_index: int) -> None:
    """Writes phishing emails and updates index"""
    logging.info(f'Processing {len(emails)} phishing emails')
    
    def write_email(args: Tuple[int, str]) -> str:
        i, email = args
        filename = f'inmail.{i + start_index}'
        filepath = os.path.join(data_path, filename)
        
        with open(filepath, 'w', encoding='latin_1') as f:
            f.write(email)
        
        return f'phishing {filepath}\n'
    
    with ThreadPoolExecutor() as executor:
        index_entries = list(executor.map(write_email, 
                                        enumerate(emails)))
    
    with open(index_path, 'a', encoding='latin_1') as index:
        index.writelines(index_entries)

def main():
    data_folder_path = 'Dataset/data'
    index_path = 'Dataset/index'
    spam_ham_data_path = 'Dataset/SpamHam/trec07p/data'
    spam_ham_index_path = 'Dataset/SpamHam/trec07p/full/index'
    phishing_folder_path = 'Dataset/Phishing/TXTs'

    setup_directories(data_folder_path, index_path)

    copy_spam_ham_files(spam_ham_data_path, data_folder_path)
    update_index_file(index_path, data_folder_path, spam_ham_index_path)

    phishing_emails = get_phishing_email_list(phishing_folder_path)
    write_phishing_emails(phishing_emails, data_folder_path, 
                         index_path, start_index=75420)
    
    logging.info('Processamento concluído com sucesso!')

if __name__ == '__main__':
    main()