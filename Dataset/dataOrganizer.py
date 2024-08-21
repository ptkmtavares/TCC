import os
import shutil

def GetPhishingEmailList(folder_path):
    email_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            header_text = ''
            try:
                with open(file_path, 'r', encoding='latin_1') as file:
                    header_text = file.read()
            except UnicodeDecodeError:
                print('Unicode Error!')
            
            email_list = email_list + header_text.split('\nFrom jose@monkey.org')
    for i in range(1, len(email_list)):
                email_list[i] = 'From jose@monkey.org' + email_list[i]
    return email_list

data_folder_path = 'Dataset/data'
index_path = 'Dataset/index'

spam_ham_data_path = 'Dataset/SpamHam/trec07p/data'
for filename in os.listdir(spam_ham_data_path):
    file_path = os.path.join(spam_ham_data_path, filename)
    shutil.copy(file_path, f'{data_folder_path}/{filename}')
        
spam_ham_index_path = 'Dataset/SpamHam/trec07p/full/index'
index_text = ''
with open(spam_ham_index_path, 'r', encoding='latin_1') as file:
        index_text = file.read()
index_text = index_text.replace('../data', data_folder_path)
with open(index_path, 'w', encoding='latin_1') as index:
    index.write(index_text)

phishing_folder_path = 'Dataset/Phishing/TXTs'
phishing_email_list = GetPhishingEmailList(phishing_folder_path)
index_str = ''
for i, email in enumerate(phishing_email_list):
    with open(f'{data_folder_path}/inmail.{i+75420}', 'w', encoding='latin_1') as inmail:
        inmail.write(email)
    index_str = index_str + f'phishing {data_folder_path}/inmail.{i+75420}\n'
with open(index_path, 'a', encoding='latin_1') as index:
    index.write(index_str)

