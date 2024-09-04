def GetPhishingEmailFromEml(file_path):
    header_text = ''
    try:
        with open(file_path, 'r', encoding='latin_1') as file:
            header_text = file.read()
    except UnicodeDecodeError:
        print('Unicode Error!')
    return header_text

data_folder_path = 'Dataset/data'
index_path = 'Dataset/index'

# Processar o arquivo Aviso_de_suspensao_da_sua_conta.eml
example_eml_path = 'Examples/Aviso_de_suspensao_da_sua_conta.eml'
phishing_email_list = []
phishing_email_list.append(GetPhishingEmailFromEml(example_eml_path))

index_str = ''
for i, email in enumerate(phishing_email_list):
    with open(f'{data_folder_path}/inmail.{i+75420}', 'w', encoding='latin_1') as inmail:
        inmail.write(email)
    index_str = index_str + f'phishing {data_folder_path}/inmail.{i+75420}\n'
with open(index_path, 'a', encoding='latin_1') as index:
    index.write(index_str)