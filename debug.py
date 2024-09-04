from dataExtractor import getEmailInfo

#email_info = getEmailInfo("C:\\Users\\Blast\\Documents\\TCC\\Dataset\\example\\Apresentacao_do_trabalho_final.eml")
email_info = getEmailInfo("C:\\Users\\Blast\\Documents\\TCC\\Dataset\\example\\Aviso_de_suspensao_da_sua_conta.eml")

# Iterar sobre os elementos do dicionário e imprimir cada um em uma nova linha
for key, value in email_info.items():
    print(f'{key}: {value}')