# Detecção de Cabeçalho SMTP Falso usando IA

Este projeto de TCC tem como objetivo detectar cabeçalhos SMTP falsos em e-mails utilizando técnicas de Inteligência Artificial. O sistema é projetado para identificar e-mails de phishing e spam, diferenciando-os de e-mails legítimos (ham). A implementação utiliza uma combinação de Redes Neurais Artificiais (MLP e GAN) e várias bibliotecas e ferramentas avançadas para otimizar o desempenho e a precisão.

## Estrutura do Projeto

A estrutura do projeto é organizada da seguinte forma:

```plaintext
.
├── cache/
├── checkpoints/
├── Dataset/
├── plots/
├── .gitignore
├── config.py
├── dataExtractor.py
├── dataOrganizer.py
├── gan.py
├── main.py
├── mlp.py
├── plot.py
├── rayParam.py
├── receivedParser.py
```

## Funcionalidades Principais

### 1. Carregamento e Pré-processamento de Dados

O projeto utiliza a função `__load_and_preprocess_data` para carregar e pré-processar os dados de treinamento e teste. Os dados são divididos em conjuntos de treinamento e teste, e os tensores são preparados para uso em modelos de aprendizado profundo.

### 2. Configuração e Treinamento do GAN

A função `__setup_gan` configura e treina uma Rede Generativa Adversarial (GAN) para gerar exemplos adversariais de e-mails de phishing. O GAN é treinado para aumentar o conjunto de dados de phishing, equilibrando a quantidade de exemplos de phishing e ham.

### 3. Geração e Aumento de Dados

A função `__generate_and_augment_data` gera exemplos adversariais usando o gerador treinado e aumenta o conjunto de dados de treinamento com esses exemplos. Isso ajuda a melhorar a precisão do modelo ao lidar com e-mails de phishing.

### 4. Treinamento e Avaliação do MLP

A função `__train_and_evaluate_mlp` treina e avalia um Perceptron Multicamadas (MLP) com e sem dados aumentados. O desempenho do modelo é comparado em termos de precisão e matriz de confusão.

### 5. Predição em Conjunto de Teste de Exemplo

A função `main` executa o processo completo de treinamento e avaliação, incluindo a predição em um conjunto de teste de exemplo. Os resultados são registrados e comparados para avaliar a eficácia do modelo.

## Tecnologias Utilizadas

### 1. Caching com Pickle

O projeto utiliza a biblioteca Pickle para armazenar em cache os recursos extraídos dos e-mails. Isso acelera o processo de carregamento de dados, evitando a necessidade de reprocessar os e-mails a cada execução.

### 2. Processamento Paralelo com concurrent.futures

Para melhorar a eficiência do processamento de e-mails, o projeto utiliza a biblioteca concurrent.futures para processar e-mails em paralelo. Isso reduz significativamente o tempo de execução ao lidar com grandes volumes de dados.

### 3. Aceleração com CUDA

O projeto utiliza a biblioteca PyTorch com aceleração CUDA para aproveitar o poder de processamento das GPUs. Isso acelera o treinamento e a inferência dos modelos de aprendizado profundo, permitindo lidar com grandes conjuntos de dados de forma eficiente.

### 4. Otimização de Hiperparâmetros com Ray Tune

Para encontrar os melhores hiperparâmetros para os modelos, o projeto utiliza a biblioteca Ray Tune. O Ray Tune realiza uma busca eficiente de hiperparâmetros, melhorando o desempenho e a precisão dos modelos treinados.

### 5. Visualização de Dados com Matplotlib

O projeto utiliza a biblioteca Matplotlib para criar visualizações dos dados e resultados dos modelos. Isso inclui a distribuição de características de e-mails, resultados de tuning de hiperparâmetros, perdas de treinamento e matrizes de confusão. As visualizações ajudam a entender melhor o desempenho dos modelos e a eficácia das técnicas aplicadas.

## Como Executar

Clone o repositório:

```bash
    git clone https://github.com/ptkmtavares/TCC
    cd https://github.com/ptkmtavares/TCC
```

Instale as dependências:

```bash
    pip install -r requirements.txt
```

Execute o script principal:

```bash
    python main.py
```

Conclusão
Este projeto demonstra a aplicação de técnicas avançadas de Inteligência Artificial para a detecção de cabeçalhos SMTP falsos em e-mails. Utilizando uma combinação de caching, processamento paralelo, aceleração com CUDA e otimização de hiperparâmetros, o sistema alcança alta precisão na identificação de e-mails de phishing e spam.
