# Setup
## Autenticação no Kaggle
Veja https://github.com/Kaggle/kagglehub?tab=readme-ov-file#authenticate 

## Clonar repositório
```
git clone https://github.com/almeidava93/kaggle_mammogram_challenge_001.git
```

## Instalação das dependências
Versão do python: `>=3.12`
```
python -m venv venv
venv/Scripts/activate
pip install -U -r requirements.txt
```

## Download do dataset
```
python download.py
```

## Preparação
- Divisão dos dados em dataset de treino, validação e teste
- Preparação dos metadados das imagens
```
python prepare.py
```

## Treino
Primeiro, é preciso configurar um experimento no arquivo `ex_config.toml`. Este arquivo concentra todos os experimentos e apresenta diversas opções, também presentes no script de treino. Por exemplo:
```
[exp_001]   # Nome do experimento
# Limte de tamanho do dataset. Use dataset_size = -1 para treinar com o dataset inteiro
dataset_size = 200          
# Alguns hiperparâmetros    
learning_rate = 0.0001
weight_decay = 0.0
dropout = 0.2
num_attn_heads = 8
num_encoder_layers = 2
feature_dim = 256
img_size = 256
num_img_channels = 1
num_img_init_features = 64
# Tamanho de batch e número de épocas
batch_size = 20
num_epochs = 10
```
A estrutura do modelo tem várias regras e opções para criar variações. Importante olhar o modelo em `model.py` para entender melhor.

# TO DO
- estrutura de resnet está pronta. Precisa adicionar camada que coleta informações do grupo de imagens daquele estudo e gera uma classificação considerando as informações do grupo de imagens. 
- uma ideia é usar o resnet como um encoder e adicionar um transformer no topo para atenção sobre as features selecionadas pelo resnet e finalmente classificar. Assim estaríamos combinando o inductive bias dos cnn com a capacidade de atenção dos transformers e a capacidade de lidar com sequências de comprimentos variados.

See https://chatgpt.com/share/67e5c9eb-6520-8012-86d0-3c2b898eb3ae

# Experiments
[X] replace transformer encoder with FFN - Worked similarly, maybe a little better
[X] increase weight decay
[] increase number of encoder layers
[] add ffn post transformer encoder
[] add ffn pre transformer encoder
[X] add image transforms
[] Change transformer norm
[] Change ResNet block norm
[] Change ResNet block activation

# Results

| Experiment | Best val AUC | Best val epoch | Best val loss |data size |lr        |weight dec|dropout   |feat dim  |
|------------|--------------|----------------|---------------|----------|----------|----------|----------|----------|
|exp_001     |              |                |               |          |          |          |          |          |
|exp_002     |              |                |               |          |          |          |          |          |
|exp_003     |              |                |               |          |          |          |          |          |