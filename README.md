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
- Baixar os ids de treino e teste do kaggle
```
kaggle competitions download -c spr-screening-mammography-recall
```

- Baixar o dataset completo com as imagens
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
# Você pode definir um checkpoint anterior como ponto de partida
start_from_checkpoint = "model/exp_001/checkpoint_last.pth"
```
A estrutura do modelo tem várias regras e opções para criar variações. Importante olhar o modelo em `model.py` para entender melhor.

Depois, basta treinar via linha de comando.
```
python train.py --exp exp_001
```

O número de `batches` pode ser definido via CLI
```
python train.py --exp exp_001 --batch-size 40
```

Você pode ajustar o número de `workers` e ativar a `pinned memory` para usar multiprocessamento e acelerar o carregamento de dados na GPU. O número recomendado de workers é `4 * n_gpus`. Este script ainda não suporta múltiplas gpus.
```
python train.py --exp exp_001 --workers 4 --pin-memory true
```

## Avaliação

Esta etapa cria o arquivo para submissão na competição do Kaggle no formato correto.

Para avaliar, utilize o comando:
```
python submit.py --exp exp_001
```

Você pode definir o tamanho dos lotes:
```
python submit.py --exp exp_001 --batch-size 40
```

Você pode definir para realizar a avaliação em uma versão específica do modelo adicionando um diretório.
```
python submit.py --exp exp_001 --path model/exp_001/checkpoint_08.pth
```

Esta etapa também suporta multiprocessamento com as opções `workers` e `pinned memory`
```
python submit.py --exp exp_001 --workers 4 --pin-memory true
``` 

# Experimentos
## Melhorias
- Reescrever código para que configurações fiquem concentradas em um único objeto e esse objeto possa ser passado para cada nn.Module ou dataset, sem precisar passar item por item

## TODO
- Usar modelos pré-treinados para processamento de imagem:
    - RADIOv2.5 NVIDIA: [repo](https://github.com/NVlabs/RADIO) / [paper](https://arxiv.org/abs/2412.07679) / [huggingface](https://huggingface.co/collections/nvidia/radio-669f77f1dd6b153f007dd1c6)
    - Digital Eye Mammography: [repo](https://github.com/cbddobvyz/digitaleye-mammography)
- Adicionar LayerNorm na camada final do classifier (igual a como RADIO foi treinado)
- Ajustar mecanismo de cropping para incluir imagens com fundo branco também. Identificar as imagens com fundo branco identificando o valor de pixel moda. Se diferente de 0, multiplicar por -1 e somar o valor máximo de pixel, conforme o código abaixo:
    ```
    img_mode = curr_img.mode().values.max().item()
    if img_mode != 0:
        curr_img = curr_img*-1 + curr_img.max()
    ```
- Testar estrutura de rede neural recorrente (GRU, LSTM) que analisa imagens de mamografia sequencialmente e atualiza a probabilidade de câncer de mama após cada imagem do estudo que analisa
- Mudar a loss function para uma que otimize melhor a AUC. Vide refs abaixo:
    - https://arxiv.org/abs/2012.03173
    - https://arxiv.org/abs/2310.11693
    