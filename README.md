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