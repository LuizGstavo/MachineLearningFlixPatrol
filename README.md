# 📺 Netflix Analytics Dashboard

![Netflix](https://img.shields.io/badge/Netflix-E50914?style=for-the-badge&logo=netflix&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF6B6B?style=for-the-badge&logo=streamlit&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)

## 🎯 Visão Geral

Dashboard interativo para análise de dados da Netflix e predição de sucesso de conteúdo usando Machine Learning. O projeto utiliza dados reais de visualização e rankings para prever o tempo de assistência de novos títulos.

### ✨ Funcionalidades Principais

- 📊 **Análise Exploratória Completa**: Visualizações interativas dos dados da Netflix
- 🧠 **Machine Learning**: Modelos preditivos para tempo de visualização
- 🔗 **Matriz de Correlação**: Análise das relações entre variáveis
- 🎯 **Predições em Tempo Real**: Interface para prever sucesso de novos conteúdos
- 📈 **Insights Automatizados**: Padrões e tendências dos dados

## 🚀 Demo

![Dashboard Screenshot](https://via.placeholder.com/800x400/E50914/FFFFFF?text=Netflix+Analytics+Dashboard)

## 📋 Pré-requisitos

- Python 3.8 ou superior
- Arquivo `flixpatrol.csv` com dados da Netflix

## 🛠️ Instalação

### 1. Clone o repositório
```bash
git clone https://github.com/seu-usuario/netflix-analytics-dashboard.git
cd netflix-analytics-dashboard
```

### 2. Crie um ambiente virtual (recomendado)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/MacOS
python -m venv venv
source venv/bin/activate
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

### 4. Execute o dashboard
```bash
streamlit run netflix_dashboard_autonomo.py
```

## 📦 Dependências

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.15.0
scikit-learn>=1.3.0
seaborn>=0.12.0
matplotlib>=3.7.0
```

## 📁 Estrutura do Projeto

```
netflix-analytics-dashboard/
│
├── netflix_dashboard_autonomo.py    # Dashboard principal
├── requirements.txt                 # Dependências Python
├── flixpatrol.csv                  # Dataset Netflix (não incluído)
├── README.md                       # Este arquivo
├── app.py                          # Aplicação alternativa
├── check_models.py                 # Verificação de modelos
├── verificar_modelos.py            # Scripts de validação
├── logs.log                        # Logs da aplicação
└── models/                         # Modelos salvos (gerados automaticamente)
    ├── netflix_success_model.pkl
    └── netflix_watchtime_model.pkl
```

## 🎮 Como Usar

### 1. Preparação dos Dados
Certifique-se de que o arquivo `flixpatrol.csv` está na pasta raiz do projeto. O arquivo deve conter as seguintes colunas:
- `Title`: Nome do título
- `Type`: Tipo (Movie/TV Show)
- `Genre`: Gênero
- `Premiere`: Ano de estreia
- `Rank`: Ranking
- `Watchtime in Million`: Tempo de visualização em milhões de horas

### 2. Execução do Dashboard
```bash
streamlit run netflix_dashboard_autonomo.py
```

### 3. Navegação
O dashboard está organizado em seções:

#### 📊 Visão Geral
- Métricas principais do dataset
- Distribuição de filmes vs séries
- Top gêneros mais populares

#### 🔍 Análise Exploratória
- Histograma de distribuição de watchtime
- Box plots comparativos
- **Matriz de Correlação** entre variáveis
- Rankings dos melhores conteúdos

#### 🤖 Machine Learning
- Treinamento de modelos (Random Forest, Gradient Boosting, Linear Regression)
- Comparação de performance dos modelos
- **Predição interativa** de novos conteúdos

#### 💡 Insights
- Padrão 80/20 nos dados
- Melhores gêneros por performance
- Correlações principais

## 🧠 Modelos de Machine Learning

### Algoritmos Implementados
1. **Random Forest Regressor**: Ensemble de árvores de decisão
2. **Gradient Boosting Regressor**: Boosting sequencial
3. **Linear Regression**: Modelo linear base

### Métricas de Avaliação
- **R² Score**: Coeficiente de determinação
- **RMSE**: Raiz do erro quadrático médio
- **MAE**: Erro absoluto médio

### Features Utilizadas
- Tipo de conteúdo (codificado)
- Gênero (codificado)
- Ano de estreia
- Ranking

## 📊 Exemplo de Uso da API de Predição

```python
# Exemplo de predição
content_type = "Movie"
genre = "Action"
premiere_year = 2024
rank = 50

# A predição retorna o tempo estimado de visualização em milhões de horas
predicted_watchtime = make_prediction(model_data, content_type, genre, premiere_year, rank)
print(f"Tempo previsto: {predicted_watchtime:.1f}M horas")
```

## 🎯 Casos de Uso

### Para Produtores de Conteúdo
- Avaliar potencial de sucesso antes da produção
- Escolher gêneros com maior potencial de audiência
- Otimizar investimentos em novos projetos

### Para Analistas de Dados
- Estudar padrões de consumo na Netflix
- Identificar fatores que influenciam o sucesso
- Desenvolver estratégias data-driven

### Para Estudantes
- Aprender análise de dados com casos reais
- Praticar Machine Learning em projetos práticos
- Entender visualização de dados interativa

## 🐛 Solução de Problemas

### Erro: "Arquivo não encontrado"
```bash
❌ Arquivo 'flixpatrol.csv' não encontrado!
```
**Solução**: Certifique-se de que o arquivo CSV está na pasta raiz do projeto.

### Erro: "Módulo não encontrado"
```bash
ModuleNotFoundError: No module named 'streamlit'
```
**Solução**: Execute `pip install -r requirements.txt`

### Erro: "Dados insuficientes"
```bash
❌ Dataset muito pequeno para análise
```
**Solução**: Verifique se o arquivo CSV tem pelo menos 100 registros válidos.

## 📈 Performance

- **Tempo de carregamento**: ~2-5 segundos
- **Treinamento dos modelos**: ~30-60 segundos
- **Predições**: Instantâneas (<1 segundo)
- **Memória utilizada**: ~50-100MB

## 🔮 Próximas Funcionalidades

- [ ] Integração com API oficial da Netflix
- [ ] Modelos de Deep Learning
- [ ] Análise de sentimentos de reviews
- [ ] Dashboard para mobile
- [ ] Export de relatórios em PDF
- [ ] Sistema de recomendação

## 🤝 Contribuição

Contribuições são bem-vindas! Por favor, siga os passos:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## 👨‍💻 Autor

**Luiz Gustavo** - Estudante de Analise e Desenvolvimento de Sistemas

- GitHub: [@seu-usuario](https://github.com/seu-usuario)
- LinkedIn: [seu-perfil](https://linkedin.com/in/seu-perfil)
- Email: seu-email@exemplo.com

## 📚 Referências

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Documentation](https://plotly.com/python/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)

