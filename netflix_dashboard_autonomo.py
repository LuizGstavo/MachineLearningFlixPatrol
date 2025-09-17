import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
warnings.filterwarnings('ignore')

# ===================================================================
# 1. CONFIGURAÇÃO E ESTILO
# ===================================================================

# Configuração da página
st.set_page_config(
    page_title="Netflix Analytics Dashboard",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado com tema Netflix
st.markdown("""
<style>
    /* Cores principais da Netflix */
    :root {
        --netflix-red: #E50914;
        --netflix-dark: #221F1F;
        --netflix-white: #ffffff;
        --netflix-gray: #564d4d;
    }
    
    /* Header principal */
    .main-header {
        font-size: 3.5rem;
        color: var(--netflix-red);
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Subtítulo */
    .subtitle {
        font-size: 1.2rem;
        color: var(--netflix-gray);
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Cards de métricas */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid var(--netflix-red);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    /* Seções */
    .section-header {
        background: linear-gradient(90deg, var(--netflix-red), #b8070f);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 2rem 0 1rem 0;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    /* Botões customizados */
    .stButton > button {
        background-color: var(--netflix-red);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid var(--netflix-red);
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Success boxes */
    .success-box {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    /* Warning boxes */
    .warning-box {
        background-color: #fff3cd;
        border-color: #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Título principal do dashboard
st.markdown('<h1 class="main-header">📺 NETFLIX ANALYTICS DASHBOARD</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Análise de Dados e Predição de Sucesso com Machine Learning</p>', unsafe_allow_html=True)
st.markdown("---")

# ===================================================================
# 2. CARREGAMENTO E LIMPEZA DE DADOS
# ===================================================================

@st.cache_data
def load_data():
    """
    Carrega e limpa os dados do arquivo CSV da Netflix de forma robusta.
    """
    try:
        df = pd.read_csv("flixpatrol.csv")
        
        # 1. Conversão 'Watchtime in Million' para numérico
        df['Watchtime in Million'] = df['Watchtime in Million'].astype(str)
        df['Watchtime_Million'] = df['Watchtime in Million'].str.replace('M', '', regex=False)
        df['Watchtime_Million'] = df['Watchtime_Million'].str.replace(',', '', regex=False)
        df['Watchtime_Million'] = pd.to_numeric(df['Watchtime_Million'], errors='coerce')
        
        # 2. Conversão 'Rank' para numérico
        df['Rank'] = df['Rank'].astype(str)
        df['Rank_Numeric'] = df['Rank'].str.replace(r'[.,]', '', regex=True)
        df['Rank_Numeric'] = pd.to_numeric(df['Rank_Numeric'], errors='coerce')
        
        # 3. Conversão 'Premiere' para numérico e criação de 'Premiere_Year'
        df['Premiere_Year'] = pd.to_numeric(df['Premiere'], errors='coerce')
        df['Content_Age'] = 2025 - df['Premiere_Year']
        
        # 4. Limpeza inicial de dados com NaN
        df_clean = df.dropna(subset=['Type', 'Genre'])
        
        # 5. Preenchimento de valores NaN restantes de forma inteligente
        df_clean['Watchtime_Million'] = df_clean['Watchtime_Million'].fillna(
            df_clean.groupby(['Type', 'Genre'])['Watchtime_Million'].transform('median')
        ).fillna(df_clean['Watchtime_Million'].median())

        df_clean['Rank_Numeric'] = df_clean['Rank_Numeric'].fillna(
            df_clean.groupby(['Type', 'Genre'])['Rank_Numeric'].transform('median')
        ).fillna(df_clean['Rank_Numeric'].median())
        
        df_clean['Premiere_Year'] = df_clean['Premiere_Year'].fillna(
            df_clean.groupby('Genre')['Premiere_Year'].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else 2020)
        ).fillna(2020)

        # 6. Reaplicar Content_Age após preencher Premiere_Year
        df_clean['Content_Age'] = 2025 - df_clean['Premiere_Year']
        
        # 7. Filtros de outliers mais permissivos
        df_clean = df_clean[(df_clean['Watchtime_Million'] >= 0) & (df_clean['Watchtime_Million'] <= 10000)]
        df_clean = df_clean[(df_clean['Premiere_Year'] >= 1980) & (df_clean['Premiere_Year'] <= 2030)]
        df_clean = df_clean[df_clean['Rank_Numeric'] > 0]
        
        # 8. Correção final e crítica do "Pior Ranking"
        total_titles_clean = len(df_clean)
        df_clean.loc[df_clean['Rank_Numeric'] > total_titles_clean, 'Rank_Numeric'] = total_titles_clean
        
        print(f"📊 Dataset final: {len(df_clean)} registros de {len(df)} originais")
        
        return df_clean
    
    except FileNotFoundError:
        st.error("❌ Arquivo 'flixpatrol.csv' não encontrado!")
        return None
    except Exception as e:
        st.error(f"❌ Erro ao carregar dados: {str(e)}")
        return None

# ===================================================================
# 3. MÓDULO DE MACHINE LEARNING
# ===================================================================

@st.cache_resource
def train_models(df):
    """Treina modelos de ML"""
    try:
        df_ml = df.copy()
        
        le_type = LabelEncoder()
        le_genre = LabelEncoder()
        
        df_ml['Type_encoded'] = le_type.fit_transform(df_ml['Type'])
        df_ml['Genre_encoded'] = le_genre.fit_transform(df_ml['Genre'])
        
        features = ['Type_encoded', 'Genre_encoded', 'Premiere_Year', 'Rank_Numeric']
        target = 'Watchtime_Million'
        
        X = df_ml[features]
        y = df_ml[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression()
        }
        
        results = {}
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'r2': r2,
                'rmse': rmse,
                'mae': mae
            }
        
        best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
        best_model = results[best_model_name]['model']
        best_model.fit(X, y)
        
        return {
            'best_model': best_model,
            'best_model_name': best_model_name,
            'all_results': results,
            'le_type': le_type,
            'le_genre': le_genre,
            'features': features
        }
    except Exception as e:
        st.error(f"❌ Erro no treinamento: {str(e)}")
        return None

def make_prediction(model_data, content_type, genre, premiere_year, rank):
    """Faz predição"""
    try:
        type_encoded = model_data['le_type'].transform([content_type])[0]
        genre_encoded = model_data['le_genre'].transform([genre])[0]
        
        features = np.array([[type_encoded, genre_encoded, premiere_year, rank]])
        prediction = model_data['best_model'].predict(features)[0]
        
        return max(0, prediction)
    except Exception as e:
        st.error(f"❌ Erro na predição: {str(e)}")
        return 0

# ===================================================================
# 4. CARREGAMENTO DOS DADOS
# ===================================================================

df = load_data()
if df is None:
    st.stop()

# ===================================================================
# SEÇÃO 1: VISÃO GERAL
# ===================================================================

st.markdown('<div class="section-header">📊 VISÃO GERAL DO DATASET</div>', unsafe_allow_html=True)

# Métricas principais
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_content = len(df)
    st.markdown(f"""
    <div class="metric-card">
        <h4>📺 Total de Conteúdo</h4>
        <h3>{total_content:,}</h3>
    </div>
    """, unsafe_allow_html=True)

with col2:
    tv_shows = len(df[df['Type'] == 'TV Show'])
    st.markdown(f"""
    <div class="metric-card">
        <h4>📺 Séries de TV</h4>
        <h3>{tv_shows:,}</h3>
    </div>
    """, unsafe_allow_html=True)

with col3:
    movies = len(df[df['Type'] == 'Movie'])
    st.markdown(f"""
    <div class="metric-card">
        <h4>🎬 Filmes</h4>
        <h3>{movies:,}</h3>
    </div>
    """, unsafe_allow_html=True)

with col4:
    genres = df['Genre'].nunique()
    st.markdown(f"""
    <div class="metric-card">
        <h4>🎭 Gêneros Únicos</h4>
        <h3>{genres:,}</h3>
    </div>
    """, unsafe_allow_html=True)

# Gráficos principais
col1, col2 = st.columns(2)

with col1:
    # Distribuição por tipo
    type_counts = df['Type'].value_counts()
    fig_pie = px.pie(
        values=type_counts.values, 
        names=type_counts.index,
        color_discrete_sequence=['#E50914', '#F5F5F1'],
        title="📊 Distribuição: Filmes vs Séries"
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    # Top 10 gêneros
    genre_counts = df['Genre'].value_counts().head(10)
    fig_bar = px.bar(
        x=genre_counts.values,
        y=genre_counts.index,
        orientation='h',
        color_discrete_sequence=['#E50914'],
        title="🎭 Top 10 Gêneros Mais Populares"
    )
    fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_bar, use_container_width=True)

# ===================================================================
# SEÇÃO 2: ANÁLISE EXPLORATÓRIA
# ===================================================================

st.markdown('<div class="section-header">🔍 ANÁLISE EXPLORATÓRIA</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Histograma de watchtime - PADRÃO 80/20
    fig_hist = px.histogram(
        df, 
        x='Watchtime_Million', 
        nbins=50,
        title="📈 Distribuição de Watchtime (Padrão 80/20)",
        color_discrete_sequence=['#E50914']
    )
    fig_hist.update_layout(
        xaxis_title="Tempo de Visualização (Milhões de Horas)",
        yaxis_title="Frequência"
    )
    fig_hist.add_annotation(
        x=df['Watchtime_Million'].quantile(0.8),
        y=df['Watchtime_Million'].value_counts().max() * 0.8,
        text="80% dos títulos<br>têm baixa audiência",
        showarrow=True,
        arrowhead=2,
        bgcolor="yellow",
        opacity=0.8
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    # Box plot por tipo
    fig_box = px.box(
        df, 
        x='Type', 
        y='Watchtime_Million',
        title="📊 Performance: Filmes vs Séries",
        color='Type',
        color_discrete_sequence=['#E50914', "#0913A8"]
    )
    st.plotly_chart(fig_box, use_container_width=True)

# MATRIZ DE CORRELAÇÃO
st.subheader("🔗 Matriz de Correlação")

# Preparar dados numéricos para correlação
df_corr = df[['Watchtime_Million', 'Rank_Numeric', 'Premiere_Year', 'Content_Age']].copy()

# Adicionar variáveis codificadas
le_type_temp = LabelEncoder()
le_genre_temp = LabelEncoder()
df_corr['Type_encoded'] = le_type_temp.fit_transform(df['Type'])
df_corr['Genre_encoded'] = le_genre_temp.fit_transform(df['Genre'])

# Renomear colunas para melhor visualização
df_corr.columns = ['Watchtime', 'Ranking', 'Ano_Estreia', 'Idade_Conteudo', 'Tipo', 'Gênero']

# Criar matriz de correlação
correlation_matrix = df_corr.corr()

# Criar heatmap com plotly
fig_corr = px.imshow(
    correlation_matrix,
    text_auto=True,
    aspect="auto",
    title="🔗 Matriz de Correlação - Relações entre Variáveis",
    color_continuous_scale='RdBu_r'
)
fig_corr.update_layout(height=500)
st.plotly_chart(fig_corr, use_container_width=True)

# Top Performers
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔥 Top 5 - Maior Watchtime")
    top_watchtime = df.nlargest(5, 'Watchtime_Million')[
        ['Title', 'Type', 'Genre', 'Watchtime_Million']
    ]
    top_watchtime.columns = ['Título', 'Tipo', 'Gênero', 'Watchtime (M)']
    st.dataframe(top_watchtime, use_container_width=True)

with col2:
    st.subheader("⭐ Top 5 - Melhor Ranking")
    top_ranking = df.nsmallest(5, 'Rank_Numeric')[
        ['Title', 'Type', 'Genre', 'Rank_Numeric']
    ]
    top_ranking.columns = ['Título', 'Tipo', 'Gênero', 'Ranking']
    st.dataframe(top_ranking, use_container_width=True)

# ===================================================================
# SEÇÃO 3: MACHINE LEARNING E PREDIÇÕES
# ===================================================================

st.markdown('<div class="section-header">🤖 MACHINE LEARNING E PREDIÇÕES</div>', unsafe_allow_html=True)

# Botão para treinar modelos
if st.button("🚀 Treinar Modelos de ML", use_container_width=True):
    with st.spinner("🔄 Treinando modelos... (30 segundos)"):
        model_data = train_models(df)
        
        if model_data:
            st.session_state.model_data = model_data
            st.session_state.models_trained = True
            st.markdown('<div class="success-box">✅ Modelos treinados com sucesso!</div>', 
                       unsafe_allow_html=True)

# Verificar se modelos foram treinados
if hasattr(st.session_state, 'models_trained') and st.session_state.models_trained:
    model_data = st.session_state.model_data
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Métricas dos modelos
        st.subheader("📊 Performance dos Modelos")
        metrics_data = []
        for name, result in model_data['all_results'].items():
            metrics_data.append({
                'Modelo': name,
                'R² Score': f"{result['r2']:.4f}",
                'RMSE': f"{result['rmse']:.1f}"
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)
        
        st.markdown(f"**🏆 Vencedor:** {model_data['best_model_name']}")
        best_r2 = model_data['all_results'][model_data['best_model_name']]['r2']
        st.markdown(f"**📈 Explica:** {best_r2*100:.1f}% da variação")
    
    with col2:
        # Formulário de predição compacto
        st.subheader("🔮 Predição Instantânea")
        
        available_types = sorted(df['Type'].unique())
        available_genres = sorted(df['Genre'].unique())
        
        content_type = st.selectbox("Tipo", available_types, key="pred_type")
        genre = st.selectbox("Gênero", available_genres, key="pred_genre")
        
        col_year, col_rank = st.columns(2)
        with col_year:
            premiere_year = st.slider("Ano", 2015, 2025, 2024, key="pred_year")
        with col_rank:
            rank = st.slider("Ranking", 1, 500, 50, key="pred_rank")
        
        if st.button("🔮 Prever Sucesso", use_container_width=True):
            predicted_watchtime = make_prediction(
                model_data, content_type, genre, premiere_year, rank
            )
            
            if predicted_watchtime > 0:
                # Categorizar resultado
                if predicted_watchtime >= 200:
                    category = "🔥 TOP HIT"
                    color = "#28a745"
                elif predicted_watchtime >= 100:
                    category = "⭐ POPULAR"
                    color = "#ffc107"
                elif predicted_watchtime >= 50:
                    category = "📊 MODERADO"
                    color = "#fd7e14"
                else:
                    category = "📉 BAIXO"
                    color = "#dc3545"
                
                st.markdown(f"""
                <div style="background: {color}; color: white; padding: 1rem; border-radius: 10px; text-align: center; margin: 1rem 0;">
                    <h3>🎯 {predicted_watchtime:.1f}M horas</h3>
                    <h4>{category}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Gráfico de gauge compacto
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = predicted_watchtime,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Performance Prevista"},
                    gauge = {
                        'axis': {'range': [None, 400]},
                        'bar': {'color': "#E50914"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 100], 'color': "yellow"},
                            {'range': [100, 200], 'color': "orange"},
                            {'range': [200, 400], 'color': "green"}
                        ]
                    }
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)

else:
    st.markdown('<div class="warning-box">⚠️ Clique no botão acima para treinar os modelos primeiro!</div>', 
               unsafe_allow_html=True)

# ===================================================================
# INSIGHTS FINAIS
# ===================================================================

st.markdown('<div class="section-header">💡 INSIGHTS PRINCIPAIS</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    watchtime_min = df['Watchtime_Million'].min()
    watchtime_mean = df['Watchtime_Million'].mean()
    watchtime_max = df['Watchtime_Million'].max()
    st.markdown(f"""
    <div class="info-box">
        <h4>📊 Padrão 80/20</h4>
        <p>• Mínimo: {watchtime_min:.1f}M horas</p>
        <p>• Média: {watchtime_mean:.1f}M horas</p>
        <p>• Máximo: {watchtime_max:.1f}M horas</p>
        <p><strong>💡 Poucos hits dominam!</strong></p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Melhor gênero por performance
    genre_performance = df.groupby('Genre')['Watchtime_Million'].mean().sort_values(ascending=False)
    best_genre = genre_performance.index[0]
    best_performance = genre_performance.iloc[0]
    
    st.markdown(f"""
    <div class="info-box">
        <h4>🎭 Melhor Gênero</h4>
        <p><strong>{best_genre}</strong></p>
        <p>Performance média:</p>
        <p><strong>{best_performance:.1f}M horas</strong></p>
        <p>💡 Aposte neste gênero!</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    # Análise de correlação principal
    correlation_value = correlation_matrix.loc['Watchtime', 'Ranking']
    st.markdown(f"""
    <div class="info-box">
        <h4>🔗 Correlação Chave</h4>
        <p>Watchtime vs Ranking:</p>
        <p><strong>{correlation_value:.3f}</strong></p>
        <p>💡 Melhor ranking = mais audiência</p>
        <p>(Correlação negativa esperada)</p>
    </div>
    """, unsafe_allow_html=True)

# ===================================================================
# FOOTER
# ===================================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        📺 <strong>Netflix Analytics Dashboard</strong> | 🤖 <strong>Machine Learning para Predição de Sucesso</strong><br>
        <strong>🎯 Decisões Data-Driven • 📊 Análise Preditiva • 💰 Otimização de Investimentos</strong>
    </div>
    """, 
    unsafe_allow_html=True
)