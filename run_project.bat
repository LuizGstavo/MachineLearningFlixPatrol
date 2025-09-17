@echo off
echo ================================================
echo    Netflix FlixPatrol Analytics Dashboard
echo ================================================
echo.
echo Iniciando o dashboard...
echo.

REM Mudar para o diretório do script
cd /d "%~dp0"

REM Verificar se o Python está instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo ERRO: Python nao encontrado! Por favor, instale o Python primeiro.
    pause
    exit /b 1
)

echo Python encontrado!
echo.

REM Verificar se o arquivo de dados existe
if not exist "flixpatrol.csv" (
    echo ERRO: Arquivo flixpatrol.csv nao encontrado!
    echo Por favor, certifique-se de que o arquivo esta no diretorio correto.
    pause
    exit /b 1
)

echo Arquivo de dados encontrado!
echo.

REM Instalar todas as dependências necessárias
echo ================================================
echo    Instalando Dependencias
echo ================================================
echo.
echo Instalando dependencias do dashboard...
pip install streamlit pandas numpy plotly scikit-learn --quiet

echo.
echo Verificando instalacao...
python -c "import streamlit, pandas, numpy, plotly, sklearn; print('✅ Todas as dependencias instaladas com sucesso!')"

if errorlevel 1 (
    echo.
    echo ❌ Erro na instalacao das dependencias!
    echo Tentando instalacao individual...
    pip install streamlit
    pip install pandas
    pip install numpy
    pip install plotly
    pip install scikit-learn
)

echo.
echo ================================================
echo    Iniciando Dashboard Streamlit
echo ================================================
echo.
echo 📺 NETFLIX ANALYTICS DASHBOARD
echo.
echo O dashboard sera aberto automaticamente no seu navegador.
echo URL: http://localhost:8501
echo.
echo ⚡ Funcionalidades disponiveis:
echo   • Visao Geral dos Dados
echo   • Analise Exploratoria Interativa
echo   • Predicoes com Machine Learning
echo.
echo 🛑 Para parar o servidor, pressione Ctrl+C
echo.

REM Iniciar o Streamlit
streamlit run netflix_dashboard_autonomo.py

echo.
echo ================================================
echo Dashboard finalizado!
echo ================================================
pause