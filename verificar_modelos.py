import pickle
import os

print("=== VERIFICANDO MODELOS SALVOS ===\n")

# Verificar modelo de sucesso
if os.path.exists('netflix_success_model.pkl'):
    try:
        with open('netflix_success_model.pkl', 'rb') as f:
            success_model = pickle.load(f)
        print(f"Modelo de Sucesso: {type(success_model).__name__}")
        
        if hasattr(success_model, 'steps'):
            print(f"Pipeline com {len(success_model.steps)} etapas:")
            for i, (nome, componente) in enumerate(success_model.steps):
                print(f"  {i+1}. {nome}: {type(componente).__name__}")
            print(f"Modelo final: {type(success_model.steps[-1][1]).__name__}")
        else:
            print(f"Modelo direto: {type(success_model).__name__}")
    except Exception as e:
        print(f"Erro ao carregar modelo de sucesso: {e}")
    print()

# Verificar modelo de tempo de visualização
if os.path.exists('netflix_watchtime_model.pkl'):
    try:
        with open('netflix_watchtime_model.pkl', 'rb') as f:
            watchtime_model = pickle.load(f)
        print(f"Modelo de Tempo de Visualização: {type(watchtime_model).__name__}")
        
        if hasattr(watchtime_model, 'steps'):
            print(f"Pipeline com {len(watchtime_model.steps)} etapas:")
            for i, (nome, componente) in enumerate(watchtime_model.steps):
                print(f"  {i+1}. {nome}: {type(componente).__name__}")
            print(f"Modelo final: {type(watchtime_model.steps[-1][1]).__name__}")
        else:
            print(f"Modelo direto: {type(watchtime_model).__name__}")
    except Exception as e:
        print(f"Erro ao carregar modelo de tempo: {e}")

print("\n=== ANÁLISE DOS LOGS ===")
print("Baseado nos logs do projeto:")
print("- Experimento de Classificação executado às 03:02:26")
print("- Experimento de Regressão executado às 03:06:36")
print("- Foram testados múltiplos algoritmos usando compare_models()")