from dotenv import load_dotenv
import mlflow
import mlflow.pyfunc
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import warnings
from openai import OpenAI
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import json
import os
from datetime import datetime
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    GPT2LMHeadModel,
    GPT2Tokenizer
)

load_dotenv()
warnings.filterwarnings('ignore')

# verifica se a gpu está disponível
device = "cuda" if torch.cuda.is_available() else "cpu"

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("genai-mlflow-3.0-demo")

experiment = mlflow.get_experiment_by_name("genai-mlflow-3.0-demo")
print(" MLflow configurado!")
print(f" Experimento: {experiment.name}")
print(f" ID do Experimento: {experiment.experiment_id}")

def register_prompts_in_mlflow():
    """Registra os prompts no MLflow Prompt Registry (novidade do MLflow 3.0)"""
    try:
        mlflow.genai.register_prompt(
            name="demo_summarization",
            template="""Resuma o seguinte texto em {{style}} estilo:

    Texto: {{text}}

    Limite: {{max_words}} palavras.""",
            commit_message="Prompt de sumarização para demonstração",
            tags={"task": "summarization", "demo": "mlflow-3.0"}
        )
        print(" Prompt de Sumarização registrado")
    except Exception as e:
        print(f" Erro ao registrar sumarização: {e}")
    

    try:
        mlflow.genai.register_prompt(
            name="demo_creative_writing",
            template="""Escreva um {{genre}} sobre {{topic}}.

    Tom: {{tone}}
    Audiência: {{audience}}

    Seja criativo e engajante.""",
            commit_message="Prompt de escrita criativa para demonstração",
            tags={"task": "creative_writing", "demo": "mlflow-3.0"}
        )
        print(" Prompt de Escrita Criativa registrado")
    except Exception as e:
        print(f" Erro: {e}")

    try:
        print(" Iniciando registro de prompts...")

        mlflow.genai.register_prompt(
            name="demo_qa_assistant",
            template="""Você é um assistente especializado em {{domain}}.
            
        Contexto: {{context}}

        Pergunta: {{question}}

        Por favor, forneça uma resposta detalhada e precisa.""",
                    commit_message="Q&A Assistant para demonstração",
                    tags={"task": "question_answering", "demo": "mlflow-3.0"}
                )
        print(" Q&A Assistant registrado")
        print(f"\n Todos os 3 prompts registrados no MLflow Registry")
        return True
    
    except Exception as e:
        print(f" Prompts já registrados ou erro: {e}")
        return False


class PromptManager:
    """Gerenciador de prompts usando MLflow Prompt Registry"""
    def __init__(self):
        self.prompt_names = {
            "qa_assistant": "demo_qa_assistant",
            "summarization": "demo_summarization",
            "creative_writing": "demo_creative_writing"
        }
        print("PromptManager inicializado")

    def get_prompt(self, prompt_type: str, **kwargs) -> str:
        if prompt_type not in self.prompt_names:
            raise ValueError(f"Prompt type '{prompt_type}' não encontrado")
        try:
            prompt_name = self.prompt_names[prompt_type]
            prompt = mlflow.genai.load_prompt(f"prompts:/{prompt_name}@latest")
            
            template = prompt.to_single_brace_format()
            return template.format(**kwargs)
        except Exception as e:
            print(f"Usando template local: {e}")
            
            templates = {
                "qa_assistant":"Você é um assistente especializado em {domain}.\n\nContexto: {context}\n\nPergunta: {question}\n\nPor favor, forneça uma resposta detalhada e precisa."
            }
            return templates.get(prompt_type, "Template não encontrado").format(**kwargs)

    def log_prompt_to_mlflow(self, prompt_type: str, filled_prompt: str, run_id: str):
        mlflow.log_param("prompt_name", self.prompt_names.get(prompt_type, prompt_type))
        mlflow.log_param("prompt_type", prompt_type)
        
        prompt_data = {
            "type": prompt_type,
            "prompt_name": self.prompt_names.get(prompt_type),
            "filled_prompt": filled_prompt,
            "timestamp": datetime.now().isoformat()
        }
        
        filename = f"prompt_{prompt_type}_{run_id}.json"
        with open(filename, "w") as f:
            json.dump(prompt_data, f, indent=4)
            
        mlflow.log_artifact(filename, "prompts")
        os.remove(filename)

# prompt_manager = PromptManager()
# example_prompt = prompt_manager.get_prompt(
#     "qa_assistant",
#     domain="MLOps",
#     context="MLflow 3.0 tem recursos para GenAI",
#     question="Quais são as novidades?"
# )
# print("\n-- Prompt gerado:")
# print("-" * 50)
# print(example_prompt)
# print("-" * 50)

class GenAIEvaluator:
    """Avaliador de métricas específicas para GenAI"""
    def __init__(self):
        self.metrics_calculated = 0
        print("GenAIEvaluator inicializado")
    
    @staticmethod
    def calculate_bleu_score(reference: str, candidate: str) -> float:
        """Calcula BLEU score simplificado"""
        ref_words = reference.lower().split()
        cand_words = candidate.lower().split()
        
        if len(cand_words) == 0:
            return 0.0
            
        matches = sum(1 for word in cand_words if word in ref_words)
        precision = matches / len(cand_words)
        
        brevity_penalty = min(1.0, len(cand_words) / len(ref_words))
        
        return precision * brevity_penalty


    @staticmethod
    def calculate_rouge_l(reference: str, candidate: str) -> float:
        """Calcula ROUGE-L score simplificado"""
        ref_words = reference.lower().split()
        cand_words = candidate.lower().split()
        
        if len(ref_words) == 0 or len(cand_words) == 0:
            return 0.0
            
        lcs_length = 0
        for ref_word in ref_words:
            if ref_word in cand_words:
                lcs_length += 1
                
        precision = lcs_length / len(cand_words) if len(cand_words) > 0 else 0
        recall = lcs_length / len(ref_words) if len(ref_words) > 0 else 0
        
        if precision + recall == 0:
            return 0.0
            
        f1 = 2 * precision * recall / (precision + recall)
        return f1

    @staticmethod
    def calculate_perplexity(text: str) -> float:
        """Calcula perplexidade simulada (menor é melhor)"""
        words = text.split()
        unique_words = len(set(words))
        total_words = len(words)
        
        if total_words == 0:
            return float('inf')
            
        diversity_ratio = unique_words / total_words
        perplexity = np.exp(-np.log(diversity_ratio + 0.1))
        return float(perplexity)

    @staticmethod
    def check_toxicity(text: str) -> float:
        toxic_keywords = ["ódio", "violência", "discriminação", "insulto", "ofensa", "ofensivo"]
        text_lower = text.lower()
        
        toxic_count = sum(1 for keyword in toxic_keywords if keyword in text_lower)

        toxicity_score = min(toxic_count / len(toxic_keywords), 1.0)
        return toxicity_score


    @staticmethod
    def evaluate_response(response: str, reference: str = None) -> Dict[str, float]:
        """Avalia resposta com múltiplas métricas"""
        evaluator = GenAIEvaluator()
        
        metrics = {
            "perplexity": evaluator.calculate_perplexity(response),
            "toxicity_score": evaluator.check_toxicity(response),
            "response_length": len(response.split()),
            "unique_words": len(set(response.lower().split()))
        }
        
        if reference:
            metrics["bleu_score"] = evaluator.calculate_bleu_score(reference, response)
            metrics["rouge_l_score"] = evaluator.calculate_rouge_l(reference, response)
        
        return metrics


# evaluator = GenAIEvaluator()
# texto_teste = "MLflow 3.0 oferece recursos avançados para GenAI"
# referencia = "MLflow 3.0 tem features para modelos generativos"
# metricas = evaluator.evaluate_response(texto_teste, referencia)
# print("\n-- Métricas calculadas:")
# print("-" * 40)
# for metrica, valor in metricas.items():
#     print(f"{metrica}: {valor:.3f}")

class GenAIModel:
    """Wrapper para modelos generativos open-source com MLflow tracking"""
    def __init__(self, model_type: str = "gpt-4o-mini", device: str = None):
        self.model_type = model_type
        self.prompt_manager = PromptManager()
        self.evaluator = GenAIEvaluator()
        self.client = OpenAI()

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"✅ GenAIModel inicializado")
        print(f"📖 Modelo: {self.model_type}")
        print(f"💻 Dispositivo: {self.device}")

    def _initialize_model(self):
        """Inicializa modelo e tokenizer do Hugging Face"""
        try:
        #     test_response = self.client.chat.completions.create(
        #         model=self.model_type,
        #         messages=[{"role": "user", "content": "Teste de conexão"}],
        #         max_completion_tokens=5
        #     )
        #     print(f"✅ Conexão com OpenAI API bem-sucedida. Modelo '{self.model_type}' está acessível.")
        #     print(f"Resposta de teste: {test_response.choices[0].message.content}")
            return True
        except Exception as e:
            print(f"🚨 Erro ao conectar com a API")
            return False

    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """Gera texto usando api OpenAi"""          

        try:
            response = self.client.chat.completions.create(
                model=self.model_type,
                messages=[
                    {'role': 'system', 'content': 'Você é um assistente de IA útil e preciso. Responda brevemente, com no máximo 100 palavras'},
                    {"role": "user", "content": prompt}
                ],
                # max_completion_tokens=max_tokens,
                # temperature=temperature,
                # top_p=0.95,
            )
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"🚨 Erro na geração: {e}")
            responses = {
                "qa": "Com base no contexto, o MLflow 3.0 introduz recursos nativos para GenAI.",
                "summary": "MLflow 3.0 oferece recursos avançados para modelos generativos.",
                "creative": "Em um laboratório de IA, cientistas descobriram o MLflow 3.0..."
            }

            if "assistente" in prompt.lower():
                return responses["qa"]
            elif "resuma" in prompt.lower():
                return responses["summary"]
            else:
                return responses["creative"]
    
    def generate_with_tracking(self, prompt_type: str, **prompt_kwargs) -> Dict[str, Any]:
        """Gera texto com tracking completo no MLflow"""
        with mlflow.start_run() as run:
            mlflow.log_param("model_type", self.model_type)
            mlflow.log_param("prompt_type", prompt_type)
            mlflow.log_param("temperature", prompt_kwargs.get("temperature", 0.7))
            mlflow.log_param("max_tokens", prompt_kwargs.get("max_tokens", 150))
            
            prompt = self.prompt_manager.get_prompt(prompt_type, **prompt_kwargs)
            
            self.prompt_manager.log_prompt_to_mlflow(prompt_type, prompt, run.info.run_id)
            mlflow.log_param("prompt_length", len(prompt))

            start_time = datetime.now()
            response = self.generate(
                prompt,
                max_tokens=prompt_kwargs.get("max_tokens", 150),
                temperature=prompt_kwargs.get("temperature", 0.7)
            )
            generation_time = (datetime.now() - start_time).total_seconds()

            metrics = self.evaluator.evaluate_response(
                response,
                reference=prompt_kwargs.get("reference")
            )
            
            mlflow.log_metric("generation_time_seconds", generation_time)
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
                
            mlflow.set_tag("task", prompt_type)
            mlflow.set_tag("model_family", "generative")
            mlflow.set_tag("mlflow.version", "3.0")
            
            return {
                "run_id": run.info.run_id,
                "prompt": prompt,
                "response": response,
                "metrics": metrics,
                "generation_time": generation_time
            }

    def generate_with_tracking_nested(self, prompt_type: str, **prompt_kwargs) -> Dict[str, Any]:
        """Gera texto com tracking em run aninhado (para pipelines)"""

        run = mlflow.active_run()
        if not run:
            raise Exception("Nenhum run ativo. Use generate_with_tracking() primeiro.")

        with mlflow.start_run(nested=True) as nested_run:
            mlflow.log_param("model_type", self.model_type)
            mlflow.log_param("prompt_type", prompt_type)

            prompt = self.prompt_manager.get_prompt(prompt_type, **prompt_kwargs)
            response = self.generate(prompt)
            
            metrics = self.evaluator.evaluate_response(response)

            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            return {
                "run_id": nested_run.info.run_id,
                "response": response,
                "metrics": metrics
            }




modelo = GenAIModel(model_type='gpt-5-nano')
modelo._initialize_model()

# if modelo.model_type is not None:
#     teste_prompt = "MLflow 3.0 is"
#     resposta = modelo.generate(teste_prompt)
#     print(f"\nTeste de geração:")
#     print(f"Prompt: '{teste_prompt}'")
#     print(f"Resposta: '{resposta}'")
# else:
#     print("\nUsando modo mock (modelo não disponível)")

# resultado_qa = modelo.generate_with_tracking(
#     prompt_type="qa_assistant",
#     domain="MLOps e Machine Learning",
#     context="MLFlow 3.0 introduz recursos nativos para modelos generativos",
#     question="Quais são as principais novidades?",
#     temperature=0.7,
#     max_tokens=100
# )

# print(f"\n- PROMPT USADO:")
# print("-" * 40)
# print(resultado_qa['prompt'][:200] + "...")

# print(f"\n- RESPOSTA GERADA:")
# print("-" * 40)
# print(resultado_qa['response'])

# print(f"\n- METRICAS DE AVALIAÇÃO:")
# print("-" * 40)
# for metrica, valor in resultado_qa['metrics'].items():
#     print(f"{metrica}: {valor:.3f}")


prompt_variations = [
    {
        "version": "v1_simple",
        "domain": "machine learning",
        "context": "MLflow é uma plataforma open-source",
        "question": "O que é MLflow?",
        "temperature": 0.5
    },
    {
        "version": "v2_detailed",
        "domain": "MLOps e machine learning",
        "context": "MLflow é uma plataforma para gerenciar o ciclo de vida de ML",
        "question": "Explique os principais componentes do MLflow",
        "temperature": 0.7
    },
    {
        "version": "v3_creative",
        "domain": "Inteligência artificial e MLOps",
        "context": "MLflow 3.0 introduz recursos específicos para GenAI",
        "question": "Como o MLflow 3.0 revoluciona modelos generativos?",
        "temperature": 0.9
    }
]

print(f"Preparadas {len(prompt_variations)} variações para teste")
print("\n" + "="*60)
print("\n COMPARAÇÃO DE PROMPTS")
print("\n" + "="*60)

comparison_results = []

for variation in prompt_variations:
    print(f"\n- Testando {variation['version']}...")

    result = modelo.generate_with_tracking(
        prompt_type="qa_assistant",
        **variation
    )

    comparison_results.append({
        "version": variation["version"],
        "run_id": result["run_id"],
        "metrics": result["metrics"],
        "temperature": variation["temperature"]
    })

    print(f" ✔ Run ID: {result['run_id'][:8]}...")
    print(f" ✔ Perplexidade: {result['metrics']['perplexity']:.2f}")
    print(f" ✔ Toxicidade: {result['metrics']['toxicity_score']:.2f}")

df_resultados = pd.DataFrame(comparison_results)

print("\n Métricas por versão:")
print("\n" + "-" * 40)

for _, row in df_resultados.iterrows():
    print(f"\n{row['version']}:")
    print(f" • Temperatura: {row['temperature']}")
    print(f" • Perplexidade: {row['metrics']['perplexity']:.2f}")
    print(f" • Comprimento: {row['metrics']['response_length']} palavras")
    print(f" • Palavras únicas: {row['metrics']['unique_words']}")

melhor_perplexidade = df_resultados.loc[
    df_resultados['metrics'].apply(lambda x: x['perplexity']).idxmin()
]

print(f"\n Melhor versão (menor perplexidade): {melhor_perplexidade['version']}")
