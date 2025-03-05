from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import json
import logging
import io
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from services.sensitivity_check import sensitivity_check
from services.toxicity_check import toxicity_check
from services.hallucination_check import HallucinationChecker
from services.politeness_check import politeness_check
from services.bias_check import bias_check
from services.pii_check import pii_check
import plotly.graph_objects as go
from mangum import Mangum

# Logging setup
logging.basicConfig(level=logging.DEBUG)

app = FastAPI(title="AI Guardrails Testing API")

# ✅ Configure CORS correctly for security
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Restrict to React frontend only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# @app.get("/")
# def read_root():
#     return {"message": "Connected to FastAPI Backend"}  # ✅ Corrected message

handler = Mangum(app)

analysis_history = [] 

# Data models
class PromptRequest(BaseModel):
    prompt: str
    response: str
    enabled_checks: Dict[str, bool]
    context: Optional[str] = ""
    threshold: Optional[float] = 0.7

class ScoreResult(BaseModel):
    result: str
    score: float

class AnalysisResponse(BaseModel):
    results: Dict[str, Dict[str, ScoreResult]]

class BatchAnalysisResponse(BaseModel):
    prompt: str
    response: str
    timestamp: str
    results: Dict[str, Dict[str, ScoreResult]]

# Guardrail Analyzer
class GuardrailAnalyzer:
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def run_selected_checks(self, text: str, enabled_checks: Dict[str, bool], context: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        results = {}
        check_mapping = {
            "toxicity": (toxicity_check, "Toxicity"),
            "bias": (bias_check, "Bias"),
            "sensitivity": (sensitivity_check, "Sensitivity"),
            "politeness": (politeness_check, "Politeness"),
            "pii": (pii_check, "PII"),
        }

        for check_name, (check_func, display_name) in check_mapping.items():
            if enabled_checks.get(check_name, False):
                result, score = check_func(text)
                results[display_name] = {"result": result, "score": round(float(score), 2)}
        
        if enabled_checks.get("hallucination", False) and context:
            checker = HallucinationChecker()
            hallucination_result, hallucination_score = checker.check_hallucination(text, context)
            results["Hallucination"] = {"result": hallucination_result, "score": round(float(hallucination_score), 2)}
        elif enabled_checks.get("hallucination", False) and not context:
            raise HTTPException(status_code=400, detail="Context is required for hallucination check.")

        return results

    def create_radar_chart(self, prompt_results: Dict[str, Dict[str, float]], response_results: Dict[str, Dict[str, float]], title: str) -> str:
        categories = list(prompt_results.keys())  
        prompt_values = [prompt_results[key]['score'] for key in categories]
        response_values = [response_results[key]['score'] for key in categories]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=prompt_values, theta=categories, fill='toself', name='Prompt Scores'))
        fig.add_trace(go.Scatterpolar(r=response_values, theta=categories, fill='toself', name='Response Scores'))

        fig.update_layout(title=title, polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)

        return fig.to_html()

analyzer = GuardrailAnalyzer()

@app.post("/analyze", response_model=AnalysisResponse)
def analyze_prompt(request: PromptRequest):
    context = request.context if request.context else ""

    if not request.prompt or not request.response:
        raise HTTPException(status_code=400, detail="Both prompt and response are required.")

    if request.enabled_checks.get("hallucination", False):
        logging.debug(f"Context received for hallucination check: '{context}'")
        if not context.strip():
            raise HTTPException(status_code=400, detail="Valid context is required for hallucination check.")

    prompt_results = analyzer.run_selected_checks(request.prompt, request.enabled_checks, context=context)
    response_results = analyzer.run_selected_checks(request.response, request.enabled_checks, context=context)

    return AnalysisResponse(
        results={
            "Prompt": {key: ScoreResult(**value) for key, value in prompt_results.items()},
            "Response": {key: ScoreResult(**value) for key, value in response_results.items()},
        },
    )

@app.post("/batch_analyze")
def batch_analyze(file: UploadFile = File(...), enabled_checks: str = Form(...)):
    try:
        df = pd.read_excel(file.file)

        required_columns = {"Model Input", "Model Output"}
        if not required_columns.issubset(df.columns):
            raise HTTPException(
                status_code=400,
                detail=f"File must contain required columns: {required_columns}",
            )

        enabled_checks = json.loads(enabled_checks)
        results = []

        for _, row in df.iterrows():
            prompt = row["Model Input"]
            response = row["Model Output"]
            context = str(row.get("Model Retrieved Context", "")).strip()

            if enabled_checks.get("hallucination", False) and not context:
                raise HTTPException(
                    status_code=400,
                    detail="Valid context is required for hallucination check."
                )

            prompt_results = analyzer.run_selected_checks(prompt, enabled_checks, context=context)
            response_results = analyzer.run_selected_checks(response, enabled_checks, context=context)


            formatted_result = {
                "prompt": prompt,
                "response": response,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),  # ✅ Correct timestamp format
                "results": {
                        "prompt": {key: {"result": value["result"], "score": value["score"]} for key, value in prompt_results.items()},
                        "response": {key: {"result": value["result"], "score": value["score"]} for key, value in response_results.items()}
                    }
            }

            results.append(formatted_result)

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
        
@app.get("/history")
def get_history():
    """
    Retrieve analysis history.
    """
    if not analysis_history:
        return {"message": "No analysis history available."}

    history_df = pd.DataFrame([
        {
            "Timestamp": entry["timestamp"],
            "Prompt": entry["prompt"],
            "Response": entry["response"],
            **{
                f"Prompt_{k}_Score": v["score"] for k, v in entry.get("prompt_results", {}).items()
            },
            **{
                f"Prompt_{k}_Result": v["result"] for k, v in entry.get("prompt_results", {}).items()
            },
            **{
                f"Response_{k}_Score": v["score"] for k, v in entry.get("response_results", {}).items()
            },
            **{
                f"Response_{k}_Result": v["result"] for k, v in entry.get("response_results", {}).items()
            },
        }
        for entry in analysis_history
    ])

    return history_df.to_dict(orient="records")

# if __name__ == '__main__':
#     uvicorn.run(app, port=8080, host='0.0.0.0')
@app.get("/dashboard_metrics")
def get_dashboard_metrics():
    """
    Retrieve overall metrics for the dashboard.
    """
    if not analysis_history:
        return {"message": "No analysis history available."}
 
    total_prompts = len(analysis_history)
    total_evaluations = total_prompts * 2  # Since each prompt has a response
 
    # Aggregate scores per category
    categories = ["Toxicity", "Bias", "Sensitivity", "Politeness", "PII", "Hallucination"]
    category_counts = {cat: 0 for cat in categories}
    total_scores = {cat: 0 for cat in categories}
    count_scores = {cat: 0 for cat in categories}
 
    for entry in analysis_history:
        for cat in categories:
            if cat in entry["prompt_results"]:
                total_scores[cat] += entry["prompt_results"][cat]["score"]
                count_scores[cat] += 1
                category_counts[cat] += 1
 
            if cat in entry["response_results"]:
                total_scores[cat] += entry["response_results"][cat]["score"]
                count_scores[cat] += 1
                category_counts[cat] += 1
 
    avg_scores = {cat: (total_scores[cat] / count_scores[cat]) if count_scores[cat] > 0 else 0 for cat in categories}
 
    return {
        "total_prompts": total_prompts,
        "total_evaluations": total_evaluations,
        "avg_scores": avg_scores,
        "most_common_issues": sorted(category_counts, key=category_counts.get, reverse=True)[:3]
    }