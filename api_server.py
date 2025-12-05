# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import Optional, List
# import pandas as pd
# import uvicorn

# import config
# from claude_copilot import ClaudeCopilot

# app = FastAPI(title="ZeroRisk ERP Health Check API", version="1.0.0")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# copilot = None
# health_scores_df = None

# class AskRequest(BaseModel):
#     question: str
#     vs_code: Optional[str] = None
#     include_history: bool = True

# class AskResponse(BaseModel):
#     answer: str
#     vs_code: Optional[str] = None
#     sources_used: int = 0
#     model: str = ""

# class HealthScoreResponse(BaseModel):
#     vs_code: str
#     process_health: float
#     system_health: float
#     readiness_score: float
#     rag_status: str
#     kpi_count: int
#     finding_count: int
#     painpoint_count: int
#     value_at_stake_usd: float

# @app.on_event("startup")
# async def startup_event():
#     global copilot, health_scores_df
#     print("Starting ZeroRisk ERP Health Check API...")
#     copilot = ClaudeCopilot()
#     copilot.initialize()
#     try:
#         health_scores_df = pd.read_parquet(config.HEALTH_SCORES_FILE)
#     except:
#         health_scores_df = pd.DataFrame()
#     print("API ready!")

# @app.get("/health")
# async def health_check():
#     return {"status": "healthy"}

# @app.post("/api/ask", response_model=AskResponse)
# async def ask_copilot(request: AskRequest):
#     if copilot is None:
#         raise HTTPException(status_code=503, detail="Copilot not initialized")
#     result = copilot.ask(request.question, request.vs_code, request.include_history)
#     if result.get("error"):
#         raise HTTPException(status_code=500, detail=result.get("answer"))
#     return AskResponse(**{k: result.get(k, "") for k in ["answer", "vs_code", "sources_used", "model"]})

# @app.get("/api/executive-summary")
# async def get_executive_summary():
#     if copilot is None:
#         raise HTTPException(status_code=503, detail="Copilot not initialized")
#     return {"summary": copilot.get_executive_summary()}

# @app.get("/api/value-stream/{vs_code}/analysis")
# async def get_value_stream_analysis(vs_code: str):
#     if copilot is None:
#         raise HTTPException(status_code=503, detail="Copilot not initialized")
#     return {"vs_code": vs_code.upper(), "analysis": copilot.get_value_stream_analysis(vs_code.upper())}

# @app.get("/api/health-scores", response_model=List[HealthScoreResponse])
# async def get_health_scores():
#     if health_scores_df is None or len(health_scores_df) == 0:
#         raise HTTPException(status_code=404, detail="Health scores not available")
#     return health_scores_df.to_dict(orient="records")

# @app.get("/api/health-scores/{vs_code}", response_model=HealthScoreResponse)
# async def get_health_score_by_vs(vs_code: str):
#     if health_scores_df is None or len(health_scores_df) == 0:
#         raise HTTPException(status_code=404, detail="Health scores not available")
#     row = health_scores_df[health_scores_df["vs_code"] == vs_code.upper()]
#     if len(row) == 0:
#         raise HTTPException(status_code=404, detail=f"Value stream {vs_code} not found")
#     return row.iloc[0].to_dict()

# @app.get("/api/value-streams")
# async def get_value_streams():
#     vs_names = {"O2C": "Order to Cash", "P2P": "Procure to Pay", "P2M": "Plan to Manufacture", 
#                 "R2R": "Record to Report", "H2R": "Hire to Retire", "A2D": "Acquire to Decommission",
#                 "R2S": "Request to Service", "I2M": "Idea to Market", "DM": "Data Management"}
#     if health_scores_df is not None and len(health_scores_df) > 0:
#         return {"value_streams": [{"code": vs, "name": vs_names.get(vs, vs)} for vs in health_scores_df["vs_code"].unique()]}
#     return {"value_streams": [{"code": k, "name": v} for k, v in vs_names.items()]}

# @app.get("/api/kpis")
# async def get_kpis(vs_code: Optional[str] = None):
#     try:
#         kpi_df = pd.read_parquet(config.KPI_TARGETS_FILE)
#         if vs_code:
#             kpi_df = kpi_df[kpi_df["vs_code"] == vs_code.upper()]
#         return {"kpis": kpi_df.to_dict(orient="records")}
#     except Exception as e:
#         raise HTTPException(status_code=404, detail=f"KPIs not available: {e}")

# @app.post("/api/clear-history")
# async def clear_history():
#     if copilot:
#         copilot.clear_history()
#     return {"message": "History cleared"}

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)


# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import Optional, List
# import pandas as pd
# import uvicorn
# import traceback
# import logging

# import config
# from claude_copilot import ClaudeCopilot

# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = FastAPI(title="ZeroRisk ERP Health Check API", version="1.0.0")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# copilot = None
# health_scores_df = None

# class AskRequest(BaseModel):
#     question: str
#     vs_code: Optional[str] = None
#     include_history: bool = True

# class AskResponse(BaseModel):
#     answer: str
#     vs_code: Optional[str] = None
#     sources_used: int = 0
#     model: str = ""

# class HealthScoreResponse(BaseModel):
#     vs_code: str
#     process_health: float
#     system_health: float
#     readiness_score: float
#     rag_status: str
#     kpi_count: int
#     finding_count: int
#     painpoint_count: int
#     value_at_stake_usd: float

# @app.on_event("startup")
# async def startup_event():
#     global copilot, health_scores_df
#     logger.info("Starting ZeroRisk ERP Health Check API...")
    
#     # Check API key
#     logger.info(f"API Key configured: {bool(config.ANTHROPIC_API_KEY and config.ANTHROPIC_API_KEY != 'your-claude-api-key-here')}")
    
#     try:
#         copilot = ClaudeCopilot()
#         copilot.initialize()
#         logger.info("Copilot initialized successfully")
#     except Exception as e:
#         logger.error(f"Failed to initialize copilot: {e}")
#         logger.error(traceback.format_exc())
    
#     try:
#         health_scores_df = pd.read_parquet(config.HEALTH_SCORES_FILE)
#         logger.info(f"Loaded {len(health_scores_df)} health scores")
#     except Exception as e:
#         logger.warning(f"Could not load health scores: {e}")
#         health_scores_df = pd.DataFrame()
    
#     logger.info("API ready!")

# @app.get("/health")
# async def health_check():
#     return {
#         "status": "healthy",
#         "copilot_initialized": copilot is not None,
#         "api_key_set": bool(config.ANTHROPIC_API_KEY and config.ANTHROPIC_API_KEY != "your-claude-api-key-here")
#     }

# @app.post("/api/ask", response_model=AskResponse)
# async def ask_copilot(request: AskRequest):
#     logger.info(f"Received question: {request.question[:50]}...")
    
#     if copilot is None:
#         logger.error("Copilot not initialized")
#         raise HTTPException(status_code=503, detail="Copilot not initialized")
    
#     try:
#         logger.info("Calling copilot.ask()...")
#         result = copilot.ask(request.question, request.vs_code, request.include_history)
#         logger.info(f"Got result: error={result.get('error', False)}")
        
#         if result.get("error"):
#             error_msg = result.get("answer", "Unknown error")
#             logger.error(f"Copilot error: {error_msg}")
#             raise HTTPException(status_code=500, detail=error_msg)
        
#         return AskResponse(
#             answer=result.get("answer", ""),
#             vs_code=result.get("vs_code"),
#             sources_used=result.get("sources_used", 0),
#             model=result.get("model", "")
#         )
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Unexpected error in ask_copilot: {e}")
#         logger.error(traceback.format_exc())
#         raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

# @app.get("/api/executive-summary")
# async def get_executive_summary():
#     if copilot is None:
#         raise HTTPException(status_code=503, detail="Copilot not initialized")
#     try:
#         return {"summary": copilot.get_executive_summary()}
#     except Exception as e:
#         logger.error(f"Error in executive summary: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/api/value-stream/{vs_code}/analysis")
# async def get_value_stream_analysis(vs_code: str):
#     if copilot is None:
#         raise HTTPException(status_code=503, detail="Copilot not initialized")
#     try:
#         return {"vs_code": vs_code.upper(), "analysis": copilot.get_value_stream_analysis(vs_code.upper())}
#     except Exception as e:
#         logger.error(f"Error in value stream analysis: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/api/health-scores", response_model=List[HealthScoreResponse])
# async def get_health_scores():
#     if health_scores_df is None or len(health_scores_df) == 0:
#         raise HTTPException(status_code=404, detail="Health scores not available")
#     return health_scores_df.to_dict(orient="records")

# @app.get("/api/health-scores/{vs_code}", response_model=HealthScoreResponse)
# async def get_health_score_by_vs(vs_code: str):
#     if health_scores_df is None or len(health_scores_df) == 0:
#         raise HTTPException(status_code=404, detail="Health scores not available")
#     row = health_scores_df[health_scores_df["vs_code"] == vs_code.upper()]
#     if len(row) == 0:
#         raise HTTPException(status_code=404, detail=f"Value stream {vs_code} not found")
#     return row.iloc[0].to_dict()

# @app.get("/api/value-streams")
# async def get_value_streams():
#     vs_names = {"O2C": "Order to Cash", "P2P": "Procure to Pay", "P2M": "Plan to Manufacture", 
#                 "R2R": "Record to Report", "H2R": "Hire to Retire", "A2D": "Acquire to Decommission",
#                 "R2S": "Request to Service", "I2M": "Idea to Market", "DM": "Data Management"}
#     if health_scores_df is not None and len(health_scores_df) > 0:
#         return {"value_streams": [{"code": vs, "name": vs_names.get(vs, vs)} for vs in health_scores_df["vs_code"].unique()]}
#     return {"value_streams": [{"code": k, "name": v} for k, v in vs_names.items()]}

# @app.get("/api/kpis")
# async def get_kpis(vs_code: Optional[str] = None):
#     try:
#         kpi_df = pd.read_parquet(config.KPI_TARGETS_FILE)
#         if vs_code:
#             kpi_df = kpi_df[kpi_df["vs_code"] == vs_code.upper()]
#         return {"kpis": kpi_df.to_dict(orient="records")}
#     except Exception as e:
#         raise HTTPException(status_code=404, detail=f"KPIs not available: {e}")

# @app.post("/api/clear-history")
# async def clear_history():
#     if copilot:
#         copilot.clear_history()
#     return {"message": "History cleared"}

# # Debug endpoint
# @app.get("/api/debug")
# async def debug_info():
#     import os
#     return {
#         "api_key_set": bool(os.getenv("ANTHROPIC_API_KEY")),
#         "api_key_in_config": bool(config.ANTHROPIC_API_KEY and config.ANTHROPIC_API_KEY != "your-claude-api-key-here"),
#         "copilot_initialized": copilot is not None,
#         "vector_store_loaded": copilot.vector_store is not None if copilot else False,
#         "health_scores_loaded": len(health_scores_df) if health_scores_df is not None else 0
#     }

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import uvicorn
import traceback
import logging

import config
from llm_copilot import LLMCopilot

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ZeroRisk ERP Health Check API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

copilot = None
health_scores_df = None

class AskRequest(BaseModel):
    question: str
    vs_code: Optional[str] = None
    include_history: bool = True

class AskResponse(BaseModel):
    answer: str
    vs_code: Optional[str] = None
    sources_used: int = 0
    model: str = ""

class HealthScoreResponse(BaseModel):
    vs_code: str
    process_health: float
    system_health: float
    readiness_score: float
    rag_status: str
    kpi_count: int
    finding_count: int
    painpoint_count: int
    value_at_stake_usd: float

@app.on_event("startup")
async def startup_event():
    global copilot, health_scores_df
    logger.info("Starting ZeroRisk ERP Health Check API...")
    logger.info(f"API Base URL: {config.API_BASE_URL}")
    logger.info(f"LLM Model: {config.LLM_MODEL_NAME}")
    
    try:
        copilot = LLMCopilot()
        
        # Test connection first
        test_result = copilot.test_connection()
        logger.info(f"LLM connection test: {test_result}")
        
        if not test_result["success"]:
            logger.error(f"LLM connection failed: {test_result.get('error')}")
        
        copilot.initialize()
        logger.info("Copilot initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize copilot: {e}")
        logger.error(traceback.format_exc())
    
    try:
        health_scores_df = pd.read_parquet(config.HEALTH_SCORES_FILE)
        logger.info(f"Loaded {len(health_scores_df)} health scores")
    except Exception as e:
        logger.warning(f"Could not load health scores: {e}")
        health_scores_df = pd.DataFrame()
    
    logger.info("API ready!")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "api_base_url": config.API_BASE_URL,
        "llm_model": config.LLM_MODEL_NAME,
        "copilot_initialized": copilot is not None
    }

@app.get("/api/test-llm")
async def test_llm():
    """Test LLM connection."""
    if copilot is None:
        return {"success": False, "error": "Copilot not initialized"}
    return copilot.test_connection()

@app.post("/api/ask", response_model=AskResponse)
async def ask_copilot(request: AskRequest):
    logger.info(f"Received question: {request.question[:50]}...")
    
    if copilot is None:
        logger.error("Copilot not initialized")
        raise HTTPException(status_code=503, detail="Copilot not initialized")
    
    try:
        logger.info("Calling copilot.ask()...")
        result = copilot.ask(request.question, request.vs_code, request.include_history)
        logger.info(f"Got result: error={result.get('error', False)}")
        
        if result.get("error"):
            error_msg = result.get("answer", "Unknown error")
            logger.error(f"Copilot error: {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
        
        return AskResponse(
            answer=result.get("answer", ""),
            vs_code=result.get("vs_code"),
            sources_used=result.get("sources_used", 0),
            model=result.get("model", "")
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/api/executive-summary")
async def get_executive_summary():
    if copilot is None:
        raise HTTPException(status_code=503, detail="Copilot not initialized")
    try:
        return {"summary": copilot.get_executive_summary()}
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/value-stream/{vs_code}/analysis")
async def get_value_stream_analysis(vs_code: str):
    if copilot is None:
        raise HTTPException(status_code=503, detail="Copilot not initialized")
    try:
        return {"vs_code": vs_code.upper(), "analysis": copilot.get_value_stream_analysis(vs_code.upper())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health-scores", response_model=List[HealthScoreResponse])
async def get_health_scores():
    if health_scores_df is None or len(health_scores_df) == 0:
        raise HTTPException(status_code=404, detail="Health scores not available")
    return health_scores_df.to_dict(orient="records")

@app.get("/api/health-scores/{vs_code}", response_model=HealthScoreResponse)
async def get_health_score_by_vs(vs_code: str):
    if health_scores_df is None or len(health_scores_df) == 0:
        raise HTTPException(status_code=404, detail="Health scores not available")
    row = health_scores_df[health_scores_df["vs_code"] == vs_code.upper()]
    if len(row) == 0:
        raise HTTPException(status_code=404, detail=f"Value stream {vs_code} not found")
    return row.iloc[0].to_dict()

@app.get("/api/value-streams")
async def get_value_streams():
    vs_names = {"O2C": "Order to Cash", "P2P": "Procure to Pay", "P2M": "Plan to Manufacture", 
                "R2R": "Record to Report", "H2R": "Hire to Retire", "A2D": "Acquire to Decommission",
                "R2S": "Request to Service", "I2M": "Idea to Market", "DM": "Data Management"}
    if health_scores_df is not None and len(health_scores_df) > 0:
        return {"value_streams": [{"code": vs, "name": vs_names.get(vs, vs)} for vs in health_scores_df["vs_code"].unique()]}
    return {"value_streams": [{"code": k, "name": v} for k, v in vs_names.items()]}

@app.get("/api/kpis")
async def get_kpis(vs_code: Optional[str] = None):
    try:
        kpi_df = pd.read_parquet(config.KPI_TARGETS_FILE)
        if vs_code:
            kpi_df = kpi_df[kpi_df["vs_code"] == vs_code.upper()]
        return {"kpis": kpi_df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"KPIs not available: {e}")

@app.post("/api/clear-history")
async def clear_history():
    if copilot:
        copilot.clear_history()
    return {"message": "History cleared"}

@app.get("/api/debug")
async def debug_info():
    return {
        "api_base_url": config.API_BASE_URL,
        "llm_model": config.LLM_MODEL_NAME,
        "embedding_model": config.EMBEDDING_MODEL_NAME,
        "embedding_dim": config.EMBEDDING_DIMENSIONS,
        "copilot_initialized": copilot is not None,
        "vector_store_loaded": copilot.vector_store is not None if copilot else False,
        "health_scores_count": len(health_scores_df) if health_scores_df is not None else 0
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)