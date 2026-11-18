"""FastAPI application for skin scan analysis."""

import json
import logging
from pathlib import Path
from typing import Any
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from .deps import get_settings, setup_logging, CORS_ORIGINS
from .schemas import ScanResponse, HealthResponse
from .utils_io import read_image_bgr

from ..pipeline.compose import run_scan
from ..pipeline.skin_tone import estimate_skin_tone

from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

settings = get_settings()
setup_logging(settings.log_level)

logger = logging.getLogger(__name__)

MAX_IMAGE_SIZE = settings.max_image_size * 1024 * 1024

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
}

app = FastAPI(
    title="Skin Scan OSS",
    description="Skin analysis and personalized recommendation API",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# RECOMMENDATION SERVICES
# --------------------------------------------------


def get_recommendation_services():
    """
    Load recommendation services only when requested.

    Create pipeline/recommendations.py and implement
    the four functions used below.
    """
    try:
        from ..pipeline.recommendations import (
            query_openbeautyfacts_safely,
            score_products,
            generate_final_report,
            build_tryon_presets,
        )

        return (
            query_openbeautyfacts_safely,
            score_products,
            generate_final_report,
            build_tryon_presets,
        )

    except ImportError as exc:
        logger.exception("Recommendation services unavailable")

        raise HTTPException(
            status_code=501,
            detail="Recommendation services are not configured.",
        ) from exc


# --------------------------------------------------
# IMAGE PROCESSING
# --------------------------------------------------


async def process_uploaded_image(image: UploadFile) -> dict[str, Any]:
    """
    Read and process an uploaded image.

    Returns the raw pipeline result, including internal
    processing data needed for skin-tone estimation.
    """

    if not image.filename:
        raise HTTPException(
            status_code=400,
            detail="Image filename is missing.",
        )

    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Unsupported image format. Use JPEG, PNG, WEBP, or BMP.",
        )

    image_data = await image.read()

    if not image_data:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty.",
        )

    if len(image_data) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large. Maximum size is {settings.max_image_size}MB.",
        )

    logger.info(
        "Received image: %s (%s bytes)",
        image.filename,
        len(image_data),
    )

    try:
        # Decode image.
        img_bgr = await run_in_threadpool(
            read_image_bgr,
            image_data,
        )

        if img_bgr is None or img_bgr.size == 0:
            raise ValueError("Unable to decode uploaded image.")

        logger.info("Image shape: %s", img_bgr.shape)

        # Run CPU-intensive scan pipeline.
        result = await run_in_threadpool(
            run_scan,
            img_bgr,
        )

        if not isinstance(result, dict):
            raise ValueError("Scan pipeline returned an invalid result.")

        if "scores" not in result:
            raise ValueError("Scan pipeline did not return scores.")

        if "overlays" not in result:
            raise ValueError("Scan pipeline did not return overlays.")

        if "regions" not in result:
            raise ValueError("Scan pipeline did not return regions.")

        logger.info("Scan completed successfully.")

        # Preserve internal data for recommendation processing.
        result["_source_image"] = img_bgr

        return result

    except HTTPException:
        raise

    except ValueError as exc:
        logger.warning("Image processing failed: %s", exc)

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Unexpected image processing error")

        raise HTTPException(
            status_code=500,
            detail="Internal server error during image analysis.",
        ) from exc

    finally:
        await image.close()


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health():
    return {"ok": True}


# --------------------------------------------------
# STANDARD SKIN SCAN
# --------------------------------------------------


@app.post("/scan", response_model=ScanResponse)
async def scan(image: UploadFile = File(...)):
    """
    Analyze an uploaded facial image.

    Returns:
        scores
        overlays
        regions

    Internal image-processing data is not exposed.
    """

    result = await process_uploaded_image(image)

    # Only return fields defined by the API response.
    return {
        "scores": result["scores"],
        "overlays": result["overlays"],
        "regions": result["regions"],
    }


# --------------------------------------------------
# SKIN SCAN + PRODUCT RECOMMENDATIONS
# --------------------------------------------------


@app.post("/scan-and-recommend")
async def scan_and_recommend(
    image: UploadFile = File(...),
    questionnaire_json: str = Form(...),
):
    """
    Analyze skin and generate personalized recommendations.

    Requires explicit biometric consent.
    """

    # 1. Validate questionnaire JSON.

    try:
        questionnaire = json.loads(questionnaire_json)

    except (json.JSONDecodeError, TypeError):
        raise HTTPException(
            status_code=400,
            detail="Invalid questionnaire JSON.",
        )

    if not isinstance(questionnaire, dict):
        raise HTTPException(
            status_code=400,
            detail="Questionnaire must be a JSON object.",
        )

    if questionnaire.get("consent_given") is not True:
        raise HTTPException(
            status_code=400,
            detail="Explicit biometric consent required.",
        )

    # 2. Run skin scan directly.
    # Do NOT call scan(image) because that is an HTTP endpoint.

    scan_result = await process_uploaded_image(image)

    # 3. Extract internal processing data.

    aligned_bgr = scan_result.get("aligned_bgr")
    cheek_mask = scan_result.get("cheek_mask")

    if aligned_bgr is None or cheek_mask is None:
        logger.error("run_scan() must return aligned_bgr and cheek_mask.")
        raise HTTPException(
            status_code=500,
            detail="Scan pipeline is missing skin-tone analysis outputs.",
        )

    if not isinstance(aligned_bgr, np.ndarray):
        raise HTTPException(
            status_code=500,
            detail="Pipeline aligned_bgr must be a NumPy array.",
        )

    if not isinstance(cheek_mask, np.ndarray):
        raise HTTPException(
            status_code=500,
            detail="Pipeline cheek_mask must be a NumPy array.",
        )

    if aligned_bgr.ndim != 3 or aligned_bgr.shape[2] != 3:
        raise HTTPException(
            status_code=500,
            detail="Pipeline aligned_bgr must be a three-channel BGR image.",
        )

    if cheek_mask.ndim == 3 and cheek_mask.shape[2] == 1:
        cheek_mask = cheek_mask[:, :, 0]

    if cheek_mask.ndim != 2 or cheek_mask.shape != aligned_bgr.shape[:2]:
        raise HTTPException(
            status_code=500,
            detail="Pipeline cheek_mask dimensions do not match aligned_bgr.",
        )

    # Estimate skin tone.
    try:
        skin_tone = await run_in_threadpool(
            estimate_skin_tone,
            aligned_bgr,
            cheek_mask,
        )
    except ValueError as exc:
        logger.warning("Invalid skin-tone input: %s", exc)
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Skin-tone estimation failed")
        raise HTTPException(
            status_code=500,
            detail="Could not estimate skin tone.",
        ) from exc

    # 5. Build user profile.

    profile = {
        "scores": scan_result["scores"],
        "regions": scan_result["regions"],
        "skin_tone": skin_tone,
        "questionnaire": questionnaire,
    }

    # 6. Load recommendation services.

    (
        query_openbeautyfacts_safely,
        score_products,
        generate_final_report,
        build_tryon_presets,
    ) = get_recommendation_services()

    # 7. Query and score products.

    try:
        candidates = await run_in_threadpool(
            query_openbeautyfacts_safely,
            profile,
        )

        top_products = await run_in_threadpool(
            score_products,
            candidates,
            profile,
        )

        report = await run_in_threadpool(
            generate_final_report,
            profile,
            top_products,
        )

        # 8. Generate virtual try-on presets.

        oiliness = float(scan_result["scores"].get("oiliness", 0))

        finish_for_skin = "matte" if oiliness > 0.6 else "natural"

        tryon_presets = await run_in_threadpool(
            build_tryon_presets,
            skin_tone,
            finish_for_skin,
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Recommendation generation failed")

        raise HTTPException(
            status_code=500,
            detail="Failed to generate skin recommendations.",
        ) from exc

    # 9. Return API response.

    return {
        "scores": scan_result["scores"],
        "overlays": scan_result["overlays"],
        "regions": scan_result["regions"],
        "skin_tone": skin_tone,
        "recommendations": top_products,
        "report_markdown": report,
        "tryon_presets": tryon_presets,
    }


# --------------------------------------------------
# WEB INTERFACE
# --------------------------------------------------

WEB_DIR = Path(__file__).resolve().parent.parent.parent / "web"


@app.get("/")
async def root():
    index_file = WEB_DIR / "index.html"

    if index_file.exists():
        return FileResponse(index_file)

    return {
        "message": "Skin Scan API is running.",
        "docs": "/docs",
    }


# --------------------------------------------------
# STATIC FILES
# --------------------------------------------------

if WEB_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(WEB_DIR)),
        name="static",
    )


# --------------------------------------------------
# STARTUP / SHUTDOWN
# --------------------------------------------------


@app.on_event("startup")
async def startup_event():
    logger.info(
        "Starting Skin Scan API in %s mode",
        settings.env,
    )


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Skin Scan API")
