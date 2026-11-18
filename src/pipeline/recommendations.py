# obf_text2sql_pipeline.py

import json
import os
import re

import duckdb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DB_PATH = os.getenv("DB_PATH", "openbeautyfacts.duckdb")

# Supports Groq or any OpenAI-compatible endpoint.
# LLM_* takes priority, with OPENAI_* retained for compatibility.
BASE_URL = (
    os.getenv("LLM_BASE_URL")
    or os.getenv("OPENAI_BASE_URL")
    or "https://api.groq.com/openai/v1"
)

API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "ollama"

MODEL = os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or "llama-3.3-70b-versatile"

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)

TABLE_NAME = "products"


# --------------------------------------------------
# EXAMPLE PROFILE
# --------------------------------------------------

scan_and_user_profile = {
    "scores": {
        "redness": 0.81,
        "oiliness": 0.74,
        "texture": 0.62,
        "pores": 0.48,
        "blemishes": 0.36,
        "hydration": 0.29,
        "pigment": 0.97,
    },
    "regions": ["forehead", "nose", "cheeks", "chin"],
    "skin_tone": {
        "L": 62.4,
        "a": 11.2,
        "b": 18.5,
        "undertone": "warm",
        "shade_hex": "#c89779",
    },
    "questionnaire": {
        "consent_given": True,
        "skin_type": "oily_dehydrated",
        "known_allergens": [
            "fragrance",
            "parfum",
            "rose",
            "linalool",
        ],
        "pregnant_or_nursing": False,
        "target_category": "skincare",
        "preferred_labels": ["vegan"],
    },
}


# --------------------------------------------------
# INGREDIENT SCORING RULES
# --------------------------------------------------

BENEFICIAL_RULES = {
    "redness": [
        "centella",
        "madecassoside",
        "panthenol",
        "allantoin",
        "ceramide",
        "bisabolol",
        "colloidal oatmeal",
    ],
    "oiliness": [
        "niacinamide",
        "salicylic acid",
        "zinc pca",
        "kaolin",
        "silica",
    ],
    "texture": [
        "lactic acid",
        "gluconolactone",
        "niacinamide",
        "panthenol",
    ],
    "pores": [
        "salicylic acid",
        "niacinamide",
        "zinc",
    ],
    "blemishes": [
        "salicylic acid",
        "azelaic acid",
        "niacinamide",
        "zinc",
    ],
    "hydration": [
        "hyaluronic acid",
        "sodium hyaluronate",
        "glycerin",
        "squalane",
        "ceramide",
        "jojoba",
    ],
    "pigment": [
        "niacinamide",
        "vitamin c",
        "ascorbic acid",
        "tocopherol",
        "vitamin-e",
        "arbutin",
        "azelaic acid",
    ],
}

CAUTION_INGREDIENTS = [
    "alcohol denat",
    "isopropyl alcohol",
    "sodium lauryl sulfate",
    "sodium hydroxide",
]

PREGNANCY_EXCLUSIONS = [
    "retinol",
    "retinal",
    "retinyl",
    "tretinoin",
    "adapalene",
]


# --------------------------------------------------
# CATEGORY SEARCH TERMS
# --------------------------------------------------

CATEGORY_TERMS = {
    "skincare": [
        "skincare",
        "skin care",
        "face cream",
        "facial cream",
        "moisturizer",
        "moisturiser",
        "serum",
        "face serum",
        "face lotion",
        "cleanser",
        "sunscreen",
        "sun cream",
        "face mask",
    ],
    "moisturizers": [
        "moisturizer",
        "moisturiser",
        "face cream",
        "facial cream",
        "hydrating cream",
    ],
    "makeup": [
        "makeup",
        "cosmetics",
        "foundation",
        "lipstick",
        "blush",
        "concealer",
        "mascara",
    ],
    "soaps": [
        "soap",
        "body wash",
        "shower gel",
        "cleanser",
    ],
    "foundation": [
        "foundation",
        "makeup",
        "face makeup",
    ],
}


# --------------------------------------------------
# DATABASE SCHEMA HELPERS
# --------------------------------------------------


def get_product_columns(con):
    """
    Read the actual products table schema from DuckDB.
    """
    rows = con.execute(f"PRAGMA table_info('{TABLE_NAME}')").fetchall()

    if not rows:
        raise RuntimeError(f"Table '{TABLE_NAME}' was not found in {DB_PATH}")

    return {row[1].lower(): row[1] for row in rows}


def quote_identifier(name):
    """
    Safely quote a database column identifier.
    """
    return '"' + name.replace('"', '""') + '"'


def find_column(columns, candidates):
    """
    Return the first existing column from a candidate list.
    """
    for candidate in candidates:
        if candidate.lower() in columns:
            return columns[candidate.lower()]

    return None


def text_expression(column):
    """
    Convert an existing column, including JSON columns, to text.
    """
    if not column:
        return "''"

    return f"COALESCE(CAST({quote_identifier(column)} AS VARCHAR), '')"


def combine_columns(columns, candidates):
    """
    Combine available columns into one searchable text expression.
    """
    existing = [columns[name.lower()] for name in candidates if name.lower() in columns]

    if not existing:
        return "''"

    expressions = [text_expression(column) for column in existing]

    return "concat_ws(' ', " + ", ".join(expressions) + ")"


# --------------------------------------------------
# BUILD QUERY AGAINST THE REAL OBF SCHEMA
# --------------------------------------------------


def build_product_query(profile, columns):
    """
    Build deterministic DuckDB SQL using the real products schema.
    No LLM-generated SQL is executed.
    """

    code_col = find_column(
        columns,
        ["_id", "code", "id"],
    )

    name_col = find_column(
        columns,
        [
            "product_name_en",
            "product_name",
            "product_name_fr",
        ],
    )

    brand_col = find_column(
        columns,
        ["brands", "brands_tags"],
    )

    completeness_col = find_column(
        columns,
        ["completeness"],
    )

    link_col = find_column(
        columns,
        ["url", "link"],
    )

    if not code_col:
        raise RuntimeError("Could not find a product ID column " "(_id, code, or id).")

    if not name_col:
        raise RuntimeError("Could not find a product name column.")

    # Real OpenBeautyFacts ingredient columns.
    ingredients_expr = combine_columns(
        columns,
        [
            "ingredients_text_with_allergens",
            "ingredients_text",
            "ingredients_text_en",
            "ingredients_text_fr",
        ],
    )

    ingredient_tags_expr = combine_columns(
        columns,
        [
            "ingredients_original_tags",
            "ingredients_tags",
        ],
    )

    allergens_expr = combine_columns(
        columns,
        [
            "allergens_tags",
            "allergens",
        ],
    )

    traces_expr = combine_columns(
        columns,
        [
            "traces_tags",
            "traces",
        ],
    )

    categories_expr = combine_columns(
        columns,
        [
            "categories_tags",
            "categories",
            "categories_en",
            "categories_raw",
            "pnns_groups_1",
            "pnns_groups_2",
        ],
    )

    labels_expr = combine_columns(
        columns,
        [
            "labels_tags",
            "labels",
        ],
    )

    # Target category.
    questionnaire = profile.get("questionnaire", {})
    target_category = questionnaire.get(
        "target_category",
        "skincare",
    ).lower()

    category_terms = CATEGORY_TERMS.get(
        target_category,
        [target_category],
    )

    where_clauses = []
    params = []

    # Search category fields.
    if categories_expr != "''":
        category_conditions = []

        for term in category_terms:
            category_conditions.append(f"({categories_expr} ILIKE ?)")
            params.append(f"%{term}%")

        where_clauses.append("(" + " OR ".join(category_conditions) + ")")

    # Search for ingredients associated with active skin concerns.
    scores = profile.get("scores", {})

    active_concerns = [
        concern
        for concern, score in scores.items()
        if (concern != "hydration" and score >= 0.45)
        or (concern == "hydration" and score <= 0.40)
    ]

    ingredient_terms = []

    for concern in active_concerns:
        ingredient_terms.extend(BENEFICIAL_RULES.get(concern, []))

    ingredient_terms = list(dict.fromkeys(ingredient_terms))

    if ingredient_terms:
        ingredient_conditions = []

        for term in ingredient_terms:
            ingredient_conditions.append(
                f"({ingredients_expr} ILIKE ? " f"OR {ingredient_tags_expr} ILIKE ?)"
            )

            params.extend(
                [
                    f"%{term}%",
                    f"%{term}%",
                ]
            )

        ingredient_filter = "(" + " OR ".join(ingredient_conditions) + ")"

        # Category and ingredient relevance.
        if where_clauses:
            where_clauses.append(ingredient_filter)

    # Allergen and trace fields are included in one searchable expression.
    safety_expr = (
        "concat_ws(' ', "
        f"{ingredients_expr}, "
        f"{ingredient_tags_expr}, "
        f"{allergens_expr}, "
        f"{traces_expr})"
    )

    allergens = [
        str(item).strip().lower()
        for item in questionnaire.get("known_allergens", [])
        if str(item).strip()
    ]

    for allergen in allergens:
        where_clauses.append(f"({safety_expr} NOT ILIKE ?)")
        params.append(f"%{allergen}%")

    # Pregnancy/nursing exclusion.
    if questionnaire.get("pregnant_or_nursing"):
        for ingredient in PREGNANCY_EXCLUSIONS:
            where_clauses.append(f"({safety_expr} NOT ILIKE ?)")
            params.append(f"%{ingredient}%")

    where_sql = " AND ".join(where_clauses) if where_clauses else "TRUE"

    completeness_expr = text_expression(completeness_col) if completeness_col else "0"

    brand_expr = text_expression(brand_col)
    link_expr = text_expression(link_col)

    query = f"""
        SELECT
            {text_expression(code_col)} AS code,
            {text_expression(name_col)} AS product_name,
            {brand_expr} AS brands,
            {categories_expr} AS categories_tags_str,
            {ingredients_expr} AS ingredients_norm,
            {ingredient_tags_expr} AS ingredients_tags_str,
            {labels_expr} AS labels_tags_str,
            TRY_CAST({completeness_expr} AS DOUBLE) AS completeness,
            {link_expr} AS link
        FROM {quote_identifier(TABLE_NAME)}
        WHERE {where_sql}
        ORDER BY completeness DESC NULLS LAST
        LIMIT 40
    """

    return query, params


# --------------------------------------------------
# SAFE PRODUCT QUERY
# --------------------------------------------------


def query_openbeautyfacts_safely(profile):
    """
    Query the real OpenBeautyFacts products table.

    All filtering is performed by parameterized DuckDB SQL.
    """

    if not profile.get("questionnaire", {}).get("consent_given"):
        raise ValueError("User consent is required before processing skin data.")

    con = duckdb.connect(
        DB_PATH,
        read_only=True,
    )

    try:
        columns = get_product_columns(con)

        print("Detected products table columns:", len(columns))

        query, params = build_product_query(
            profile,
            columns,
        )

        print("\nExecuting SQL:\n", query)
        print("\nSQL parameters:", params)

        rows = con.execute(
            query,
            params,
        ).fetchall()

        result_columns = [
            "code",
            "product_name",
            "brands",
            "categories_tags_str",
            "ingredients_norm",
            "ingredients_tags_str",
            "labels_tags_str",
            "completeness",
            "link",
        ]

        results = []

        for row in rows:
            item = dict(zip(result_columns, row))

            item["completeness"] = item["completeness"] or 0.0

            results.append(item)

        print(f"Found {len(results)} candidate products.")

        return results

    finally:
        con.close()


# --------------------------------------------------
# DETERMINISTIC PRODUCT SCORING
# --------------------------------------------------


def score_products(candidates, profile):
    """
    Score products using ingredient matches and skin-scan metrics.
    """

    scores = profile["scores"]

    active_concerns = [
        concern
        for concern, value in scores.items()
        if (concern != "hydration" and value >= 0.40)
        or (concern == "hydration" and value <= 0.45)
    ]

    scored = []

    for item in candidates:
        haystack = (
            f"{item.get('ingredients_norm', '')} "
            f"{item.get('ingredients_tags_str', '')}"
        ).lower()

        rule_score = 0.0
        matched_beneficial = {}
        matched_cautions = []

        for concern in active_concerns:
            weight = (
                1.0 - scores[concern] if concern == "hydration" else scores[concern]
            )

            hits = [
                ingredient
                for ingredient in BENEFICIAL_RULES.get(concern, [])
                if ingredient in haystack
            ]

            if hits:
                matched_beneficial[concern] = hits

                rule_score += len(hits) * (1.0 + weight)

        for ingredient in CAUTION_INGREDIENTS:
            if ingredient in haystack:
                matched_cautions.append(ingredient)
                rule_score -= 1.5

        scored.append(
            {
                **item,
                "rule_score": round(rule_score, 2),
                "matched_beneficial": matched_beneficial,
                "matched_cautions": matched_cautions,
            }
        )

    scored.sort(
        key=lambda item: (
            item["rule_score"],
            item["completeness"],
        ),
        reverse=True,
    )

    return scored[:5]


# --------------------------------------------------
# VIRTUAL TRY-ON PRESETS
# --------------------------------------------------


def build_tryon_presets(skin_tone, finish):
    """
    Build OpenMakeupSDK presets from detected undertone.
    """

    lip_by_undertone = {
        "warm": {
            "color": "#b84336",
            "finish": "glossy",
        },
        "cool": {
            "color": "#9e1b44",
            "finish": "matte",
        },
        "neutral": {
            "color": "#a8384b",
            "finish": "shimmer",
        },
    }

    blush_by_undertone = {
        "warm": {
            "color": "#e07a5f",
            "finish": "matte",
        },
        "cool": {
            "color": "#d86c8a",
            "finish": "matte",
        },
        "neutral": {
            "color": "#d97b73",
            "finish": "shimmer",
        },
    }

    undertone = skin_tone.get(
        "undertone",
        "neutral",
    )

    return [
        {
            "label": f"Foundation ({undertone})",
            "category": "foundation",
            "opts": {
                "color": skin_tone.get(
                    "shade_hex",
                    "#c89779",
                ),
                "finish": finish,
            },
        },
        {
            "label": f"Lipstick ({undertone})",
            "category": "lipstick",
            "opts": lip_by_undertone.get(
                undertone,
                lip_by_undertone["neutral"],
            ),
        },
        {
            "label": f"Blush ({undertone})",
            "category": "blush",
            "opts": blush_by_undertone.get(
                undertone,
                blush_by_undertone["neutral"],
            ),
        },
    ]


# --------------------------------------------------
# FINAL RECOMMENDATION REPORT
# --------------------------------------------------


def generate_final_report(profile, top_products):
    """
    Generate a readable cosmetic recommendation report.
    """

    prompt = f"""
Write a clear cosmetic recommendation report.

Use ONLY the verified scan metrics and products below.

Skin scan and questionnaire:
{json.dumps(profile, indent=2)}

Verified products:
{json.dumps(top_products, indent=2)}

Include:

1. Plain-language summary of all seven skin metrics.
2. For each product:
   - Product name, brand and code.
   - Why it matches based on matched_beneficial.
   - Any matched_cautions.
3. Explain that products containing the user-listed
   allergen terms were excluded from the database results.
4. Mention detected undertone:
   {profile.get('skin_tone', {}).get('undertone', 'unknown')}
5. Mention the detected shade:
   {profile.get('skin_tone', {}).get('shade_hex', 'unknown')}
6. Invite the user to preview shades using Virtual Try-On.
7. Include this disclaimer:

   Cosmetic guidance only, not a medical diagnosis.
   Consult a dermatologist for persistent skin concerns.

Do not invent product details, ingredients or medical claims.

Important:
The database text search cannot guarantee that a product
is allergen-free. Explain that users should verify the
complete ingredient list and packaging.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful cosmetic product "
                    "recommendation assistant. "
                    "Be accurate and avoid medical claims."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
        max_tokens=1500,
    )

    return (response.choices[0].message.content or "").strip()


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    candidates = query_openbeautyfacts_safely(scan_and_user_profile)

    top_products = score_products(
        candidates,
        scan_and_user_profile,
    )

    print("\nTOP PRODUCTS:")
    print(
        json.dumps(
            top_products,
            indent=2,
            ensure_ascii=False,
        )
    )

    report = generate_final_report(
        scan_and_user_profile,
        top_products,
    )

    print("\nFINAL REPORT:")
    print(report)

    presets = build_tryon_presets(
        scan_and_user_profile["skin_tone"],
        finish="natural",
    )

    print("\nTRY-ON PRESETS:")
    print(
        json.dumps(
            presets,
            indent=2,
        )
    )
