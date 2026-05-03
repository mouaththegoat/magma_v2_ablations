"""Safe literature-research helpers for MAGMA v2.

This module intentionally exposes narrow HTTP-provider adapters instead of
shell/curl/free-form browsing. The literature worker can retrieve metadata and
write a canonical research contract, but it cannot execute arbitrary commands
or read/write arbitrary project files.
"""

from __future__ import annotations

import json
import re
import hashlib
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from typing import Any


LITERATURE_TRIGGER_KEYWORDS = (
    "literature",
    "sota",
    "state of the art",
    "latest",
    "paper",
    "evidence",
    "foundation model",
    "pretrained",
    "transfer learning",
    "multimodal",
    "fusion",
)

MODALITY_WORDS = ("text", "ehr", "tabular", "cxr", "x-ray", "xray", "image", "ecg", "waveform")


def should_run_literature_stage(user_request: str, data_handoff: dict[str, Any] | None = None) -> tuple[bool, list[str]]:
    """Return whether a research pass is useful and why.

    This is advisory. The research stage is non-blocking and should not be a
    universal hard gate for ordinary baseline runs.
    """
    reasons: list[str] = []
    request = user_request.lower()
    matched_keywords = [word for word in LITERATURE_TRIGGER_KEYWORDS if word in request]
    if matched_keywords:
        reasons.append(f"request mentions research/modeling evidence terms: {', '.join(sorted(set(matched_keywords)))}")

    request_modalities = [word for word in MODALITY_WORDS if word in request]
    if len(set(request_modalities)) > 1:
        reasons.append(f"request references multiple modality terms: {', '.join(sorted(set(request_modalities)))}")

    handoff = data_handoff or {}
    normalized_layout = str(handoff.get("normalized_layout") or "")
    observed_layout = str(handoff.get("observed_layout") or "")
    modalities = handoff.get("modalities") or []
    if normalized_layout == "multimodal_alignment_manifest" or observed_layout == "multimodal_joined_inputs":
        reasons.append(f"data handoff indicates multimodal layout: {normalized_layout or observed_layout}")
    if isinstance(modalities, list) and len({str(item) for item in modalities}) > 1:
        reasons.append(f"data handoff lists multiple modalities: {', '.join(map(str, modalities))}")

    return bool(reasons), reasons


def compact_research_context(task_input: dict[str, Any] | None, data_handoff: dict[str, Any] | None) -> dict[str, Any]:
    """Return only bounded, research-relevant context from canonical artifacts."""
    request = str((task_input or {}).get("raw_prompt") or (task_input or {}).get("request") or "")
    handoff = data_handoff or {}
    should_run, reasons = should_run_literature_stage(request, handoff)
    return {
        "user_request": request[:4000],
        "trigger_recommended": should_run,
        "trigger_reasons": reasons,
        "data_semantics": {
            "observed_layout": handoff.get("observed_layout"),
            "normalized_layout": handoff.get("normalized_layout"),
            "modalities": handoff.get("modalities"),
            "task_type": handoff.get("task_type"),
            "prediction_unit": handoff.get("prediction_unit"),
            "target": handoff.get("target"),
            "label_spec": handoff.get("label_spec"),
            "split_source": handoff.get("split_source"),
            "limitations": handoff.get("limitations"),
            "unresolved_questions": handoff.get("unresolved_questions"),
        },
        "data_handoff_fingerprint": handoff_fingerprint(handoff),
    }


def search_literature_providers(query_plan: dict[str, Any], max_results: int = 8) -> dict[str, Any]:
    """Search bounded biomedical/ML metadata providers.

    Providers are fixed allowlist adapters. The caller supplies semantic query
    terms, not URLs. Network failures become warnings so research never blocks
    the pipeline.
    """
    max_results = max(1, min(int(max_results or 8), 20))
    queries = build_queries(query_plan)
    provider_results: list[dict[str, Any]] = []
    warnings: list[str] = []
    providers = (_search_europe_pmc, _search_openalex, _search_semantic_scholar, _search_arxiv, _search_crossref)

    per_query_limit = max(2, min(6, max_results))
    for query in queries:
        for provider in providers:
            try:
                provider_results.extend(provider(query, max_results=per_query_limit))
            except Exception as exc:
                warnings.append(f"{provider.__name__} failed for query '{query[:80]}': {exc}")

    papers = rank_papers(dedupe_papers(provider_results), query_plan)[:max_results]
    return {
        "status": "success",
        "query": queries[0],
        "queries": queries,
        "providers": ["europe_pmc", "openalex", "semantic_scholar", "arxiv", "crossref"],
        "papers": papers,
        "warnings": warnings,
        "retrieved_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }


def build_queries(query_plan: dict[str, Any]) -> list[str]:
    explicit = query_plan.get("queries")
    if isinstance(explicit, list):
        queries = [str(item).strip() for item in explicit if str(item).strip()]
        if queries:
            return [query[:500] for query in queries[:5]]

    task = str(query_plan.get("task") or query_plan.get("clinical_task") or "clinical prediction model")
    modalities = [str(item) for item in (query_plan.get("modalities") or []) if item]
    modality_text = " ".join(modalities)
    base = build_query(query_plan)
    variants = [base]
    if len(modalities) > 1:
        variants.extend([
            f"{task} {modality_text} multimodal fusion deep learning clinical prediction",
            f"{task} {modality_text} late fusion early fusion calibration clinical machine learning",
            f"{task} medical image text fusion clinical prediction external validation calibration",
        ])
    return list(dict.fromkeys(re.sub(r"\s+", " ", query).strip()[:500] for query in variants if query.strip()))[:5]


def build_query(query_plan: dict[str, Any]) -> str:
    task = str(query_plan.get("task") or query_plan.get("clinical_task") or "clinical prediction model")
    modalities = query_plan.get("modalities") or []
    modality_text = " ".join(str(item) for item in modalities if item)
    methods = query_plan.get("method_terms") or []
    method_text = " ".join(str(item) for item in methods if item)
    extra = str(query_plan.get("extra_terms") or "")
    query = f"{task} {modality_text} {method_text} {extra} machine learning clinical prediction calibration validation"
    query = re.sub(r"\s+", " ", query).strip()
    return query[:500]


def handoff_fingerprint(payload: dict[str, Any] | None) -> str | None:
    if not isinstance(payload, dict):
        return None
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def research_is_stale(research_handoff: dict[str, Any] | None, data_handoff: dict[str, Any] | None) -> bool:
    if not isinstance(research_handoff, dict):
        return False
    expected = handoff_fingerprint(data_handoff)
    recorded = ((research_handoff.get("context") or {}).get("data_handoff_fingerprint"))
    return bool(expected and recorded and expected != recorded)


def dedupe_papers(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for paper in papers:
        key = str(paper.get("doi") or paper.get("pmid") or paper.get("arxiv_id") or paper.get("title") or "").lower()
        key = re.sub(r"\W+", "", key)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(paper)
    return unique


def rank_papers(papers: list[dict[str, Any]], query_plan: dict[str, Any]) -> list[dict[str, Any]]:
    terms = _relevance_terms(query_plan)
    for paper in papers:
        paper["relevance_score"] = _paper_relevance_score(paper, terms)
    return sorted(
        papers,
        key=lambda item: (
            float(item.get("relevance_score") or 0),
            int(item.get("citation_count") or 0) if str(item.get("citation_count") or "").isdigit() else 0,
            int(item.get("year") or 0) if str(item.get("year") or "").isdigit() else 0,
        ),
        reverse=True,
    )


def _relevance_terms(query_plan: dict[str, Any]) -> set[str]:
    raw_terms: list[str] = []
    raw_terms.append(str(query_plan.get("task") or query_plan.get("clinical_task") or ""))
    raw_terms.extend(str(item) for item in (query_plan.get("modalities") or []))
    raw_terms.extend(str(item) for item in (query_plan.get("method_terms") or []))
    raw_terms.extend(str(item) for item in (query_plan.get("must_include_terms") or []))
    tokens = set(re.findall(r"[a-zA-Z][a-zA-Z0-9+-]{2,}", " ".join(raw_terms).lower()))
    return tokens | {"multimodal", "fusion", "clinical", "prediction", "calibration", "validation"}


def _paper_relevance_score(paper: dict[str, Any], terms: set[str]) -> float:
    text = " ".join(str(paper.get(key) or "") for key in ("title", "abstract", "venue", "source")).lower()
    if not text:
        return 0.0
    hits = sum(1 for term in terms if term in text)
    score = hits / max(1, len(terms))
    if "multimodal" in text or "fusion" in text:
        score += 0.15
    if "clinical" in text or "medical" in text or "patient" in text:
        score += 0.05
    return round(min(score, 1.0), 4)


def normalize_research_handoff(payload: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Normalize a model-supplied research handoff into a stable contract."""
    now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    papers = payload.get("papers") if isinstance(payload.get("papers"), list) else []
    method_cards = payload.get("method_cards") if isinstance(payload.get("method_cards"), list) else []
    recommendations = payload.get("recommendation") or payload.get("recommended_primary_method") or {}
    if isinstance(recommendations, str):
        recommendations = {"summary": recommendations}

    handoff = {
        "status": payload.get("status") or "available",
        "scope": payload.get("scope") or "non_blocking_literature_context",
        "query_plan": _json_object(payload.get("query_plan")),
        "papers": [_normalize_paper(item) for item in papers if isinstance(item, dict)],
        "method_cards": [_normalize_method_card(item, index) for index, item in enumerate(method_cards) if isinstance(item, dict)],
        "sota_summary_by_modality": _json_object(payload.get("sota_summary_by_modality")),
        "recommended_baselines": payload.get("recommended_baselines") if isinstance(payload.get("recommended_baselines"), list) else [],
        "recommendation": _json_object(recommendations),
        "evidence_gaps": payload.get("evidence_gaps") if isinstance(payload.get("evidence_gaps"), list) else [],
        "limitations": payload.get("limitations") if isinstance(payload.get("limitations"), list) else [],
        "quality_gate": _json_object(payload.get("quality_gate")),
        "freshness_stamp": {
            **_json_object(payload.get("freshness_stamp")),
            "retrieved_at": _json_object(payload.get("freshness_stamp")).get("retrieved_at") or now,
        },
        "context": context or payload.get("context") or {},
    }
    handoff["quality_gate"] = {
        "status": handoff["quality_gate"].get("status") or _quality_status(handoff),
        "warnings": handoff["quality_gate"].get("warnings") or _quality_warnings(handoff),
        **{k: v for k, v in handoff["quality_gate"].items() if k not in {"status", "warnings"}},
    }
    return handoff


def research_report_markdown(handoff: dict[str, Any]) -> str:
    lines = [
        "# Research Handoff",
        "",
        f"Status: `{handoff.get('status')}`",
        f"Retrieved at: `{(handoff.get('freshness_stamp') or {}).get('retrieved_at')}`",
        "",
        "## Recommendation",
        str((handoff.get("recommendation") or {}).get("summary") or handoff.get("recommendation") or "No primary recommendation recorded."),
        "",
        "## Method Cards",
    ]
    method_cards = handoff.get("method_cards") or []
    if method_cards:
        for card in method_cards:
            lines.append(f"- `{card.get('method_id')}`: {card.get('summary') or card.get('method_family') or 'method card'}")
    else:
        lines.append("- No method cards were produced.")

    lines.extend(["", "## Evidence Gaps"])
    gaps = handoff.get("evidence_gaps") or []
    lines.extend([f"- {gap}" for gap in gaps] or ["- No evidence gaps recorded."])

    lines.extend(["", "## Supporting Papers"])
    papers = handoff.get("papers") or []
    if papers:
        for paper in papers[:12]:
            title = paper.get("title") or "Untitled"
            year = paper.get("year") or "unknown year"
            source = paper.get("source") or paper.get("venue") or "unknown source"
            url = paper.get("url") or paper.get("doi") or ""
            lines.append(f"- {title} ({year}, {source}) {url}".rstrip())
    else:
        lines.append("- No papers were retrieved or selected.")

    warnings = (handoff.get("quality_gate") or {}).get("warnings") or []
    if warnings:
        lines.extend(["", "## Quality Warnings"])
        lines.extend([f"- {warning}" for warning in warnings])
    return "\n".join(lines) + "\n"


def _search_europe_pmc(query: str, max_results: int) -> list[dict[str, Any]]:
    encoded = urllib.parse.urlencode({"query": query, "format": "json", "pageSize": str(max_results)})
    payload = _read_json_url(f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?{encoded}")
    results = (((payload or {}).get("resultList") or {}).get("result")) or []
    papers: list[dict[str, Any]] = []
    for item in results:
        full_text_urls = []
        if isinstance(item.get("fullTextUrlList"), dict):
            full_text_urls = item.get("fullTextUrlList", {}).get("fullTextUrl") or []
        first_url = full_text_urls[0].get("url") if full_text_urls and isinstance(full_text_urls[0], dict) else None
        papers.append({
            "source": "Europe PMC",
            "title": item.get("title"),
            "year": item.get("pubYear"),
            "venue": item.get("journalTitle"),
            "doi": item.get("doi"),
            "pmid": item.get("pmid"),
            "abstract": item.get("abstractText"),
            "url": first_url or item.get("doi"),
            "citation_count": item.get("citedByCount"),
            "evidence_level": "biomedical_index",
        })
    return papers


def _search_openalex(query: str, max_results: int) -> list[dict[str, Any]]:
    encoded = urllib.parse.urlencode({"search": query, "per-page": str(max_results)})
    payload = _read_json_url(f"https://api.openalex.org/works?{encoded}")
    results = (payload or {}).get("results") or []
    papers: list[dict[str, Any]] = []
    for item in results:
        doi = item.get("doi")
        papers.append({
            "source": "OpenAlex",
            "title": item.get("title"),
            "year": item.get("publication_year"),
            "venue": ((item.get("primary_location") or {}).get("source") or {}).get("display_name"),
            "doi": doi.replace("https://doi.org/", "") if isinstance(doi, str) else doi,
            "abstract": _openalex_abstract(item.get("abstract_inverted_index")),
            "url": item.get("id"),
            "citation_count": item.get("cited_by_count"),
            "evidence_level": item.get("type") or "metadata_index",
        })
    return papers


def _search_semantic_scholar(query: str, max_results: int) -> list[dict[str, Any]]:
    fields = "title,year,venue,abstract,citationCount,url,externalIds,publicationTypes"
    encoded = urllib.parse.urlencode({"query": query, "limit": str(max_results), "fields": fields})
    payload = _read_json_url(f"https://api.semanticscholar.org/graph/v1/paper/search?{encoded}")
    results = (payload or {}).get("data") or []
    papers: list[dict[str, Any]] = []
    for item in results:
        external = item.get("externalIds") or {}
        papers.append({
            "source": "Semantic Scholar",
            "title": item.get("title"),
            "year": item.get("year"),
            "venue": item.get("venue"),
            "doi": external.get("DOI"),
            "pmid": external.get("PubMed"),
            "arxiv_id": external.get("ArXiv"),
            "abstract": item.get("abstract"),
            "url": item.get("url"),
            "citation_count": item.get("citationCount"),
            "evidence_level": ",".join(item.get("publicationTypes") or []) or "citation_graph",
        })
    return papers


def _search_arxiv(query: str, max_results: int) -> list[dict[str, Any]]:
    encoded = urllib.parse.urlencode({
        "search_query": f"all:{query}",
        "start": "0",
        "max_results": str(max_results),
        "sortBy": "relevance",
        "sortOrder": "descending",
    })
    xml_text = _read_text_url(f"https://export.arxiv.org/api/query?{encoded}")
    root = ET.fromstring(xml_text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    papers: list[dict[str, Any]] = []
    for entry in root.findall("atom:entry", ns):
        arxiv_url = _xml_text(entry, "atom:id", ns)
        arxiv_id = arxiv_url.rsplit("/", 1)[-1] if arxiv_url else None
        papers.append({
            "source": "arXiv",
            "title": _xml_text(entry, "atom:title", ns),
            "year": (_xml_text(entry, "atom:published", ns) or "")[:4] or None,
            "venue": "arXiv",
            "doi": None,
            "arxiv_id": arxiv_id,
            "abstract": _xml_text(entry, "atom:summary", ns),
            "url": arxiv_url,
            "citation_count": None,
            "evidence_level": "preprint",
        })
    return papers


def _search_crossref(query: str, max_results: int) -> list[dict[str, Any]]:
    encoded = urllib.parse.urlencode({"query": query, "rows": str(max_results), "select": "DOI,title,container-title,published-print,published-online,is-referenced-by-count,URL,abstract,type"})
    payload = _read_json_url(f"https://api.crossref.org/works?{encoded}")
    items = (((payload or {}).get("message") or {}).get("items")) or []
    papers: list[dict[str, Any]] = []
    for item in items:
        title = item.get("title") or []
        venue = item.get("container-title") or []
        year = _crossref_year(item)
        papers.append({
            "source": "Crossref",
            "title": title[0] if title else None,
            "year": year,
            "venue": venue[0] if venue else None,
            "doi": item.get("DOI"),
            "abstract": _strip_markup(item.get("abstract")),
            "url": item.get("URL"),
            "citation_count": item.get("is-referenced-by-count"),
            "evidence_level": item.get("type") or "metadata_index",
        })
    return papers


def _read_json_url(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "MAGMA-v2-literature-worker/0.1"})
    with urllib.request.urlopen(request, timeout=12) as response:
        data = response.read(2_000_000)
    return json.loads(data.decode("utf-8", errors="replace"))


def _read_text_url(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "MAGMA-v2-literature-worker/0.1"})
    with urllib.request.urlopen(request, timeout=12) as response:
        data = response.read(2_000_000)
    return data.decode("utf-8", errors="replace")


def _xml_text(element: ET.Element, path: str, ns: dict[str, str]) -> str | None:
    node = element.find(path, ns)
    if node is None or node.text is None:
        return None
    return re.sub(r"\s+", " ", node.text).strip()


def _openalex_abstract(index: Any) -> str | None:
    if not isinstance(index, dict):
        return None
    positions: list[tuple[int, str]] = []
    for word, slots in index.items():
        if isinstance(slots, list):
            positions.extend((int(slot), str(word)) for slot in slots if isinstance(slot, int))
    return " ".join(word for _, word in sorted(positions))[:3000] if positions else None


def _crossref_year(item: dict[str, Any]) -> Any:
    for key in ("published-print", "published-online"):
        parts = ((item.get(key) or {}).get("date-parts") or [])
        if parts and parts[0]:
            return parts[0][0]
    return None


def _strip_markup(value: Any) -> str | None:
    if value is None:
        return None
    return re.sub(r"<[^>]+>", "", str(value))


def _json_object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _normalize_paper(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "paper_id": item.get("paper_id") or item.get("doi") or item.get("pmid") or item.get("arxiv_id") or item.get("title"),
        "source": item.get("source"),
        "title": item.get("title"),
        "venue": item.get("venue"),
        "year": item.get("year"),
        "doi": item.get("doi"),
        "pmid": item.get("pmid"),
        "arxiv_id": item.get("arxiv_id"),
        "url": item.get("url"),
        "abstract": str(item.get("abstract") or "")[:3000],
        "citation_count": item.get("citation_count"),
        "evidence_level": item.get("evidence_level"),
    }


def _normalize_method_card(item: dict[str, Any], index: int) -> dict[str, Any]:
    method_id = item.get("method_id") or f"method_{index + 1}"
    return {
        "method_id": method_id,
        "method_family": item.get("method_family") or item.get("fusion_type") or item.get("name"),
        "modalities": item.get("modalities") if isinstance(item.get("modalities"), list) else [],
        "summary": item.get("summary") or item.get("rationale"),
        "missing_modality_handling": item.get("missing_modality_handling"),
        "imbalance_strategy": item.get("imbalance_strategy"),
        "evaluation_protocol": item.get("evaluation_protocol"),
        "calibration": item.get("calibration"),
        "external_validation": item.get("external_validation"),
        "supporting_paper_ids": item.get("supporting_paper_ids") if isinstance(item.get("supporting_paper_ids"), list) else [],
        "evidence_basis": item.get("evidence_basis") or ("retrieved_paper" if item.get("supporting_paper_ids") else "practice_prior"),
        "limitations": item.get("limitations") if isinstance(item.get("limitations"), list) else [],
    }


def _quality_status(handoff: dict[str, Any]) -> str:
    method_cards = handoff.get("method_cards") or []
    papers = handoff.get("papers") or []
    linked_cards = [
        card for card in method_cards
        if isinstance(card, dict) and card.get("supporting_paper_ids")
    ]
    relevant_papers = [
        paper for paper in papers
        if isinstance(paper, dict) and float(paper.get("relevance_score") or 0) >= 0.15
    ]
    if linked_cards and relevant_papers:
        return "available"
    if method_cards or papers:
        return "limited"
    return "thin"


def _quality_warnings(handoff: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if not handoff.get("papers"):
        warnings.append("No supporting papers were recorded.")
    if not handoff.get("method_cards"):
        warnings.append("No method cards were recorded.")
    if handoff.get("method_cards") and not any(card.get("supporting_paper_ids") for card in handoff.get("method_cards", []) if isinstance(card, dict)):
        warnings.append("Method cards are not linked to supporting paper IDs; treat recommendations as practice priors.")
    low_relevance = [
        paper for paper in handoff.get("papers", [])
        if isinstance(paper, dict) and float(paper.get("relevance_score") or 0) < 0.15
    ]
    if handoff.get("papers") and len(low_relevance) == len(handoff.get("papers", [])):
        warnings.append("Retrieved papers have low lexical relevance to the query plan.")
    if not handoff.get("recommendation"):
        warnings.append("No recommendation was recorded.")
    return warnings
