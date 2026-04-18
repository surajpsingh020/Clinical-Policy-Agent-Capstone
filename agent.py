"""
agent.py — Clinical Policy & Factuality Agent
Capstone Project | Agentic AI Course
Architecture: LangGraph + ChromaDB + SentenceTransformer + ChatGroq + MemorySaver
"""

import os
import re
import json
import datetime
from typing import TypedDict, List

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ══════════════════════════════════════════════════════════════════════════════
# 1. STATE  (must be defined before any nodes)
# ══════════════════════════════════════════════════════════════════════════════

class CapstoneState(TypedDict):
    messages:    List           # conversation history (HumanMessage / AIMessage)
    query:       str            # current user query
    retrieved:   str            # concatenated retrieved context from ChromaDB
    sources:     List[str]      # titles of retrieved source documents
    answer:      str            # generated answer
    eval_score:  float          # faithfulness score  0.0 – 1.0
    retry_count: int            # number of retry attempts so far (max 2)
    route:       str            # "retrieve" | "memory_only" | "tool"
    tool_output: str            # output from tool_node

# ══════════════════════════════════════════════════════════════════════════════
# 2. SINGLETONS
# ══════════════════════════════════════════════════════════════════════════════

_embedder       = None
_chroma_client  = None
_collection     = None
_llm            = None

GROQ_MODEL = "llama-3.3-70b-versatile"


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(model=GROQ_MODEL, temperature=0)
    return _llm


def get_collection():
    global _chroma_client, _collection
    if _collection is None:
        _chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
        _collection    = _chroma_client.get_or_create_collection("clinical_policies")
        _seed_knowledge_base(_collection)
    return _collection

# ══════════════════════════════════════════════════════════════════════════════
# 3. KNOWLEDGE BASE  (10 documents, 100-300 words each)
# ══════════════════════════════════════════════════════════════════════════════

DOCUMENTS = [
    {
        "id":      "doc_001",
        "title":   "Hand Hygiene Protocol",
        "content": (
            "Hand Hygiene Protocol — St. Mercy Hospital\n\n"
            "Hand hygiene is the single most effective measure to prevent healthcare-associated "
            "infections (HAIs). All healthcare workers must perform hand hygiene at the WHO Five "
            "Moments: (1) before touching a patient, (2) before an aseptic procedure, (3) after "
            "body-fluid exposure, (4) after touching a patient, and (5) after touching patient "
            "surroundings.\n\n"
            "Technique: Use alcohol-based hand rub (ABHR, ≥60% ethanol or isopropanol) for "
            "visibly clean hands — rub for a minimum of 20 seconds covering all surfaces. Use "
            "soap and water when hands are visibly soiled or after Clostridioides difficile "
            "contact (ABHR is ineffective against C. diff spores). Surgical hand scrub must last "
            "a minimum of 2–3 minutes.\n\n"
            "Gloves are not a substitute for hand hygiene. Hands must be cleaned before donning "
            "and immediately after removing gloves. Artificial nails and nail extensions are "
            "prohibited in all clinical areas. Natural nails must not exceed 0.25 inches beyond "
            "the fingertip.\n\n"
            "Non-compliance is a Patient Safety Event and must be reported. Department managers "
            "conduct monthly audits using the Hand Hygiene Observation Tool. Hospital-wide "
            "compliance target: ≥90%."
        ),
    },
    {
        "id":      "doc_002",
        "title":   "Medication Administration Policy",
        "content": (
            "Medication Administration Policy — St. Mercy Hospital\n\n"
            "All medication administration must follow the Ten Rights: Right Patient, Right Drug, "
            "Right Dose, Right Route, Right Time, Right Reason, Right Documentation, Right to "
            "Refuse, Right Assessment, and Right Evaluation.\n\n"
            "Patient Identification: Scan the barcode wristband and confirm with two identifiers "
            "(full name + date of birth) before every administration. Never administer based "
            "solely on room or bed number.\n\n"
            "High-Alert Medications: Insulin, anticoagulants (heparin, warfarin), opioids, and "
            "concentrated electrolytes (potassium chloride >2 mEq/mL) require an independent "
            "double-check by a second licensed nurse before administration. These are stored in "
            "segregated, labeled bins in the Pyxis dispensing cabinet.\n\n"
            "Verbal and Telephone Orders: Must be repeated back and confirmed (read-back), "
            "documented in the EMR within 60 minutes, and co-signed by the ordering provider "
            "within 24 hours.\n\n"
            "Medication Errors: Any near-miss or error must be reported immediately via RL "
            "Solutions. A root-cause analysis (RCA) is initiated for all serious events. "
            "Pharmacist reconciliation is mandatory upon admission, transfer, and discharge."
        ),
    },
    {
        "id":      "doc_003",
        "title":   "Fall Prevention Protocol",
        "content": (
            "Fall Prevention Protocol — St. Mercy Hospital\n\n"
            "All patients are assessed using the Morse Fall Scale upon admission, at each shift "
            "change, after a fall, and after any condition change. Scores: 0–24 = low risk; "
            "25–44 = moderate risk; ≥45 = high risk.\n\n"
            "High-Risk Interventions (mandatory for score ≥45): Place a yellow fall-risk "
            "wristband; activate the bed alarm; ensure non-slip footwear at all times; conduct "
            "hourly rounding; post a 'Fall Risk' sign outside the room; keep the bed in the "
            "lowest position with side rails up; complete a pharmacy medication review flagging "
            "sedatives, diuretics, and antihypertensives.\n\n"
            "Moderate-Risk Interventions: Educate the patient and family; keep call light within "
            "reach; ensure a clear path to the bathroom.\n\n"
            "Post-Fall Management: Do NOT move the patient until assessed by the RN and "
            "physician. Perform a neurological assessment. Document using the Post-Fall Huddle "
            "Tool within 1 hour. File an incident report. If injury is suspected, initiate "
            "diagnostic imaging per physician order. Target: Zero patient falls with injury "
            "(NDNQI benchmark)."
        ),
    },
    {
        "id":      "doc_004",
        "title":   "Sepsis Management Bundle (Hour-1 Bundle)",
        "content": (
            "Sepsis Management Bundle (Hour-1 Bundle) — St. Mercy Hospital\n\n"
            "Sepsis is a life-threatening emergency. Suspect sepsis when a patient has suspected "
            "infection PLUS ≥2 SIRS criteria or meets qSOFA criteria (altered mental status, "
            "respiratory rate ≥22/min, systolic BP ≤100 mmHg).\n\n"
            "Within 1 Hour of Recognition (mandatory Hour-1 Bundle):\n"
            "1. Measure lactate level; re-measure if initial lactate >2 mmol/L.\n"
            "2. Obtain blood cultures (≥2 sets from 2 sites) BEFORE antibiotics.\n"
            "3. Administer broad-spectrum antibiotics immediately.\n"
            "4. Begin 30 mL/kg IV crystalloid resuscitation for hypotension (MAP <65 mmHg) "
            "   or lactate ≥4 mmol/L.\n"
            "5. Apply vasopressors (norepinephrine first-line) if hypotension persists after "
            "   fluid resuscitation to maintain MAP ≥65 mmHg.\n\n"
            "Sepsis Alert: Any nurse may activate by calling 5-SEPSIS (5-73747) or via EMR. "
            "The Rapid Response Team responds within 15 minutes.\n\n"
            "Documentation: Complete the Sepsis Screening Tool every 4 hours (ICU) or every "
            "shift (step-down). All bundle elements must be time-stamped in the EMR."
        ),
    },
    {
        "id":      "doc_005",
        "title":   "Blood Transfusion Policy",
        "content": (
            "Blood Transfusion Policy — St. Mercy Hospital\n\n"
            "Informed Consent: Written consent is required before all non-emergent transfusions "
            "and must be documented in the EMR. In life-threatening emergencies, transfusion may "
            "proceed and consent obtained as soon as practical.\n\n"
            "Ordering: A physician or authorized APP must specify product type, quantity, and "
            "any special requirements (irradiated, CMV-negative, leukoreduced).\n\n"
            "Pre-Transfusion Verification (bedside, independent double-check by TWO nurses): "
            "patient identity (scan wristband + name + DOB), blood product label (unit number, "
            "blood type, expiration), and the physician order.\n\n"
            "Administration: Use a blood administration set with a 170–260 micron filter. Normal "
            "saline (0.9% NaCl) is the ONLY compatible IV flush. No medications may be added to "
            "blood products or co-infused through the same line.\n\n"
            "Vital Signs: Obtain baseline vitals, then at 15 minutes post-start, at 1 hour, and "
            "at completion. PRBCs infuse over 2–4 hours (maximum 4 hours). If a transfusion "
            "reaction is suspected, STOP immediately, maintain IV access with NS, notify "
            "physician and blood bank, and complete the Transfusion Reaction Report."
        ),
    },
    {
        "id":      "doc_006",
        "title":   "Pain Assessment and Management Policy",
        "content": (
            "Pain Assessment and Management Policy — St. Mercy Hospital\n\n"
            "Pain is the fifth vital sign and must be assessed and documented at every patient "
            "encounter, each shift, 1 hour after any pain intervention, and upon patient request.\n\n"
            "Assessment Tools: Use the Numeric Rating Scale (NRS 0–10) for patients who can "
            "self-report. Use the CPOT (Critical-Care Pain Observation Tool) or FLACC scale for "
            "non-verbal, sedated, or cognitively impaired patients.\n\n"
            "Multimodal Analgesia: A multimodal approach combining non-opioid analgesics "
            "(acetaminophen, NSAIDs, gabapentin) with non-pharmacological methods (ice, heat, "
            "positioning, distraction) is preferred to minimize opioid requirements.\n\n"
            "Opioid Safety: Before administering any opioid, assess respiratory rate (must be "
            "≥10/min), sedation level (RASS scale), and pain score. Naloxone must be at bedside "
            "for all patients on continuous opioid infusions. PCA requires a dedicated PCA "
            "agreement and daily physician review.\n\n"
            "Documentation: Record pain score, intervention, patient response, and reassessment "
            "score in the EMR. A pain score >4 requires a care plan update and reassessment "
            "within 1 hour."
        ),
    },
    {
        "id":      "doc_007",
        "title":   "Isolation Precautions Policy",
        "content": (
            "Isolation Precautions Policy — St. Mercy Hospital\n\n"
            "Standard Precautions apply to ALL patients at ALL times regardless of diagnosis.\n\n"
            "Transmission-Based Precautions:\n\n"
            "1. Contact Precautions (MRSA, VRE, C. diff, scabies): Gloves and gown upon room "
            "entry. Use dedicated patient equipment (stethoscope, BP cuff). C. diff requires "
            "soap-and-water hand hygiene — ABHR is ineffective against spores. Patient transport: "
            "cover wounds; notify receiving department in advance.\n\n"
            "2. Droplet Precautions (influenza, pertussis, bacterial meningitis): Surgical mask "
            "upon room entry. Patient wears surgical mask during transport.\n\n"
            "3. Airborne Precautions (tuberculosis, measles, varicella, COVID-19): Fit-tested "
            "N95 respirator upon room entry. Patient must be in a negative-pressure Airborne "
            "Infection Isolation Room (AIIR). Patient transport is minimized; patient wears "
            "surgical mask if transport is required.\n\n"
            "Signage: Post the appropriate precaution sign outside the room. PPE is donned "
            "before entering and doffed immediately outside the room. Discontinuation requires "
            "an Infectious Disease (ID) physician order."
        ),
    },
    {
        "id":      "doc_008",
        "title":   "Code Blue / Cardiac Arrest Protocol",
        "content": (
            "Code Blue / Cardiac Arrest Protocol — St. Mercy Hospital\n\n"
            "Activation: Any staff member discovering an unresponsive patient calls 'Code Blue' "
            "via overhead paging and dials 5-CODE (5-2633). State location clearly.\n\n"
            "First Responder Actions: Begin high-quality CPR immediately — rate 100–120/min, "
            "depth ≥2 inches, allow full chest recoil, minimize interruptions (<10 sec). Attach "
            "AED or defibrillator as soon as available. AEDs are located at every nursing "
            "station.\n\n"
            "Code Team Roles: Team Leader (physician/intensivist), Compressor (nurse/tech), "
            "Airway Manager (RT/anesthesia), Medication Nurse (RN), Recorder (RN), Family "
            "Liaison (social work/chaplain).\n\n"
            "Medications (ACLS protocol): Epinephrine 1 mg IV/IO every 3–5 minutes for all "
            "rhythms. Amiodarone 300 mg IV/IO for refractory VF/pulseless VT; second dose "
            "150 mg.\n\n"
            "Post-ROSC Care: Target SpO₂ 94–99%, PaCO₂ 35–45 mmHg, systolic BP ≥90 mmHg. "
            "Initiate Targeted Temperature Management (TTM) per physician order. Debrief within "
            "30 minutes using the Code Blue Debrief Checklist."
        ),
    },
    {
        "id":      "doc_009",
        "title":   "Patient Identification Policy",
        "content": (
            "Patient Identification Policy — St. Mercy Hospital\n\n"
            "Accurate patient identification is mandatory before every clinical intervention to "
            "prevent wrong-patient errors — a Joint Commission Sentinel Event category.\n\n"
            "Two-Identifier Rule: Every interaction (medication, specimen, procedures, "
            "transfusions, dietary) requires verification of TWO identifiers: (1) Full legal "
            "name and (2) Date of birth. Room number and bed number are NEVER acceptable "
            "identifiers.\n\n"
            "Wristbands: A hospital-issued ID wristband must be on the patient at all times — "
            "wrist preferred; ankle if not feasible. Alert wristbands (red = allergy, yellow = "
            "fall risk, purple = DNR) are placed on the same arm as the ID wristband.\n\n"
            "Barcode Medication Administration (BCMA): Scan the patient wristband barcode AND "
            "medication barcode before every administration. BCMA override is permitted only in "
            "emergencies and requires documented justification.\n\n"
            "Unidentified Patients: Assigned a temporary unique identifier (e.g., 'John Doe "
            "04-17') and issued a wristband. Identity must be confirmed and updated in the EMR "
            "within 24 hours.\n\n"
            "Specimen Labeling: Label at the bedside in the patient's presence immediately after "
            "collection. Pre-labeled tubes are strictly prohibited."
        ),
    },
    {
        "id":      "doc_010",
        "title":   "Antibiotic Stewardship Policy",
        "content": (
            "Antibiotic Stewardship Policy — St. Mercy Hospital\n\n"
            "The Antimicrobial Stewardship Program (ASP) is led by the Infectious Disease (ID) "
            "pharmacist and physician to optimize antibiotic use, minimize resistance, and reduce "
            "C. difficile infections.\n\n"
            "Prospective Audit and Feedback: The ASP team reviews all broad-spectrum antibiotic "
            "orders (meropenem, vancomycin, piperacillin-tazobactam, ceftriaxone, "
            "fluoroquinolones) within 48–72 hours via the mandatory 'Antibiotic Time-Out.'\n\n"
            "48–72 Hour Antibiotic Time-Out: Reassess the indication, de-escalate to the "
            "narrowest effective agent, transition to oral therapy if clinically appropriate, "
            "and define the duration of therapy in the order.\n\n"
            "Restricted Antibiotics: Carbapenems (meropenem, ertapenem), daptomycin, and "
            "linezolid require prior authorization from the ID physician or ASP pharmacist. "
            "Unauthorized orders are auto-discontinued at 24 hours.\n\n"
            "Culture-Guided Therapy: Empiric therapy must be adjusted based on culture and "
            "sensitivity results. Failure to de-escalate without documented reasoning is a "
            "protocol deviation.\n\n"
            "Education: All prescribers complete mandatory annual ASP training. Antibiogram data "
            "is published quarterly on the hospital intranet."
        ),
    },
]


def _seed_knowledge_base(collection) -> None:
    """Idempotently populate ChromaDB with the 10 clinical policy documents."""
    if collection.count() > 0:
        return
    embedder = get_embedder()
    for doc in DOCUMENTS:
        embedding = embedder.encode(doc["content"]).tolist()
        collection.add(
            ids=[doc["id"]],
            embeddings=[embedding],
            documents=[doc["content"]],
            metadatas=[{"title": doc["title"]}],
        )

# ══════════════════════════════════════════════════════════════════════════════
# 4. TOOLS
# ══════════════════════════════════════════════════════════════════════════════

_DRUG_INTERACTIONS: dict = {
    frozenset(["warfarin",      "aspirin"]):        "HIGH RISK: Warfarin + aspirin significantly increases bleeding risk. Monitor INR closely; consider GI prophylaxis.",
    frozenset(["metformin",     "contrast dye"]):   "MODERATE RISK: Hold metformin 48 h before and after iodinated contrast to prevent lactic acidosis.",
    frozenset(["ssri",          "tramadol"]):        "HIGH RISK: SSRIs + tramadol increases serotonin syndrome risk. Monitor for agitation, hyperthermia, tachycardia.",
    frozenset(["metronidazole", "alcohol"]):         "HIGH RISK: Disulfiram-like reaction. Avoid alcohol during and 48 h after metronidazole therapy.",
    frozenset(["ciprofloxacin", "antacids"]):        "MODERATE RISK: Antacids reduce ciprofloxacin absorption significantly. Administer ciprofloxacin 2 h before or 6 h after antacids.",
    frozenset(["lisinopril",    "potassium"]):       "MODERATE RISK: ACE inhibitors cause potassium retention; potassium supplementation may cause hyperkalemia.",
    frozenset(["heparin",       "nsaids"]):          "HIGH RISK: NSAIDs + heparin increases bleeding and GI ulceration risk. Avoid combination when possible.",
    frozenset(["amiodarone",    "digoxin"]):         "MODERATE RISK: Amiodarone raises digoxin levels ~70%. Reduce digoxin dose by 50% and monitor plasma levels.",
    frozenset(["clopidogrel",   "omeprazole"]):      "MODERATE RISK: Omeprazole reduces clopidogrel antiplatelet effect via CYP2C19 inhibition. Consider pantoprazole instead.",
    frozenset(["phenytoin",     "warfarin"]):        "MODERATE RISK: Phenytoin may initially increase then decrease warfarin effect; monitor INR closely.",
}


def drug_interaction_checker(drug1: str, drug2: str) -> str:
    """Mock Drug Interaction Checker. Returns a string, never raises an exception."""
    try:
        key = frozenset([drug1.lower().strip(), drug2.lower().strip()])
        result = _DRUG_INTERACTIONS.get(key)
        if result:
            return f"[Drug Interaction Checker]\n{result}"
        return (
            f"[Drug Interaction Checker]\n"
            f"No significant interaction found between '{drug1}' and '{drug2}' in the local "
            f"database. Always verify with the clinical pharmacist for patient-specific context."
        )
    except Exception:
        return "[Drug Interaction Checker]\nService temporarily unavailable. Consult the clinical pharmacist."


def get_current_datetime() -> str:
    """Returns current date and time. Never raises."""
    try:
        now = datetime.datetime.now()
        return f"[Datetime Tool]\nCurrent date and time: {now.strftime('%A, %B %d, %Y at %H:%M:%S')}"
    except Exception:
        return "[Datetime Tool]\nDatetime unavailable."

# ══════════════════════════════════════════════════════════════════════════════
# 5. NODES  (fully isolated — each returns only its own keys)
# ══════════════════════════════════════════════════════════════════════════════

def router_node(state: CapstoneState) -> dict:
    """
    Routes query to one of three paths.
    Output key: route  ∈ {"retrieve", "memory_only", "tool"}
    """
    llm   = get_llm()
    query = state["query"]

    system = (
        "You are a routing agent for a hospital clinical policy assistant. "
        "Classify the user query into EXACTLY ONE of these categories and respond with only that single word:\n\n"
        "  retrieve    — query is about clinical policies, protocols, procedures, medications, "
                       "infection control, patient safety, or any hospital guideline topic.\n"
        "  tool        — query asks about a drug-drug interaction (mentions two specific drug names) "
                       "OR asks for the current date or time.\n"
        "  memory_only — query is casual conversation (greetings, thank-you), asks to repeat or "
                       "clarify a previous answer, or is completely off-topic (sports, weather, "
                       "general knowledge unrelated to clinical care).\n\n"
        "Output ONLY the single word: retrieve, memory_only, or tool."
    )
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=query)])
    raw = response.content.strip().lower()

    if "tool" in raw:
        route = "tool"
    elif "memory" in raw:
        route = "memory_only"
    else:
        route = "retrieve"

    return {"route": route}


def retrieve_node(state: CapstoneState) -> dict:
    """
    Retrieves top-3 relevant documents from ChromaDB using semantic search.
    Clears tool_output to maintain path isolation.
    """
    embedder   = get_embedder()
    collection = get_collection()
    query      = state["query"]

    query_embedding = embedder.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "metadatas"],
    )
    docs    = results["documents"][0]
    metas   = results["metadatas"][0]

    retrieved_text = "\n\n---\n\n".join(docs)
    sources        = [m.get("title", "Unknown") for m in metas]

    return {"retrieved": retrieved_text, "sources": sources, "tool_output": ""}


def skip_retrieval_node(state: CapstoneState) -> dict:
    """
    Explicit no-op for non-retrieval paths.
    Returns empty retrieved and sources per specification.
    """
    return {"retrieved": "", "sources": [], "tool_output": ""}


def memory_node(state: CapstoneState) -> dict:
    """
    Applies sliding-window memory: retains only the last 6 messages.
    Runs on ALL paths (after retrieve / skip_retrieval / tool).
    """
    messages = state.get("messages", [])
    windowed = messages[-6:] if len(messages) > 6 else messages
    return {"messages": windowed}


def tool_node(state: CapstoneState) -> dict:
    """
    Executes Drug Interaction Checker or Datetime tool.
    Clears retrieved/sources to maintain path isolation.
    """
    query = state["query"].lower()
    llm   = get_llm()

    # Datetime branch
    if any(kw in query for kw in ["current date", "today", "what time", "what day", "datetime"]):
        return {"tool_output": get_current_datetime(), "retrieved": "", "sources": []}

    # Drug-interaction branch: extract two drug names via LLM
    extraction_prompt = (
        f"Extract exactly two drug names from this query: \"{state['query']}\"\n"
        "Respond ONLY with a JSON object (no markdown, no preamble):\n"
        '{"drug1": "<name1>", "drug2": "<name2>"}\n'
        "If two drug names cannot be found, use \"unknown\" as the value."
    )
    resp = llm.invoke([HumanMessage(content=extraction_prompt)])
    try:
        raw = re.sub(r"```(?:json)?|```", "", resp.content.strip()).strip()
        parsed = json.loads(raw)
        drug1  = parsed.get("drug1", "unknown")
        drug2  = parsed.get("drug2", "unknown")
    except Exception:
        drug1, drug2 = "unknown", "unknown"

    return {
        "tool_output": drug_interaction_checker(drug1, drug2),
        "retrieved":   "",
        "sources":     [],
    }


def generate_node(state: CapstoneState) -> dict:
    """
    Generates the final answer.
    Adds a retry-awareness instruction on subsequent attempts.
    """
    llm         = get_llm()
    query       = state["query"]
    retrieved   = state.get("retrieved",   "")
    tool_output = state.get("tool_output", "")
    messages    = state.get("messages",    [])
    retry_count = state.get("retry_count", 0)

    # Build context
    ctx_parts = []
    if retrieved:
        ctx_parts.append(f"CLINICAL POLICY CONTEXT:\n{retrieved}")
    if tool_output:
        ctx_parts.append(f"TOOL OUTPUT:\n{tool_output}")
    context = "\n\n".join(ctx_parts) if ctx_parts else "No specific context available."

    retry_note = (
        "\n\nIMPORTANT — RETRY ATTEMPT: Your previous answer scored below the faithfulness "
        "threshold. Be more precise, cite only details explicitly stated in the context, and "
        "do NOT add any information not present in the provided context."
        if retry_count > 0 else ""
    )

    system = (
        "You are a Clinical Policy Assistant for St. Mercy Hospital. "
        "Answer using ONLY the provided context. "
        "If the answer is not in the context, respond: "
        "'This falls outside my clinical policy knowledge base. "
        "Please consult the relevant department or supervisor.' "
        "Be precise and cite specific policy details. Never fabricate clinical information."
        + retry_note
    )

    lc_msgs = [SystemMessage(content=system)]
    # Include last 4 conversation messages for short-term context
    for msg in messages[-4:]:
        if isinstance(msg, (HumanMessage, AIMessage)):
            lc_msgs.append(msg)

    lc_msgs.append(HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"))

    response = llm.invoke(lc_msgs)
    answer   = response.content.strip()

    # Append this turn to messages
    updated_messages = list(messages)
    updated_messages.append(HumanMessage(content=query))
    updated_messages.append(AIMessage(content=answer))

    return {"answer": answer, "messages": updated_messages}


def eval_node(state: CapstoneState) -> dict:
    """
    Scores answer faithfulness (0.0–1.0).
    Increments retry_count when score < 0.7.
    Max effective retries: 2 (enforced by eval_decision).
    """
    llm         = get_llm()
    answer      = state.get("answer",      "")
    retrieved   = state.get("retrieved",   "")
    tool_output = state.get("tool_output", "")
    retry_count = state.get("retry_count", 0)

    context = retrieved or tool_output
    if not context:
        # Conversational / memory-only path — assume faithful
        return {"eval_score": 1.0, "retry_count": retry_count}

    eval_prompt = (
        "You are a medical QA evaluator assessing answer faithfulness.\n\n"
        f"CONTEXT (source documents):\n{context[:2500]}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Score: How well is the answer grounded in the context only, without fabrication?\n"
        "1.0 = perfectly faithful (every claim traceable to context)\n"
        "0.0 = entirely fabricated (no grounding in context)\n\n"
        "Respond with ONLY a decimal number between 0.0 and 1.0. No explanation."
    )
    try:
        resp     = llm.invoke([HumanMessage(content=eval_prompt)])
        score_str = resp.content.strip()
        match    = re.search(r"[01]?\.\d+|[01]", score_str)
        score    = float(match.group()) if match else 0.75
        score    = max(0.0, min(1.0, score))
    except Exception:
        score = 0.75  # default pass when evaluator fails

    new_retry_count = retry_count + 1 if score < 0.7 else retry_count
    return {"eval_score": score, "retry_count": new_retry_count}

# ══════════════════════════════════════════════════════════════════════════════
# 6. STANDALONE ROUTING FUNCTIONS  (used in add_conditional_edges)
# ══════════════════════════════════════════════════════════════════════════════

def route_decision(state: CapstoneState) -> str:
    """Maps router_node output to the correct branch key."""
    return state.get("route", "retrieve")


def eval_decision(state: CapstoneState) -> str:
    """
    After eval_node:
      • score < 0.7 AND retry_count < 2  → 'retry'  (go back to generate)
      • otherwise                         → 'end'
    """
    score       = state.get("eval_score",  1.0)
    retry_count = state.get("retry_count", 0)
    if score < 0.7 and retry_count < 2:
        return "retry"
    return "end"

# ══════════════════════════════════════════════════════════════════════════════
# 7. GRAPH COMPILATION
# ══════════════════════════════════════════════════════════════════════════════

def build_graph():
    memory = MemorySaver()
    graph  = StateGraph(CapstoneState)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    graph.add_node("router",        router_node)
    graph.add_node("retrieve",      retrieve_node)
    graph.add_node("skip_retrieval", skip_retrieval_node)
    graph.add_node("tool",          tool_node)
    graph.add_node("memory",        memory_node)
    graph.add_node("generate",      generate_node)
    graph.add_node("eval",          eval_node)

    # ── Entry ─────────────────────────────────────────────────────────────────
    graph.set_entry_point("router")

    # ── Conditional edges from router ─────────────────────────────────────────
    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "retrieve":    "retrieve",
            "memory_only": "skip_retrieval",
            "tool":        "tool",
        },
    )

    # ── Fixed edges ───────────────────────────────────────────────────────────
    graph.add_edge("retrieve",       "memory")
    graph.add_edge("skip_retrieval", "memory")
    graph.add_edge("tool",           "memory")
    graph.add_edge("memory",         "generate")
    graph.add_edge("generate",       "eval")

    # ── Conditional edges from eval (retry loop) ─────────────────────────────
    graph.add_conditional_edges(
        "eval",
        eval_decision,
        {
            "retry": "generate",   # re-run generate (max 2 times)
            "end":   END,
        },
    )

    return graph.compile(checkpointer=memory)


# ── Compiled singleton ────────────────────────────────────────────────────────
_app = None


def get_app():
    global _app
    if _app is None:
        _app = build_graph()
    return _app


def run_query(query: str, thread_id: str = "default", prior_messages: list = None) -> dict:
    """
    Public interface. Pass prior_messages from session state for multi-turn memory.
    Returns the full CapstoneState after graph execution.
    """
    app    = get_app()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: CapstoneState = {
        "messages":    prior_messages or [],
        "query":       query,
        "retrieved":   "",
        "sources":     [],
        "answer":      "",
        "eval_score":  0.0,
        "retry_count": 0,
        "route":       "",
        "tool_output": "",
    }
    return app.invoke(initial_state, config=config)


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    questions = [
        "What are the five WHO moments for hand hygiene?",
        "Describe the Hour-1 Sepsis Bundle steps.",
        "What PPE is required for airborne precautions?",
        "How should a suspected blood transfusion reaction be managed?",
        "Check interaction between warfarin and aspirin.",
    ]
    for q in questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        r = run_query(q, thread_id="smoke-test")
        print(f"Route:      {r['route']}")
        print(f"Sources:    {r['sources']}")
        print(f"Eval Score: {r['eval_score']:.2f}")
        print(f"Retries:    {r['retry_count']}")
        print(f"A: {r['answer'][:300]}...")
