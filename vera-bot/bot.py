"""
Vera 2.0 — magicpin merchant AI assistant (challenge submission)

Endpoints: /v1/context, /v1/tick, /v1/reply, /v1/healthz, /v1/metadata  (+ /v1/teardown, /v1/submission)
Composer: LLM (any OpenAI-compatible provider, auto-detected from key) -> validation -> deterministic template fallback.
Replies: rule-based router first (auto-reply, stop, hostile, off-topic, commit-intent), LLM for everything else.
"""
import os, re, json, time, uuid, glob, hashlib
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, wait
from urllib import request as urlrequest

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse

app = FastAPI(title="Vera 2.0")
START = time.time()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------- LLM config
LLM_API_KEY = os.environ.get("LLM_API_KEY", "").strip()


def _provider():
    base = os.environ.get("LLM_BASE_URL")
    model = os.environ.get("LLM_MODEL")
    k = LLM_API_KEY
    if k.startswith("gsk_"):
        d = ("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile")
    elif k.startswith("AIza"):
        d = ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.0-flash")
    elif k.startswith("sk-or-"):
        d = ("https://openrouter.ai/api/v1", "meta-llama/llama-3.3-70b-instruct")
    else:
        d = ("https://api.openai.com/v1", "gpt-4o-mini")
    return (base or d[0]).rstrip("/"), (model or d[1])


LLM_BASE, LLM_MODEL = _provider()


def llm_json(system: str, user: str, timeout: float = 20.0):
    """Call an OpenAI-compatible chat endpoint, return parsed JSON dict or None."""
    if not LLM_API_KEY:
        return None
    body = {
        "model": LLM_MODEL,
        "temperature": 0,
        "max_tokens": 600,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }
    try:
        req = urlrequest.Request(
            LLM_BASE + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {LLM_API_KEY}"},
        )
        with urlrequest.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode())
        text = data["choices"][0]["message"]["content"]
        text = re.sub(r"^```(json)?|```$", "", text.strip()).strip()
        return json.loads(text)
    except Exception as e:  # noqa
        print("LLM error:", repr(e)[:300])
        return None


# ----------------------------------------------------------------------------- state
contexts: dict = {}          # (scope, id) -> {"version": int, "payload": dict}
conversations: dict = {}     # conv_id -> state dict
sent_suppression: set = set()


def ctx(scope, cid):
    v = contexts.get((scope, cid))
    return v["payload"] if v else None


def now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ----------------------------------------------------------------------------- helpers
def wants_hinglish(merchant, customer=None):
    if customer:
        lp = str((customer.get("identity") or {}).get("language_pref", "")).lower()
        return "hi" in lp
    langs = (merchant.get("identity") or {}).get("languages") or []
    return "hi" in [str(x).lower() for x in langs]


def salutation(category, merchant):
    ident = merchant.get("identity") or {}
    first = ident.get("owner_first_name") or ident.get("name", "").split("'")[0].split()[0]
    slug = (category or {}).get("slug", "")
    if slug == "dentists" and not first.lower().startswith("dr"):
        return f"Dr. {first}"
    return first


def find_digest(category, item_id):
    for d in (category or {}).get("digest") or []:
        if d.get("id") == item_id:
            return d
    return None


def pct(x):
    try:
        return f"{abs(float(x)) * 100:.0f}%"
    except Exception:
        return str(x)


def active_offer(merchant, category):
    for o in merchant.get("offers") or []:
        if o.get("status") == "active":
            return o.get("title")
    cat = (category or {}).get("offer_catalog") or []
    return cat[0]["title"] if cat else None


# ----------------------------------------------------------------------------- COMPOSER
SYSTEM_COMPOSE = """You are Vera, magicpin's WhatsApp assistant for Indian local merchants. You write ONE WhatsApp message.

HARD RULES
1. Use ONLY facts present in the provided JSON. Never invent numbers, research, competitor names, offers, prices, dates or slots. If a fact isn't there, don't mention it.
2. Open with the "why now": the trigger event, anchored on a concrete, checkable fact (a number, date, source, price) from the data.
3. Personalise to THIS merchant: use their name, locality, their own numbers, their active offer, peer benchmark comparison, and recent conversation history if relevant (don't repeat what was already said).
4. Voice: peer/colleague, never salesy hype. Follow category voice: use allowed vocab, NEVER use taboo words. Dentists/pharmacies = clinical-peer. Offers as "Service @ ₹price", never "X% off".
5. Language: follow the LANGUAGE instruction exactly. Hinglish means Roman-script Hindi-English code-mix, natural, not formal Hindi.
6. Short: 2-4 sentences, ~40-80 words. No preamble ("Hope you're well"), no self-introduction, no hashtags, max 1 emoji.
7. End with exactly ONE clear call to action in the last sentence. Prefer effort-externalisation ("I've drafted X, reply YES and I'll publish") or a single yes/no. Customer booking messages may offer numbered slot choices.
8. Use at least one lever: loss aversion, social proof from peer_stats, curiosity, effort externalisation, reciprocity, or asking the merchant a question.
9. If send_as is merchant_on_behalf, you write AS the merchant's business to their customer: warm, respectful of consent scope, no medical claims, use the customer's name and history.

Return JSON only: {"body": "...", "cta": "binary_yes_stop" | "open_ended" | "slot_choice" | "none", "rationale": "1-2 sentences: which trigger fact + which merchant fact + which lever"}"""


def build_prompt(category, merchant, trigger, customer):
    payload = trigger.get("payload") or {}
    item_id = payload.get("top_item_id") or payload.get("digest_item_id") or payload.get("alert_id")
    cat_view = {
        "slug": category.get("slug"),
        "voice": category.get("voice"),
        "offer_catalog": category.get("offer_catalog"),
        "peer_stats": category.get("peer_stats"),
        "referenced_digest_item": find_digest(category, item_id) if item_id else None,
        "digest_other": [d for d in (category.get("digest") or []) if d.get("id") != item_id][:3],
        "seasonal_beats": category.get("seasonal_beats"),
        "trend_signals": category.get("trend_signals"),
        "patient_content_library": [{"id": c.get("id"), "title": c.get("title")} for c in (category.get("patient_content_library") or [])][:4],
    }
    m_view = dict(merchant)
    m_view["conversation_history"] = (merchant.get("conversation_history") or [])[-4:]
    send_as = "merchant_on_behalf" if (trigger.get("scope") == "customer" and customer) else "vera"
    lang = "Hinglish (Hindi-English code-mix, Roman script)" if wants_hinglish(merchant, customer) else "English"
    who = (f"Address the customer {customer.get('identity', {}).get('name')} as the business '{merchant.get('identity', {}).get('name')}'."
           if send_as == "merchant_on_behalf" else f"Address the merchant as '{salutation(category, merchant)}'.")
    user = (
        f"send_as: {send_as}\nLANGUAGE: {lang}\n{who}\n\n"
        f"TRIGGER (why now):\n{json.dumps(trigger, ensure_ascii=False)}\n\n"
        f"MERCHANT:\n{json.dumps(m_view, ensure_ascii=False)}\n\n"
        f"CATEGORY:\n{json.dumps(cat_view, ensure_ascii=False)}\n\n"
        + (f"CUSTOMER:\n{json.dumps(customer, ensure_ascii=False)}\n\n" if customer else "")
        + "Write the message now."
    )
    return send_as, user


def validate(out, category):
    if not isinstance(out, dict) or not str(out.get("body", "")).strip():
        return False
    body = out["body"].lower()
    for t in ((category or {}).get("voice") or {}).get("vocab_taboo") or []:
        t0 = t.split("(")[0].strip().lower()
        if t0 and t0 in body:
            return False
    if re.search(r"\b\d{1,2}% off\b", body):
        return False
    return True


def compose(category: dict, merchant: dict, trigger: dict, customer: dict | None = None) -> dict:
    """Public contract from the brief: returns body, cta, send_as, suppression_key, rationale."""
    category = category or {}
    send_as, user = build_prompt(category, merchant, trigger, customer)
    out = llm_json(SYSTEM_COMPOSE, user)
    if not validate(out, category):
        out2 = llm_json(SYSTEM_COMPOSE, user + "\n\nYour previous draft broke a rule (taboo word, % off, or empty). Rewrite strictly.") if out else None
        out = out2 if validate(out2, category) else None
    if out is None:
        out = fallback_compose(category, merchant, trigger, customer)
    cta = out.get("cta") or "open_ended"
    return {
        "body": out["body"].strip(),
        "cta": cta,
        "send_as": send_as,
        "suppression_key": trigger.get("suppression_key") or f"{trigger.get('kind')}:{merchant.get('merchant_id')}",
        "rationale": out.get("rationale") or "Trigger-anchored, merchant-specific message with single CTA.",
    }


# ----------------------------------------------------------------------------- deterministic fallback
def fallback_compose(category, merchant, trigger, customer):
    """No-LLM composer: specific, fact-anchored templates per trigger kind."""
    hi = wants_hinglish(merchant, customer)
    name = salutation(category, merchant)
    ident = merchant.get("identity") or {}
    perf = merchant.get("performance") or {}
    peer = (category or {}).get("peer_stats") or {}
    p = trigger.get("payload") or {}
    kind = trigger.get("kind", "")
    offer = active_offer(merchant, category)
    loc = ident.get("locality", "")
    biz = ident.get("name", "")

    # customer-facing
    if trigger.get("scope") == "customer" and customer:
        cname = (customer.get("identity") or {}).get("name", "there")
        slots = p.get("available_slots") or p.get("next_session_options") or []
        slot_txt = " ya ".join(s.get("label", "") for s in slots[:2]) if hi else " or ".join(s.get("label", "") for s in slots[:2])
        last = (customer.get("relationship") or {}).get("last_visit")
        if kind == "chronic_refill_due" and p.get("molecule_list"):
            meds = ", ".join(p.get("molecule_list") or [])
            body = (f"Namaste {cname}, {biz} se. Aapki {meds} ki supply {str(p.get('stock_runs_out_iso',''))[:10]} tak khatam ho rahi hai. "
                    f"Same order saved address pe bhej dein? Reply YES." if hi else
                    f"Hi {cname}, {biz} here. Your {meds} supply runs out around {str(p.get('stock_runs_out_iso',''))[:10]}. "
                    f"Shall we deliver the same order to your saved address? Reply YES.")
            return {"body": body, "cta": "binary_yes_stop", "rationale": "Refill due date from trigger; one-tap reorder."}
        base = (f"Hi {cname}, {biz} se 🙂 " if hi else f"Hi {cname}, {biz} here 🙂 ")
        if last:
            base += (f"Aapki last visit {last} ko thi. " if hi else f"Your last visit was on {last}. ")
        if kind in ("recall_due",):
            base += ("Aapki next visit ab due hai. " if hi else "Your next visit is now due. ")
        elif kind == "appointment_tomorrow":
            base += ("Kal aapka appointment hai, bas confirm kar rahe hain. " if hi else "Just confirming your appointment tomorrow. ")
        elif kind in ("customer_lapsed_soft", "customer_lapsed_hard", "winback_eligible"):
            base += ("Kaafi time ho gaya, hum aapko miss kar rahe hain. " if hi else "It's been a while, and we'd love to see you again. ")
        elif kind == "trial_followup":
            base += ("Trial kaisa laga? Next session book kar dein? " if hi else "Hope you enjoyed the trial. Want to book your next session? ")
        if offer:
            base += (f"{offer} abhi available hai. " if hi else f"{offer} is available right now. ")
        if slot_txt:
            base += (f"Slots: {slot_txt}. Reply 1 ya 2, ya apna time batayein." if hi else f"Open slots: {slot_txt}. Reply 1 or 2, or tell us a time that suits you.")
            cta = "slot_choice"
        else:
            base += ("Book karna ho toh reply YES." if hi else "Reply YES and we'll book you in.")
            cta = "binary_yes_stop"
        return {"body": base, "cta": cta, "rationale": f"Customer-facing {kind}: uses customer history + merchant's real offer/slots."}

    # merchant-facing: only use kind-specific template if its key facts exist
    REQUIRED = {"perf_dip": ["delta_pct"], "seasonal_perf_dip": ["delta_pct"], "perf_spike": ["delta_pct"],
                "competitor_opened": ["competitor_name", "their_offer"], "renewal_due": ["days_remaining"],
                "milestone_reached": ["value_now", "milestone_value"], "review_theme_emerged": ["theme", "occurrences_30d"],
                "festival_upcoming": ["festival", "date"], "dormant_with_vera": ["days_since_last_merchant_message"],
                "gbp_unverified": ["estimated_uplift_pct"]}
    if any(p.get(k) in (None, "", []) for k in REQUIRED.get(kind, [])):
        kind = "_generic"
    item = find_digest(category, p.get("top_item_id") or p.get("digest_item_id") or p.get("alert_id"))
    peer_ctr = peer.get("avg_ctr")
    ctr = perf.get("ctr")
    if kind in ("research_digest", "regulation_change", "cde_opportunity", "supply_alert") and item:
        fact = f"{item.get('title')} ({item.get('source','')})"
        extra = f" Deadline: {p.get('deadline_iso')}." if p.get("deadline_iso") else ""
        body = (f"{name}, is hafte ka ek relevant update: {fact}.{extra} {(item.get('actionable','').rstrip('.') + '.') if item.get('actionable') else ''} "
                f"Main 2-line summary + patient/customer WhatsApp draft bana doon? Reply YES." if hi else
                f"{name}, one item from this week worth your 2 minutes: {fact}.{extra} {(item.get('actionable','').rstrip('.') + '.') if item.get('actionable') else ''} "
                f"Want me to send a 2-line summary plus a ready-to-share customer note? Reply YES.")
        return {"body": body, "cta": "binary_yes_stop", "rationale": f"{kind} anchored on cited digest item; effort externalised."}
    if kind in ("perf_dip", "seasonal_perf_dip"):
        m, d = p.get("metric", "calls"), pct(p.get("delta_pct"))
        seasonal = " Yeh season mein expected hai, par" if (hi and p.get("is_expected_seasonal")) else (" This is partly seasonal, but" if p.get("is_expected_seasonal") else "")
        body = (f"{name}, pichhle {p.get('window','7d')} mein aapke {m} {d} gire hain.{seasonal} {loc} mein jo listings fresh post + offer chala rahi hain woh hold kar rahi hain. "
                f"Maine {offer or 'ek offer'} ke saath ek Google post draft kiya hai. Publish kar doon? Reply YES." if hi else
                f"{name}, your {m} are down {d} over the last {p.get('window','7d')}.{seasonal} listings in {loc} that post weekly with a clear offer are holding up. "
                f"I've drafted a Google post featuring {offer or 'your top service'}. Publish it? Reply YES.")
        return {"body": body, "cta": "binary_yes_stop", "rationale": "Loss aversion on real dip + effort externalisation."}
    if kind == "perf_spike":
        m, d = p.get("metric", "views"), pct(p.get("delta_pct"))
        drv = str(p.get("likely_driver", "")).replace("_", " ")
        body = (f"{name}, good news: {m} {d} upar hain is hafte" + (f", lagta hai '{drv}' kaam kar gaya" if drv else "") + ". Isi momentum pe ek follow-up post daal dein? Draft ready hai, reply YES." if hi else
                f"{name}, nice one: {m} are up {d} this week" + (f", looks like the '{drv}' drove it" if drv else "") + ". Want to ride the momentum with a follow-up post? Draft's ready, reply YES.")
        return {"body": body, "cta": "binary_yes_stop", "rationale": "Positive reinforcement on real spike; low-effort next step."}
    if kind == "competitor_opened":
        body = (f"{name}, {p.get('competitor_name','ek naya competitor')} {p.get('distance_km','')}km door khula hai ({p.get('opened_date','')}) aur '{p.get('their_offer','')}' chala raha hai. "
                f"Aapka {offer or 'offer'} highlight karke ek post + review push kar dein? Reply YES." if hi else
                f"{name}, {p.get('competitor_name','a new competitor')} opened {p.get('distance_km','')}km away on {p.get('opened_date','')} with '{p.get('their_offer','')}'. "
                f"Want me to push a post highlighting your {offer or 'offer'} and your reviews? Reply YES.")
        return {"body": body, "cta": "binary_yes_stop", "rationale": "Competitor facts from trigger; loss aversion."}
    if kind == "renewal_due":
        body = (f"{name}, aapka {p.get('plan','')} plan {p.get('days_remaining','')} din mein khatam ho raha hai (₹{p.get('renewal_amount','')}). Last 30 din: {perf.get('views')} views, {perf.get('calls')} calls. Renew link bhej doon? Reply YES." if hi else
                f"{name}, your {p.get('plan','')} plan ends in {p.get('days_remaining','')} days (₹{p.get('renewal_amount','')}). Last 30 days it brought {perf.get('views')} views and {perf.get('calls')} calls. Shall I send the renewal link? Reply YES.")
        return {"body": body, "cta": "binary_yes_stop", "rationale": "Renewal framed on value delivered."}
    if kind == "milestone_reached":
        body = (f"{name}, aap {p.get('value_now')} {str(p.get('metric','')).replace('_',' ')} pe ho, {p.get('milestone_value')} bas thoda door hai! Recent happy customers ko ek review-request WhatsApp bhej dein? Reply YES." if hi else
                f"{name}, you're at {p.get('value_now')} {str(p.get('metric','')).replace('_',' ')}, just short of {p.get('milestone_value')}! Want me to send a review request to your recent happy customers? Reply YES.")
        return {"body": body, "cta": "binary_yes_stop", "rationale": "Milestone proximity + effort externalisation."}
    if kind == "review_theme_emerged":
        body = (f"{name}, pichhle 30 din mein {p.get('occurrences_30d')} reviews mein '{str(p.get('theme','')).replace('_',' ')}' aaya hai, jaise: \"{p.get('common_quote','')}\". Ek polite public reply draft kar doon? Reply YES." if hi else
                f"{name}, '{str(p.get('theme','')).replace('_',' ')}' came up in {p.get('occurrences_30d')} reviews in the last 30 days, e.g. \"{p.get('common_quote','')}\". Want me to draft a calm public reply for these? Reply YES.")
        return {"body": body, "cta": "binary_yes_stop", "rationale": "Review theme from trigger; reputation loss aversion."}
    if kind == "festival_upcoming":
        body = (f"{name}, {p.get('festival')} {p.get('date')} ko hai. Maine {offer or 'aapki top service'} pe ek festive post draft kiya hai. Schedule kar doon? Reply YES." if hi else
                f"{name}, {p.get('festival')} is on {p.get('date')}. I've drafted a festive post around {offer or 'your top service'}. Want me to schedule it? Reply YES.")
        return {"body": body, "cta": "binary_yes_stop", "rationale": "Festival timing + ready draft."}
    if kind == "curious_ask_due":
        body = (f"{name}, quick sawaal: is hafte {loc} mein sabse zyada kaunsi service maangi ja rahi hai? Aap batao, main uspe ek post + offer set kar dungi." if hi else
                f"{name}, quick one: which service are customers in {loc} asking for most this week? Tell me and I'll turn it into a post and offer for you.")
        return {"body": body, "cta": "open_ended", "rationale": "Asking-the-merchant lever; builds engagement habit."}
    if kind == "dormant_with_vera":
        body = (f"{name}, {p.get('days_since_last_merchant_message')} din ho gaye baat kiye. Is beech aapki listing pe {perf.get('views')} views aaye (30 din). Ek 2-min profile check kar doon? Reply YES." if hi else
                f"{name}, it's been {p.get('days_since_last_merchant_message')} days. Meanwhile your listing got {perf.get('views')} views in 30 days. Want a quick 2-minute profile check? Reply YES.")
        return {"body": body, "cta": "binary_yes_stop", "rationale": "Re-engagement with a real number, low-effort ask."}
    if kind == "gbp_unverified":
        body = (f"{name}, aapka Google profile abhi unverified hai. Verify hone pe typically ~{pct(p.get('estimated_uplift_pct'))} zyada visibility milti hai. Process ({str(p.get('verification_path','')).replace('_',' ')}) main step-by-step karwa doon? Reply YES." if hi else
                f"{name}, your Google profile is still unverified. Verification typically adds ~{pct(p.get('estimated_uplift_pct'))} visibility. Want me to walk you through it ({str(p.get('verification_path','')).replace('_',' ')})? Reply YES.")
        return {"body": body, "cta": "binary_yes_stop", "rationale": "Unverified status + uplift estimate from trigger."}
    # generic but still data-anchored
    comp = ""
    if ctr and peer_ctr:
        comp = (f" Aapka CTR {ctr*100:.1f}% hai vs peer avg {peer_ctr*100:.1f}%." if hi else f" Your CTR is {ctr*100:.1f}% vs a peer average of {peer_ctr*100:.1f}%.")
    topic = str(p.get("intent_topic") or p.get("season") or p.get("match") or trigger.get("kind", "")).replace("_", " ")
    body = (f"{name}, '{topic}' ke liye maine ek ready plan draft kiya hai.{comp} {offer + ' ko lead offer rakhte hain. ' if offer else ''}Bhej doon? Reply YES." if hi else
            f"{name}, I've put together a ready plan for '{topic}'.{comp} {'Leading with ' + offer + '. ' if offer else ''}Want me to send it over? Reply YES.")
    return {"body": body, "cta": "binary_yes_stop", "rationale": f"{kind}: trigger topic + merchant metrics, effort externalised."}


# ----------------------------------------------------------------------------- REPLY ROUTER
AUTO_PATTERNS = [
    r"thank(s| you) for (contacting|reaching|your message)", r"automated (assistant|message|reply)", r"auto.?reply",
    r"we will (get back|respond|contact)", r"will get back to you", r"our team will", r"currently (unavailable|closed|away)",
    r"out of office", r"business hours", r"jaankari ke liye.*shukriya", r"team tak pahuncha", r"hum jald hi", r"aapka sandesh",
]
STOP_PATTERNS = [r"\bstop\b", r"not interested", r"unsubscribe", r"don'?t (message|contact|text)", r"band karo", r"mat bhejo",
                 r"nahi chahiye", r"leave me alone", r"no thanks", r"remove me", r"interest nahi"]
HOSTILE_PATTERNS = [r"\bidiot\b", r"\bstupid\b", r"\bfool\b", r"useless", r"bakwas", r"\bspam\b", r"\bscam\b", r"shut up", r"bewakoof",
                    r"\bf+u+c+k", r"\bbc\b", r"\bmc\b", r"chutiya", r"pagal", r"harass", r"get lost", r"\bdamn\b", r"bloody"]
COMMIT_PATTERNS = [r"\byes\b", r"\bhaan\b", r"\bha\b", r"\bhan\b", r"\bok(ay)?\b", r"let'?s do it", r"go ahead", r"\bdo it\b", r"\bkaro\b",
                   r"kar do", r"chalo", r"\bsure\b", r"\bdone\b", r"\bproceed\b", r"sounds good", r"\bsend\b", r"bhej do",
                   r"i want to join", r"judna hai", r"judrna", r"sign me up", r"\bstart\b", r"publish", r"theek hai", r"thik hai"]
OFFTOPIC_PATTERNS = [r"\bgst\b", r"income tax", r"\bitr\b", r"\bloan\b", r"insurance", r"\bvisa\b", r"passport", r"cricket score",
                     r"stock (tip|market)", r"crypto", r"\bpolitic", r"homework", r"\bbitcoin\b"]


def _any(pats, text):
    return any(re.search(p, text, re.I) for p in pats)


def _norm(t):
    return re.sub(r"\W+", " ", t.lower()).strip()


SYSTEM_REPLY = """You are Vera, magicpin's WhatsApp assistant for Indian local merchants, continuing a live conversation.
Rules: never re-introduce yourself; 1-3 short sentences; use only facts from the provided context (never invent numbers, names, prices);
peer tone; follow the LANGUAGE given (match what the merchant just used); answer their actual question first, then move ONE step forward toward the goal of the original trigger;
if the merchant has agreed to something, DO it (confirm what you did / are doing with specifics) — do not ask qualifying questions;
end with at most one clear question or CTA. Never repeat an earlier message verbatim.
Return JSON only: {"body": "...", "cta": "binary_yes_stop" | "open_ended" | "none", "rationale": "..."}"""


def reply_lang_hinglish(msg, default):
    hindi_markers = r"\b(hai|haan|nahi|kya|karo|kar|mujhe|aap|hum|bhai|ji|theek|acha|accha|kaise|kab|kitna|chahiye|mera|meri|dekho|batao|abhi)\b"
    if re.search(r"[\u0900-\u097F]", msg) or re.search(hindi_markers, msg, re.I):
        return True
    if re.fullmatch(r"[A-Za-z0-9 ,.'!?\-]+", msg.strip()) and len(msg.split()) >= 4:
        return False
    return default


def respond(conv: dict, message: str) -> dict:
    """Route the merchant/customer reply. conv holds merchant/category/trigger/customer + turns."""
    merchant = conv.get("merchant") or {}
    category = conv.get("category") or {}
    trigger = conv.get("trigger") or {}
    customer = conv.get("customer")
    text = message or ""
    hi = reply_lang_hinglish(text, wants_hinglish(merchant, customer))
    name = salutation(category, merchant) if merchant else ""
    prior_inbound = [_norm(t["msg"]) for t in conv["turns"] if t["from"] != "bot"][:-1]

    # 1. auto-reply: canned pattern or verbatim repeat
    is_auto = _any(AUTO_PATTERNS, text) or (_norm(text) in prior_inbound and len(text) > 15)
    if is_auto:
        conv["auto_count"] = conv.get("auto_count", 0) + 1
        if conv["auto_count"] == 1:
            return {"action": "send",
                    "body": ("Lagta hai yeh auto-reply hai 🙂 Jab owner/manager free hon, bas 'YES' reply kar dein, main 2 minute mein baaki sab kar dungi."
                             if hi else "Looks like an auto-reply 🙂 Whenever the owner or manager sees this, just reply 'YES' and I'll handle the rest in 2 minutes."),
                    "cta": "binary_yes_stop",
                    "rationale": "Auto-reply detected (canned text/verbatim repeat); one short owner-directed nudge, no more turns wasted."}
        if conv["auto_count"] == 2:
            return {"action": "wait", "wait_seconds": 86400,
                    "rationale": "Second consecutive auto-reply; backing off 24h instead of burning turns."}
        return {"action": "end", "rationale": "Repeated auto-replies (3+); exiting gracefully to avoid spam."}

    # 2. explicit stop
    if _any(STOP_PATTERNS, text) and not re.search(r"\b(yes|haan|ok)\b", text, re.I):
        return {"action": "end", "rationale": "Merchant opted out / not interested; ending respectfully, no further messages."}

    hostile = _any(HOSTILE_PATTERNS, text)
    offtopic = _any(OFFTOPIC_PATTERNS, text)

    # 3. hostile
    if hostile:
        conv["hostile_count"] = conv.get("hostile_count", 0) + 1
        if conv["hostile_count"] >= 2:
            return {"action": "end", "rationale": "Repeated hostility; exiting politely to protect the relationship."}
        extra = ""
        if offtopic:
            extra = (" GST/tax jaise kaam mein main help nahi kar sakti, main sirf aapki listing aur customers pe kaam karti hoon."
                     if hi else " I can't help with GST/tax, I only work on your listing and customers.")
        return {"action": "send",
                "body": (f"Sorry agar message pareshaan karne wala laga, {name}.{extra} Aap chahein toh main band kar deti hoon, warna ek line mein batayein kya kaam ka hoga."
                         if hi else f"Sorry if that felt intrusive, {name}.{extra} Happy to stop messaging, or tell me in one line what would actually be useful.").replace(" ,", ","),
                "cta": "open_ended", "rationale": "Hostile reply: de-escalate once, offer opt-out, stay on mission."}

    # 4. off-topic
    if offtopic:
        return {"action": "send",
                "body": ("Yeh mere scope se bahar hai, iske liye CA/expert behtar rahega. Main aapki Google listing, offers aur customer follow-ups sambhalti hoon. Jo pehle baat ho rahi thi, woh aage badhaun? Reply YES."
                         if hi else "That's outside what I can help with, a CA would be the right person. I handle your Google listing, offers and customer follow-ups. Shall I continue with what we were discussing? Reply YES."),
                "cta": "binary_yes_stop", "rationale": "Off-topic request declined politely; redirected to original trigger."}

    # 5. commitment -> action mode
    committed = _any(COMMIT_PATTERNS, text) and len(text.split()) <= 25 and "?" not in text
    mode = "ACTION MODE: the merchant just agreed. Confirm you are doing it now, state concretely what you've done/drafted and when it goes live. Do NOT ask any qualifying question." if committed else \
        "Respond to what they said; if it's a question, answer it from the context; then one step forward."

    history = "\n".join(f"{'VERA' if t['from']=='bot' else 'THEM'}: {t['msg']}" for t in conv["turns"][-8:])
    user = (f"LANGUAGE: {'Hinglish (Roman-script Hindi-English mix)' if hi else 'English'}\n{mode}\n\n"
            f"TRIGGER: {json.dumps(trigger, ensure_ascii=False)}\n"
            f"MERCHANT: {json.dumps({k: merchant.get(k) for k in ['identity','performance','offers','signals','customer_aggregate']}, ensure_ascii=False)}\n"
            f"CATEGORY voice/offers/peers: {json.dumps({k: category.get(k) for k in ['voice','offer_catalog','peer_stats']}, ensure_ascii=False)}\n"
            + (f"CUSTOMER: {json.dumps(customer, ensure_ascii=False)}\n" if customer else "")
            + f"\nCONVERSATION SO FAR:\n{history}\n\nWrite Vera's next message.")
    out = llm_json(SYSTEM_REPLY, user, timeout=18)
    if not (isinstance(out, dict) and str(out.get("body", "")).strip()):
        offer = active_offer(merchant, category)
        if committed:
            body = (f"Done 👍 Kaam shuru kar diya hai. {('Draft ' + offer + ' ke saath') if offer else 'Draft'} 10 minute mein yahin bhejti hoon, aap bas approve kar dena."
                    if hi else f"Done 👍 On it now. I'll share the {('draft featuring ' + offer) if offer else 'draft'} here within 10 minutes; you just approve it.")
            cta = "none"
        else:
            body = ("Samajh gayi. Aapke liye sabse simple next step: main draft bana ke bhejti hoon, aap sirf approve karna. Chalega? Reply YES."
                    if hi else "Got it. Simplest next step: I prepare the draft and you just approve it. Shall I? Reply YES.")
            cta = "binary_yes_stop"
        out = {"body": body, "cta": cta, "rationale": "Fallback reply: action if committed, else single low-effort CTA."}
    # anti-repetition
    sent = [t["msg"] for t in conv["turns"] if t["from"] == "bot"]
    if out["body"].strip() in sent:
        out["body"] = out["body"].strip() + (" (Aapke reply ka wait kar rahi hoon.)" if hi else " (Standing by for your go-ahead.)")
    return {"action": "send", "body": out["body"].strip(), "cta": out.get("cta", "open_ended"),
            "rationale": (("[action-mode] " if committed else "") + str(out.get("rationale", ""))).strip()}


# ----------------------------------------------------------------------------- ENDPOINTS
@app.get("/")
async def root():
    return {"service": "Vera 2.0 — magicpin AI challenge bot", "endpoints": ["/v1/healthz", "/v1/metadata", "/v1/context", "/v1/tick", "/v1/reply"]}


@app.get("/v1/healthz")
async def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _) in contexts:
        counts[scope] = counts.get(scope, 0) + 1
    return {"status": "ok", "uptime_seconds": int(time.time() - START), "contexts_loaded": counts}


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": os.environ.get("TEAM_NAME", "Vera 2.0"),
        "team_members": [m.strip() for m in os.environ.get("TEAM_MEMBERS", "Candidate").split(",")],
        "model": LLM_MODEL if LLM_API_KEY else "template-fallback",
        "approach": "Trigger-routed LLM composer over 4 contexts with fact-grounding + taboo/format validation and deterministic template fallback; rule-based reply router (auto-reply, opt-out, hostile, off-topic, commit->action mode) + LLM for open replies.",
        "contact_email": os.environ.get("CONTACT_EMAIL", "candidate@example.com"),
        "version": "1.0.0",
        "submitted_at": "2026-09-26T00:00:00Z",
    }


@app.post("/v1/context")
async def push_context(req: Request):
    try:
        b = await req.json()
    except Exception:
        return JSONResponse(status_code=400, content={"accepted": False, "reason": "malformed_json"})
    scope, cid, ver, payload = b.get("scope"), b.get("context_id"), b.get("version"), b.get("payload")
    if scope not in ("category", "merchant", "customer", "trigger"):
        return JSONResponse(status_code=400, content={"accepted": False, "reason": "invalid_scope", "details": str(scope)})
    if not cid or not isinstance(payload, dict):
        return JSONResponse(status_code=400, content={"accepted": False, "reason": "missing_fields"})
    try:
        ver = int(ver or 1)
    except Exception:
        ver = 1
    cur = contexts.get((scope, cid))
    if cur and cur["version"] > ver:
        return JSONResponse(status_code=409, content={"accepted": False, "reason": "stale_version", "current_version": cur["version"]})
    if cur and cur["version"] == ver:  # idempotent no-op
        return {"accepted": True, "ack_id": f"ack_{cid}_v{ver}", "stored_at": cur.get("stored_at", now_iso())}
    stored = now_iso()
    contexts[(scope, cid)] = {"version": ver, "payload": payload, "stored_at": stored}
    return {"accepted": True, "ack_id": f"ack_{cid}_v{ver}", "stored_at": stored}


def _resolve(trg_id):
    trg = ctx("trigger", trg_id)
    if not trg:
        return None
    mid = trg.get("merchant_id") or (trg.get("payload") or {}).get("merchant_id")
    merchant = ctx("merchant", mid)
    if not merchant:
        return None
    category = ctx("category", merchant.get("category_slug")) or {}
    cid = trg.get("customer_id") or (trg.get("payload") or {}).get("customer_id")
    customer = ctx("customer", cid) if cid else None
    if trg.get("scope") == "customer" and not customer:
        return None
    return trg, merchant, category, customer


def _expired(trg, now):
    try:
        exp = datetime.fromisoformat(str(trg.get("expires_at")).replace("Z", "+00:00"))
        n = datetime.fromisoformat(str(now).replace("Z", "+00:00"))
        return exp < n
    except Exception:
        return False


def _template_params(msg, merchant, customer):
    who = (customer or {}).get("identity", {}).get("name") if customer else salutation(None, merchant)
    return [who or "", merchant.get("identity", {}).get("name", ""), msg["body"][:120]]


pool = ThreadPoolExecutor(max_workers=10)


@app.post("/v1/tick")
async def tick(req: Request):
    try:
        b = await req.json()
    except Exception:
        b = {}
    now = b.get("now") or now_iso()
    jobs = []
    busy_merchants = set()
    ids = list(b.get("available_triggers") or [])
    # higher urgency first
    ids.sort(key=lambda t: -int((ctx("trigger", t) or {}).get("urgency", 0) or 0))
    for trg_id in ids:
        r = _resolve(trg_id)
        if not r:
            continue
        trg, merchant, category, customer = r
        key = trg.get("suppression_key") or trg_id
        if key in sent_suppression or _expired(trg, now):
            continue
        pair = (merchant.get("merchant_id"), (customer or {}).get("customer_id"))
        if pair in busy_merchants:
            continue
        busy_merchants.add(pair)
        jobs.append((trg_id, r))
        if len(jobs) >= 20:
            break

    futures = {pool.submit(compose, r[2], r[1], r[0], r[3]): (tid, r) for tid, r in jobs}
    done, _ = wait(futures, timeout=22)
    actions = []
    for fut, (tid, (trg, merchant, category, customer)) in futures.items():
        if fut in done and not fut.exception():
            msg = fut.result()
        else:
            fb = fallback_compose(category, merchant, trg, customer)
            msg = {"body": fb["body"], "cta": fb["cta"], "rationale": fb["rationale"],
                   "send_as": "merchant_on_behalf" if (trg.get("scope") == "customer" and customer) else "vera",
                   "suppression_key": trg.get("suppression_key") or tid}
        conv_id = f"conv_{uuid.uuid4().hex[:10]}"
        conversations[conv_id] = {"merchant": merchant, "category": category, "trigger": trg, "customer": customer,
                                  "turns": [{"from": "bot", "msg": msg["body"]}], "auto_count": 0}
        sent_suppression.add(msg["suppression_key"])
        actions.append({
            "conversation_id": conv_id,
            "merchant_id": merchant.get("merchant_id"),
            "customer_id": (customer or {}).get("customer_id"),
            "send_as": msg["send_as"],
            "trigger_id": tid,
            "template_name": f"vera_{trg.get('kind','generic')}_v1",
            "template_params": _template_params(msg, merchant, customer),
            "body": msg["body"],
            "cta": msg["cta"],
            "suppression_key": msg["suppression_key"],
            "rationale": msg["rationale"],
        })
    return {"actions": actions}


@app.post("/v1/reply")
async def reply(req: Request):
    try:
        b = await req.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "malformed_json"})
    cid = b.get("conversation_id") or f"conv_unknown_{uuid.uuid4().hex[:6]}"
    conv = conversations.get(cid)
    if not conv:  # unknown conversation (e.g. replay test) — rebuild what we can
        merchant = ctx("merchant", b.get("merchant_id")) or {"merchant_id": b.get("merchant_id"), "identity": {}}
        conv = {"merchant": merchant, "category": ctx("category", merchant.get("category_slug")) or {},
                "trigger": {}, "customer": ctx("customer", b.get("customer_id")) if b.get("customer_id") else None,
                "turns": [], "auto_count": 0}
        conversations[cid] = conv
    if conv.get("ended"):
        return {"action": "end", "rationale": "Conversation already closed."}
    msg = str(b.get("message", ""))
    conv["turns"].append({"from": b.get("from_role", "merchant"), "msg": msg})
    out = respond(conv, msg)
    if out["action"] == "send":
        conv["turns"].append({"from": "bot", "msg": out["body"]})
    elif out["action"] == "end":
        conv["ended"] = True
    return out


@app.post("/v1/teardown")
async def teardown():
    contexts.clear(); conversations.clear(); sent_suppression.clear()
    return {"ok": True}


# ----------------------------------------------------------------------------- submission generator
def load_local_dataset():
    x = json.load(open(os.path.join(BASE_DIR, "dataset.json"), encoding="utf-8"))
    ms = {m["merchant_id"]: m for m in x["merchants"]}
    cs = {c["customer_id"]: c for c in x["customers"]}
    ts = {t["id"]: t for t in x["triggers"]}
    return x["categories"], ms, cs, ts, x["pairs"]


def build_submission():
    cats, ms, cs, ts, pairs = load_local_dataset()

    def one(p):
        m = ms[p["merchant_id"]]; t = ts[p["trigger_id"]]
        c = cs.get(p.get("customer_id")) if p.get("customer_id") else None
        out = compose(cats.get(m["category_slug"], {}), m, t, c)
        return {"test_id": p["test_id"], **out}

    rows = list(pool.map(one, pairs))
    rows.sort(key=lambda r: r["test_id"])
    return rows


@app.get("/v1/submission")
async def submission():
    rows = build_submission()
    return PlainTextResponse("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                             headers={"Content-Disposition": "attachment; filename=submission.jsonl"})


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "submission":
        with open("submission.jsonl", "w") as f:
            for r in build_submission():
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("wrote submission.jsonl")
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
