"""
Vera 2.0 — magicpin merchant AI assistant (challenge submission)

Endpoints: /v1/context, /v1/tick, /v1/reply, /v1/healthz, /v1/metadata  (+ /v1/teardown, /v1/submission)
Composer: LLM (any OpenAI-compatible provider, auto-detected from key) -> validation -> deterministic template fallback.
Replies: rule-based router first (auto-reply, stop, hostile, off-topic, commit-intent), LLM for everything else.
"""
import os, re, json, time, uuid, threading, asyncio
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, wait
from urllib import request as urlrequest, error as urlerror

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, HTMLResponse

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
        d = ("https://api.groq.com/openai/v1", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "openai/gpt-oss-20b"])
    elif k.startswith("AIza"):
        d = ("https://generativelanguage.googleapis.com/v1beta/openai", ["gemini-flash-lite-latest"])
    elif k.startswith("sk-or-"):
        d = ("https://openrouter.ai/api/v1", ["meta-llama/llama-3.3-70b-instruct:free"])
    else:
        d = ("https://api.openai.com/v1", ["gpt-4o-mini"])
    models = [m.strip() for m in model.split(",")] if model else d[1]
    return (base or d[0]).rstrip("/"), models


LLM_BASE, LLM_MODELS = _provider()
LLM_MODEL = LLM_MODELS[0]
_llm_gate = threading.Semaphore(int(os.environ.get("LLM_CONCURRENCY", "2")))
_cooldown = {}  # model -> unix time until which we skip it (rate-limited)
LLM_STATS = {"ok": 0, "fail": 0, "last_error": None}


PREFERRED_MODELS = ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "meta-llama/llama-4-maverick-17b-128e-instruct",
                    "meta-llama/llama-4-scout-17b-16e-instruct", "moonshotai/kimi-k2-instruct", "qwen/qwen3-32b",
                    "openai/gpt-oss-20b", "llama-3.1-8b-instant", "gemma2-9b-it"]
_SKIP = ("whisper", "tts", "guard", "playai", "compound", "distil", "embed", "vision", "orpheus", "allam")
MODEL_ERRORS: dict = {}
_discovered = {"done": False, "available": None}


def discover_models():
    """Ask the provider which models exist and rank them, so a retired model name never breaks the bot."""
    global LLM_MODELS, LLM_MODEL
    if _discovered["done"] or not LLM_API_KEY or os.environ.get("LLM_MODEL"):
        _discovered["done"] = True
        return
    try:
        req = urlrequest.Request(LLM_BASE + "/models", headers={"Authorization": f"Bearer {LLM_API_KEY}"})
        with urlrequest.urlopen(req, timeout=10) as r:
            ids = [m.get("id", "") for m in json.loads(r.read().decode()).get("data", [])]
        chat = [i for i in ids if i and not any(k in i.lower() for k in _SKIP)]
        _discovered["available"] = chat
        ranked = [m for m in PREFERRED_MODELS if m in chat] + [m for m in chat if m not in PREFERRED_MODELS]
        if ranked:
            LLM_MODELS = ranked[:4]
            LLM_MODEL = LLM_MODELS[0]
        print("LLM models:", LLM_MODELS)
    except Exception as e:
        print("model discovery failed:", repr(e)[:200])
    _discovered["done"] = True


def _call(model, system, user, timeout, json_mode=True, extras=True):
    body = {"model": model, "temperature": 0, "max_tokens": 500,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if extras and "gpt-oss" in model:
        body.update(reasoning_effort="low", max_tokens=1500)
    if extras and "qwen3" in model:
        body.update(reasoning_format="hidden", max_tokens=1500)
    req = urlrequest.Request(LLM_BASE + "/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json", "Authorization": f"Bearer {LLM_API_KEY}"})
    with urlrequest.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode())
    text = data["choices"][0]["message"].get("content") or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"^```(json)?|```$", "", text.strip()).strip()
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0) if m else text)


def llm_json(system: str, user: str, timeout: float = 18.0):
    """OpenAI-compatible call with model cascade + rate-limit cooldown. Returns dict or None (-> template fallback)."""
    if not LLM_API_KEY:
        return None
    if not _discovered["done"]:
        discover_models()
    deadline = time.time() + timeout
    with _llm_gate:
        ready = [_cooldown.get(m, 0) for m in LLM_MODELS]
        wait = min(ready) - time.time()
        if 0 < wait < deadline - time.time() - 5:
            time.sleep(wait)  # every model rate-limited: wait it out if the caller's budget allows
        for model in list(LLM_MODELS):
            if _cooldown.get(model, 0) > time.time():
                continue
            attempts = [(True, True), (False, False)]  # full request, then plain (no json mode / extras)
            for json_mode, extras in attempts:
                left = deadline - time.time()
                if left < 3:
                    break
                try:
                    out = _call(model, system, user, min(left, 15), json_mode, extras)
                    LLM_STATS["ok"] += 1
                    MODEL_ERRORS.pop(model, None)
                    return out
                except urlerror.HTTPError as e:
                    try:
                        detail = e.read().decode()[:180]
                    except Exception:
                        detail = ""
                    LLM_STATS["last_error"] = MODEL_ERRORS[model] = f"{e.code} {detail}"
                    print("LLM warn:", model, e.code, detail)
                    if e.code == 429:
                        ra = e.headers.get("retry-after") if e.headers else None
                        try:
                            wait = min(float(ra), 60) if ra else 20
                        except Exception:
                            wait = 20
                        _cooldown[model] = time.time() + wait
                        break
                    if e.code in (404, 403):
                        _cooldown[model] = time.time() + 3600  # model gone / not allowed: skip it for an hour
                        break
                    if e.code == 400 and json_mode:
                        continue
                    break
                except Exception as e:
                    LLM_STATS["last_error"] = MODEL_ERRORS[model] = repr(e)[:150]
                    print("LLM warn:", model, repr(e)[:150])
                    break
    LLM_STATS["fail"] += 1
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


def nice_date(iso):
    """'2026-05-12' / '2026-04-28T00:00:00+05:30' -> '12 May'. Falls back to the raw string."""
    try:
        return datetime.fromisoformat(str(iso).replace("Z", "+00:00")).strftime("%-d %b")
    except Exception:
        return str(iso)[:10]


def _is_pct(title):
    return bool(re.search(r"\d+\s*%", str(title)))


def active_offer(merchant, category):
    """Prefer 'Service @ ₹price' offers (brief: service+price beats % off)."""
    act = [o.get("title", "") for o in (merchant.get("offers") or []) if o.get("status") == "active"]
    cat = [o.get("title", "") for o in ((category or {}).get("offer_catalog") or [])]
    for pool in (act, cat):
        for t in pool:
            if "@" in t and "%" not in t:
                return t
    for pool in (act, cat):
        for t in pool:
            if "%" not in t:
                return t
    return (act or cat or [None])[0]


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


def _j(x):
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def build_prompt(category, merchant, trigger, customer):
    payload = trigger.get("payload") or {}
    item_id = payload.get("top_item_id") or payload.get("digest_item_id") or payload.get("alert_id")
    voice = category.get("voice") or {}
    cat_view = {
        "slug": category.get("slug"),
        "tone": voice.get("tone"), "taboo_words": voice.get("vocab_taboo"),
        "allowed_vocab": (voice.get("vocab_allowed") or [])[:10],
        "offer_catalog": [o.get("title") for o in (category.get("offer_catalog") or [])][:6],
        "peer_stats": category.get("peer_stats"),
        "referenced_digest_item": find_digest(category, item_id) if item_id else None,
        "seasonal_beats": (category.get("seasonal_beats") or [])[:2],
        "trend_signals": (category.get("trend_signals") or [])[:2],
    }
    ident = merchant.get("identity") or {}
    m_view = {
        "name": ident.get("name"), "owner_first_name": ident.get("owner_first_name"),
        "locality": ident.get("locality"), "city": ident.get("city"), "verified": ident.get("verified"),
        "subscription": merchant.get("subscription"), "performance": merchant.get("performance"),
        "offers": [f"{o.get('title')} ({o.get('status')})" for o in (merchant.get("offers") or [])],
        "customer_aggregate": merchant.get("customer_aggregate"), "signals": merchant.get("signals"),
        "review_themes": merchant.get("review_themes"),
        "recent_conversation": [f"{t.get('from')}: {t.get('body')}" for t in (merchant.get("conversation_history") or [])[-2:]],
    }
    send_as = "merchant_on_behalf" if (trigger.get("scope") == "customer" and customer) else "vera"
    lang = "Hinglish (Hindi-English code-mix, Roman script)" if wants_hinglish(merchant, customer) else "English"
    who = (f"Address the customer {customer.get('identity', {}).get('name')} as the business '{ident.get('name')}'."
           if send_as == "merchant_on_behalf" else f"Address the merchant as '{salutation(category, merchant)}'.")
    c_view = None
    if customer:
        c_view = {k: customer.get(k) for k in ("identity", "relationship", "state", "preferences", "consent")}
        (c_view.get("identity") or {}).pop("phone_redacted", None)
    user = (f"send_as: {send_as}\nLANGUAGE: {lang}\n{who}\n"
            f"TRIGGER: {_j({k: trigger.get(k) for k in ('kind', 'scope', 'source', 'payload', 'urgency', 'expires_at')})}\n"
            f"MERCHANT: {_j(m_view)}\nCATEGORY: {_j(cat_view)}\n"
            + (f"CUSTOMER: {_j(c_view)}\n" if c_view else "") + "Write the message now.")
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


def compose(category: dict, merchant: dict, trigger: dict, customer: dict | None = None, llm_timeout: float = 18.0) -> dict:
    """Public contract from the brief: returns body, cta, send_as, suppression_key, rationale."""
    category = category or {}
    send_as, user = build_prompt(category, merchant, trigger, customer)
    out = llm_json(SYSTEM_COMPOSE, user, timeout=llm_timeout)
    if not validate(out, category):
        out2 = llm_json(SYSTEM_COMPOSE, user + "\n\nYour previous draft broke a rule (taboo word, % off, or empty). Rewrite strictly.", timeout=llm_timeout) if out else None
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
            body = (f"Namaste {cname}, {biz} se. Aapki {meds} ki supply {nice_date(p.get('stock_runs_out_iso',''))} tak khatam ho rahi hai. "
                    f"Same order saved address pe bhej dein? Reply YES." if hi else
                    f"Hi {cname}, {biz} here. Your {meds} supply runs out around {nice_date(p.get('stock_runs_out_iso',''))}. "
                    f"Shall we deliver the same order to your saved address? Reply YES.")
            return {"body": body, "cta": "binary_yes_stop", "rationale": "Refill due date from trigger; one-tap reorder."}
        base = (f"Hi {cname}, {biz} se 🙂 " if hi else f"Hi {cname}, {biz} here 🙂 ")
        if last:
            base += (f"Aapki last visit {nice_date(last)} ko thi. " if hi else f"Your last visit was on {nice_date(last)}. ")
        if kind in ("recall_due",):
            base += ("Aapki next visit ab due hai. " if hi else "Your next visit is now due. ")
        elif kind == "appointment_tomorrow":
            base += ("Kal aapka appointment hai, bas confirm kar rahe hain. " if hi else "Just confirming your appointment tomorrow. ")
        elif kind == "chronic_refill_due" and (category or {}).get("slug") == "pharmacies":
            base += ("Aapki regular dawaiyon ka refill time aa gaya hai. " if hi else "It's time for your regular medicine refill. ")
            offer = None
        elif kind == "chronic_refill_due":
            base += ("Aapka routine check-in due hai. " if hi else "You're due for a routine check-in. ")
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
            if (category or {}).get("slug") == "pharmacies":
                base += ("Order ready karein? Reply YES." if hi else "Shall we keep your order ready? Reply YES.")
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
    # data-anchored generic (trigger payload missing details): lean on merchant's own numbers vs peers
    views, calls = perf.get("views"), perf.get("calls")
    d7 = perf.get("delta_7d") or {}
    pc = peer.get("avg_calls_30d")
    stat = ""
    if views is not None and calls is not None:
        stat = (f"Pichhle 30 din: {views} views, {calls} calls" + (f" (peer avg {pc} calls)" if pc else "") + "."
                if hi else f"Last 30 days: {views} views, {calls} calls" + (f" (peer avg {pc})" if pc else "") + ".")
    off_txt = (f" {offer} ko lead rakh ke" if hi else f" leading with {offer}") if offer else ""
    tk = trigger.get("kind", "")
    cat_word = {"dentists": "clinic", "salons": "salon", "gyms": "gym", "restaurants": "restaurant", "pharmacies": "pharmacy"}.get((category or {}).get("slug"), "business")
    if tk in ("perf_dip", "perf_spike"):
        want_up = tk == "perf_spike"
        cands = [k for k in ("calls_pct", "views_pct", "ctr_pct") if isinstance(d7.get(k), (int, float)) and (d7[k] > 0) == want_up and d7[k] != 0]
        key = cands[0] if cands else None
        if key:
            val = d7[key]; met = key.split("_")[0]
            up = val > 0
            body = (f"{name}, is hafte aapke {met} {pct(val)} {'upar' if up else 'neeche'} gaye hain. {stat} "
                    f"Maine{off_txt} ek Google post draft kiya hai, {'momentum' if up else 'recovery'} ke liye. Publish kar doon? Reply YES." if hi else
                    f"{name}, your {met} are {'up' if up else 'down'} {pct(val)} this week. {stat} "
                    f"I've drafted a Google post{off_txt} to {'build on it' if up else 'win them back'}. Publish it? Reply YES.")
            return {"body": body, "cta": "binary_yes_stop", "rationale": f"{tk}: real 7-day delta from merchant performance + peer benchmark; effort externalised."}
        if tk == "perf_dip" and calls is not None and pc and calls < pc:
            body = (f"{name}, pichhle 30 din mein aapko {calls} calls aaye, jabki {cat_word}s ka peer avg {pc} hai. "
                    f"Maine{off_txt} ek Google post draft kiya hai taaki search se calls badhein. Publish kar doon? Reply YES." if hi else
                    f"{name}, you got {calls} calls in the last 30 days against a peer average of {pc}. "
                    f"I've drafted a Google post{off_txt} to turn more of those searches into calls. Publish it? Reply YES.")
            return {"body": body, "cta": "binary_yes_stop", "rationale": "perf_dip: merchant calls vs peer average (loss aversion); effort externalised."}
    if tk == "competitor_opened":
        body = (f"{name}, {loc} mein ek naya {cat_word} Google pe live hua hai. {stat} Customers compare karne se pehle{off_txt} ek fresh post + review request push kar dein? Reply YES." if hi else
                f"{name}, a new {cat_word} just went live on Google in {loc}. {stat} Before customers start comparing, shall I push a fresh post{off_txt} plus a review request? Reply YES.")
        return {"body": body, "cta": "binary_yes_stop", "rationale": "Competitor trigger (no name in payload, so none invented) + merchant vs peer numbers; loss aversion."}
    if tk == "milestone_reached":
        tot = (merchant.get("customer_aggregate") or {}).get("total_unique_ytd")
        body = (f"{name}, is saal ab tak {tot} unique customers aapke {cat_word} aa chuke hain 🎉 Ek thank-you post + happy customers ko review request bhej dein? Draft ready hai, reply YES." if hi and tot else
                f"{name}, {tot} unique customers have visited this year 🎉 Want me to send a thank-you post and a review request to happy customers? Draft's ready, reply YES." if tot else
                (f"{name}, aapki listing ek milestone pe pahunchi hai. {stat} Ek thank-you post + review request bhej dein? Reply YES." if hi else
                 f"{name}, your listing just hit a milestone. {stat} Shall I send a thank-you post and review request? Reply YES."))
        return {"body": body, "cta": "binary_yes_stop", "rationale": "Milestone framed on merchant's real customer count; reciprocity + effort externalised."}
    if tk == "festival_upcoming":
        body = (f"{name}, festive season shuru ho raha hai aur log abhi se {cat_word} search kar rahe hain. {stat} Maine{off_txt} ek festive post draft kiya hai. Schedule kar doon? Reply YES." if hi else
                f"{name}, festive season is starting and people are already searching for a {cat_word}. {stat} I've drafted a festive post{off_txt}. Schedule it? Reply YES.")
        return {"body": body, "cta": "binary_yes_stop", "rationale": "Festival trigger + merchant numbers; timeliness + ready draft."}
    if tk == "dormant_with_vera":
        body = (f"{name}, kaafi din se baat nahi hui. Is beech: {stat[len('Pichhle 30 din: '):] if stat else ''} Ek 2-minute profile check karke bataun kya improve ho sakta hai? Reply YES." if hi else
                f"{name}, it's been a while. Meanwhile: {stat[len('Last 30 days: '):] if stat else ''} Want a 2-minute profile check with what to fix first? Reply YES.")
        return {"body": body, "cta": "binary_yes_stop", "rationale": "Re-engagement anchored on the merchant's own recent numbers; low-effort ask."}
    comp = ""
    if ctr and peer_ctr:
        comp = (f" Aapka CTR {ctr*100:.1f}% hai vs peer avg {peer_ctr*100:.1f}%." if hi else f" Your CTR is {ctr*100:.1f}% vs a peer average of {peer_ctr*100:.1f}%.")
    topic = str(p.get("intent_topic") or p.get("season") or p.get("match") or tk).replace("_", " ")
    body = (f"{name}, '{topic}' ke liye maine ek ready plan draft kiya hai.{comp} {offer + ' ko lead offer rakhte hain. ' if offer else ''}Bhej doon? Reply YES." if hi else
            f"{name}, I've put together a ready plan for '{topic}'.{comp} {'Leading with ' + offer + '. ' if offer else ''}Want me to send it over? Reply YES.")
    return {"body": body, "cta": "binary_yes_stop", "rationale": f"{tk}: trigger topic + merchant metrics, effort externalised."}


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
@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return HTMLResponse(render_landing())


@app.get("/v1/healthz")
async def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _) in contexts:
        counts[scope] = counts.get(scope, 0) + 1
    return {"status": "ok", "uptime_seconds": int(time.time() - START), "contexts_loaded": counts,
            "llm": {"model": LLM_MODEL if LLM_API_KEY else None, "calls_ok": LLM_STATS["ok"], "calls_failed": LLM_STATS["fail"]}}


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": os.environ.get("TEAM_NAME", "Aryan Maheshwari — Vera 2.0"),
        "team_members": [m.strip() for m in os.environ.get("TEAM_MEMBERS", "Aryan Maheshwari").split(",")],
        "model": LLM_MODEL if LLM_API_KEY else "template-fallback",
        "approach": "Trigger-routed LLM composer over 4 contexts with fact-grounding + taboo/format validation and deterministic template fallback; rule-based reply router (auto-reply, opt-out, hostile, off-topic, commit->action mode) + LLM for open replies.",
        "contact_email": os.environ.get("CONTACT_EMAIL", "candidate@example.com"),
        "version": "1.1.0",
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



# ----------------------------------------------------------------------------- landing page (for humans)
import html as _html

EXAMPLE_IDS = [("T30", "Dentist · regulation change"), ("T09", "Dentist · competitor opened"), ("T28", "Patient recall · sent as the clinic")]


def _examples():
    rows = {r["test_id"]: r for r in build_submission()}
    try:
        cats, ms, cs, ts, pairs = load_local_dataset()
        names = {p["test_id"]: ms[p["merchant_id"]]["identity"]["name"] for p in pairs}
    except Exception:
        names = {}
    out = []
    for tid, label in EXAMPLE_IDS:
        r = rows.get(tid)
        if r:
            out.append((names.get(tid, ""), label, r["body"], r["rationale"]))
    return out


def render_landing():
    e = _html.escape
    bubbles = "".join(
        f'<figure class="msg"><figcaption><strong>{e(n)}</strong><span>{e(l)}</span></figcaption>'
        f'<p class="bubble">{e(b)}</p><p class="why">Why: {e(w)}</p></figure>' for n, l, b, w in _examples())
    llm = f"{e(LLM_MODEL)}" if LLM_API_KEY else "template composer"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Vera 2.0 — merchant WhatsApp assistant</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>💬</text></svg>">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Hind:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{{--ink:#17202c;--muted:#5d6878;--bg:#f2f4f7;--line:#d9dee6;--wall:#e9e1d6;--out:#d9fdd3;--violet:#4a2fb0;}}
*{{box-sizing:border-box}}html,body{{margin:0}}
body{{background:var(--bg);color:var(--ink);font:17px/1.6 Hind,"Segoe UI",system-ui,sans-serif;
padding:env(safe-area-inset-top,0) 0 env(safe-area-inset-bottom,0)}}
a{{color:var(--violet)}}a:focus-visible{{outline:3px solid var(--violet);outline-offset:2px}}
.wrap{{max-width:1080px;margin:0 auto;padding:48px 24px 72px}}
.hero{{display:grid;grid-template-columns:1fr 1.05fr;gap:48px;align-items:start}}
h1{{font-size:clamp(2rem,4.2vw,3.1rem);line-height:1.08;font-weight:700;margin:0 0 20px;letter-spacing:-.01em}}
.lede{{font-size:1.12rem;color:var(--muted);max-width:34em;margin:0 0 24px}}
.live{{display:inline-flex;gap:10px;align-items:center;font-weight:500;background:#fff;border:1px solid var(--line);padding:8px 14px;border-radius:999px}}
.dot{{width:9px;height:9px;border-radius:50%;background:#1f9d55}}
.chat{{background:var(--wall);border-radius:22px;padding:22px 18px;border:1px solid #ddd2c4}}
.msg{{margin:0 0 18px}}.msg:last-child{{margin:0}}
figcaption{{font-size:.84rem;color:#6b5f52;margin:0 0 4px 4px;display:flex;gap:8px;flex-wrap:wrap}}
figcaption strong{{color:#3d342b}}
.bubble{{background:var(--out);margin:0;padding:10px 14px;border-radius:14px 14px 4px 14px;box-shadow:0 1px 0 rgba(0,0,0,.08);font-size:.98rem;line-height:1.5}}
.why{{font-size:.78rem;color:#7a6e61;margin:6px 4px 0}}
@media (prefers-reduced-motion:no-preference){{.msg{{animation:in .5s ease both}}.msg:nth-child(2){{animation-delay:.35s}}.msg:nth-child(3){{animation-delay:.7s}}
@keyframes in{{from{{opacity:0;transform:translateY(8px)}}to{{opacity:1;transform:none}}}}}}
h2{{font-size:1.5rem;margin:64px 0 16px;font-weight:600}}
.table{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;background:#fff;border:1px solid var(--line);border-radius:12px;overflow:hidden}}
td,th{{text-align:left;padding:12px 16px;border-bottom:1px solid var(--line);vertical-align:top}}th{{font-weight:600;background:#f8f9fb}}
tr:last-child td{{border-bottom:0}}td:first-child{{font-style:italic;color:var(--muted);width:34%}}
ol{{padding-left:1.3em;max-width:44em}}ol li{{margin:0 0 10px}}
code{{background:#fff;border:1px solid var(--line);padding:1px 6px;border-radius:6px;font-size:.88em}}
.ep{{display:flex;flex-wrap:wrap;gap:10px}}.ep a{{background:#fff;border:1px solid var(--line);padding:8px 14px;border-radius:10px;text-decoration:none}}
footer{{margin-top:56px;color:var(--muted);font-size:.9rem}}
@media (max-width:820px){{.hero{{grid-template-columns:1fr}}.wrap{{padding-top:28px}}}}
{DEMO_CSS}
</style></head><body><main class="wrap">
<section class="hero"><div>
<h1>The one WhatsApp a busy merchant actually replies to.</h1>
<p class="lede">Vera 2.0 is my rebuild of magicpin's merchant assistant for the AI Challenge. It reads four layers of context
(category, merchant, trigger, customer), decides whether a message is worth sending, writes it in the merchant's own language
with their real numbers, and handles the reply, including auto-replies, a sudden "let's do it", and a hard no.</p>
<span class="live"><span class="dot" aria-hidden="true"></span>Live · composing with {llm}</span>
<p style="margin:18px 0 0"><a href="#try" style="font-weight:600">Chat with it yourself ↓</a></p>
</div>
<div class="chat" aria-label="Messages composed by this bot from the challenge dataset">{bubbles}</div></section>

{DEMO_HTML}

<h2>When the merchant replies</h2>
<div class="table"><table><tr><th>Merchant says</th><th>What Vera 2.0 does</th></tr>
<tr><td>"Thank you for contacting us, we'll get back to you" (again)</td><td>Spots the canned auto-reply, sends one short note for the owner, backs off 24h, then exits. No wasted turns.</td></tr>
<tr><td>"Ok, let's do it" / "haan karo"</td><td>Switches straight to action and confirms what it's doing. No more qualifying questions.</td></tr>
<tr><td>Abuse, then "can you file my GST?"</td><td>De-escalates once, offers to stop, declines the off-topic ask and steers back to the listing.</td></tr>
<tr><td>"Not interested, stop"</td><td>Ends the conversation and never messages on that thread again.</td></tr>
<tr><td>A real question, in Hindi after starting in English</td><td>Answers from the data it has, in the language used in that turn, then moves one step forward.</td></tr>
</table></div>

<h2>How a message gets written</h2>
<ol>
<li>A trigger arrives (research digest, dip in calls, Diwali, a patient's recall window). Expired or already-sent triggers are dropped.</li>
<li>Only the context that matters is pulled: the cited digest item, peer benchmarks, the active service@price offer, last conversation turns.</li>
<li>An LLM writes the message at temperature 0 under hard rules: lead with why-now, use only facts present, category voice, one CTA.</li>
<li>A validator rejects taboo words ("guaranteed", "cure") and "% off" framing, and re-prompts once.</li>
<li>If the model is slow or unavailable, a deterministic, fact-anchored template takes over, so the bot never times out or sends an empty message.</li>
</ol>

<h2>Endpoints</h2>
<p>This service is scored by magicpin's judge harness over HTTP. Useful read-only links:</p>
<div class="ep"><a href="/v1/healthz">/v1/healthz</a><a href="/v1/metadata">/v1/metadata</a><a href="/v1/status">/v1/status</a><a href="/v1/submission">/v1/submission</a><a href="https://github.com/Hellster666/vera-bot">Source on GitHub</a></div>
<p>Judge-facing: <code>POST /v1/context</code> <code>POST /v1/tick</code> <code>POST /v1/reply</code></p>
<footer>Built by Aryan Maheshwari for the magicpin Tech / Product AI Analyst challenge. Synthetic dataset; no real merchant data.</footer>
</main></body></html>"""



# ----------------------------------------------------------------------------- live demo (for humans; separate from judge state)
DEMO_CONVS: dict = {}


def _scenario_label(p, ms, ts, cs):
    m = ms[p["merchant_id"]]; t = ts[p["trigger_id"]]
    who = m["identity"]["name"]
    kind = t["kind"].replace("_", " ")
    tail = f" → customer {cs[p['customer_id']]['identity']['name']}" if p.get("customer_id") and p["customer_id"] in cs else ""
    return f"{who} ({m['category_slug'][:-1] if m['category_slug'].endswith('s') else m['category_slug']}) · {kind}{tail}"


@app.get("/demo/scenarios")
def demo_scenarios():
    cats, ms, cs, ts, pairs = load_local_dataset()
    return [{"test_id": p["test_id"], "label": _scenario_label(p, ms, ts, cs)} for p in pairs]


@app.post("/demo/start")
async def demo_start(req: Request):
    b = await req.json()
    cats, ms, cs, ts, pairs = load_local_dataset()
    p = next((x for x in pairs if x["test_id"] == b.get("test_id")), pairs[0])
    m = ms[p["merchant_id"]]; t = ts[p["trigger_id"]]
    c = cs.get(p.get("customer_id")) if p.get("customer_id") else None
    cat = cats.get(m["category_slug"], {})
    before = LLM_STATS["ok"]
    msg = await asyncio.to_thread(compose, cat, m, t, c)
    engine = "llm" if LLM_STATS["ok"] > before else "template"
    cid = "demo_" + uuid.uuid4().hex[:10]
    DEMO_CONVS[cid] = {"merchant": m, "category": cat, "trigger": t, "customer": c,
                       "turns": [{"from": "bot", "msg": msg["body"]}], "auto_count": 0}
    if len(DEMO_CONVS) > 500:
        DEMO_CONVS.pop(next(iter(DEMO_CONVS)))
    return {"conversation_id": cid, "engine": engine, "who": ("customer" if c else "merchant"), **msg}


@app.post("/demo/reply")
async def demo_reply(req: Request):
    b = await req.json()
    conv = DEMO_CONVS.get(b.get("conversation_id"))
    if not conv:
        return JSONResponse(status_code=404, content={"error": "unknown conversation, start a new one"})
    if conv.get("ended"):
        return {"action": "end", "rationale": "Conversation already closed."}
    text = str(b.get("message", ""))[:1000]
    conv["turns"].append({"from": "merchant", "msg": text})
    before = LLM_STATS["ok"]
    out = await asyncio.to_thread(respond, conv, text)
    out["engine"] = "llm" if LLM_STATS["ok"] > before else "rules/template"
    if out["action"] == "send":
        conv["turns"].append({"from": "bot", "msg": out["body"]})
    elif out["action"] == "end":
        conv["ended"] = True
    return out


DEMO_HTML = """
<h2 id="try">Try it: chat with Vera as the merchant</h2>
<p class="lede" style="margin-bottom:16px">Pick a scenario from the challenge dataset. Vera writes the opening WhatsApp; you reply as the merchant (or customer) and see how it handles you. Try an auto-reply, a "haan karo", or asking it to file your GST.</p>
<div class="demo">
  <div class="demo-bar">
    <label for="scn" class="sr">Scenario</label>
    <select id="scn"><option>Loading scenarios…</option></select>
    <button id="startBtn" type="button">Start conversation</button>
  </div>
  <div id="thread" class="thread" aria-live="polite"><p class="hint">Vera's first message will appear here.</p></div>
  <div class="chips" id="chips" hidden>
    <button type="button" data-t="Haan, karo">Haan, karo</button>
    <button type="button" data-t="What exactly will you post?">What exactly will you post?</button>
    <button type="button" data-t="Thank you for contacting us. We will get back to you shortly.">Auto-reply</button>
    <button type="button" data-t="Can you also help me file my GST?">GST question</button>
    <button type="button" data-t="Not interested, stop messaging">Not interested</button>
  </div>
  <form id="replyForm" class="reply" hidden>
    <label for="replyIn" class="sr">Your reply</label>
    <input id="replyIn" autocomplete="off" placeholder="Reply as the merchant…">
    <button type="submit">Send</button>
  </form>
</div>
<script>
(function(){
  var scn=document.getElementById('scn'), thread=document.getElementById('thread'), chips=document.getElementById('chips'),
      form=document.getElementById('replyForm'), input=document.getElementById('replyIn'), start=document.getElementById('startBtn'), conv=null, busy=false;
  fetch('/demo/scenarios').then(function(r){return r.json()}).then(function(list){
    scn.innerHTML='';
    list.forEach(function(s){var o=document.createElement('option');o.value=s.test_id;o.textContent=s.test_id+' · '+s.label;scn.appendChild(o)});
    scn.value='T09';
  }).catch(function(){scn.innerHTML='<option>Could not load scenarios</option>'});
  function add(cls,text,meta){
    var w=document.createElement('div');w.className='row '+cls;
    var b=document.createElement('p');b.className='b';b.textContent=text;w.appendChild(b);
    if(meta){var m=document.createElement('p');m.className='meta';m.textContent=meta;w.appendChild(m)}
    thread.appendChild(w);thread.scrollTop=thread.scrollHeight;return w;
  }
  function typing(){return add('bot typing','Vera is typing…')}
  function lock(v){busy=v;start.disabled=v;form.querySelector('button').disabled=v;chips.querySelectorAll('button').forEach(function(x){x.disabled=v})}
  start.addEventListener('click',function(){
    if(busy)return;lock(true);thread.innerHTML='';var t=typing();
    fetch('/demo/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({test_id:scn.value})})
    .then(function(r){return r.json()}).then(function(d){
      t.remove();conv=d.conversation_id;
      add('bot',d.body,'Why: '+d.rationale+'  ·  written by '+(d.engine==='llm'?'the LLM':'the fallback template'));
      input.placeholder='Reply as the '+d.who+'…';chips.hidden=false;form.hidden=false;input.focus();
    }).catch(function(){t.remove();add('sys','Something went wrong. Please try again.')}).finally(function(){lock(false)});
  });
  function send(text){
    if(busy||!conv||!text.trim())return;lock(true);add('me',text);input.value='';var t=typing();
    fetch('/demo/reply',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({conversation_id:conv,message:text})})
    .then(function(r){return r.json()}).then(function(d){
      t.remove();
      if(d.action==='send'){add('bot',d.body,'Why: '+d.rationale)}
      else if(d.action==='wait'){add('sys','Vera backs off for '+Math.round((d.wait_seconds||0)/3600)+'h. '+d.rationale)}
      else{add('sys','Vera ended the conversation. '+(d.rationale||''));chips.hidden=true;form.hidden=true}
    }).catch(function(){t.remove();add('sys','Something went wrong. Please try again.')}).finally(function(){lock(false)});
  }
  form.addEventListener('submit',function(e){e.preventDefault();send(input.value)});
  chips.addEventListener('click',function(e){var b=e.target.closest('button');if(b)send(b.getAttribute('data-t'))});
})();
</script>
"""

DEMO_CSS = """
.sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0)}
.demo{background:#fff;border:1px solid var(--line);border-radius:18px;overflow:hidden}
.demo-bar{display:flex;gap:10px;flex-wrap:wrap;padding:14px;border-bottom:1px solid var(--line);background:#f8f9fb}
.demo select{flex:1 1 320px;min-width:0;font:inherit;padding:9px 10px;border:1px solid var(--line);border-radius:10px;background:#fff;color:var(--ink)}
.demo button{font:inherit;font-weight:600;border:0;border-radius:10px;padding:9px 16px;background:var(--violet);color:#fff;cursor:pointer}
.demo button:disabled{opacity:.55;cursor:default}
.thread{background:var(--wall);height:420px;overflow-y:auto;padding:18px;display:flex;flex-direction:column;gap:12px}
.hint{color:#7a6e61;margin:auto;text-align:center}
.row{max-width:82%}.row .b{margin:0;padding:10px 14px;border-radius:14px;line-height:1.5;box-shadow:0 1px 0 rgba(0,0,0,.08);white-space:pre-wrap}
.row.bot{align-self:flex-start}.row.bot .b{background:#fff;border-radius:14px 14px 14px 4px}
.row.me{align-self:flex-end}.row.me .b{background:var(--out);border-radius:14px 14px 4px 14px}
.row.sys{align-self:center;max-width:92%}.row.sys .b{background:#fff7e0;color:#6b5000;font-size:.9rem;text-align:center}
.row.typing .b{color:#7a6e61;font-style:italic}
.meta{font-size:.75rem;color:#7a6e61;margin:5px 4px 0}
.chips{display:flex;gap:8px;flex-wrap:wrap;padding:12px 14px 0}
.chips button{background:#efeafd;color:var(--violet);font-weight:500;padding:6px 12px;border-radius:999px}
.reply{display:flex;gap:10px;padding:12px 14px 14px}
.reply input{flex:1;min-width:0;font:inherit;padding:10px 12px;border:1px solid var(--line);border-radius:10px}
"""


# ----------------------------------------------------------------------------- submission generator
def load_local_dataset():
    x = json.load(open(os.path.join(BASE_DIR, "dataset.json"), encoding="utf-8"))
    ms = {m["merchant_id"]: m for m in x["merchants"]}
    cs = {c["customer_id"]: c for c in x["customers"]}
    ts = {t["id"]: t for t in x["triggers"]}
    return x["categories"], ms, cs, ts, x["pairs"]


SUB_ROWS: dict = {}
SUB_STATE = {"running": False, "started": None, "finished": None}


def _submission_worker():
    """Builds the 30 canonical messages slowly in the background so free-tier LLM rate limits aren't hit."""
    if SUB_STATE["running"]:
        return
    SUB_STATE.update(running=True, started=now_iso(), finished=None)
    try:
        cats, ms, cs, ts, pairs = load_local_dataset()
        for p in pairs:
            m = ms[p["merchant_id"]]; t = ts[p["trigger_id"]]
            c = cs.get(p.get("customer_id")) if p.get("customer_id") else None
            SUB_ROWS[p["test_id"]] = {"test_id": p["test_id"], **compose(cats.get(m["category_slug"], {}), m, t, c, llm_timeout=45)}
            time.sleep(float(os.environ.get("SUBMISSION_PACE_S", "8")))
    finally:
        SUB_STATE.update(running=False, finished=now_iso())


def build_submission():
    """Rows from the background run where available; deterministic template for any not yet composed."""
    cats, ms, cs, ts, pairs = load_local_dataset()
    rows = []
    for p in pairs:
        if p["test_id"] in SUB_ROWS:
            rows.append(SUB_ROWS[p["test_id"]]); continue
        m = ms[p["merchant_id"]]; t = ts[p["trigger_id"]]
        c = cs.get(p.get("customer_id")) if p.get("customer_id") else None
        cat = cats.get(m["category_slug"], {})
        fb = fallback_compose(cat, m, t, c)
        rows.append({"test_id": p["test_id"], "body": fb["body"], "cta": fb["cta"],
                     "send_as": "merchant_on_behalf" if (t.get("scope") == "customer" and c) else "vera",
                     "suppression_key": t.get("suppression_key"), "rationale": fb["rationale"]})
    return rows


@app.on_event("startup")
def _kickoff():
    if LLM_API_KEY:
        def _go():
            discover_models()
            _submission_worker()
        threading.Thread(target=_go, daemon=True).start()


@app.api_route("/v1/submission", methods=["GET", "HEAD"])
def submission(refresh: int = 0):
    """submission.jsonl for the 30 canonical pairs. Background-composed by the LLM at startup; ?refresh=1 re-runs."""
    if refresh and LLM_API_KEY and not SUB_STATE["running"]:
        SUB_ROWS.clear()
        threading.Thread(target=_submission_worker, daemon=True).start()
    rows = build_submission()
    return PlainTextResponse("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                             media_type="text/plain; charset=utf-8",
                             headers={"X-Composed-By-LLM": str(len(SUB_ROWS)), "X-Worker-Running": str(SUB_STATE["running"])})


@app.get("/v1/status")
async def status():
    return {"llm_model": LLM_MODEL if LLM_API_KEY else None, "llm_models_in_use": LLM_MODELS if LLM_API_KEY else [],
            "provider_models_available": _discovered["available"], "per_model_errors": MODEL_ERRORS,
            "llm_calls_ok": LLM_STATS["ok"], "llm_calls_failed": LLM_STATS["fail"],
            "last_llm_error": LLM_STATS["last_error"], "submission_rows_composed": len(SUB_ROWS), "submission_worker": SUB_STATE}


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
