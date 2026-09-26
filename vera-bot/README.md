# Vera 2.0 — magicpin AI Challenge

**Live endpoints:** `/v1/context`, `/v1/tick`, `/v1/reply`, `/v1/healthz`, `/v1/metadata` (plus `/v1/teardown`, and `/v1/submission` to regenerate `submission.jsonl`).

## Approach
1. **Composer (`compose()`)**: the 4 contexts are trimmed to what matters (the digest item the trigger points to, peer stats, voice rules, active offer, last 4 conversation turns) and sent to an LLM at temperature 0 with hard rules: open with the *why-now* fact, use only data present in context, service+price offers (never "% off"), category voice and taboo list, the merchant's language (Hinglish when `hi` is in languages), one CTA in the last sentence, at least one compulsion lever.
2. **Validation**: output is checked for taboo words, "% off" framing and empty body; one strict re-prompt, then a **deterministic template fallback** per trigger kind (also used if the LLM is slow or unavailable), so the bot never times out or returns an empty message.
3. **Reply router** (rules first, LLM second): auto-reply detection (canned phrases + verbatim repeats) → one owner-directed nudge, then wait 24h, then exit; explicit opt-out → end; hostility → one de-escalation, then exit; off-topic (GST, loans) → polite decline + redirect; **commitment ("ok let's do it", "haan karo") → action mode**, confirming what's being done with no further qualifying questions. Language is re-detected every turn.
4. **Tick policy**: urgency-sorted, suppression-key dedup, expired triggers skipped, one message per merchant per tick, parallel composition within a 22s budget.

## Tradeoffs
- Rules for routing (fast, predictable, auditable) vs. LLM for wording (quality). Commit intent requires no question mark, so "ok?" isn't mistaken for a yes.
- Restraint over volume: expired or duplicate triggers are never sent.
- In-memory state is enough for a 60-minute test; production would use Redis.

## What would have helped most
Reply-rate history per trigger kind and category (to learn which levers actually work), real slot availability for more merchants, and the list of already Meta-approved templates.
