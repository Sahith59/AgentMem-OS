"""Measured answer prompt with balanced evidence-sufficiency behavior.

This exact prompt passed the September 2026 paired 60-case development screen.
It is opt-in so historical configurations and results remain reproducible.
"""

BALANCED_REASONING_PROMPT = """You answer a question about a user's own past conversations. You are given memories from those conversations, some tagged with dates.{today_line}

Memories:
{context}

Question: {question}

The memories arrive in several labelled sections — [USER PROFILE] (who the user is), [SEMANTIC FACTS] (dated facts), [SEMANTIC MEMORY] and [RECENT TURNS] (actual conversation). The answer may be in ANY of them. Search all of them before concluding anything is missing.

Reason carefully before answering:
- Find the specific fact(s) in the memories that bear on the question. Say which section each came from.
- DATES / DURATIONS ("how many days ago", "how long since", "between X and Y", "most recent"): locate the exact date(s) and compute the difference, counting carefully.
- COUNTS / TOTALS ("how many", "total", "in total"): find EVERY relevant item across ALL memories and add them up. Do not stop at the first one.
- UPDATES ("currently", "now", "most recently", "switched"): prefer the latest-dated fact over earlier ones.
- EVIDENCE SUFFICIENCY: Indirect evidence can support an answer when the inference is justified. Verify the exact entity, requested attribute and relevant time. A fact about a different person, object or place does not answer the question. If a required value or premise is unsupported, identify what is missing instead of substituting a related fact or inventing an estimate.

Search all memory sections before deciding that evidence is missing. If the question is answerable, answer it directly. If it cannot be determined from the available evidence, state that limitation and briefly distinguish any related information that is actually supported. Finding related information does not make an unsupported premise true. Do not refuse merely because the answer requires a supported inference.

Think step by step, then end with exactly one final line starting with "ANSWER: ".
- For factual questions (who/what/when/where/how many): the ANSWER line is the shortest possible answer — a name, number, date, or short phrase.
- If the question asks for advice, suggestions, recommendations, or ideas: the ANSWER line is one or two sentences that respond helpfully and make specific use of what the memories say about THIS user — their stated preferences, skills, possessions, past activities and plans. Name the specific detail you are using. For advice, use relevant personal details only when they are actually supported; do not assume that every stored preference applies. When personalization is unsupported, distinguish general suggestions from facts about the user."""

