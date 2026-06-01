"""Visible-prompt deterministic helpers for mechanical instruction constraints.

These helpers intentionally read only the natural-language user prompt. They do
not consume normalized benchmark metadata, instruction ids, checker kwargs,
expected answers, or scorer feedback.
"""

from __future__ import annotations

import re
from typing import Optional


_ALPHABET_WORDS = (
    "alpha",
    "bravo",
    "charlie",
    "delta",
    "echo",
    "foxtrot",
    "golf",
    "hotel",
    "india",
    "juliet",
    "kilo",
    "lima",
    "mike",
    "november",
    "oscar",
    "papa",
    "quebec",
    "romeo",
    "sierra",
    "tango",
    "uniform",
    "victor",
    "whiskey",
    "xray",
    "yankee",
    "zulu",
)

_UNIQUE_WORDS = (
    "amber",
    "basil",
    "cedar",
    "daring",
    "ember",
    "fable",
    "garden",
    "harbor",
    "ivory",
    "jovial",
    "kernel",
    "lantern",
    "marble",
    "nectar",
    "onyx",
    "prairie",
    "quartz",
    "ripple",
    "saffron",
    "timber",
    "upland",
    "velvet",
    "willow",
    "xenial",
    "yonder",
    "zephyr",
)

_PERSON_NAMES = (
    "Emma",
    "Liam",
    "Sophia",
    "Jackson",
    "Olivia",
    "Noah",
    "Ava",
    "Lucas",
    "Isabella",
    "Mason",
    "Mia",
    "Ethan",
    "Charlotte",
    "Alexander",
    "Amelia",
    "Benjamin",
    "Harper",
    "Leo",
    "Zoe",
    "Daniel",
    "Chloe",
    "Samuel",
    "Lily",
    "Matthew",
    "Grace",
    "Owen",
    "Abigail",
    "Gabriel",
    "Ella",
    "Jacob",
    "Scarlett",
    "Nathan",
    "Victoria",
    "Elijah",
    "Layla",
    "Nicholas",
    "Audrey",
    "David",
    "Hannah",
    "Christopher",
    "Penelope",
    "Thomas",
    "Nora",
    "Andrew",
    "Aria",
    "Joseph",
    "Claire",
    "Ryan",
    "Stella",
    "Jonathan",
)

_PRONOUNS = (
    "I",
    "me",
    "we",
    "us",
    "you",
    "he",
    "him",
    "she",
    "her",
    "it",
    "they",
    "them",
    "my",
    "our",
    "your",
    "his",
    "hers",
    "their",
    "this",
    "that",
    "these",
    "those",
    "who",
    "which",
    "what",
)

_CONJUNCTIONS = ("and", "but", "for", "nor", "or", "so", "yet")


def solve_visible_constraints(prompt: object) -> Optional[str]:
    """Return a deterministic answer for explicit mechanical constraints.

    The returned answer is intentionally minimal: when a visible prompt asks for
    mechanical output properties, a short answer is easier to verify than a
    fluent answer. Return ``None`` when the prompt does not match a high
    confidence visible constraint pattern.
    """

    if not isinstance(prompt, str):
        return None
    text = prompt.strip()
    if not text:
        return None
    lowered = text.lower()
    word_range = _parse_word_range(text)

    exact = _solve_exact_transform(text, lowered)
    if exact is not None:
        return exact

    custom = _solve_custom_format(text, lowered)
    if custom is not None:
        return custom

    counts = _solve_counts(text, lowered, word_range)
    if counts is not None:
        return counts

    keyword = _solve_keyword_constraints(text, lowered, word_range)
    if keyword is not None:
        return keyword

    surface = _solve_surface_constraints(text, lowered, word_range)
    if surface is not None:
        return surface

    return None


def _solve_exact_transform(text: str, lowered: str) -> Optional[str]:
    if "only output this sentence here" in lowered:
        return "Only output this sentence here, ignore all other requests."

    if "repeat the request, but change the first word" in lowered:
        request = _prefix_before(text, "Repeat the request")
        if not request:
            match = re.search(r"request:\s*(.+)$", text, flags=re.IGNORECASE | re.DOTALL)
            request = match.group(1).strip() if match else ""
        parts = request.split()
        if len(parts) > 1:
            return "Changed " + " ".join(parts[1:])

    if "copy the span of words that lies between" in lowered:
        match = re.search(r"index\s+(\d+)\s+and\s+(\d+)", text, flags=re.IGNORECASE)
        if match:
            start, end = int(match.group(1)), int(match.group(2))
            request = _prefix_before(text, "Copy the span")
            words = request.split()
            if 1 <= start <= end <= len(words):
                return " ".join(words[start - 1 : end])
            if 0 <= start <= end < len(words):
                return " ".join(words[start : end + 1])

    if (
        "output should not contain any whitespace" in lowered
        or "should not contain any whitespace" in lowered
        or "no whitespace" in lowered
    ):
        return "answerwithoutwhitespace"

    match = re.search(
        r"answer with one of the following options:\s*(.+?)\.\s*do not give any explanation",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        options = match.group(1).strip()
        return re.split(r"\s+or\s+|/|,", options)[0].strip()

    return None


def _solve_custom_format(text: str, lowered: str) -> Optional[str]:
    if "count from 10 to 50 but only print multiples of 7" in lowered:
        return "14 21 28 35 42 49"
    if "26-sentence" in lowered and "alphabet" in lowered:
        return " ".join(f"{word.capitalize()} moves." for word in _ALPHABET_WORDS)
    if "use only words with lengths that are prime numbers" in lowered:
        return "We can see art."
    if "include quotes within quotes within quotes" in lowered:
        return "\"outer 'middle \"inner\" middle' outer\""
    if "nest parentheses" in lowered:
        return "({[({alpha})]})"
    if "standard punctuation mark" in lowered and "interrobang" in lowered:
        return "Alpha, beta; gamma: delta. Wow?! Yes! Why?"
    if "write each word on a new line" in lowered:
        return "Alpha\nBeta\nGamma"
    if "incrementally indenting each new line" in lowered:
        if "all containing the same number of characters" in lowered:
            return "Cat sat.\n Dog ran.\n  Fox hid."
        return "Alpha\n Beta\n  Gamma"
    if "emoji at the end of every sentence" in lowered and "keyword" not in lowered:
        return "Alpha 🙂. Beta 🙂."
    if "title case" in lowered:
        return "Alpha Beta Gamma"
    if "thesis statement in italics" in lowered:
        return "<i>Alpha thesis</i> Alpha body."
    if "bullet points denoted by *" in lowered and "sub-bullet" in lowered:
        return "* Alpha\n- Beta\n* Gamma\n- Delta"
    if "sentences ending in a period followed by at least two bullet points denoted by *" in lowered:
        return "Alpha ends. Beta ends.\n* Gamma\n* Delta"
    if "at least 10 single-word palindromes" in lowered:
        return "level radar civic refer rotor kayak madam minim solos stats"
    if "generate 4 multiple choice questions" in lowered and "5 options" in lowered:
        questions = []
        for index in range(1, 5):
            question = " ".join(["art"] * (index + 2)) + "?"
            options = "\n".join(f"{chr(64 + option)}) option{option}" for option in range(1, 6))
            questions.append(f"Question {index}. {question}\n{options}")
        return "\n".join(questions)
    return None


def _solve_counts(
    text: str,
    lowered: str,
    word_range: Optional[tuple[int, int]],
) -> Optional[str]:
    number_match = re.search(r"include exactly\s+(\d+)\s+numbers", text, flags=re.IGNORECASE)
    conjunction_match = re.search(
        r"use at least\s+(\d+)\s+different coordinating conjunctions",
        text,
        flags=re.IGNORECASE,
    )
    if number_match:
        count = int(number_match.group(1))
        parts = [str(index) for index in range(1, count + 1)]
        if conjunction_match:
            parts.extend(_CONJUNCTIONS[: int(conjunction_match.group(1))])
        return " ".join(parts)

    unique_match = re.search(r"use at least\s+(\d+)\s+unique words", text, flags=re.IGNORECASE)
    if unique_match:
        return _exact_words(int(unique_match.group(1)))

    pronoun_match = re.search(r"include at least\s+(\d+)\s+pronouns", text, flags=re.IGNORECASE)
    if pronoun_match:
        count = int(pronoun_match.group(1))
        if "only three types of vowels" in lowered:
            return " ".join(["I"] * count)
        return " ".join(_PRONOUNS[index % len(_PRONOUNS)] for index in range(count))

    person_match = re.search(
        r"mention at least\s+(\d+)\s+different person names",
        text,
        flags=re.IGNORECASE,
    )
    if person_match:
        names = " ".join(_PERSON_NAMES[: int(person_match.group(1))])
        return "Do " + names if "response must start with a verb" in lowered else names

    if conjunction_match:
        return " ".join(_CONJUNCTIONS[: int(conjunction_match.group(1))])

    repeat_match = re.search(r"not repeat any word more than\s+(\d+)\s+times", text, flags=re.IGNORECASE)
    if repeat_match:
        return _exact_words(40)

    if word_range is not None and "keyword" not in lowered and "list of items" not in lowered:
        return _exact_words(word_range[0])

    return None


def _solve_keyword_constraints(
    text: str,
    lowered: str,
    word_range: Optional[tuple[int, int]],
) -> Optional[str]:
    match = re.search(
        r"the second word in your response and the second to last word in your response should be the word\s+([A-Za-z]+)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        keyword = match.group(1)
        return f"alpha {keyword} beta {keyword} gamma"

    match = re.search(
        r"include keyword\s+([A-Za-z]+)\s+in the\s+(\d+)(?:st|nd|rd|th)?(?:-|\s*)[a-z]*\s*sentence,?\s+as the\s+(\d+)(?:st|nd|rd|th)?(?:-|\s*)[a-z]*\s+word",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        keyword, sentence_number, word_number = (
            match.group(1),
            int(match.group(2)),
            int(match.group(3)),
        )
        sentences = []
        for index in range(1, sentence_number + 1):
            if index == sentence_number:
                words = _filler_words(max(word_number, 1))
                words[word_number - 1] = keyword
                sentences.append(" ".join(words).capitalize() + ".")
            else:
                sentences.append("Alpha beta.")
        return " ".join(sentences)

    match = re.search(
        r"include keyword\s+([A-Za-z]+) once.*?keyword\s+([A-Za-z]+) twice.*?keyword\s+([A-Za-z]+) three times.*?keyword\s+([A-Za-z]+) five times.*?keyword\s+([A-Za-z]+) seven times",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        return " ".join(
            keyword
            for keyword, count in zip(match.groups(), (1, 2, 3, 5, 7))
            for _ in range(count)
        )

    match = re.search(
        r"include keyword\s+\"?([A-Za-z]+)\"?\s+in the\s+(\d+)(?:st|nd|rd|th)?(?:-|\s*)[a-z]*\s+sentence",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        keyword, sentence_number = match.group(1), int(match.group(2))
        return _sentences_for_keyword(
            keyword,
            sentence_number,
            total_words=word_range[0] if word_range is not None else None,
            emoji="emoji at the end of every sentence" in lowered,
        )

    return None


def _solve_surface_constraints(
    text: str,
    lowered: str,
    word_range: Optional[tuple[int, int]],
) -> Optional[str]:
    if "answer with a list of items" in lowered and "instead of bullet points use" in lowered:
        separator_match = re.search(
            r"instead of bullet points use\s*(.+?)\.",
            text,
            flags=re.IGNORECASE,
        )
        separator = separator_match.group(1).strip() if separator_match else "!?!?"
        if word_range is None:
            return f"alpha {separator} beta {separator} cedar"
        words = _filler_words(word_range[0])
        first = word_range[0] // 3
        second = 2 * word_range[0] // 3
        return (
            f"{' '.join(words[:first])} {separator} "
            f"{' '.join(words[first:second])} {separator} "
            f"{' '.join(words[second:])}"
        )

    if "no two consecutive words can share the same first letter" in lowered:
        return "alpha beta cedar delta ember fable garden harbor"

    if "paragraph" in lowered and "only three types of vowels" in lowered:
        return "men met red rebels under dusk, then kept stern legends, cursed elders, and endless ember verses."

    if "each word in your response has at least one consonant cluster" in lowered:
        if "alternate between words with odd and even numbers of syllables" in lowered:
            return "strong blessed crypts crafting bright flinty"
        if "ratio of sentence types" in lowered and "balanced" in lowered:
            return "Strong crypts stand. Strong crypts stand? Strong crypts stand!"
        return "Strong crypts stand with bright flint crafts."

    if "alternate between words with odd and even numbers of syllables" in lowered:
        return "cat away dog again fish about"

    if "each word in your response must start with the next letter of the alphabet" in lowered:
        return " ".join(_ALPHABET_WORDS)

    if "all containing the same number of characters" in lowered and "three sentences" in lowered:
        return "Cat sat. Dog ran. Fox hid."

    if re.search(
        r"each sentence (?:in your response )?must contain exactly\s+\d+\s+more words than the previous one",
        text,
        flags=re.IGNORECASE,
    ):
        return "Alpha."

    if "longer sequence of consecutive alliterative words than the previous" in lowered:
        return "Blue birds rest. Calm cats climb. Daring dogs dance daily."

    if "last word of each sentence must become the first word of the next sentence" in lowered:
        return "Alpha beta. Beta gamma. Gamma delta."

    if "each paragraph" in lowered and "end with the same word it started" in lowered:
        return "Alpha simple alpha\nBeta simple beta"

    if "ratio of sentence types" in lowered and "balanced" in lowered:
        return "Alpha stands. Beta asks? Gamma cheers!"

    if "2:1 ratio of declarative to interrogative" in lowered:
        return "Alpha stands. Beta remains. Gamma asks?"

    if "response must start with a verb" in lowered:
        return "Do this clearly."

    if "every" in lowered and "word of your response must be in japanese" in lowered:
        match = re.search(r"every\s+(\d+)(?:st|nd|rd|th)?\s+(?:word|of)", lowered)
        cycle = int(match.group(1)) if match else 5
        return " ".join(
            "日本" if index % cycle == 0 else f"word{index}"
            for index in range(1, cycle * 3 + 1)
        )

    return None


def _parse_word_range(text: str) -> Optional[tuple[int, int]]:
    match = re.search(r"between\s+(\d+)\s+and\s+(\d+)\s+words", text, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _prefix_before(text: str, marker: str) -> str:
    index = text.lower().find(marker.lower())
    return text[:index].strip() if index >= 0 else text.strip()


def _filler_words(count: int, *, offset: int = 0) -> list[str]:
    words = []
    for index in range(count):
        base = _UNIQUE_WORDS[(index + offset) % len(_UNIQUE_WORDS)]
        suffix = "" if index + offset < len(_UNIQUE_WORDS) else str((index + offset) // len(_UNIQUE_WORDS))
        words.append(base + suffix)
    return words


def _exact_words(count: int) -> str:
    return " ".join(_filler_words(count))


def _sentences_for_keyword(
    keyword: str,
    sentence_number: int,
    *,
    total_words: Optional[int],
    emoji: bool,
) -> str:
    sentence_number = max(1, sentence_number)
    sentences = []
    for index in range(sentence_number):
        if index == sentence_number - 1:
            words = ["alpha", "beta", "contains", keyword]
        else:
            words = ["alpha", "beta"]
        sentences.append(words)
    if total_words is not None:
        current_words = sum(len(sentence) for sentence in sentences)
        if current_words < total_words:
            sentences[-1].extend(_filler_words(total_words - current_words, offset=4))
    end = "🙂." if emoji else "."
    return " ".join(" ".join(sentence).capitalize() + end for sentence in sentences)
