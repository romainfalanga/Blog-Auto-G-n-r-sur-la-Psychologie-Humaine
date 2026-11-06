"""Build short, extractive explanations before the public fictional stories.

The source is each article's explanatory definition section, never its opening
fiction. No model, API, invented claim or new source is involved. Article Markdown
is read-only. Canonical IDs match build_examples.py and article_keywords.json.
"""
from __future__ import annotations

import hashlib
import html
import json
import pathlib
import re
import sys
import unicodedata

ROOT = pathlib.Path(__file__).resolve().parents[1]
TARGET = ROOT / 'data/article_intros.json'
HEADING = re.compile(r'^(#{2,6})[ \t]+(.+?)[ \t]*#*[ \t]*$', re.MULTILINE)
WORD = re.compile(r"[^\W_]+(?:[’'-][^\W_]+)*")
MIN_OPENING_WORDS = 180
DEFINITION_MARKERS = ('definition', "qu'est-ce", 'comprendre', 'concept', 'mecanisme', 'au-dela', "coup d'")
CHARACTER_NAMES = [p['prenom'] for p in json.loads((ROOT / 'scripts/personnages.json').read_text())['personnages']]
PERSONALIZED_SENTENCE = re.compile(r'\b(?:Pour|Chez|Dans le cas de|Comme)\s+(?:' + '|'.join(re.escape(name) for name in CHARACTER_NAMES) + r')\b')
ABBREVIATIONS = {'m', 'mme', 'mlle', 'dr', 'pr', 'prof', 'etc', 'cf', 'ex', 'av', 'env', 'vol', 'p', 'pp', 'al'}

def normalized(text: str) -> str:
    text = unicodedata.normalize('NFD', text.lower().replace('’', "'"))
    return ''.join(c for c in text if unicodedata.category(c) != 'Mn')


def plain_text(markdown: str) -> str:
    """Remove presentation markers without rewriting prose or punctuation."""
    text = re.sub(r'!\[([^]]*)\]\([^)]+\)', r'\1', markdown)
    text = re.sub(r'\[([^]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'(?<!\w)[*_`]+|[*_`]+(?!\w)', '', text)
    text = html.unescape(text)
    return re.sub(r'\s+', ' ', text).strip()


def sections(body: str) -> list[dict]:
    headings = list(HEADING.finditer(body))
    result = []
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(body)
        paragraphs = [plain_text(p) for p in re.split(r'\n\s*\n', body[heading.end():end].strip()) if p.strip()]
        result.append({'heading': plain_text(heading.group(2)), 'paragraphs': paragraphs, 'level': len(heading.group(1))})
    return result


def sentences(paragraph: str) -> list[str]:
    """Split at completed sentences, retaining French closing quotes.

    Author initials (James J. Gross), abbreviations, decimals and ellipses are not
    mistaken for arbitrary cut points. No sentence is shortened in mid-word.
    """
    result, start, index = [], 0, 0
    while index < len(paragraph):
        char = paragraph[index]
        if char not in '.!?':
            index += 1
            continue
        if char == '.':
            before = re.search(r'([A-Za-zÀ-ÖØ-öø-ÿ]+)$', paragraph[:index])
            word = before.group(1) if before else ''
            if (index and index + 1 < len(paragraph) and paragraph[index - 1].isdigit() and paragraph[index + 1].isdigit()) or word.lower() in ABBREVIATIONS or (len(word) == 1 and word.isupper()):
                index += 1
                continue
        end = index + 1
        while end < len(paragraph) and paragraph[end] in '.!?':
            end += 1
        while end < len(paragraph) and paragraph[end] in '\"»”’)]':
            end += 1
        next_start = end
        while next_start < len(paragraph) and paragraph[next_start].isspace():
            next_start += 1
        # A capital after punctuation is a new sentence; a closing quote may be
        # followed by a lower-case continuation, which must remain attached.
        if next_start == len(paragraph) or (next_start > end and (paragraph[next_start].isupper() or paragraph[next_start] in '«\"“')):
            result.append(paragraph[start:end].strip())
            start = next_start
            index = next_start
        else:
            index = end
    remainder = paragraph[start:].strip()
    if remainder:
        result.append(remainder)
    return result


def excerpt(paragraph: str) -> str:
    parts = sentences(paragraph)
    if not parts:
        raise ValueError('Empty explanatory paragraph')
    if PERSONALIZED_SENTENCE.search(parts[0]):
        raise ValueError('The selected paragraph starts with a fictional case; review manually')
    chosen = [parts[0]]
    for part in parts[1:3]:
        # Some definitions immediately segue back to their fictional character.
        # Keep the complete generic definition instead of padding with biography.
        if PERSONALIZED_SENTENCE.search(part):
            break
        if len(' '.join(chosen)) >= 200:
            break
        candidate = ' '.join(chosen + [part])
        if len(candidate) > 450:
            break
        chosen.append(part)
    return ' '.join(chosen)


def build(write: bool = True) -> tuple[dict, dict]:
    output, report = {}, {'articles': 0, 'h3_definitions': [], 'short_complete_definitions': [], 'long_complete_definitions': []}
    article_topics = json.loads((ROOT / 'data/parcours_topics.json').read_text())
    for path in sorted((ROOT / 'content/posts').rglob('*.md')):
        if path.name == '_index.md':
            continue
        source = path.read_text()
        split = source.split('---', 2)
        if len(split) != 3:
            raise ValueError(f'Frontmatter missing: {path}')
        # The generated corpus has explicit draft booleans. Do not include drafts.
        if re.search(r'^draft:\s*true\s*$', split[1], re.MULTILINE):
            continue
        body = split[2]
        identifier = path.relative_to(ROOT / 'content').with_suffix('').as_posix()
        first_heading = HEADING.search(body)
        if not first_heading:
            raise ValueError(f'No section after the opening scene: {identifier}')
        opening_words = len(WORD.findall(plain_text(body[:first_heading.start()])))
        if opening_words < MIN_OPENING_WORDS:
            raise ValueError(f'Opening scene too short ({opening_words} words, minimum {MIN_OPENING_WORDS}): {identifier}')
        topics = article_topics.get(identifier)
        if not isinstance(topics, list) or not 2 <= len(topics) <= 3 or any(
            not isinstance(topic, str) or not topic.strip() or topic.strip() != topic for topic in topics
        ) or len({normalized(topic) for topic in topics}) != len(topics):
            raise ValueError(f'Expected two or three distinct reading concepts: {identifier}')
        candidates = sections(body)
        definition = next((s for s in candidates if any(marker in normalized(s['heading']) for marker in DEFINITION_MARKERS)), None)
        if not definition or not definition['paragraphs']:
            raise ValueError(f'No explanatory definition section; review manually: {identifier}')
        paragraph = definition['paragraphs'][0]
        if paragraph.startswith(('-', '*', '#')):
            raise ValueError(f'Expected explanatory prose, not list: {identifier}')
        text = excerpt(paragraph)
        # Exactly one contiguous excerpt from one explanatory paragraph.
        if text not in paragraph:
            raise ValueError(f'Excerpt is not verbatim after Markdown removal: {identifier}')
        entry = {'text': text, 'source_heading': definition['heading'], 'source_paragraph': 1, 'method': 'definition_excerpt'}
        if definition['level'] == 3:
            report['h3_definitions'].append(identifier)
        entry['source_body_sha256'] = hashlib.sha256(body.encode()).hexdigest()
        if len(entry['text']) < 200:
            report['short_complete_definitions'].append({'id': identifier, 'length': len(entry['text'])})
        if len(entry['text']) > 450:
            report['long_complete_definitions'].append({'id': identifier, 'length': len(entry['text'])})
        if not entry['text'] or not re.search(r'[.!?][\"»”’)]*$', entry['text']):
            raise ValueError(f'Excerpt does not end with a complete sentence: {identifier}')
        if len(sentences(entry['text'])) > 3:
            raise ValueError(f'Excerpt exceeds three sentences: {identifier}')
        output[identifier] = entry
    if stale_topics := article_topics.keys() - output.keys():
        raise ValueError('Reading concepts without a published article: ' + ', '.join(sorted(stale_topics)))
    report['articles'] = len(output)
    if write:
        TARGET.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    return output, report


if __name__ == '__main__':
    values, report = build(write='--check' not in sys.argv)
    if '--check' in sys.argv:
        if not TARGET.exists() or json.loads(TARGET.read_text()) != values:
            raise SystemExit('article_intros.json is stale; run python scripts/build_article_intros.py')
    print(json.dumps(report, ensure_ascii=False, indent=2))
