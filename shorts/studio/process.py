# 쇼츠 메이커 처리 스크립트 (GitHub Actions에서 실행)
# 입력: input.json(장면 사진, 길이), audio.wav(16kHz 모노, 없을 수도 있음)
# 출력: result.json  {lines:[{start,end,text}], speech, titles:[{line1,line2}], error?}
import json
import os
import sys
import traceback
import urllib.request

OUT = 'result.json'
MODEL = os.environ.get('WHISPER_MODEL', 'small')
# 무음·음악 구간에서 Whisper가 지어내는 대표 문구
BOGUS = ('MBC 뉴스', '자막 제공', 'Amara.org', '한글자막 by', '자막 by')


def save(d):
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False)


def load_model():
    from faster_whisper import WhisperModel
    return WhisperModel(MODEL, device='cpu', compute_type='int8')


def transcribe(path):
    segs, _ = load_model().transcribe(
        path, language='ko', vad_filter=True, word_timestamps=True,
        beam_size=5, condition_on_previous_text=False)
    words = []
    for s in segs:
        if s.no_speech_prob > 0.6 and s.avg_logprob < -1.0:
            continue
        for w in s.words or []:
            if w.word.strip():
                words.append((w.start, w.end, w.word))
    return words


def to_lines(words, maxlen=14, gap=0.6):
    """단어를 쇼츠용 짧은 자막 줄로 묶는다. 쉼·문장부호·글자 수 기준으로 끊음."""
    lines, cur = [], []

    def text_of(ws):
        return ''.join(w[2] for w in ws).strip()

    def flush():
        t = text_of(cur).rstrip('.,')
        if t and not any(b in t for b in BOGUS):
            lines.append({'start': cur[0][0], 'end': cur[-1][1], 'text': t})
        cur.clear()

    for w in words:
        if cur and (w[0] - cur[-1][1] > gap or len(text_of(cur + [w])) > maxlen):
            flush()
        cur.append(w)
        if w[2].strip()[-1:] in '.?!,':
            flush()
    if cur:
        flush()

    # 너무 짧게 스치는 자막은 조금 늘리되 다음 줄과 겹치지 않게
    for i, ln in enumerate(lines):
        end = max(ln['end'] + 0.25, ln['start'] + 0.8)
        if i + 1 < len(lines):
            end = min(end, lines[i + 1]['start'])
        ln['end'] = round(max(end, ln['start'] + 0.3), 2)
        ln['start'] = round(ln['start'], 2)
    return lines


def ai_titles(frames, transcript):
    """GitHub Models(무료)로 상단 제목 후보를 만든다. 실패하면 (None, 이유)."""
    token = os.environ.get('GITHUB_TOKEN')
    if not token:
        return None, 'GITHUB_TOKEN 없음'
    prompt = (
        '유튜브 쇼츠 위쪽 검은 띠에 크게 들어갈 제목을 지어줘.\n'
        '장면 사진' + (' 과 영상 속 말' if transcript else '') + '을 보고 영상 내용에 딱 맞게.\n'
        '- 2줄. 각 줄은 공백 포함 12자 이내\n'
        '- 첫 줄은 상황, 둘째 줄은 궁금증을 만들거나 핵심을 찍는 말\n'
        '- 실제 사람이 쓴 것처럼 자연스러운 한국어 구어체\n'
        '- 이모지, 해시태그, 따옴표, 과장된 낚시 금지\n'
        '- 서로 느낌이 다른 후보 3개\n'
        '답은 JSON만: {"titles":[{"line1":"","line2":""},...]}'
    )
    if transcript:
        prompt += '\n\n영상 속 말: ' + transcript[:1500]
    content = [{'type': 'text', 'text': prompt}]
    for f in frames[:4]:
        content.append({'type': 'image_url', 'image_url': {'url': f, 'detail': 'low'}})

    last = None
    for model in ('openai/gpt-4.1-mini', 'openai/gpt-4o-mini'):
        body = json.dumps({
            'model': model,
            'messages': [{'role': 'user', 'content': content}],
            'response_format': {'type': 'json_object'},
            'temperature': 0.8,
        }).encode()
        req = urllib.request.Request(
            'https://models.github.ai/inference/chat/completions', data=body,
            headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json',
                     'Accept': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                data = json.load(r)
            txt = data['choices'][0]['message']['content']
            out = []
            for t in json.loads(txt).get('titles', []):
                l1, l2 = str(t.get('line1', '')).strip(), str(t.get('line2', '')).strip()
                if l1 or l2:
                    out.append({'line1': l1, 'line2': l2})
            if out:
                return out[:3], None
            last = model + ': 빈 응답'
        except Exception as e:  # 다음 모델로
            detail = ''
            if hasattr(e, 'read'):
                try:
                    detail = e.read().decode()[:300]
                except Exception:
                    pass
            last = f'{model}: {e} {detail}'.strip()
    return None, last


def main():
    with open('input.json', encoding='utf-8') as f:
        inp = json.load(f)
    words = []
    if inp.get('hasAudio') and os.path.exists('audio.wav'):
        words = transcribe('audio.wav')
    lines = to_lines(words)
    speech = sum(len(l['text'].replace(' ', '')) for l in lines) >= 6
    if not speech:
        lines = []
    transcript = ' '.join(l['text'] for l in lines)
    titles, err = ai_titles(inp.get('frames', []), transcript)
    res = {'version': 1, 'speech': speech, 'lines': lines, 'titles': titles or []}
    if err and not titles:
        res['titleError'] = err
    save(res)


if __name__ == '__main__':
    if sys.argv[1:] == ['warm']:
        load_model()  # 모델을 미리 받아 캐시에 저장
        sys.exit(0)
    try:
        main()
    except Exception as e:
        save({'error': f'{type(e).__name__}: {e}', 'trace': traceback.format_exc()[-1500:]})
