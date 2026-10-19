# 쇼츠 메이커 처리 스크립트 (GitHub Actions에서 실행)
# 입력: input.json(장면 사진, 길이), audio.wav(16kHz 모노, 없을 수도 있음)
# 출력: result.json  {lines:[{start,end,text}], speech, titles:[{line1,line2}], error?}
#       mode=style 이면 {style:{title,sub,tone,examples}} (예시 쇼츠의 글자 스타일·말투)
import json
import os
import sys
import traceback
import urllib.request

OUT = 'result.json'
MODEL = os.environ.get('WHISPER_MODEL', 'small')
# 무음·음악 구간에서 Whisper가 지어내는 대표 문구
BOGUS = ('MBC 뉴스', '자막 제공', 'Amara.org', '한글자막 by', '자막 by')
# 영상 끝 무음·음악에서 Whisper가 자주 지어내는 마무리 인사 (맨 끝 줄일 때만 지움)
OUTRO = ('시청해주셔서감사합니다', '시청해주셔서고맙습니다', '구독과좋아요부탁드립니다', '구독좋아요부탁드립니다',
         '다음영상에서만나요', '감사합니다')


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

    while lines and lines[-1]['text'].replace(' ', '').rstrip('.!~') in OUTRO:
        lines.pop()
    # 너무 짧게 스치는 자막은 조금 늘리되 다음 줄과 겹치지 않게
    for i, ln in enumerate(lines):
        end = max(ln['end'] + 0.25, ln['start'] + 0.8)
        if i + 1 < len(lines):
            end = min(end, lines[i + 1]['start'])
        ln['end'] = round(max(end, ln['start'] + 0.3), 2)
        ln['start'] = round(ln['start'], 2)
    return lines


def ask_json(prompt, frames, temperature):
    """GitHub Models(무료)에 사진과 함께 물어보고 JSON으로 받는다. 실패하면 (None, 이유)."""
    token = os.environ.get('GITHUB_TOKEN')
    if not token:
        return None, 'GITHUB_TOKEN 없음'
    content = [{'type': 'text', 'text': prompt}]
    for f in frames[:6]:
        content.append({'type': 'image_url', 'image_url': {'url': f, 'detail': 'low'}})
    last = None
    for model in ('openai/gpt-4.1-mini', 'openai/gpt-4o-mini'):
        body = json.dumps({
            'model': model,
            'messages': [{'role': 'user', 'content': content}],
            'response_format': {'type': 'json_object'},
            'temperature': temperature,
        }).encode()
        req = urllib.request.Request(
            'https://models.github.ai/inference/chat/completions', data=body,
            headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json',
                     'Accept': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                data = json.load(r)
            return json.loads(data['choices'][0]['message']['content']), None
        except Exception as e:  # 다음 모델로
            detail = ''
            if hasattr(e, 'read'):
                try:
                    detail = e.read().decode()[:300]
                except Exception:
                    pass
            last = f'{model}: {e} {detail}'.strip()
    return None, last


def ai_titles(frames, transcript, tone='', examples=(), channel=''):
    """상단 제목 후보 3개 + 유튜브 업로드 제목·해시태그. 예시 쇼츠 말투가 있으면 그 말투를 따라 함."""
    prompt = (
        '유튜브 쇼츠 위쪽 검은 띠에 크게 들어갈 제목을 지어줘.\n'
        '장면 사진' + (' 과 영상 속 말' if transcript else '') + '을 보고 영상 내용에 딱 맞게.\n'
        '- 2줄. 각 줄은 공백 포함 13자 이내\n'
        '- 첫 줄은 상황, 둘째 줄은 궁금증을 만들거나 핵심을 찍는 말 (둘째 줄은 색을 바꿔 강조됨)\n'
        '- 이런 느낌: "강아지 두마리 키우는 / 견주의 흔한 일상", "○○과 힘대결하면 / 벌어지는 일", '
        '"○○한테 찐으로 / 반해버린 순간 ㅋㅋ", "사람 대우도 못 하면서 / 이게 왜 제 탓이죠?"\n'
        '- 실제 사람이 쓴 것처럼 자연스러운 한국어 구어체\n'
        '- 이모지, 해시태그, 따옴표, 과장된 낚시 금지\n'
        '- 서로 느낌이 다른 후보 3개\n'
        '그리고 유튜브에 올릴 때 쓸 upload 도 만들어줘:\n'
        '- upload.title: 검색에 잘 걸리는 업로드 제목 (40자 이내, 첫 번째 후보와 같은 내용)\n'
        '- upload.tags: 영상 내용에 맞는 해시태그 4~6개 (#shorts 포함)\n'
        '답은 JSON만: {"titles":[{"line1":"","line2":""},...],"upload":{"title":"","tags":["#..."]}}'
    )
    if tone:
        prompt += '\n\n이 말투를 따라 해: ' + tone[:300]
    if examples:
        prompt += '\n참고할 예시 제목(그대로 베끼지는 말 것): ' + ' / '.join(e[:40] for e in examples[:3])
    if channel:
        prompt += '\n채널 이름: ' + channel[:40]
    if transcript:
        prompt += '\n\n영상 속 말: ' + transcript[:1500]
    data, err = ask_json(prompt, frames[:4], 0.8)
    if data is None:
        return None, None, err
    out = []
    for t in data.get('titles', []) if isinstance(data, dict) else []:
        l1, l2 = str(t.get('line1', '')).strip(), str(t.get('line2', '')).strip()
        if l1 or l2:
            out.append({'line1': l1, 'line2': l2})
    upload = None
    u = data.get('upload') if isinstance(data, dict) else None
    if isinstance(u, dict) and u.get('title'):
        tags = [str(t).strip() for t in (u.get('tags') or []) if str(t).strip()][:8]
        upload = {'title': str(u['title']).strip()[:100], 'tags': tags}
    return (out[:3], upload, None) if out else (None, None, '빈 응답')


def ai_style(frames):
    """예시 쇼츠 장면을 보고 제목·자막 글자 스타일과 제목 말투를 뽑는다."""
    prompt = (
        '이 사진들은 어떤 유튜브 쇼츠의 장면이야. 화면에 들어간 글자 스타일을 똑같이 따라 하려고 해.\n'
        '- band: 영상 바깥(위쪽 제목 띠) 배경색\n'
        '- title: 위쪽 띠에 크게 고정된 제목 글자. 없으면 비슷하게 어울리는 값으로\n'
        'title.box2: 둘째 줄 글자 뒤에 색 상자가 있으면 그 색, 없으면 ""\n'
        '- sub: 말하는 내용이 바뀌며 나오는 자막 글자. 없으면 기본값\n'
        '색은 실제 보이는 색을 #RRGGBB로. 두 줄 제목이면 color1은 첫 줄, color2는 둘째 줄(강조) 색.\n'
        'weight: 아주 두껍고 각진 제목용 글꼴이면 "heavy", 일반 굵은 고딕이면 "bold"\n'
        'size: 화면 폭 대비 제목 크기 "S"/"M"/"L"\n'
        'stroke(제목): 글자에 검은 테두리가 있으면 true\n'
        'sub.stroke: 자막 테두리 색 #RRGGBB, 없으면 "none". sub.box: 자막 뒤에 배경 상자가 있으면 true, boxColor는 그 색\n'
        'sub.position: 자막 세로 위치 "high"/"middle"/"low"\n'
        'sub.font: 자막이 각진 굵은 글꼴이면 "heavy", 일반 고딕이면 "bold"\n'
        'tone: 제목 말투를 한 문장으로 설명 (예: 반말로 짧게 놀라움을 강조)\n'
        'examples: 화면에 보이는 제목 글자를 그대로 최대 2개 (안 보이면 빈 배열)\n'
        '답은 JSON만: {"style":{"band":"","title":{"color1":"","color2":"","box2":"","weight":"","size":"","stroke":false},'
        '"sub":{"color":"","stroke":"","box":false,"boxColor":"","position":"","font":""},"tone":"","examples":[]}}'
    )
    data, err = ask_json(prompt, frames, 0.2)
    if data is None:
        return None, err
    st = data.get('style') if isinstance(data, dict) else None
    return (st, None) if isinstance(st, dict) else (None, '빈 응답')


def main():
    with open('input.json', encoding='utf-8') as f:
        inp = json.load(f)
    if inp.get('mode') == 'style':
        st, err = ai_style(inp.get('frames', []))
        save({'version': 1, 'style': st} if st else {'version': 1, 'styleError': err})
        return
    words = []
    if inp.get('hasAudio') and os.path.exists('audio.wav'):
        words = transcribe('audio.wav')
    lines = to_lines(words)
    speech = sum(len(l['text'].replace(' ', '')) for l in lines) >= 6
    if not speech:
        lines = []
    transcript = ' '.join(l['text'] for l in lines)
    titles, upload, err = ai_titles(inp.get('frames', []), transcript,
                                    inp.get('tone', ''), inp.get('examples', []), inp.get('channel', ''))
    res = {'version': 1, 'speech': speech, 'lines': lines, 'titles': titles or []}
    if upload:
        res['upload'] = upload
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
