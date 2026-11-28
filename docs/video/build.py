"""brag video for tiny-sql-rl: frames drawn with PIL (each a pure function of t), music synthesised with numpy,
encoded with the ffmpeg binary from imageio-ffmpeg.

    uv run --no-project --with imageio-ffmpeg --with pillow --with numpy python docs/video/build.py [--stills t1,t2,...]

Writes docs/video/work/render.mp4 (the poster frame is then baked in as frame 0 with ffmpeg).
"""
import math, subprocess, sys, wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = Path(__file__).resolve().parent
WORK = OUT / "work"
WORK.mkdir(exist_ok=True)
W, H, FPS, DUR = 1920, 1080, 30, 20.0
N = int(DUR * FPS)
BPM = 96
BEAT = 60 / BPM
BAR = 4 * BEAT  # 2.5 s

BG = (15, 15, 17)
PANEL = (24, 24, 27)
FIELD = (39, 39, 42)
BORDER = (63, 63, 70)
TEXT = (244, 244, 245)
MUTED = (161, 161, 170)
ORANGE = (194, 65, 12)
ORANGE_HI = (234, 88, 12)
F = "C:/Windows/Fonts/"


def font(name, size):
    return ImageFont.truetype(F + name, size)


TITLE = lambda s: font("segoeuib.ttf", s)
SEMI = lambda s: font("seguisb.ttf", s) if Path(F + "seguisb.ttf").exists() else font("segoeuib.ttf", s)
LIGHT = lambda s: font("segoeuil.ttf", s)
REG = lambda s: font("segoeui.ttf", s)
ARIAL = lambda s: font("arial.ttf", s)
ARIALB = lambda s: font("arialbd.ttf", s)
MONO = lambda s: font("consola.ttf", s)


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease_out(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def ease_io(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def appear(t, t0, dur=0.55):
    """(alpha, rise offset px) for an element entering at t0."""
    p = ease_out((t - t0) / dur)
    return p, int(round((1 - p) * 26))


# ---------- background (static) ----------
def make_bg():
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    base = np.zeros((H, W, 3), np.float32) + np.array(BG, np.float32)
    g = np.exp(-(((x - W * 0.78) / 900) ** 2 + ((y - H * 1.05) / 620) ** 2))
    base += g[..., None] * np.array([38, 12, 2], np.float32)
    g2 = np.exp(-(((x - W * 0.12) / 800) ** 2 + ((y + H * 0.1) / 500) ** 2))
    base += g2[..., None] * np.array([10, 10, 14], np.float32)
    base += np.random.default_rng(1).uniform(-1.5, 1.5, base.shape)  # dither: no banding in the glow
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).convert("RGBA")


BGIMG = make_bg()


def rgba(c, a):
    return (*c, int(round(255 * clamp(a))))


def text(d, xy, s, f, c, a, anchor="la"):
    if a > 0:
        d.text(xy, s, font=f, fill=rgba(c, a), anchor=anchor)


# ---------- scenes ----------
def scene1(t, d):
    a1, r1 = appear(t, 0.25)
    a2, r2 = appear(t, 1.25)
    a3, r3 = appear(t, 2.5)
    x = 210
    text(d, (x, 300 + r1), "1.7B parameters.", TITLE(120), TEXT, a1)
    text(d, (x, 450 + r2), "One 8 GB laptop GPU.", TITLE(120), TEXT, a2)
    w = int(560 * ease_io((t - 1.9) / 0.7))
    if w > 0:
        d.rectangle([x + 4, 628, x + 4 + w, 635], fill=rgba(ORANGE_HI, 1))
    text(d, (x, 680 + r3), "Reinforcement learning that writes SQL.", LIGHT(56), MUTED, a3)


QUESTION = "Which continent has the most countries?"
SQL = ["SELECT c.Continent, COUNT(*) as CountryCount", "FROM country c", "GROUP BY c.Continent",
       "ORDER BY CountryCount DESC", "LIMIT 1;"]
KW = {"SELECT", "COUNT", "as", "FROM", "GROUP", "BY", "ORDER", "DESC", "LIMIT"}
T_TYPE0, T_TYPE1 = 6.25, 7.55
T_PRESS = 7.8125
T_SQL = [8.4375 + 0.25 * i for i in range(5)]
T_ANS, T_ROW, T_TIME = 9.6875, 10.0, 10.3125


def field(d, box, a, fill=FIELD):
    d.rounded_rectangle(box, radius=6, fill=rgba(fill, a), outline=rgba(BORDER, a), width=1)


def block_label(d, x, y, s, a):
    text(d, (x, y), s, ARIAL(22), TEXT, a)


def sql_line(d, x, y, s, a):
    f = MONO(28)
    for tok in s.replace("(", " ( ").replace(")", " ) ").split(" "):
        pass
    cx = x
    i = 0
    while i < len(s):
        j = i
        if s[i].isalpha():
            while j < len(s) and (s[j].isalnum() or s[j] in "._"):
                j += 1
        else:
            j = i + 1
        tok = s[i:j]
        col = (147, 168, 255) if tok in KW else (253, 186, 116) if tok.isdigit() else (125, 211, 252) if "." in tok else (228, 228, 231)
        text(d, (cx, y), tok, f, col, a)
        cx += d.textlength(tok, font=f)
        i = j


def scene2(t, d):
    ac, rc = appear(t, 5.3)
    text(d, (210, 70 + rc), "Ask in plain English. It writes the SQL.", SEMI(58), TEXT, ac)
    ap, rp = appear(t, 5.55, 0.7)
    if ap <= 0:
        return
    ox, oy = 210, 190 + rp
    pw = W - 2 * 210
    d.rounded_rectangle([ox, oy, ox + pw, oy + 860], radius=14, fill=rgba(BG, ap), outline=rgba(BORDER, ap), width=2)
    x0, y = ox + 36, oy + 26
    text(d, (x0, y), "tiny-sql-rl", ARIALB(38), TEXT, ap)
    y += 58
    # Database + Question row
    field(d, [x0, y, x0 + 330, y + 104], ap, PANEL)
    block_label(d, x0 + 14, y + 12, "Database", ap)
    field(d, [x0 + 14, y + 48, x0 + 316, y + 92], ap)
    text(d, (x0 + 30, y + 58), "world_1", ARIAL(23), TEXT, ap)
    d.polygon([(x0 + 284, y + 66), (x0 + 298, y + 66), (x0 + 291, y + 75)], fill=rgba(TEXT, ap))
    qx = x0 + 346
    field(d, [qx, y, ox + pw - 36, y + 104], ap, PANEL)
    block_label(d, qx + 14, y + 12, "Question", ap)
    field(d, [qx + 14, y + 48, ox + pw - 50, y + 92], ap)
    n = int(len(QUESTION) * clamp((t - T_TYPE0) / (T_TYPE1 - T_TYPE0)))
    typed = QUESTION[:n]
    text(d, (qx + 30, y + 58), typed, ARIAL(23), TEXT, ap)
    if T_TYPE0 - 0.4 < t < T_PRESS and int(t * 2.4) % 2 == 0:
        cx = qx + 31 + d.textlength(typed, font=ARIAL(23))
        d.rectangle([cx, y + 57, cx + 2, y + 84], fill=rgba(TEXT, ap))
    y += 116
    # Ask button
    pressed = 0 <= t - T_PRESS < 0.18
    bcol = (154, 52, 18) if pressed else ORANGE
    inset = 3 if pressed else 0
    d.rounded_rectangle([x0 + inset, y + inset, ox + pw - 36 - inset, y + 54 - inset], radius=7, fill=rgba(bcol, ap))
    text(d, ((x0 + ox + pw - 36) // 2, y + 27), "Ask", ARIALB(24), TEXT, ap, anchor="mm")
    y += 68
    # Generated SQL
    field(d, [x0, y, ox + pw - 36, y + 240], ap, PANEL)
    d.rounded_rectangle([x0 + 8, y + 8, x0 + 200, y + 40], radius=4, fill=rgba(FIELD, ap))
    text(d, (x0 + 18, y + 13), "‹›  Generated SQL", ARIAL(19), TEXT, ap)
    for i, line in enumerate(SQL):
        la, lr = appear(t, T_SQL[i], 0.3)
        ly = y + 54 + 36 * i
        text(d, (x0 + 22, ly + lr // 3), str(i + 1), MONO(26), (113, 113, 122), la)
        sql_line(d, x0 + 64, ly + lr // 3, line, la * ap)
    y += 258
    # Answer
    field(d, [x0, y, ox + pw - 36, y + 104], ap, PANEL)
    block_label(d, x0 + 14, y + 12, "Answer", ap)
    field(d, [x0 + 14, y + 48, ox + pw - 50, y + 92], ap)
    aa, _ = appear(t, T_ANS, 0.35)
    text(d, (x0 + 30, y + 58), "Found 1 row.", ARIAL(23), TEXT, aa * ap)
    y += 112
    # Result table
    ra, rr = appear(t, T_ROW, 0.4)
    if ra > 0:
        text(d, (x0, y), "Result", ARIAL(21), TEXT, ra * ap)
        tb = [x0, y + 34, ox + pw - 36, y + 34 + 104]
        d.rounded_rectangle(tb, radius=6, fill=rgba(PANEL, ra * ap), outline=rgba(BORDER, ra * ap))
        d.line([x0, y + 34 + 52, ox + pw - 36, y + 34 + 52], fill=rgba(BORDER, ra * ap))
        cw = ox + pw - 36 - 260
        text(d, (x0 + 18, y + 50), "Continent", font("consolab.ttf", 25), TEXT, ra * ap)
        text(d, (cw + 18, y + 50), "CountryCount", font("consolab.ttf", 25), TEXT, ra * ap)
        text(d, (x0 + 18, y + 100), "Africa", MONO(26), TEXT, ra * ap)
        text(d, (cw + 18, y + 100), "58", MONO(26), TEXT, ra * ap)
    ta, _ = appear(t, T_TIME, 0.4)
    text(d, (x0, oy + 860 - 22), "text-to-SQL + query: 11.2 s  ·  on a laptop CPU", ARIAL(22), MUTED, ta * ap, anchor="lb")


def count(t, t0, t1, a, b):
    return a + (b - a) * ease_io((t - t0) / (t1 - t0))


def scene3(t, d):
    a0, r0 = appear(t, 12.75)
    text(d, (W // 2, 200 + r0), "Spider test  ·  2,147 held-out questions  ·  official execution accuracy", REG(36), MUTED, a0, anchor="mm")
    an, rn = appear(t, 12.9, 0.6)
    big = TITLE(190)
    text(d, (560, 420 + rn), "56.3%", big, (113, 113, 122), an, anchor="mm")
    text(d, (W // 2, 420 + rn), "→", LIGHT(120), MUTED, an, anchor="mm")
    v = count(t, 13.125, 13.75, 56.3, 61.7)
    text(d, (1360, 420 + rn), f"{v:.1f}%", big, ORANGE_HI, an, anchor="mm")
    text(d, (560, 545 + rn), "Qwen3-1.7B, zero-shot", REG(34), MUTED, an, anchor="mm")
    text(d, (1360, 545 + rn), "after GRPO", REG(34), TEXT, an, anchor="mm")
    a1, r1 = appear(t, 14.375)
    text(d, (W // 2, 720 + r1), "Same gain with 3 training seeds.", SEMI(54), TEXT, a1, anchor="mm")
    a2, r2 = appear(t, 15.0)
    text(d, (W // 2, 810 + r2), "66.3% with the full inference stack.", LIGHT(48), MUTED, a2, anchor="mm")


def scene4(t, d):
    a1, r1 = appear(t, 17.6, 0.7)
    text(d, (W // 2, 420 + r1), "tiny-sql-rl", TITLE(150), TEXT, a1, anchor="mm")
    w = int(300 * ease_io((t - 17.9) / 0.6))
    if w > 0:
        d.rectangle([W // 2 - w, 520, W // 2 + w, 526], fill=rgba(ORANGE_HI, 1))
    a2, r2 = appear(t, 18.125)
    text(d, (W // 2, 610 + r2), "huggingface.co/Alpericon/tiny-sql-rl", REG(46), TEXT, a2, anchor="mm")
    a3, r3 = appear(t, 18.4375)
    text(d, (W // 2, 680 + r3), "github.com/alpericon/tiny-sql-rl", REG(46), MUTED, a3, anchor="mm")


# scene, start, end; each fades out over the last FO seconds, the next fades in after its start (dip through bg)
SCENES = [(scene1, 0.0, 5.0), (scene2, 5.0, 12.5), (scene3, 12.5, 17.5), (scene4, 17.5, 20.01)]
FO = 0.35


def frame(t):
    img = BGIMG.copy()
    for fn, s, e in SCENES:
        if not (s <= t < e):
            continue
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        fn(t, ImageDraw.Draw(layer))
        if e < DUR and t > e - FO:
            k = clamp((e - t) / FO)
            al = np.asarray(layer).copy()
            al[..., 3] = (al[..., 3] * ease_io(k)).astype(np.uint8)
            layer = Image.fromarray(al)
        img = Image.alpha_composite(img, layer)
    return img.convert("RGB")


# ---------- audio ----------
SR = 48000


def note(n):
    return 440.0 * 2 ** ((n - 69) / 12)


def env_adsr(n, a, r, sr=SR):
    e = np.ones(n)
    na, nr = int(a * sr), int(r * sr)
    e[:na] = np.linspace(0, 1, na) if na else 1
    if nr:
        e[-nr:] *= np.linspace(1, 0, nr)
    return e


def music():
    L = int(DUR * SR)
    out = np.zeros(L)
    tt = np.arange(L) / SR
    chords = [[57, 60, 64], [53, 57, 60], [48, 52, 55, 60], [55, 59, 62]]  # Am F C G
    roots = [45, 41, 48, 43]
    rng = np.random.default_rng(7)
    for b in range(8):
        s = int(b * BAR * SR)
        n = int((BAR + 0.6) * SR) if b < 7 else L - s
        n = min(n, L - s)
        x = np.arange(n) / SR
        ch = chords[b % 4] if b < 7 else chords[0]
        pad = np.zeros(n)
        for m in ch:
            for det in (-0.06, 0.06):
                f0 = note(m + 12) * (1 + det / 100)
                for h, g in ((1, 1.0), (2, 0.28), (3, 0.1)):
                    pad += g * np.sin(2 * np.pi * f0 * h * x + rng.uniform(0, 6.28))
        pad *= env_adsr(n, 0.5, 0.7 if b < 7 else 1.6) * 0.035
        out[s:s + n] += pad
        if b >= 1:
            bx = env_adsr(n, 0.02, 0.5) * np.exp(-x * 0.6) * 0.12
            out[s:s + n] += bx * np.sin(2 * np.pi * note(roots[b % 4] if b < 7 else 45) * x)
    # kick on beats 1 and 3, bars 3-7; soft offbeat ticks
    for k in range(int(DUR / BEAT)):
        tb = k * BEAT
        if 2 * BAR <= tb < 7 * BAR and k % 2 == 0:
            s = int(tb * SR); n = int(0.3 * SR); x = np.arange(n) / SR
            f = 45 + 70 * np.exp(-x * 30)
            out[s:s + n] += 0.22 * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-x * 13)
        if 2 * BAR <= tb < 7 * BAR:
            s = int((tb + BEAT / 2) * SR); n = int(0.05 * SR)
            nz = rng.normal(0, 1, n); nz = np.diff(np.concatenate([[0], nz]))  # brighter
            out[s:s + n] += 0.012 * nz * np.exp(-np.arange(n) / SR * 90)
    # whoosh into each cut (filtered noise swelling to the bar line)
    for c in (5.0, 12.5, 17.5):
        n = int(0.9 * SR); s = int(c * SR) - int(0.7 * SR)
        nz = rng.normal(0, 1, n)
        k = np.ones(400) / 400
        nz = np.convolve(nz, k, mode="same") * 6
        e = np.sin(np.pi * np.clip(np.arange(n) / n, 0, 1)) ** 2
        out[s:s + n] += 0.05 * nz * e
    # click on the Ask press (in key: A, short)
    s = int(T_PRESS * SR); n = int(0.08 * SR); x = np.arange(n) / SR
    out[s:s + n] += 0.07 * np.sin(2 * np.pi * note(81) * x) * np.exp(-x * 70)
    # soft bells on the numbers and lines (A minor pentatonic)
    for tb, m in ((13.75, 76), (14.375, 79), (15.0, 81), (18.125, 76), (18.4375, 72)):
        s = int(tb * SR); n = int(1.6 * SR); n = min(n, L - s); x = np.arange(n) / SR
        b = np.sin(2 * np.pi * note(m) * x) + 0.25 * np.sin(2 * np.pi * note(m) * 2.76 * x) * np.exp(-x * 4)
        out[s:s + n] += 0.05 * b * np.exp(-x * 2.6) * env_adsr(n, 0.004, 0.05)
    out *= np.clip(tt / 0.3, 0, 1) * np.clip((DUR - tt) / 1.2, 0, 1)
    # gentle reverb-ish smear: a few quiet delayed copies
    wet = np.zeros_like(out)
    for dly, g in ((0.071, 0.25), (0.113, 0.2), (0.167, 0.15), (0.241, 0.1)):
        d = int(dly * SR); wet[d:] += g * out[:-d]
    out = out + 0.5 * wet
    out = np.tanh(out * 1.2) / 1.2
    out *= 0.89 / np.max(np.abs(out))
    st = np.stack([out, out], 1)
    with wave.open(str(WORK / "music.wav"), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((st * 32767).astype(np.int16).tobytes())


def main():
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    if "--stills" in sys.argv:
        for s in sys.argv[sys.argv.index("--stills") + 1].split(","):
            frame(float(s)).save(WORK / f"still_{float(s):05.2f}.png")
        return
    music()
    p = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                          "-i", "-", "-i", str(WORK / "music.wav"), "-c:v", "libx264", "-preset", "slow", "-crf", "17",
                          "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
                          "-t", str(DUR), str(WORK / "render.mp4")], stdin=subprocess.PIPE)
    for i in range(N):
        p.stdin.write(frame(i / FPS).tobytes())
    p.stdin.close()
    p.wait()
    print("rendered", p.returncode)


if __name__ == "__main__":
    main()
