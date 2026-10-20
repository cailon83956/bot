"""
Phần lõi dùng chung cho các bot (Discord / Telegram) của AICoverGen.

Gồm: cấu hình qua biến môi trường, lưu trữ JSON, tiện ích format,
tải & cài đặt voice model RVC.

Không import discord hay telegram ở đây để mọi bot đều dùng được.
"""
import asyncio
import functools
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import uuid
import zipfile
from pathlib import Path
from urllib.parse import urlparse

SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent

log = logging.getLogger('aicover')


# --------------------------------------------------------------------------------------
# Cấu hình
# --------------------------------------------------------------------------------------
def load_dotenv(path):
    if not path.exists():
        return
    for line in path.read_text(encoding='utf8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_dotenv(BASE_DIR / '.env')


def env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (ValueError, TypeError):
        return default


def env_ids(name):
    return {int(x) for x in re.findall(r'-?\d+', os.environ.get(name, ''))}


OWNER_IDS = env_ids('OWNER_IDS')                    # toàn quyền (cleanup, model...)
ADMIN_IDS = env_ids('ADMIN_IDS') | env_ids('ADMIN_ROLE_IDS')
MAX_DURATION_MIN = env_int('MAX_DURATION_MIN', 10)  # độ dài bài tối đa (phút), 0 = không giới hạn
MAX_JOBS_PER_USER = env_int('MAX_JOBS_PER_USER', 3)
MAX_QUEUE = env_int('MAX_QUEUE', 20)
CLEAN_AFTER_HOURS = env_int('CLEAN_AFTER_HOURS', 12)
ALLOW_USER_MODELS = os.environ.get('ALLOW_USER_MODELS', '0') == '1'
UPLOAD_FALLBACK = os.environ.get('UPLOAD_FALLBACK', '0') == '1'
MAX_MODEL_BYTES = 2 * 1024 ** 3
DEFAULT_UPLOAD_LIMIT = 50 * 1024 * 1024             # Telegram Bot API: 50 MB

DATA_DIR = BASE_DIR / 'bot_data'
UPLOAD_DIR = BASE_DIR / 'uploads'
for _d in (DATA_DIR, UPLOAD_DIR):
    _d.mkdir(exist_ok=True)

AUDIO_EXTS = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.opus', '.aac', '.wma', '.mp4', '.webm'}

# Giá trị mặc định + giới hạn (key = tên tham số của song_cover_pipeline)
SETTING_SPECS = {
    'voice':             (str, None, None, None),
    'pitch_change':      (int, -3, 3, 0),
    'pitch_change_all':  (int, -12, 12, 0),
    'f0_method':         (str, None, None, 'rmvpe'),
    'crepe_hop_length':  (int, 32, 512, 128),
    'index_rate':        (float, 0, 1, 0.5),
    'filter_radius':     (int, 0, 7, 3),
    'rms_mix_rate':      (float, 0, 1, 0.25),
    'protect':           (float, 0, 0.5, 0.33),
    'main_gain':         (int, -20, 20, 0),
    'backup_gain':       (int, -20, 20, 0),
    'inst_gain':         (int, -20, 20, 0),
    'reverb_rm_size':    (float, 0, 1, 0.15),
    'reverb_wet':        (float, 0, 1, 0.2),
    'reverb_dry':        (float, 0, 1, 0.8),
    'reverb_damping':    (float, 0, 1, 0.7),
    'output_format':     (str, None, None, 'mp3'),
}
DEFAULT_SETTINGS = {k: v[3] for k, v in SETTING_SPECS.items()}
SETTING_LABELS = {
    'voice': 'Voice model', 'pitch_change': 'Pitch giọng (octave)', 'pitch_change_all': 'Pitch toàn bài (semitone)',
    'f0_method': 'Thuật toán pitch', 'crepe_hop_length': 'Crepe hop', 'index_rate': 'Index rate',
    'filter_radius': 'Filter radius', 'rms_mix_rate': 'RMS mix', 'protect': 'Protect',
    'main_gain': 'Vol giọng chính (dB)', 'backup_gain': 'Vol giọng phụ (dB)', 'inst_gain': 'Vol nhạc nền (dB)',
    'reverb_rm_size': 'Reverb size', 'reverb_wet': 'Reverb wet', 'reverb_dry': 'Reverb dry',
    'reverb_damping': 'Reverb damping', 'output_format': 'Định dạng xuất',
}

STAGE_VI = [
    ('Starting', '🚀 Khởi động pipeline'),
    ('Downloading', '⬇️ Đang tải bài hát'),
    ('Separating Vocals from Instrumental', '🎼 Tách vocal / nhạc nền'),
    ('Separating Main Vocals', '🎤 Tách giọng chính / giọng phụ'),
    ('DeReverb', '🧼 Khử vang (DeReverb)'),
    ('Converting voice', '🧠 Đổi giọng bằng RVC'),
    ('audio effects', '🎛️ Thêm hiệu ứng cho giọng'),
    ('overall pitch', '🎹 Đổi key toàn bài'),
    ('Combining', '🎚️ Trộn giọng AI + nhạc nền'),
    ('Removing', '🧹 Dọn file tạm'),
    ('Done', '✅ Hoàn tất'),
]


def stage_vi(msg):
    for key, vi in STAGE_VI:
        if key.lower() in msg.lower():
            return vi
    return msg


# --------------------------------------------------------------------------------------
# Tiện ích
# --------------------------------------------------------------------------------------
class JsonStore:
    def __init__(self, path, default):
        self.path = Path(path)
        self.default = default
        try:
            self.data = json.loads(self.path.read_text(encoding='utf8'))
        except Exception:
            self.data = default() if callable(default) else default

    def save(self):
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding='utf8')
        os.replace(tmp, self.path)


def fmt_duration(sec):
    sec = int(sec or 0)
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f'{h}:{m:02d}:{s:02d}' if h else f'{m}:{s:02d}'


def fmt_size(n):
    for unit in ('B', 'KB', 'MB', 'GB'):
        if n < 1024 or unit == 'GB':
            return f'{n:.1f} {unit}' if unit != 'B' else f'{n} B'
        n /= 1024


def progress_bar(p, width=18):
    p = max(0.0, min(1.0, p))
    filled = int(round(p * width))
    return '▰' * filled + '▱' * (width - filled) + f' {int(p * 100)}%'


def safe_name(name, maxlen=50):
    name = re.sub(r'[^\w\-. ]', '_', str(name), flags=re.UNICODE).strip(' .')
    return name[:maxlen]


def is_url(text):
    return urlparse(str(text).strip()).scheme in ('http', 'https')


async def run_blocking(func, *args, executor=None, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, functools.partial(func, *args, **kwargs))


def list_models():
    d = BASE_DIR / 'rvc_models'
    if not d.exists():
        return []
    out = []
    for p in sorted(d.iterdir(), key=lambda x: x.name.lower()):
        if p.is_dir() and any(f.suffix == '.pth' for f in p.iterdir()):
            out.append(p.name)
    return out


def probe_duration(path):
    try:
        out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(path)],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        return float(out)
    except Exception:
        return 0


def shrink_audio(src, limit):
    """Nén lại thành mp3 với bitrate giảm dần cho đến khi vừa giới hạn upload. Trả về path hoặc None."""
    for kbps in (192, 128, 96, 64, 48):
        dst = Path(tempfile.gettempdir()) / f'{uuid.uuid4().hex}_{kbps}.mp3'
        r = subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(src), '-vn', '-b:a', f'{kbps}k', str(dst)])
        if r.returncode == 0 and dst.exists() and dst.stat().st_size <= limit:
            return dst
        dst.unlink(missing_ok=True)
    return None


def to_mp3(src, kbps=192):
    dst = Path(tempfile.gettempdir()) / f'{uuid.uuid4().hex}_{Path(src).stem[:60]}.mp3'
    r = subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(src), '-vn', '-b:a', f'{kbps}k', str(dst)])
    return dst if r.returncode == 0 else None


def upload_litterbox(path):
    import requests
    with open(path, 'rb') as f:
        r = requests.post('https://litterbox.catbox.moe/resources/internals/api.php',
                          data={'reqtype': 'fileupload', 'time': '72h'}, files={'fileToUpload': f}, timeout=300)
    r.raise_for_status()
    return r.text.strip()


# ---- tải / giải nén model --------------------------------------------------------------
def normalize_model_url(url):
    p = urlparse(url)
    if p.hostname and 'huggingface.co' in p.hostname:
        url = url.replace('/blob/', '/resolve/').split('?')[0] + '?download=true'
    elif p.hostname and 'pixeldrain.com' in p.hostname:
        m = re.search(r'/(?:u|api/file)/([\w-]+)', p.path)
        if m:
            url = f'https://pixeldrain.com/api/file/{m.group(1)}'
    elif p.hostname and 'dropbox.com' in p.hostname:
        url = re.sub(r'dl=0', 'dl=1', url) if 'dl=' in url else url + ('&' if '?' in url else '?') + 'dl=1'
    return url


def download_file(url, dest_dir):
    """Tải file về dest_dir, trả về path. Hỗ trợ link trực tiếp / HuggingFace / PixelDrain / Dropbox / Google Drive (cần gdown)."""
    import requests
    dest_dir = Path(dest_dir)
    if 'drive.google.com' in url or 'docs.google.com' in url:
        try:
            import gdown
        except ImportError:
            raise Exception('Link Google Drive cần cài `gdown` (pip install gdown). Hoặc dùng link trực tiếp / HuggingFace.')
        out = gdown.download(url, str(dest_dir) + os.sep, quiet=True, fuzzy=True)
        if not out:
            raise Exception('Không tải được từ Google Drive (file riêng tư hoặc hết lượt tải).')
        return Path(out)

    url = normalize_model_url(url)
    with requests.get(url, stream=True, timeout=30, headers={'User-Agent': 'Mozilla/5.0'}, allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length') or 0)
        if total > MAX_MODEL_BYTES:
            raise Exception(f'File quá lớn ({fmt_size(total)}).')
        name = None
        cd = r.headers.get('content-disposition', '')
        m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)', cd)
        if m:
            name = m.group(1)
        name = safe_name(name or os.path.basename(urlparse(r.url).path) or 'model.zip', 100) or 'model.zip'
        path = dest_dir / name
        size = 0
        with open(path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                size += len(chunk)
                if size > MAX_MODEL_BYTES:
                    raise Exception('File quá lớn (>2GB).')
                f.write(chunk)
    return path


def install_model(file_path, dir_name):
    """Từ file zip/.pth -> rvc_models/<dir_name>/{model.pth, model.index}. Trả về (pth_name, có_index)."""
    models_dir = BASE_DIR / 'rvc_models'
    target = models_dir / dir_name
    if target.exists():
        raise Exception(f'Model `{dir_name}` đã tồn tại. Hãy chọn tên khác.')
    file_path = Path(file_path)
    work = Path(tempfile.mkdtemp(prefix='mdl_'))
    try:
        if zipfile.is_zipfile(file_path):
            total = 0
            with zipfile.ZipFile(file_path) as z:
                for info in z.infolist():
                    total += info.file_size
                    dest = (work / info.filename).resolve()
                    if work.resolve() not in dest.parents and dest != work.resolve():
                        raise Exception('Zip chứa đường dẫn không hợp lệ.')
                if total > MAX_MODEL_BYTES * 2:
                    raise Exception('Zip giải nén quá lớn.')
                z.extractall(work)
        elif file_path.suffix.lower() in ('.pth', '.index'):
            shutil.copy(file_path, work / file_path.name)
        else:
            raise Exception('Chỉ hỗ trợ file .zip hoặc .pth.')

        pths, idxs = [], []
        for root, _, files in os.walk(work):
            for n in files:
                p = Path(root) / n
                if n.endswith('.pth') and not n.startswith(('G_', 'D_')):
                    pths.append(p)
                elif n.endswith('.index') and not n.startswith('trained_'):
                    idxs.append(p)
        if not pths:
            raise Exception('Không tìm thấy file .pth trong file đã tải (cần model RVC v1/v2).')
        pth = max(pths, key=lambda p: p.stat().st_size)
        idx = max(idxs, key=lambda p: p.stat().st_size) if idxs else None
        target.mkdir(parents=True)
        shutil.move(str(pth), target / safe_name(pth.name, 100))
        if idx:
            shutil.move(str(idx), target / safe_name(idx.name, 100))
        return pth.name, bool(idx)
    finally:
        shutil.rmtree(work, ignore_errors=True)
