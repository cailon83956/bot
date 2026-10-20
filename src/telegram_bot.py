"""
AICoverGen — Telegram bot (không có Web UI).

Chạy:  python src/telegram_bot.py      (hoặc bash run_telegram_bot.sh)
Token: biến môi trường TELEGRAM_BOT_TOKEN, file .env, hoặc Kaggle Secret tên TELEGRAM_BOT_TOKEN.

Tính năng
  /cover      tạo AI cover từ link YouTube/SoundCloud/..., tên bài hát, file đính kèm
  /search     tìm bài trên YouTube rồi bấm nút để tạo cover
  /separate   chỉ tách vocal / nhạc nền (stems)
  /settings   lưu thông số mặc định riêng cho từng người dùng
  /model      xem / thêm / tải lên / xoá / đổi tên voice model RVC
  /queue /cancel /history /resend
  /gpu /ping /cleanup /help

Khác biệt so với bản Discord: Telegram không có voice channel cho bot,
nên kết quả được gửi thẳng dạng file âm thanh (mp3/wav/flac).
"""
import asyncio
import gc
import logging
import os
import re
import shutil
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from core import (  # noqa: E402
    ADMIN_IDS, ALLOW_USER_MODELS, AUDIO_EXTS, BASE_DIR as _BD, CLEAN_AFTER_HOURS, DATA_DIR,
    DEFAULT_SETTINGS, DEFAULT_UPLOAD_LIMIT, MAX_DURATION_MIN, MAX_JOBS_PER_USER, MAX_QUEUE,
    OWNER_IDS, SETTING_LABELS, SETTING_SPECS, UPLOAD_DIR, UPLOAD_FALLBACK,
    JsonStore, download_file, fmt_duration, fmt_size, install_model, is_url, list_models,
    progress_bar, probe_duration, run_blocking, safe_name, shrink_audio, stage_vi, to_mp3,
    upload_litterbox,
)

log = logging.getLogger('aicover.tg')

try:
    from telegram import (
        InlineKeyboardButton, InlineKeyboardMarkup, Update,
    )
    from telegram.ext import (
        Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters,
    )
except ImportError:  # pragma: no cover
    sys.exit('Thiếu python-telegram-bot. Cài bằng: pip install "python-telegram-bot>=20.0"')


# --------------------------------------------------------------------------------------
# Cấu hình
# --------------------------------------------------------------------------------------
def get_token():
    token = os.environ.get('TELEGRAM_BOT_TOKEN')
    if token:
        return token
    try:  # Kaggle Secrets
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret('TELEGRAM_BOT_TOKEN')
    except Exception:
        pass
    try:  # Colab Secrets
        from google.colab import userdata
        return userdata.get('TELEGRAM_BOT_TOKEN')
    except Exception:
        return None


# --------------------------------------------------------------------------------------
# Job / hàng đợi
# --------------------------------------------------------------------------------------
@dataclass
class Job:
    id: str
    kind: str                      # 'cover' | 'separate'
    user_id: int
    user_name: str
    chat_id: int
    song_input: str                # URL hoặc đường dẫn file local
    title: str
    params: dict = field(default_factory=dict)
    keep_files: bool = False
    cleanup_paths: list = field(default_factory=list)
    status: str = 'pending'        # pending/running/done/failed/cancelled
    progress: float = 0.0
    stage: str = ''
    cancelled: bool = False
    created: float = field(default_factory=time.time)
    started: float = 0.0
    thumbnail: Optional[str] = None
    status_msg = None              # telegram.Message


class CoverBot:
    """Chứa logic chung, không phụ thuộc framework."""

    def __init__(self):
        self.settings = JsonStore(DATA_DIR / 'settings.json', dict)
        self.history = JsonStore(DATA_DIR / 'history.json', list)
        self.queue = asyncio.Queue()
        self.pending = []
        self.current: Optional[Job] = None
        self.jobs = {}
        self.executor = None
        self.worker_task = None
        self.started_at = time.time()
        self.stats = {'done': 0, 'failed': 0}

    # ---------------- quyền ----------------
    def is_admin(self, user_id: int) -> bool:
        return user_id in OWNER_IDS or user_id in ADMIN_IDS

    def can_add_models(self, user_id: int) -> bool:
        return ALLOW_USER_MODELS or self.is_admin(user_id)

    # ---------------- cài đặt người dùng ----------------
    def get_settings(self, user_id):
        s = dict(DEFAULT_SETTINGS)
        s.update(self.settings.data.get(str(user_id), {}))
        return s

    def update_settings(self, user_id, changes: dict):
        cur = self.settings.data.setdefault(str(user_id), {})
        cur.update({k: v for k, v in changes.items() if v is not None})
        self.settings.save()

    # ---------------- housekeeping ----------------
    def cleanup_old_files(self, hours, force=False):
        cutoff = time.time() - hours * 3600
        freed = 0
        busy = self.current.song_input if self.current else None
        for root in (BASE_DIR / 'song_output', BASE_DIR / 'downloads', UPLOAD_DIR):
            if not root.exists():
                continue
            for p in root.iterdir():
                try:
                    if p.stat().st_mtime < cutoff or force:
                        if self.current and not force and busy and str(p) in str(busy):
                            continue
                        size = sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) if p.is_dir() else p.stat().st_size
                        shutil.rmtree(p) if p.is_dir() else p.unlink()
                        freed += size
                except Exception as e:
                    log.warning('cleanup %s: %s', p, e)
        return freed

    # ---------------- worker ----------------
    async def worker(self):
        while True:
            job = await self.queue.get()
            if job.cancelled:
                self.finish_job(job)
                continue
            self.pending = [j for j in self.pending if j.id != job.id]
            self.current = job
            try:
                await self.run_job(job)
            except Exception:
                log.exception('Worker error')
            finally:
                self.current = None
                self.finish_job(job)

    def finish_job(self, job):
        self.jobs.pop(job.id, None)
        self.pending = [j for j in self.pending if j.id != job.id]
        for p in job.cleanup_paths:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass

    @staticmethod
    def job_text(job: Job, queue_pos=None, error=None):
        title = {'pending': '⏳ Đang chờ trong hàng đợi', 'running': '⚙️ Đang xử lý', 'done': '✅ Hoàn tất',
                 'failed': '❌ Lỗi', 'cancelled': '🛑 Đã huỷ'}[job.status]
        lines = [f'**{title}**', f'🎵 {job.title[:120]}']
        if job.kind == 'cover':
            lines.append(f'🎙️ Voice: `{job.params.get("voice")}` • Pitch: {job.params.get("pitch_change", 0):+d} oct')
        else:
            lines.append('🎼 Chế độ: tách stems')
        lines.append(f'🆔 Job: `{job.id}`')
        if job.status == 'pending' and queue_pos:
            lines.append(f'📍 Vị trí hàng đợi: #{queue_pos}')
        if job.status == 'running':
            lines.append(f'{job.stage or "Đang xử lý"} {progress_bar(job.progress)}')
        if error:
            lines.append(f'\n```\n{error[:900]}\n```')
        lines.append(f'\n👤 {job.user_name}')
        return '\n'.join(lines)

    async def edit_status(self, job: Job, text: str):
        if not job.status_msg:
            return
        try:
            await job.status_msg.edit_text(text, parse_mode='Markdown')
        except Exception:
            pass

    # ---------------- thực thi job ----------------
    async def run_job(self, job: Job):
        import main as pipeline  # import muộn: nặng (torch, fairseq)
        job.status = 'running'
        job.started = time.time()
        job.stage = stage_vi('Starting')

        def cb(message, percent):
            if job.cancelled:
                raise pipeline.JobCancelled()
            job.progress = percent
            job.stage = stage_vi(message)

        async def updater():
            last = None
            while True:
                await asyncio.sleep(5)
                state = (job.stage, round(job.progress, 2))
                if state != last:
                    last = state
                    await self.edit_status(job, self.job_text(job))

        await self.edit_status(job, self.job_text(job))
        upd = asyncio.create_task(updater())
        try:
            if job.kind == 'separate':
                result = await run_blocking(pipeline.separate_only, job.song_input, cb, executor=self.executor)
            else:
                p = job.params
                result = await run_blocking(
                    pipeline.song_cover_pipeline, job.song_input, p['voice'], p['pitch_change'], job.keep_files,
                    main_gain=p['main_gain'], backup_gain=p['backup_gain'], inst_gain=p['inst_gain'],
                    index_rate=p['index_rate'], filter_radius=p['filter_radius'], rms_mix_rate=p['rms_mix_rate'],
                    f0_method=p['f0_method'], crepe_hop_length=p['crepe_hop_length'], protect=p['protect'],
                    pitch_change_all=p['pitch_change_all'], reverb_rm_size=p['reverb_rm_size'],
                    reverb_wet=p['reverb_wet'], reverb_dry=p['reverb_dry'], reverb_damping=p['reverb_damping'],
                    output_format=p['output_format'], progress_cb=cb, executor=self.executor)
            job.status = 'done'
            job.progress = 1.0
            self.stats['done'] += 1
        except pipeline.JobCancelled:
            job.status = 'cancelled'
        except Exception as e:
            log.exception('Job %s failed', job.id)
            job.status = 'failed'
            self.stats['failed'] += 1
            err = str(e) or e.__class__.__name__
            if 'sign in' in err.lower():
                err += '\n\n→ YouTube chặn IP máy chủ. Đặt YT_COOKIES_FILE=cookies.txt hoặc dùng file đính kèm.'
            await self.edit_status(job, self.job_text(job, error=err))
        finally:
            upd.cancel()
            await run_blocking(self.free_gpu, executor=self.executor)

        if job.status == 'cancelled':
            await self.edit_status(job, self.job_text(job))
            return
        if job.status == 'done':
            elapsed = time.time() - job.started
            await self.edit_status(job, self.job_text(job))
            try:
                if job.kind == 'separate':
                    await self.deliver_stems(job, result, elapsed)
                else:
                    await self.deliver_cover(job, result, elapsed)
            except Exception as e:
                log.exception('Deliver failed')
                await self.send(job.chat_id, f'⚠️ Xong nhưng không gửi được file: `{e}`')

    @staticmethod
    def free_gpu():
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ---------------- gửi kết quả ----------------
    async def send(self, chat_id, text, **kw):
        try:
            return await self.app.bot.send_message(chat_id, text, **kw)
        except Exception:
            log.warning('Cannot send to %s', chat_id)
            return None

    async def prepare_file(self, path, name_hint=None):
        """Trả về (path_gửi, tmp_path). Nén nếu vượt giới hạn upload."""
        limit = int(DEFAULT_UPLOAD_LIMIT * 0.97)
        path = Path(path)
        if path.stat().st_size > limit:
            tmp = await run_blocking(shrink_audio, path, limit)
            if tmp:
                return tmp, tmp
            if UPLOAD_FALLBACK:
                link = await run_blocking(upload_litterbox, path)
                return None, link
            raise Exception(f'File {fmt_size(path.stat().st_size)} vượt giới hạn upload ({fmt_size(limit)}) '
                            f'ngay cả khi nén. Bật UPLOAD_FALLBACK=1 để gửi qua link.')
        return path, None

    async def deliver_cover(self, job, cover_path, elapsed):
        title = job.title
        voice = job.params['voice']
        header = (f'🎤 **AI Cover đã sẵn sàng**\n'
                  f'🎵 {title[:120]}\n'
                  f'🎙️ Voice: `{voice}` • Pitch: {job.params["pitch_change"]:+d} oct / {job.params["pitch_change_all"]:+d} st\n'
                  f'⏱️ {elapsed:.0f}s • Job `{job.id}`')
        send_path, tmp = await self.prepare_file(cover_path, f'{safe_name(title, 60) or "cover"}.mp3')
        try:
            if send_path is None:
                await self.send(job.chat_id, f'{header}\n🔗 {tmp}')
            else:
                with open(send_path, 'rb') as fh:
                    # Telegram send_audio chỉ nhận mp3/m4a/ogg; wav/flac phải gửi dạng document
                    if Path(send_path).suffix.lower() in ('.mp3', '.m4a', '.ogg', '.oga'):
                        await self.app.bot.send_audio(chat_id=job.chat_id, audio=fh, caption=header)
                    else:
                        await self.app.bot.send_document(chat_id=job.chat_id, document=fh, caption=header)
        finally:
            if tmp and Path(tmp).exists() and str(tmp) != str(cover_path):
                Path(tmp).unlink(missing_ok=True)

        rec = {'id': job.id, 'user_id': job.user_id, 'title': title, 'voice': voice, 'path': str(cover_path),
               'created': int(time.time()), 'chat_id': job.chat_id}
        self.history.data.append(rec)
        self.history.data = self.history.data[-500:]
        self.history.save()

    async def deliver_stems(self, job, res, elapsed):
        header = f'🎼 **Đã tách stems**\n🎵 {job.title[:120]}\n⏱️ {elapsed:.0f}s • Job `{job.id}`'
        for key, label in (('vocals', 'Vocals'), ('instrumental', 'Instrumental'),
                           ('main_vocals_dereverb', 'MainVocals_DeReverb'), ('backup_vocals', 'BackupVocals')):
            p = res.get(key)
            if p and os.path.exists(p):
                mp3 = await run_blocking(to_mp3, p)
                if not mp3:
                    continue
                try:
                    with open(mp3, 'rb') as fh:
                        await self.app.bot.send_audio(chat_id=job.chat_id, audio=fh,
                                                      caption=f'{header}\n🔹 {label}')
                finally:
                    Path(mp3).unlink(missing_ok=True)
    # ---------------- thêm job ----------------
    def user_active_jobs(self, user_id):
        return sum(1 for j in self.jobs.values() if j.user_id == user_id)

    async def resolve_source(self, song, file):
        """Trả về (song_input, title, duration, thumbnail, cleanup_paths)."""
        import main as pipeline
        if file is not None:
            fname = getattr(file, 'file_name', None) or 'audio'
            ext = Path(fname).suffix.lower()
            if ext not in AUDIO_EXTS:
                raise Exception('File đính kèm phải là âm thanh (mp3, wav, flac, m4a, ogg...).')
            dest = UPLOAD_DIR / f'{uuid.uuid4().hex[:8]}_{safe_name(fname, 80) or ("audio" + ext)}'
            await file.download_to_drive(custom_path=dest)
            dur = await run_blocking(probe_duration, dest)
            return str(dest), Path(fname).stem, dur, None, [dest]
        if not song:
            raise Exception('Hãy nhập link / tên bài hát hoặc đính kèm file âm thanh.')
        info = await run_blocking(pipeline.get_media_info, song.strip())
        return info['url'], info['title'], info['duration'], info.get('thumbnail'), []

    async def submit(self, chat_id, user_id, user_name, kind, song_input, title, duration, thumb, cleanup, overrides,
                     keep_files=False):
        try:
            if MAX_DURATION_MIN and duration and duration > MAX_DURATION_MIN * 60:
                raise Exception(f'Bài dài {fmt_duration(duration)} — giới hạn là {MAX_DURATION_MIN} phút.')
            if len(self.pending) >= MAX_QUEUE:
                raise Exception(f'Hàng đợi đã đầy ({MAX_QUEUE}). Thử lại sau.')
            if self.user_active_jobs(user_id) >= MAX_JOBS_PER_USER:
                raise Exception(f'Bạn đang có {MAX_JOBS_PER_USER} job chưa xong. Hãy đợi hoặc /cancel.')
            params = {}
            if kind == 'cover':
                params = self.get_settings(user_id)
                params.update({k: v for k, v in overrides.items() if v is not None})
                if not params.get('voice'):
                    raise Exception('Bạn chưa chọn voice model. Thêm tham số `voice=...` hoặc dùng /settings set voice=...')
                if params['voice'] not in list_models():
                    raise Exception(f"Voice model `{params['voice']}` không tồn tại. Xem /model list.")
        except Exception:
            for p in cleanup:
                Path(p).unlink(missing_ok=True)
            raise

        job = Job(id=uuid.uuid4().hex[:6], kind=kind, user_id=user_id, user_name=user_name, chat_id=chat_id,
                  song_input=song_input, title=title, params=params, keep_files=keep_files,
                  cleanup_paths=cleanup, thumbnail=thumb)
        self.jobs[job.id] = job
        self.pending.append(job)
        await self.queue.put(job)
        msg = await self.send(chat_id, self.job_text(job, queue_pos=len(self.pending)))
        job.status_msg = msg
        return job


# --------------------------------------------------------------------------------------
# Bot Telegram
# --------------------------------------------------------------------------------------
class TgBot(CoverBot):
    def __init__(self):
        super().__init__()
        self.app = None

    # ---------------- đăng ký lệnh ----------------
    def build(self, token):
        app = Application.builder().token(token).build()
        self.app = app
        app.add_handler(CommandHandler('start', self.cmd_start))
        app.add_handler(CommandHandler('help', self.cmd_help))
        app.add_handler(CommandHandler('cover', self.cmd_cover))
        app.add_handler(CommandHandler('search', self.cmd_search))
        app.add_handler(CommandHandler('separate', self.cmd_separate))
        app.add_handler(CommandHandler('queue', self.cmd_queue))
        app.add_handler(CommandHandler('cancel', self.cmd_cancel))
        app.add_handler(CommandHandler('history', self.cmd_history))
        app.add_handler(CommandHandler('resend', self.cmd_resend))
        app.add_handler(CommandHandler('settings', self.cmd_settings))
        app.add_handler(CommandHandler('model', self.cmd_model))
        app.add_handler(CommandHandler('gpu', self.cmd_gpu))
        app.add_handler(CommandHandler('ping', self.cmd_ping))
        app.add_handler(CommandHandler('cleanup', self.cmd_cleanup))
        app.add_handler(CallbackQueryHandler(self.on_search_pick, pattern=r'^pick:'))
        # upload model (.zip kèm caption /modeladd) phải đứng trước handler file âm thanh
        app.add_handler(MessageHandler(filters.Document.ALL & filters.CAPTION, self.on_model_upload))
        # file đính kèm để tạo cover / tách stems
        app.add_handler(MessageHandler(filters.Document.ALL | filters.AUDIO | filters.VOICE, self.on_audio_file))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.on_text))
        return app

    async def post_init(self, _):
        from concurrent.futures import ThreadPoolExecutor
        from telegram import BotCommand
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='gpu')
        try:
            await self.app.bot.set_my_commands([
                BotCommand('cover', 'Tạo AI cover từ link/tên bài/file'),
                BotCommand('search', 'Tìm bài trên YouTube'),
                BotCommand('separate', 'Tách vocal / nhạc nền'),
                BotCommand('queue', 'Xem hàng đợi'),
                BotCommand('cancel', 'Huỷ job của bạn'),
                BotCommand('history', 'Lịch sử cover'),
                BotCommand('resend', 'Gửi lại cover cũ'),
                BotCommand('settings', 'Xem/đặt cài đặt cá nhân'),
                BotCommand('model', 'Quản lý voice model RVC'),
                BotCommand('gpu', 'Trạng thái hệ thống'),
                BotCommand('ping', 'Độ trễ bot'),
                BotCommand('cleanup', 'Dọn file tạm (admin)'),
                BotCommand('help', 'Hướng dẫn'),
            ])
        except Exception:
            log.warning('Không đặt được danh sách lệnh', exc_info=True)
        try:
            await asyncio.to_thread(__import__('importlib').import_module, 'main')
        except Exception:
            log.exception('Không import được pipeline (main.py)')
        self.worker_task = asyncio.create_task(self.worker())
        asyncio.create_task(self.housekeeping())

    async def post_shutdown(self, _):
        if self.worker_task:
            self.worker_task.cancel()
        if self.executor:
            self.executor.shutdown(wait=False)

    async def housekeeping(self):
        while True:
            await asyncio.sleep(600)
            try:
                await run_blocking(self.cleanup_old_files, CLEAN_AFTER_HOURS)
            except Exception:
                log.warning('housekeeping lỗi')

    # ---------------- helpers ----------------
    @staticmethod
    def parse_args(args):
        """Chia tham số dạng key=value. Phần còn lại là vị trí đầu làm 'song'."""
        song_parts, kv = [], {}
        for a in args:
            if '=' in a and re.match(r'^[a-z_][a-z0-9_]*=', a):
                k, v = a.split('=', 1)
                kv[k.strip().lower()] = v.strip()
            else:
                song_parts.append(a)
        return ' '.join(song_parts).strip(), kv

    @staticmethod
    def coerce(spec, raw):
        typ = spec[0]
        try:
            if typ is int:
                return int(raw)
            if typ is float:
                return float(raw)
        except (TypeError, ValueError):
            return None
        return raw

    @staticmethod
    def user_name(update: Update) -> str:
        u = update.effective_user
        return (u.full_name or u.username or str(u.id))[:64]

    # ---------------- lệnh: cơ bản ----------------
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            '🎤 **AICoverGen Telegram Bot**\n\n'
            'Biến bất kỳ bài hát nào thành bản cover bằng giọng RVC.\n\n'
            'Bắt đầu: `/cover <link hoặc tên bài>` hoặc gửi file âm thanh kèm caption `/cover`.\n'
            'Xem `/help` để biết đầy đủ.')
        if update.effective_user.id not in OWNER_IDS:
            log.info('User mới: %s (%s)', update.effective_user.id, self.user_name(update))

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        text = (
            '🎤 **AICoverGen**\n\n'
            '**Tạo cover**\n'
            '`/cover <link|tên bài> [voice=ten] [pitch=+1] [pitch_all=-2] [f0=rmvpe] '
            '[index=0.5] [main_vol=0] [format=mp3]`\n'
            '`/search <từ khoá>` → bấm nút chọn bài\n'
            '`/separate <link|tên bài>` → chỉ tách vocal/nhạc nền\n'
            'Gửi file âm thanh kèm caption `/cover` cũng được.\n\n'
            '**Cá nhân hoá**\n'
            '`/settings show` • `/settings set voice=ten pitch=1` • `/settings reset`\n\n'
            '**Model RVC**\n'
            '`/model list` • `/model info <tên>`\n'
            '`/model add <link> <tên>` • `/model remove <tên>` • `/model rename <cũ> <mới>`\n'
            'Gửi file .zip kèm caption `/modeladd <tên>` để upload.\n\n'
            '**Hàng đợi**\n'
            '`/queue` • `/cancel [job_id]` • `/history` • `/resend [job_id]`\n\n'
            '**Khác**\n'
            '`/gpu` • `/ping` • `/cleanup` (admin)\n\n'
            f'_Giới hạn: {MAX_DURATION_MIN or "∞"} phút/bài • {MAX_JOBS_PER_USER} job/người_'
        )
        await update.message.reply_text(text, parse_mode='Markdown')

    async def cmd_ping(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(f'🏓 Bot đang hoạt động. ⏱ {time.time() - self.started_at:.0f}s uptime.')

    # ---------------- lệnh: cover / separate / search ----------------
    async def cmd_cover(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self._cover_or_separate(update, context, 'cover')

    async def cmd_separate(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self._cover_or_separate(update, context, 'separate')

    async def _cover_or_separate(self, update: Update, context: ContextTypes.DEFAULT_TYPE, kind):
        args = context.args or []
        song, kv = self.parse_args(args)
        overrides = {}
        if kind == 'cover':
            mapping = {
                'voice': 'voice', 'pitch': 'pitch_change', 'pitch_all': 'pitch_change_all',
                'f0': 'f0_method', 'crepe': 'crepe_hop_length', 'index': 'index_rate',
                'filter': 'filter_radius', 'rms': 'rms_mix_rate', 'protect': 'protect',
                'main_vol': 'main_gain', 'backup_vol': 'backup_gain', 'inst_vol': 'inst_gain',
                'reverb_size': 'reverb_rm_size', 'reverb_wet': 'reverb_wet', 'reverb_dry': 'reverb_dry',
                'reverb_damping': 'reverb_damping', 'format': 'output_format',
            }
            for k, v in kv.items():
                if k not in mapping:
                    continue
                target = mapping[k]
                val = self.coerce(SETTING_SPECS[target], v)
                if val is None:
                    await update.message.reply_text(f'❌ Giá trị `{k}={v}` không hợp lệ.')
                    return
                if target in ('f0_method',) and val not in ('rmvpe', 'mangio-crepe'):
                    await update.message.reply_text('❌ f0 phải là `rmvpe` hoặc `mangio-crepe`.')
                    return
                if target == 'output_format' and val not in ('mp3', 'wav', 'flac'):
                    await update.message.reply_text('❌ format phải là `mp3`, `wav` hoặc `flac`.')
                    return
                overrides[target] = val
            keep = kv.get('keep', '0') in ('1', 'true', 'yes')

        status = await update.message.reply_text('⏳ Đang chuẩn bị...')
        try:
            src, title, dur, thumb, cleanup = await self.resolve_source(song, None)
        except Exception as e:
            await status.edit_text(f'❌ {e}')
            return
        try:
            await self.submit(update.effective_chat.id, update.effective_user.id, self.user_name(update),
                              kind, src, title, dur, thumb, cleanup, overrides,
                              keep_files=locals().get('keep', False))
        except Exception as e:
            await status.edit_text(f'❌ {e}')

    async def on_audio_file(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """File đính kèm: mặc định xử lý như /cover (trừ khi caption chỉ định /separate)."""
        caption = (update.message.caption or '').strip()
        parts = caption.split()
        cmd = parts[0].lower() if parts else ''
        if cmd in ('/cover', '/separate'):
            kind = 'cover' if cmd == '/cover' else 'separate'
            rest = parts[1:]
        elif cmd == '':
            kind = 'cover'
            rest = []
        else:
            kind = 'cover'
            rest = parts
        song, kv = self.parse_args(rest)
        overrides = {}
        if kind == 'cover' and kv.get('voice'):
            overrides['voice'] = kv['voice']
        status = await update.message.reply_text('⏳ Đang chuẩn bị file...')
        try:
            src, title, dur, thumb, cleanup = await self.resolve_source(song, update.message.document or
                                                                        update.message.voice or
                                                                        update.message.audio)
            await self.submit(update.effective_chat.id, update.effective_user.id, self.user_name(update),
                              kind, src, title, dur, thumb, cleanup, overrides)
        except Exception as e:
            await status.edit_text(f'❌ {e}')

    async def cmd_search(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        import main as pipeline
        query = ' '.join(context.args or []).strip()
        if not query:
            await update.message.reply_text('Dùng: `/search <từ khoá>`')
            return
        status = await update.message.reply_text('🔎 Đang tìm...')
        try:
            results = await run_blocking(pipeline.search_youtube, query, 8)
        except Exception as e:
            await status.edit_text(f'❌ Tìm kiếm lỗi: {e}')
            return
        if not results:
            await status.edit_text('Không có kết quả.')
            return
        lines = []
        buttons = []
        for i, r in enumerate(results, 1):
            lines.append(f'{i}. {r["title"][:70]}\n   {fmt_duration(r["duration"])} • {r["uploader"][:30]}\n   {r["url"]}')
            buttons.append([InlineKeyboardButton(f'{i}. {r["title"][:40]}', callback_data=f'pick:{i-1}')])
        text = f'🔎 Kết quả cho “{query[:80]}”\n\n' + '\n'.join(lines) + '\n\nBấm nút để tạo cover:'
        await status.edit_text(text, parse_mode='Markdown',
                              reply_markup=InlineKeyboardMarkup(buttons))
        context.user_data['search_results'] = results

    async def on_search_pick(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        results = context.user_data.get('search_results') or []
        idx = int(query.data.split(':')[1])
        if not results or idx >= len(results):
            await query.edit_message_text('Kết quả đã hết hạn. Hãy /search lại.')
            return
        r = results[idx]
        await query.edit_message_text(f'✅ Đã chọn: {r["title"][:80]}\n⏳ Đang tạo job...')
        try:
            await self.submit(query.message.chat_id, query.from_user.id,
                              (query.from_user.full_name or str(query.from_user.id))[:64],
                              'cover', r['url'], r['title'], r['duration'], None, [], {})
        except Exception as e:
            await query.edit_message_text(f'❌ {e}')

    async def on_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Tin nhắn thường: nếu là số -> chọn kết quả tìm kiếm; nếu là link -> gợi ý /cover."""
        text = (update.message.text or '').strip()
        if text.isdigit():
            results = context.user_data.get('search_results') or []
            i = int(text) - 1
            if 0 <= i < len(results):
                r = results[i]
                await update.message.reply_text(f'✅ Đã chọn: {r["title"][:80]}\n⏳ Đang tạo job...')
                try:
                    await self.submit(update.effective_chat.id, update.effective_user.id,
                                      self.user_name(update), 'cover', r['url'], r['title'],
                                      r['duration'], None, [], {})
                except Exception as e:
                    await update.message.reply_text(f'❌ {e}')
                return
        if is_url(text) and ('youtube.com' in text or 'youtu.be' in text):
            await update.message.reply_text('💡 Dùng `/cover <link>` để tạo cover từ link này.')

    # ---------------- lệnh: queue / cancel / history / resend ----------------
    async def cmd_queue(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        lines = []
        if self.current:
            c = self.current
            lines.append(f'▶️ **{c.title[:70]}** (`{c.id}`) — {c.user_name}\n{c.stage} {progress_bar(c.progress, 12)}')
        if self.pending:
            lines.append('\n⏳ *Đang chờ:*')
            for i, j in enumerate(self.pending[:15], 1):
                lines.append(f'{i}. `{j.id}` {j.title[:55]} — {j.user_name}')
        if not lines:
            lines = ['Hàng đợi trống 🎉']
        await update.message.reply_text('\n'.join(lines), parse_mode='Markdown')

    async def cmd_cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        args = context.args or []
        job_id = args[0].strip() if args else None
        mine = [j for j in self.jobs.values() if j.user_id == update.effective_user.id or self.is_admin(update.effective_user.id)]
        if job_id:
            job = next((j for j in mine if j.id == job_id), None)
        else:
            own = [j for j in self.jobs.values() if j.user_id == update.effective_user.id]
            job = own[-1] if own else None
        if not job:
            await update.message.reply_text('Không tìm thấy job nào của bạn để huỷ.')
            return
        job.cancelled = True
        if job.status == 'pending':
            job.status = 'cancelled'
            self.pending = [j for j in self.pending if j.id != job.id]
            self.jobs.pop(job.id, None)
            await self.edit_status(job, self.job_text(job))
            await update.message.reply_text(f'🛑 Đã huỷ job `{job.id}`.')
        else:
            await update.message.reply_text(f'🛑 Job `{job.id}` sẽ dừng ở bước kế tiếp.')

    async def cmd_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        mine = [h for h in self.history.data if h['user_id'] == update.effective_user.id][-10:][::-1]
        if not mine:
            await update.message.reply_text('Bạn chưa tạo cover nào.')
            return
        lines = [f'`{h["id"]}` **{h["title"][:50]}** — `{h["voice"]}`' +
                 ('' if os.path.exists(h['path']) else ' *(đã hết hạn)*') for h in mine]
        await update.message.reply_text('🕘 *Lịch sử cover*\n' + '\n'.join(lines) +
                                        '\n\nDùng `/resend <id>` để gửi lại', parse_mode='Markdown')

    async def cmd_resend(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        args = context.args or []
        mine = [h for h in self.history.data if h['user_id'] == update.effective_user.id]
        rec = next((h for h in mine if h['id'] == (args[0].strip() if args else '')), None) if args \
            else (mine[-1] if mine else None)
        if not rec:
            await update.message.reply_text('Không tìm thấy cover.')
            return
        if not os.path.exists(rec['path']):
            await update.message.reply_text('File đã bị dọn (hết hạn). Hãy tạo lại bằng /cover.')
            return
        try:
            send_path, tmp = await self.prepare_file(rec['path'])
            if send_path is None:
                await update.message.reply_text(f'🔗 {tmp}')
                return
            try:
                with open(send_path, 'rb') as fh:
                    if Path(send_path).suffix.lower() in ('.mp3', '.m4a', '.ogg', '.oga'):
                        await update.message.reply_audio(audio=fh,
                                                         caption=f'**{rec["title"][:100]}** — `{rec["voice"]}`')
                    else:
                        await update.message.reply_document(document=fh,
                                                            caption=f'**{rec["title"][:100]}** — `{rec["voice"]}`')
            finally:
                if tmp and str(tmp) != str(rec['path']):
                    Path(tmp).unlink(missing_ok=True)
        except Exception as e:
            await update.message.reply_text(f'❌ {e}')

    # ---------------- lệnh: settings ----------------
    async def cmd_settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        args = context.args or []
        sub = (args[0].lower() if args else 'show')
        uid = update.effective_user.id
        if sub == 'show':
            s = self.get_settings(uid)
            lines = [f'**{SETTING_LABELS[k]}**: `{s[k] if s[k] is not None else "(chưa đặt)"}`' for k in SETTING_SPECS]
            await update.message.reply_text('⚙️ *Cài đặt của bạn*\n' + '\n'.join(lines), parse_mode='Markdown')
            return
        if sub == 'reset':
            self.settings.data.pop(str(uid), None)
            self.settings.save()
            await update.message.reply_text('♻️ Đã đặt lại cài đặt về mặc định.')
            return
        if sub == 'set':
            _, kv = self.parse_args(args[1:])
            mapping = {
                'voice': 'voice', 'pitch': 'pitch_change', 'pitch_all': 'pitch_change_all',
                'f0': 'f0_method', 'crepe': 'crepe_hop_length', 'index': 'index_rate',
                'filter': 'filter_radius', 'rms': 'rms_mix_rate', 'protect': 'protect',
                'main_vol': 'main_gain', 'backup_vol': 'backup_gain', 'inst_vol': 'inst_gain',
                'reverb_size': 'reverb_rm_size', 'reverb_wet': 'reverb_wet', 'reverb_dry': 'reverb_dry',
                'reverb_damping': 'reverb_damping', 'format': 'output_format',
            }
            changed = {}
            for k, v in kv.items():
                if k not in mapping:
                    continue
                target = mapping[k]
                val = self.coerce(SETTING_SPECS[target], v)
                if val is None:
                    await update.message.reply_text(f'❌ Giá trị `{k}={v}` không hợp lệ.')
                    return
                changed[target] = val
            if not changed:
                await update.message.reply_text('Dùng: `/settings set voice=ten pitch=1 index=0.6`')
                return
            bad_voice = None
            if 'voice' in changed and changed['voice'] not in list_models():
                bad_voice = changed.pop('voice')
            if changed:
                self.update_settings(uid, changed)
                await update.message.reply_text('✅ Đã lưu: ' +
                                                 ', '.join(f'**{SETTING_LABELS[k]}** = `{v}`' for k, v in changed.items()),
                                                 parse_mode='Markdown')
            if bad_voice:
                await update.message.reply_text(f'❌ Voice `{bad_voice}` không tồn tại. Xem /model list.')
            return
        await update.message.reply_text('Dùng: `/settings show` • `/settings set key=value ...` • `/settings reset`')

    # ---------------- lệnh: model ----------------
    async def cmd_model(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        args = context.args or []
        sub = (args[0].lower() if args else 'list')
        uid = update.effective_user.id
        if sub in ('list', ''):
            models = list_models()
            if not models:
                await update.message.reply_text('Chưa có model nào. Dùng `/model add <link> <tên>`.')
                return
            text = '🎙️ *Voice models*\n' + '\n'.join(f'• `{m}`' for m in models[:60])
            if len(models) > 60:
                text += f'\n… và {len(models) - 60} model khác'
            await update.message.reply_text(text, parse_mode='Markdown')
            return
        if sub == 'info':
            if len(args) < 2:
                await update.message.reply_text('Dùng: `/model info <tên>`')
                return
            name = args[1]
            if name not in list_models():
                await update.message.reply_text('Không có model này.')
                return
            d = BASE_DIR / 'rvc_models' / name
            files = sorted(d.iterdir())
            lines = [f'• `{f.name}` — {fmt_size(f.stat().st_size)}' for f in files]
            has_idx = any(f.suffix == '.index' for f in files)
            lines.append('')
            lines.append('Có file .index' if has_idx else '⚠️ Không có .index (index_rate không có tác dụng)')
            await update.message.reply_text(f'🎙️ *{name}*\n' + '\n'.join(lines), parse_mode='Markdown')
            return
        if sub == 'add':
            if len(args) < 3:
                await update.message.reply_text('Dùng: `/model add <link> <tên>`')
                return
            if not self.can_add_models(uid):
                await update.message.reply_text('🚫 Chỉ admin mới được thêm model.')
                return
            url, name = args[1], safe_name(args[2])
            if not name:
                await update.message.reply_text('Tên model không hợp lệ.')
                return
            if (BASE_DIR / 'rvc_models' / name).exists():
                await update.message.reply_text(f'Model `{name}` đã tồn tại.')
                return
            status = await update.message.reply_text('⬇️ Đang tải model...')
            tmp = Path(tempfile.mkdtemp(prefix='dl_'))
            try:
                f = await run_blocking(download_file, url, tmp)
                pth, has_idx = await run_blocking(install_model, f, name)
                await status.edit_text(f'✅ Đã cài model `{name}` ({pth}){"" if has_idx else " — ⚠️ không có file .index"}.')
            except Exception as e:
                await status.edit_text(f'❌ {e}')
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            return
        if sub in ('upload', 'addfile'):
            await update.message.reply_text('Gửi file .zip kèm caption `/modeladd <tên>`.')
            return
        if sub == 'remove':
            if len(args) < 2:
                await update.message.reply_text('Dùng: `/model remove <tên>`')
                return
            if not self.is_admin(uid):
                await update.message.reply_text('🚫 Chỉ admin.')
                return
            name = args[1]
            if name not in list_models():
                await update.message.reply_text('Không có model này.')
                return
            shutil.rmtree(BASE_DIR / 'rvc_models' / name, ignore_errors=True)
            await update.message.reply_text(f'🗑️ Đã xoá `{name}`.')
            return
        if sub == 'rename':
            if len(args) < 3:
                await update.message.reply_text('Dùng: `/model rename <cũ> <mới>`')
                return
            if not self.is_admin(uid):
                await update.message.reply_text('🚫 Chỉ admin.')
                return
            old, new = args[1], safe_name(args[2])
            src, dst = BASE_DIR / 'rvc_models' / old, BASE_DIR / 'rvc_models' / new
            if old not in list_models() or not new or dst.exists():
                await update.message.reply_text('Tên cũ không tồn tại hoặc tên mới không hợp lệ / đã tồn tại.')
                return
            src.rename(dst)
            await update.message.reply_text(f'✅ `{old}` → `{new}`')
            return
        await update.message.reply_text('Dùng: `/model list|info|add|remove|rename`')

    async def on_model_upload(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Xử lý file .zip kèm caption `/modeladd <tên>`."""
        caption = (update.message.caption or '').strip()
        if not caption.lower().startswith('/modeladd'):
            return
        uid = update.effective_user.id
        if not self.can_add_models(uid):
            await update.message.reply_text('🚫 Chỉ admin mới được thêm model.')
            return
        name = safe_name(caption.split(None, 1)[1] if len(caption.split(None, 1)) > 1 else '')
        if not name:
            await update.message.reply_text('Dùng: gửi file .zip kèm caption `/modeladd <tên>`.')
            return
        if (BASE_DIR / 'rvc_models' / name).exists():
            await update.message.reply_text(f'Model `{name}` đã tồn tại.')
            return
        status = await update.message.reply_text('⬇️ Đang cài model...')
        tmp = Path(tempfile.mkdtemp(prefix='up_'))
        try:
            dest = tmp / 'model.zip'
            await update.message.document.download_to_drive(custom_path=dest)
            pth, has_idx = await run_blocking(install_model, dest, name)
            await status.edit_text(f'✅ Đã cài model `{name}` ({pth}){"" if has_idx else " — ⚠️ không có file .index"}.')
        except Exception as e:
            await status.edit_text(f'❌ {e}')
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # ---------------- lệnh: gpu / cleanup ----------------
    async def cmd_gpu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        def info():
            lines = []
            try:
                import torch
                if torch.cuda.is_available():
                    i = torch.cuda.current_device()
                    p = torch.cuda.get_device_properties(i)
                    used = torch.cuda.memory_reserved(i) / 1024 ** 3
                    lines.append(f'**GPU:** {p.name} — {p.total_memory / 1024 ** 3:.1f} GB (torch giữ {used:.1f} GB)')
                else:
                    lines.append('**GPU:** ❌ không có CUDA (sẽ rất chậm / lỗi)')
            except Exception as e:
                lines.append(f'**GPU:** lỗi {e}')
            try:
                import onnxruntime as ort
                lines.append(f'**ONNX providers:** {", ".join(ort.get_available_providers())}')
            except Exception:
                pass
            du = shutil.disk_usage(BASE_DIR)
            so = BASE_DIR / 'song_output'
            used = sum(f.stat().st_size for f in so.rglob('*') if f.is_file()) if so.exists() else 0
            lines.append(f'**Ổ đĩa:** trống {fmt_size(du.free)} / {fmt_size(du.total)} • song_output {fmt_size(used)}')
            return '\n'.join(lines)

        text = await run_blocking(info)
        up = int(time.time() - self.started_at)
        text += (f'\n**Hàng đợi:** {len(self.pending)} chờ{" + 1 đang chạy" if self.current else ""}'
                 f' • xong {self.stats["done"]} • lỗi {self.stats["failed"]}')
        text += f'\n**Uptime:** {up // 3600}h{up % 3600 // 60:02d}m • **Models:** {len(list_models())}'
        await update.message.reply_text('🖥️ *Trạng thái hệ thống*\n' + text, parse_mode='Markdown')

    async def cmd_cleanup(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id
        if not self.is_admin(uid):
            await update.message.reply_text('🚫 Chỉ admin.')
            return
        all_files = (context.args and context.args[0].lower() in ('all', 'full')) or False
        if all_files and (self.current or self.pending):
            await update.message.reply_text('Đang có job, hãy đợi xong.')
            return
        freed = await run_blocking(self.cleanup_old_files, 0 if all_files else 1, all_files)
        await update.message.reply_text(f'🧹 Đã giải phóng {fmt_size(freed)}.')


# --------------------------------------------------------------------------------------
def check_environment():
    problems = []
    mdx = BASE_DIR / 'mdxnet_models'
    for f in ('UVR-MDX-NET-Voc_FT.onnx', 'UVR_MDXNET_KARA_2.onnx', 'Reverb_HQ_By_FoxJoy.onnx', 'model_data.json'):
        if not (mdx / f).exists():
            problems.append(f'thiếu mdxnet_models/{f}')
    for f in ('hubert_base.pt', 'rmvpe.pt'):
        if not (BASE_DIR / 'rvc_models' / f).exists():
            problems.append(f'thiếu rvc_models/{f}')
    if problems:
        log.warning('Cần chạy `python src/download_models.py`: %s', '; '.join(problems))
    if not shutil.which('ffmpeg'):
        log.warning('Không tìm thấy ffmpeg — cài bằng apt-get install ffmpeg sox')


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    token = get_token()
    if not token:
        sys.exit('Chưa có TELEGRAM_BOT_TOKEN (biến môi trường, file .env hoặc Kaggle Secret).')
    check_environment()
    bot = TgBot()
    app = bot.build(token)
    app.post_init = bot.post_init
    app.post_shutdown = bot.post_shutdown
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()
