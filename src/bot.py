"""
AICoverGen Discord bot (không có Web UI).

Chạy:  python src/bot.py        (hoặc  bash run_bot.sh)
Token: biến môi trường DISCORD_TOKEN, file .env, hoặc Kaggle Secret tên DISCORD_TOKEN.

Tính năng chính
  /cover      tạo AI cover từ link YouTube/SoundCloud/..., tên bài hát (tự tìm) hoặc file đính kèm
  /search     tìm bài trên YouTube rồi chọn để tạo cover
  /separate   chỉ tách vocal / nhạc nền (stems)
  /settings   lưu thông số mặc định riêng cho từng người dùng
  /model      xem / thêm / tải lên / xoá / đổi tên voice model RVC
  /queue /cancel /history /resend
  /voice      phát cover trong voice channel (join, play, skip, stop, volume, queue, leave)
  /gpu /ping /cleanup /help
"""
import asyncio
import functools
import gc
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import discord
import requests
from discord import app_commands
from discord.ext import commands, tasks

SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from core import (  # noqa: E402
    ADMIN_IDS, ALLOW_USER_MODELS, AUDIO_EXTS, CLEAN_AFTER_HOURS, DATA_DIR, DEFAULT_SETTINGS,
    DEFAULT_UPLOAD_LIMIT, MAX_DURATION_MIN, MAX_JOBS_PER_USER, MAX_QUEUE, OWNER_IDS, SETTING_LABELS,
    SETTING_SPECS, UPLOAD_DIR, UPLOAD_FALLBACK, JsonStore, download_file, env_ids, env_int, fmt_duration,
    fmt_size, install_model, is_url, list_models, load_dotenv, progress_bar, probe_duration, run_blocking,
    safe_name, shrink_audio, stage_vi, to_mp3, upload_litterbox,
)

# --- riêng cho Discord ---
GUILD_IDS = env_ids('GUILD_IDS') | env_ids('GUILD_ID')  # sync lệnh tức thì cho các server này
ADMIN_ROLE_IDS = env_ids('ADMIN_ROLE_IDS')               # role được quản lý model


# --------------------------------------------------------------------------------------
# Cấu hình
# --------------------------------------------------------------------------------------
def get_token():
    token = os.environ.get('DISCORD_TOKEN')
    if token:
        return token
    try:  # Kaggle Secrets
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret('DISCORD_TOKEN')
    except Exception:
        pass
    try:  # Colab Secrets
        from google.colab import userdata
        return userdata.get('DISCORD_TOKEN')
    except Exception:
        return None


log = logging.getLogger('aicover')

# Tiện ích
# --------------------------------------------------------------------------------------
# Job / hàng đợi
# --------------------------------------------------------------------------------------
@dataclass
class Job:
    id: str
    kind: str                      # 'cover' | 'separate'
    user_id: int
    user_name: str
    channel_id: Optional[int]
    guild_id: Optional[int]
    song_input: str                # URL hoặc đường dẫn file local
    title: str
    params: dict = field(default_factory=dict)
    play_voice: bool = False
    voice_channel_id: Optional[int] = None
    keep_files: bool = False
    cleanup_paths: list = field(default_factory=list)
    status: str = 'pending'        # pending/running/done/failed/cancelled
    progress: float = 0.0
    stage: str = ''
    cancelled: bool = False
    created: float = field(default_factory=time.time)
    started: float = 0.0
    thumbnail: Optional[str] = None
    status_msg: Optional[discord.Message] = None


class StatusMessage:
    """Tin nhắn trạng thái: sửa được lâu hơn 15 phút của interaction bằng cách gửi tin mới khi token hết hạn."""

    def __init__(self, bot, channel_id, msg):
        self.bot = bot
        self.channel_id = channel_id
        self.msg = msg

    async def edit(self, **kw):
        try:
            if self.msg:
                await self.msg.edit(**kw)
                return
        except discord.HTTPException:
            pass
        ch = self.bot.get_channel(self.channel_id) if self.channel_id else None
        if ch:
            try:
                self.msg = await ch.send(**kw)
            except discord.HTTPException:
                self.msg = None


class GuildPlayer:
    def __init__(self):
        self.queue = deque()   # (title, path)
        self.current = None
        self.volume = 0.7
        self.last_active = time.time()


# --------------------------------------------------------------------------------------
# Cog chính
# --------------------------------------------------------------------------------------
class AICover(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.settings = JsonStore(DATA_DIR / 'settings.json', dict)
        self.history = JsonStore(DATA_DIR / 'history.json', list)
        self.queue = asyncio.Queue()
        self.pending = []              # list[Job] đang chờ (để hiển thị / huỷ)
        self.current: Optional[Job] = None
        self.jobs = {}                 # id -> Job (đang chờ / đang chạy)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='gpu')
        self.players = {}              # guild_id -> GuildPlayer
        self.worker_task = None
        self.started_at = time.time()
        self.stats = {'done': 0, 'failed': 0}

    # ---------------- vòng đời ----------------
    async def cog_load(self):
        # import trước các thư viện nặng (torch, fairseq) để lệnh đầu tiên không bị chậm / lỗi muộn
        try:
            import importlib
            await asyncio.to_thread(importlib.import_module, 'main')
        except Exception:
            log.exception('Không import được pipeline (main.py) — kiểm tra lại cài đặt phụ thuộc')
        self.worker_task = asyncio.create_task(self.worker())
        self.housekeeping.start()

    async def cog_unload(self):
        self.housekeeping.cancel()
        if self.worker_task:
            self.worker_task.cancel()
        self.executor.shutdown(wait=False)

    # ---------------- quyền ----------------
    def is_admin(self, user) -> bool:
        if user.id in OWNER_IDS:
            return True
        if isinstance(user, discord.Member):
            if user.guild_permissions.administrator or user.guild_permissions.manage_guild:
                return True
            if ADMIN_ROLE_IDS and any(r.id in ADMIN_ROLE_IDS for r in user.roles):
                return True
        return False

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
    @tasks.loop(minutes=1)
    async def housekeeping(self):
        n = len(self.pending) + (1 if self.current else 0)
        try:
            await self.bot.change_presence(activity=discord.Game(f'/cover • {n} job' if n else '/cover • rảnh'))
        except Exception:
            pass
        # tự rời voice khi không hoạt động
        for gid, player in list(self.players.items()):
            guild = self.bot.get_guild(gid)
            vc = guild.voice_client if guild else None
            if vc and not vc.is_playing() and not player.queue and time.time() - player.last_active > 300:
                await vc.disconnect()
        # dọn file cũ mỗi ~30 phút
        if int(time.time() // 60) % 30 == 0:
            await run_blocking(self.cleanup_old_files, CLEAN_AFTER_HOURS)

    @housekeeping.before_loop
    async def _hk_wait(self):
        await self.bot.wait_until_ready()

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
                self.finish_job_bookkeeping(job)
                continue
            self.pending = [j for j in self.pending if j.id != job.id]
            self.current = job
            try:
                await self.run_job(job)
            except Exception:
                log.exception('Worker error')
            finally:
                self.current = None
                self.finish_job_bookkeeping(job)
                await self.refresh_queue_positions()

    def finish_job_bookkeeping(self, job):
        self.jobs.pop(job.id, None)
        self.pending = [j for j in self.pending if j.id != job.id]
        for p in job.cleanup_paths:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass

    async def refresh_queue_positions(self):
        for i, j in enumerate(self.pending, 1):
            if j.status_msg and not j.cancelled:
                await j.status_msg.edit(embed=self.job_embed(j, queue_pos=i))

    def job_embed(self, job: Job, queue_pos=None, error=None):
        title = {'pending': '⏳ Đang chờ trong hàng đợi', 'running': '⚙️ Đang xử lý', 'done': '✅ Hoàn tất',
                 'failed': '❌ Lỗi', 'cancelled': '🛑 Đã huỷ'}[job.status]
        color = {'pending': 0xF1C40F, 'running': 0x3498DB, 'done': 0x2ECC71, 'failed': 0xE74C3C, 'cancelled': 0x95A5A6}[job.status]
        e = discord.Embed(title=title, color=color)
        e.description = f'**{discord.utils.escape_markdown(job.title)[:200]}**'
        if job.kind == 'cover':
            e.add_field(name='Voice', value=f"`{job.params.get('voice')}`", inline=True)
            e.add_field(name='Pitch', value=f"{job.params.get('pitch_change', 0):+d} oct / {job.params.get('pitch_change_all', 0):+d} st", inline=True)
        else:
            e.add_field(name='Chế độ', value='Tách stems', inline=True)
        e.add_field(name='Job', value=f'`{job.id}`', inline=True)
        if job.status == 'pending' and queue_pos:
            e.add_field(name='Vị trí hàng đợi', value=f'#{queue_pos}/{len(self.pending)}', inline=False)
        if job.status == 'running':
            e.add_field(name=job.stage or 'Đang xử lý', value=progress_bar(job.progress), inline=False)
        if error:
            e.add_field(name='Chi tiết', value=f'```{error[:900]}```', inline=False)
        if job.thumbnail:
            e.set_thumbnail(url=job.thumbnail)
        e.set_footer(text=f'Yêu cầu bởi {job.user_name}')
        return e

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
                await asyncio.sleep(4)
                state = (job.stage, round(job.progress, 2))
                if state != last and job.status_msg:
                    last = state
                    await job.status_msg.edit(embed=self.job_embed(job))

        if job.status_msg:
            await job.status_msg.edit(embed=self.job_embed(job))
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
            if 'sign in' in err.lower() or 'bot' in err.lower() and 'youtube' in err.lower():
                err += '\n\n→ YouTube chặn IP của máy chủ. Đặt YT_COOKIES_FILE=cookies.txt (xem README) hoặc dùng file đính kèm.'
            if job.status_msg:
                await job.status_msg.edit(embed=self.job_embed(job, error=err))
            await self.notify(job, content=f'<@{job.user_id}> job `{job.id}` thất bại.')
        finally:
            upd.cancel()
            await run_blocking(self.free_gpu, executor=self.executor)

        if job.status == 'cancelled':
            if job.status_msg:
                await job.status_msg.edit(embed=self.job_embed(job))
            return
        if job.status == 'done':
            elapsed = time.time() - job.started
            if job.status_msg:
                await job.status_msg.edit(embed=self.job_embed(job))
            try:
                if job.kind == 'separate':
                    await self.deliver_stems(job, result, elapsed)
                else:
                    await self.deliver_cover(job, result, elapsed)
            except Exception as e:
                log.exception('Deliver failed')
                await self.notify(job, content=f'<@{job.user_id}> ⚠️ Xong nhưng không gửi được file: `{e}`')

    @staticmethod
    def free_gpu():
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    async def notify(self, job, **kw):
        """Gửi tin vào kênh; nếu không có quyền thì DM người dùng."""
        ch = self.bot.get_channel(job.channel_id) if job.channel_id else None
        try:
            if ch is None:
                raise discord.Forbidden(None, 'no channel')
            return await ch.send(**kw)
        except (discord.Forbidden, discord.HTTPException):
            try:
                user = await self.bot.fetch_user(job.user_id)
                return await user.send(**kw)
            except Exception:
                log.warning('Cannot notify user %s', job.user_id)

    async def prepare_file(self, path, name_hint=None, guild_id=None):
        """Đảm bảo file nhỏ hơn giới hạn upload. Trả về (discord.File | None, link | None, tmp_path | None)."""
        guild = self.bot.get_guild(guild_id) if guild_id else None
        limit = int((guild.filesize_limit if guild else DEFAULT_UPLOAD_LIMIT) * 0.97)
        return await run_blocking(self._prepare_file_sync, Path(path), limit, name_hint)

    @staticmethod
    def _prepare_file_sync(path, limit, name_hint):
        tmp = None
        send_path = path
        if path.stat().st_size > limit:
            tmp = shrink_audio(path, limit)
            if tmp:
                send_path = tmp
            elif UPLOAD_FALLBACK:
                return None, upload_litterbox(path), None
            else:
                raise Exception(f'File {fmt_size(path.stat().st_size)} vượt giới hạn upload ({fmt_size(limit)}) '
                                f'ngay cả khi nén. Bật UPLOAD_FALLBACK=1 để gửi qua link.')
        fname = name_hint or (path.name if send_path == path else path.stem + '.mp3')
        return discord.File(send_path, filename=fname), None, tmp

    async def deliver_cover(self, job, cover_path, elapsed):
        title = job.title
        voice = job.params['voice']
        e = discord.Embed(title='🎤 AI Cover đã sẵn sàng', description=f'**{discord.utils.escape_markdown(title)[:200]}**', color=0x9B59B6)
        e.add_field(name='Voice', value=f'`{voice}`', inline=True)
        e.add_field(name='Pitch', value=f"{job.params['pitch_change']:+d} oct / {job.params['pitch_change_all']:+d} st", inline=True)
        e.add_field(name='Thuật toán', value=job.params['f0_method'], inline=True)
        e.add_field(name='Thời gian xử lý', value=f'{elapsed:.0f}s', inline=True)
        e.set_footer(text=f'Job {job.id} • {job.user_name}')
        if job.thumbnail:
            e.set_thumbnail(url=job.thumbnail)

        file, link, tmp = await self.prepare_file(cover_path, f'{safe_name(title, 80) or "cover"} ({safe_name(voice)}).{Path(cover_path).suffix.lstrip(".")}', job.guild_id)
        try:
            view = ResultView(self, job, cover_path)
            msg = await self.notify(job, content=f'<@{job.user_id}>' + (f'\n{link}' if link else ''), embed=e,
                                    **({'file': file} if file else {}), view=view)
        finally:
            if tmp:
                tmp.unlink(missing_ok=True)

        rec = {'id': job.id, 'user_id': job.user_id, 'title': title, 'voice': voice, 'path': str(cover_path),
               'created': int(time.time()), 'guild_id': job.guild_id}
        self.history.data.append(rec)
        self.history.data = self.history.data[-500:]
        self.history.save()

        if job.play_voice and job.guild_id:
            guild = self.bot.get_guild(job.guild_id)
            member = guild.get_member(job.user_id) if guild else None
            if member and member.voice and member.voice.channel:
                try:
                    await self.enqueue_voice(guild, member.voice.channel, title, cover_path)
                except Exception as ex:
                    await self.notify(job, content=f'⚠️ Không phát được trong voice: {ex}')

    async def deliver_stems(self, job, res, elapsed):
        files, tmps = [], []
        try:
            for key, label in (('vocals', 'Vocals'), ('instrumental', 'Instrumental'),
                               ('main_vocals_dereverb', 'MainVocals_DeReverb'), ('backup_vocals', 'BackupVocals')):
                p = res.get(key)
                if p and os.path.exists(p):
                    mp3 = await run_blocking(to_mp3, p)
                    if mp3:
                        tmps.append(mp3)
                        files.append(discord.File(mp3, filename=f'{safe_name(job.title, 60)}_{label}.mp3'))
            e = discord.Embed(title='🎼 Đã tách stems', description=f'**{discord.utils.escape_markdown(job.title)[:200]}**', color=0x1ABC9C)
            e.set_footer(text=f'Job {job.id} • {elapsed:.0f}s')
            # Discord tối đa 10 file / 25MB tuỳ server: gửi từng file nhỏ lẻ nếu lỗi
            try:
                await self.notify(job, content=f'<@{job.user_id}>', embed=e, files=files)
            except discord.HTTPException:
                await self.notify(job, content=f'<@{job.user_id}> (gửi từng file do vượt giới hạn)', embed=e)
                for f, mp3 in zip(files, tmps):
                    f.reset()
                    await self.notify(job, file=f)
        finally:
            for t in tmps:
                t.unlink(missing_ok=True)

    # ---------------- thêm job ----------------
    def user_active_jobs(self, user_id):
        return sum(1 for j in self.jobs.values() if j.user_id == user_id)

    async def resolve_source(self, inter, song, file):
        """Trả về (song_input, title, duration, thumbnail, cleanup_paths) hoặc raise Exception với thông báo thân thiện."""
        import main as pipeline
        if file is not None:
            ext = Path(file.filename).suffix.lower()
            if ext not in AUDIO_EXTS and not (file.content_type or '').startswith(('audio/', 'video/')):
                raise Exception('File đính kèm phải là âm thanh (mp3, wav, flac, m4a, ogg...).')
            dest = UPLOAD_DIR / f'{uuid.uuid4().hex[:8]}_{safe_name(file.filename, 80) or "audio" + ext}'
            await file.save(dest)
            dur = await run_blocking(probe_duration, dest)
            title = Path(file.filename).stem
            return str(dest), title, dur, None, [dest]
        if not song:
            raise Exception('Hãy nhập link / tên bài hát hoặc đính kèm file âm thanh.')
        info = await run_blocking(pipeline.get_media_info, song.strip())
        return info['url'], info['title'], info['duration'], info.get('thumbnail'), []

    async def submit(self, inter: discord.Interaction, kind, song_input, title, duration, thumb, cleanup, overrides, play_voice=False, keep_files=False):
        """Kiểm tra giới hạn, tạo job, đẩy vào hàng đợi, gửi tin trạng thái. `inter` đã được defer/response."""
        try:
            if MAX_DURATION_MIN and duration and duration > MAX_DURATION_MIN * 60:
                raise Exception(f'Bài dài {fmt_duration(duration)} — giới hạn là {MAX_DURATION_MIN} phút.')
            if len(self.pending) >= MAX_QUEUE:
                raise Exception(f'Hàng đợi đã đầy ({MAX_QUEUE}). Thử lại sau.')
            if self.user_active_jobs(inter.user.id) >= MAX_JOBS_PER_USER:
                raise Exception(f'Bạn đang có {MAX_JOBS_PER_USER} job chưa xong. Hãy đợi hoặc /cancel.')
            params = {}
            if kind == 'cover':
                params = self.get_settings(inter.user.id)
                params.update({k: v for k, v in overrides.items() if v is not None})
                if not params.get('voice'):
                    raise Exception('Bạn chưa chọn voice model. Dùng tham số `voice` hoặc `/settings set voice:...`.')
                if params['voice'] not in list_models():
                    raise Exception(f"Voice model `{params['voice']}` không tồn tại. Xem `/model list`.")
        except Exception:
            for p in cleanup:
                Path(p).unlink(missing_ok=True)
            raise

        job = Job(id=uuid.uuid4().hex[:6], kind=kind, user_id=inter.user.id, user_name=inter.user.display_name,
                  channel_id=inter.channel_id, guild_id=inter.guild_id, song_input=song_input, title=title,
                  params=params, play_voice=play_voice, keep_files=keep_files, cleanup_paths=cleanup, thumbnail=thumb)
        self.jobs[job.id] = job
        self.pending.append(job)
        await self.queue.put(job)
        pos = len(self.pending) + (1 if self.current else 0)
        msg = await inter.followup.send(embed=self.job_embed(job, queue_pos=len(self.pending)), wait=True)
        job.status_msg = StatusMessage(self.bot, inter.channel_id, msg)
        return job

    # ---------------- autocomplete ----------------
    async def voice_autocomplete(self, inter: discord.Interaction, current: str):
        cur = current.lower()
        return [app_commands.Choice(name=m[:100], value=m) for m in list_models() if cur in m.lower()][:25]

    async def public_autocomplete(self, inter: discord.Interaction, current: str):
        cur = current.lower()
        return [app_commands.Choice(name=m['name'][:100], value=m['name'][:100]) for m in self.load_public()
                if cur in m['name'].lower()][:25]

    def load_public(self):
        p = BASE_DIR / 'rvc_models' / 'public_models.json'
        try:
            return json.loads(p.read_text(encoding='utf8')).get('voice_models', [])
        except Exception:
            return []

    # ======================================================================================
    # Lệnh: /cover /search /separate
    # ======================================================================================
    @app_commands.command(name='cover', description='Tạo AI cover từ link YouTube, tên bài hát hoặc file đính kèm')
    @app_commands.describe(
        song='Link (YouTube/SoundCloud/...) hoặc tên bài hát để tự tìm', file='Hoặc đính kèm file âm thanh',
        voice='Voice model RVC (mặc định lấy từ /settings)', pitch='Pitch GIỌNG (octave): +1 nam→nữ, -1 nữ→nam',
        pitch_all='Đổi key toàn bài (semitone, làm giảm chất lượng nhẹ)', f0_method='Thuật toán dò pitch',
        index_rate='0-1: càng cao càng giống giọng gốc của model', main_vol='Âm lượng giọng chính (dB)',
        backup_vol='Âm lượng giọng phụ (dB)', inst_vol='Âm lượng nhạc nền (dB)', output_format='Định dạng file xuất',
        play_in_voice='Tự phát trong voice channel của bạn khi xong', keep_files='Giữ file trung gian (tách lại stems)')
    @app_commands.autocomplete(voice=voice_autocomplete)
    @app_commands.choices(
        f0_method=[app_commands.Choice(name='rmvpe (rõ nhất)', value='rmvpe'),
                   app_commands.Choice(name='mangio-crepe (mượt hơn)', value='mangio-crepe')],
        output_format=[app_commands.Choice(name='mp3 (nhẹ)', value='mp3'), app_commands.Choice(name='wav (chất lượng)', value='wav'),
                       app_commands.Choice(name='flac', value='flac')])
    async def cover(self, inter: discord.Interaction, song: Optional[str] = None, file: Optional[discord.Attachment] = None,
                    voice: Optional[str] = None, pitch: Optional[app_commands.Range[int, -3, 3]] = None,
                    pitch_all: Optional[app_commands.Range[int, -12, 12]] = None, f0_method: Optional[str] = None,
                    index_rate: Optional[app_commands.Range[float, 0, 1]] = None,
                    main_vol: Optional[app_commands.Range[int, -20, 20]] = None,
                    backup_vol: Optional[app_commands.Range[int, -20, 20]] = None,
                    inst_vol: Optional[app_commands.Range[int, -20, 20]] = None,
                    output_format: Optional[str] = None, play_in_voice: bool = False, keep_files: bool = False):
        await inter.response.defer(thinking=True)
        try:
            src, title, dur, thumb, cleanup = await self.resolve_source(inter, song, file)
            overrides = dict(voice=voice, pitch_change=pitch, pitch_change_all=pitch_all, f0_method=f0_method,
                             index_rate=index_rate, main_gain=main_vol, backup_gain=backup_vol, inst_gain=inst_vol,
                             output_format=output_format)
            await self.submit(inter, 'cover', src, title, dur, thumb, cleanup, overrides, play_in_voice, keep_files)
        except Exception as e:
            await inter.followup.send(f'❌ {e}', ephemeral=True)

    @app_commands.command(name='search', description='Tìm bài trên YouTube rồi chọn để tạo AI cover')
    @app_commands.describe(query='Từ khoá tìm kiếm', voice='Voice model (mặc định lấy từ /settings)')
    @app_commands.autocomplete(voice=voice_autocomplete)
    async def search(self, inter: discord.Interaction, query: str, voice: Optional[str] = None):
        import main as pipeline
        await inter.response.defer(ephemeral=True)
        try:
            results = await run_blocking(pipeline.search_youtube, query, 8)
        except Exception as e:
            return await inter.followup.send(f'❌ Tìm kiếm lỗi: {e}', ephemeral=True)
        if not results:
            return await inter.followup.send('Không có kết quả.', ephemeral=True)
        e = discord.Embed(title=f'🔎 Kết quả cho “{query[:80]}”', color=0xE67E22,
                          description='\n'.join(f'**{i}.** [{r["title"][:70]}]({r["url"]}) — `{fmt_duration(r["duration"])}` • {r["uploader"][:25]}'
                                                for i, r in enumerate(results, 1)))
        await inter.followup.send(embed=e, view=SearchView(self, inter.user.id, results, voice), ephemeral=True)

    @app_commands.command(name='separate', description='Chỉ tách giọng hát và nhạc nền (không đổi giọng)')
    @app_commands.describe(song='Link hoặc tên bài hát', file='Hoặc đính kèm file âm thanh')
    async def separate(self, inter: discord.Interaction, song: Optional[str] = None, file: Optional[discord.Attachment] = None):
        await inter.response.defer(thinking=True)
        try:
            src, title, dur, thumb, cleanup = await self.resolve_source(inter, song, file)
            await self.submit(inter, 'separate', src, title, dur, thumb, cleanup, {})
        except Exception as e:
            await inter.followup.send(f'❌ {e}', ephemeral=True)

    # ======================================================================================
    # Lệnh: queue / cancel / history / resend
    # ======================================================================================
    @app_commands.command(name='queue', description='Xem hàng đợi xử lý')
    async def queue_cmd(self, inter: discord.Interaction):
        e = discord.Embed(title='📋 Hàng đợi', color=0x3498DB)
        if self.current:
            c = self.current
            e.add_field(name='▶️ Đang chạy', value=f'`{c.id}` {c.title[:60]} — {c.user_name}\n{c.stage} {progress_bar(c.progress, 12)}', inline=False)
        if self.pending:
            e.add_field(name='⏳ Đang chờ', value='\n'.join(f'**{i}.** `{j.id}` {j.title[:55]} — {j.user_name}' for i, j in enumerate(self.pending[:15], 1)), inline=False)
        if not self.current and not self.pending:
            e.description = 'Hàng đợi trống 🎉'
        await inter.response.send_message(embed=e)

    @app_commands.command(name='cancel', description='Huỷ job của bạn (mặc định: job mới nhất)')
    @app_commands.describe(job_id='ID job (xem trong /queue)')
    async def cancel(self, inter: discord.Interaction, job_id: Optional[str] = None):
        mine = [j for j in self.jobs.values() if j.user_id == inter.user.id or self.is_admin(inter.user)]
        if job_id:
            job = next((j for j in mine if j.id == job_id.strip()), None)
        else:
            mine_own = [j for j in self.jobs.values() if j.user_id == inter.user.id]
            job = mine_own[-1] if mine_own else None
        if not job:
            return await inter.response.send_message('Không tìm thấy job nào của bạn để huỷ.', ephemeral=True)
        job.cancelled = True
        if job.status == 'pending':
            job.status = 'cancelled'
            self.pending = [j for j in self.pending if j.id != job.id]
            self.jobs.pop(job.id, None)
            if job.status_msg:
                await job.status_msg.edit(embed=self.job_embed(job))
            await self.refresh_queue_positions()
            await inter.response.send_message(f'🛑 Đã huỷ job `{job.id}`.', ephemeral=True)
        else:
            await inter.response.send_message(f'🛑 Job `{job.id}` sẽ dừng ở bước kế tiếp (GPU không ngắt giữa chừng được).', ephemeral=True)

    @app_commands.command(name='history', description='Xem các cover bạn đã tạo gần đây')
    async def history_cmd(self, inter: discord.Interaction):
        mine = [h for h in self.history.data if h['user_id'] == inter.user.id][-10:][::-1]
        if not mine:
            return await inter.response.send_message('Bạn chưa tạo cover nào.', ephemeral=True)
        lines = [f'`{h["id"]}` <t:{h["created"]}:R> **{h["title"][:50]}** — `{h["voice"]}`' + ('' if os.path.exists(h['path']) else ' *(đã hết hạn)*') for h in mine]
        await inter.response.send_message(embed=discord.Embed(title='🕘 Lịch sử cover', description='\n'.join(lines), color=0x9B59B6)
                                          .set_footer(text='Dùng /resend <id> để gửi lại'), ephemeral=True)

    @app_commands.command(name='resend', description='Gửi lại một cover cũ')
    @app_commands.describe(job_id='ID trong /history (bỏ trống = cover gần nhất)')
    async def resend(self, inter: discord.Interaction, job_id: Optional[str] = None):
        mine = [h for h in self.history.data if h['user_id'] == inter.user.id]
        rec = next((h for h in mine if h['id'] == (job_id or '').strip()), None) if job_id else (mine[-1] if mine else None)
        if not rec:
            return await inter.response.send_message('Không tìm thấy cover.', ephemeral=True)
        if not os.path.exists(rec['path']):
            return await inter.response.send_message('File đã bị dọn (hết hạn). Hãy tạo lại bằng /cover.', ephemeral=True)
        await inter.response.defer()
        try:
            file, link, tmp = await self.prepare_file(rec['path'], None, inter.guild_id)
            try:
                await inter.followup.send(content=link or f'**{rec["title"]}** — `{rec["voice"]}`', **({'file': file} if file else {}))
            finally:
                if tmp:
                    tmp.unlink(missing_ok=True)
        except Exception as e:
            await inter.followup.send(f'❌ {e}', ephemeral=True)

    # ======================================================================================
    # /settings
    # ======================================================================================
    settings_group = app_commands.Group(name='settings', description='Thông số mặc định cá nhân cho /cover')

    @settings_group.command(name='show', description='Xem cài đặt hiện tại của bạn')
    async def settings_show(self, inter: discord.Interaction):
        s = self.get_settings(inter.user.id)
        e = discord.Embed(title='⚙️ Cài đặt của bạn', color=0x95A5A6,
                          description='\n'.join(f'**{SETTING_LABELS[k]}**: `{s[k] if s[k] is not None else "(chưa đặt)"}`' for k in SETTING_SPECS))
        await inter.response.send_message(embed=e, ephemeral=True)

    @settings_group.command(name='set', description='Đặt thông số mặc định (chỉ các mục bạn nhập sẽ đổi)')
    @app_commands.autocomplete(voice=voice_autocomplete)
    @app_commands.choices(
        f0_method=[app_commands.Choice(name='rmvpe', value='rmvpe'), app_commands.Choice(name='mangio-crepe', value='mangio-crepe')],
        output_format=[app_commands.Choice(name='mp3', value='mp3'), app_commands.Choice(name='wav', value='wav'), app_commands.Choice(name='flac', value='flac')])
    async def settings_set(self, inter: discord.Interaction, voice: Optional[str] = None,
                           pitch: Optional[app_commands.Range[int, -3, 3]] = None,
                           pitch_all: Optional[app_commands.Range[int, -12, 12]] = None, f0_method: Optional[str] = None,
                           crepe_hop: Optional[app_commands.Range[int, 32, 512]] = None,
                           index_rate: Optional[app_commands.Range[float, 0, 1]] = None,
                           filter_radius: Optional[app_commands.Range[int, 0, 7]] = None,
                           rms_mix: Optional[app_commands.Range[float, 0, 1]] = None,
                           protect: Optional[app_commands.Range[float, 0.0, 0.5]] = None,
                           main_vol: Optional[app_commands.Range[int, -20, 20]] = None,
                           backup_vol: Optional[app_commands.Range[int, -20, 20]] = None,
                           inst_vol: Optional[app_commands.Range[int, -20, 20]] = None,
                           reverb_size: Optional[app_commands.Range[float, 0, 1]] = None,
                           reverb_wet: Optional[app_commands.Range[float, 0, 1]] = None,
                           reverb_dry: Optional[app_commands.Range[float, 0, 1]] = None,
                           reverb_damping: Optional[app_commands.Range[float, 0, 1]] = None,
                           output_format: Optional[str] = None):
        if voice is not None and voice not in list_models():
            return await inter.response.send_message(f'❌ Voice `{voice}` không tồn tại.', ephemeral=True)
        changes = dict(voice=voice, pitch_change=pitch, pitch_change_all=pitch_all, f0_method=f0_method,
                       crepe_hop_length=crepe_hop, index_rate=index_rate, filter_radius=filter_radius,
                       rms_mix_rate=rms_mix, protect=protect, main_gain=main_vol, backup_gain=backup_vol,
                       inst_gain=inst_vol, reverb_rm_size=reverb_size, reverb_wet=reverb_wet, reverb_dry=reverb_dry,
                       reverb_damping=reverb_damping, output_format=output_format)
        changed = {k: v for k, v in changes.items() if v is not None}
        if not changed:
            return await inter.response.send_message('Bạn chưa nhập mục nào để đổi.', ephemeral=True)
        self.update_settings(inter.user.id, changed)
        await inter.response.send_message('✅ Đã lưu: ' + ', '.join(f'**{SETTING_LABELS[k]}** = `{v}`' for k, v in changed.items()), ephemeral=True)

    @settings_group.command(name='reset', description='Đặt lại toàn bộ cài đặt về mặc định')
    async def settings_reset(self, inter: discord.Interaction):
        self.settings.data.pop(str(inter.user.id), None)
        self.settings.save()
        await inter.response.send_message('♻️ Đã đặt lại cài đặt về mặc định.', ephemeral=True)

    # ======================================================================================
    # /model
    # ======================================================================================
    model_group = app_commands.Group(name='model', description='Quản lý voice model RVC')

    def can_add_models(self, user):
        return ALLOW_USER_MODELS or self.is_admin(user)

    @model_group.command(name='list', description='Danh sách voice model đang có')
    async def model_list(self, inter: discord.Interaction):
        models = list_models()
        if not models:
            return await inter.response.send_message('Chưa có model nào. Dùng `/model add`.', ephemeral=True)
        pages = [models[i:i + 20] for i in range(0, len(models), 20)]
        embeds = []
        for n, chunk in enumerate(pages, 1):
            embeds.append(discord.Embed(title=f'🎙️ Voice models ({len(models)})', color=0x9B59B6,
                                        description='\n'.join(f'• `{m}`' for m in chunk)).set_footer(text=f'Trang {n}/{len(pages)}'))
        await inter.response.send_message(embed=embeds[0], view=PagerView(inter.user.id, embeds) if len(embeds) > 1 else discord.utils.MISSING, ephemeral=True)

    @model_group.command(name='info', description='Thông tin một voice model')
    @app_commands.autocomplete(name=voice_autocomplete)
    async def model_info(self, inter: discord.Interaction, name: str):
        d = BASE_DIR / 'rvc_models' / name
        if name not in list_models():
            return await inter.response.send_message('Không có model này.', ephemeral=True)
        files = sorted(d.iterdir())
        e = discord.Embed(title=f'🎙️ {name}', color=0x9B59B6)
        for f in files:
            e.add_field(name=f.name[:250], value=fmt_size(f.stat().st_size), inline=False)
        e.set_footer(text='Có file .index' if any(f.suffix == '.index' for f in files) else 'Không có .index (index_rate sẽ không có tác dụng)')
        await inter.response.send_message(embed=e, ephemeral=True)

    @model_group.command(name='add', description='Tải voice model từ link (zip/.pth; HuggingFace, PixelDrain, Drive...)')
    @app_commands.describe(url='Link tải trực tiếp tới .zip chứa .pth (+ .index)', name='Tên model (chữ, số, _ -)')
    async def model_add(self, inter: discord.Interaction, url: str, name: str):
        if not self.can_add_models(inter.user):
            return await inter.response.send_message('🚫 Chỉ admin mới được thêm model.', ephemeral=True)
        name = safe_name(name)
        if not name:
            return await inter.response.send_message('Tên model không hợp lệ.', ephemeral=True)
        if (BASE_DIR / 'rvc_models' / name).exists():
            return await inter.response.send_message(f'Model `{name}` đã tồn tại.', ephemeral=True)
        await inter.response.defer(thinking=True)
        tmp = Path(tempfile.mkdtemp(prefix='dl_'))
        try:
            f = await run_blocking(download_file, url, tmp)
            pth, has_idx = await run_blocking(install_model, f, name)
            await inter.followup.send(f'✅ Đã cài model `{name}` ({pth}){"" if has_idx else " — ⚠️ không có file .index"}.')
        except Exception as e:
            await inter.followup.send(f'❌ {e}')
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @model_group.command(name='upload', description='Tải model lên trực tiếp bằng file zip đính kèm')
    async def model_upload(self, inter: discord.Interaction, file: discord.Attachment, name: str):
        if not self.can_add_models(inter.user):
            return await inter.response.send_message('🚫 Chỉ admin mới được thêm model.', ephemeral=True)
        name = safe_name(name)
        if not name:
            return await inter.response.send_message('Tên model không hợp lệ.', ephemeral=True)
        await inter.response.defer(thinking=True)
        tmp = Path(tempfile.mkdtemp(prefix='up_'))
        try:
            dest = tmp / (safe_name(file.filename, 100) or 'model.zip')
            await file.save(dest)
            pth, has_idx = await run_blocking(install_model, dest, name)
            await inter.followup.send(f'✅ Đã cài model `{name}` ({pth}){"" if has_idx else " — ⚠️ không có file .index"}.')
        except Exception as e:
            await inter.followup.send(f'❌ {e}')
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @model_group.command(name='public', description='Cài model từ danh sách public_models.json (nếu có)')
    @app_commands.autocomplete(name=public_autocomplete)
    async def model_public(self, inter: discord.Interaction, name: str):
        if not self.can_add_models(inter.user):
            return await inter.response.send_message('🚫 Chỉ admin mới được thêm model.', ephemeral=True)
        m = next((x for x in self.load_public() if x['name'][:100] == name), None)
        if not m:
            return await inter.response.send_message('Không tìm thấy trong danh sách public (cần rvc_models/public_models.json).', ephemeral=True)
        await self.model_add.callback(self, inter, m['url'], m['name'])

    @model_group.command(name='remove', description='Xoá một voice model (admin)')
    @app_commands.autocomplete(name=voice_autocomplete)
    async def model_remove(self, inter: discord.Interaction, name: str):
        if not self.is_admin(inter.user):
            return await inter.response.send_message('🚫 Chỉ admin.', ephemeral=True)
        if name not in list_models():
            return await inter.response.send_message('Không có model này.', ephemeral=True)
        await inter.response.send_message(f'Xoá vĩnh viễn model `{name}`?', view=ConfirmView(inter.user.id, lambda: shutil.rmtree(BASE_DIR / 'rvc_models' / name), f'🗑️ Đã xoá `{name}`.'), ephemeral=True)

    @model_group.command(name='rename', description='Đổi tên voice model (admin)')
    @app_commands.autocomplete(name=voice_autocomplete)
    async def model_rename(self, inter: discord.Interaction, name: str, new_name: str):
        if not self.is_admin(inter.user):
            return await inter.response.send_message('🚫 Chỉ admin.', ephemeral=True)
        new_name = safe_name(new_name)
        src, dst = BASE_DIR / 'rvc_models' / name, BASE_DIR / 'rvc_models' / new_name
        if name not in list_models() or not new_name or dst.exists():
            return await inter.response.send_message('Tên cũ không tồn tại hoặc tên mới không hợp lệ / đã tồn tại.', ephemeral=True)
        src.rename(dst)
        await inter.response.send_message(f'✅ `{name}` → `{new_name}`', ephemeral=True)

    # ======================================================================================
    # /voice  (phát cover trong voice channel)
    # ======================================================================================
    voice_group = app_commands.Group(name='voice', description='Phát cover trong voice channel', guild_only=True)

    def player(self, guild_id):
        return self.players.setdefault(guild_id, GuildPlayer())

    @staticmethod
    def voice_supported():
        try:
            from discord import voice_client
            return bool(getattr(voice_client, 'has_nacl', False)) and shutil.which('ffmpeg') is not None
        except Exception:
            return False

    async def ensure_voice(self, guild: discord.Guild, channel: discord.VoiceChannel):
        if not self.voice_supported():
            raise Exception('Máy chủ chưa cài PyNaCl/ffmpeg nên không dùng được voice (`pip install PyNaCl`).')
        vc = guild.voice_client
        if vc and vc.is_connected():
            if vc.channel.id != channel.id:
                await vc.move_to(channel)
            return vc
        try:
            if not discord.opus.is_loaded():
                for name in ('libopus.so.0', 'libopus.so', 'opus'):
                    try:
                        discord.opus.load_opus(name)
                        break
                    except Exception:
                        continue
            return await channel.connect(timeout=30, self_deaf=True)
        except asyncio.TimeoutError:
            raise Exception('Kết nối voice bị timeout (mạng của máy chủ có thể chặn UDP — Kaggle thường như vậy).')

    async def enqueue_voice(self, guild, channel, title, path):
        vc = await self.ensure_voice(guild, channel)
        pl = self.player(guild.id)
        pl.queue.append((title, str(path)))
        pl.last_active = time.time()
        if not vc.is_playing() and not vc.is_paused():
            self.play_next(guild)

    def play_next(self, guild):
        vc = guild.voice_client
        pl = self.player(guild.id)
        if not vc or not vc.is_connected():
            pl.queue.clear()
            pl.current = None
            return
        while pl.queue:
            title, path = pl.queue.popleft()
            if not os.path.exists(path):
                continue
            pl.current = title
            src = discord.PCMVolumeTransformer(discord.FFmpegPCMAudio(path), volume=pl.volume)

            def after(err, g=guild):
                if err:
                    log.warning('voice error: %s', err)
                self.player(g.id).last_active = time.time()
                self.bot.loop.call_soon_threadsafe(self.play_next, g)

            vc.play(src, after=after)
            return
        pl.current = None
        pl.last_active = time.time()

    async def _need_user_voice(self, inter):
        if not isinstance(inter.user, discord.Member) or not inter.user.voice or not inter.user.voice.channel:
            await inter.response.send_message('Bạn cần vào một voice channel trước.', ephemeral=True)
            return None
        return inter.user.voice.channel

    @voice_group.command(name='join', description='Cho bot vào voice channel của bạn')
    async def voice_join(self, inter: discord.Interaction):
        ch = await self._need_user_voice(inter)
        if not ch:
            return
        await inter.response.defer(ephemeral=True)
        try:
            await self.ensure_voice(inter.guild, ch)
            await inter.followup.send(f'🔊 Đã vào {ch.mention}.', ephemeral=True)
        except Exception as e:
            await inter.followup.send(f'❌ {e}', ephemeral=True)

    @voice_group.command(name='play', description='Phát một cover của bạn trong voice (mặc định: cover mới nhất)')
    @app_commands.describe(job_id='ID trong /history', file='Hoặc đính kèm file âm thanh bất kỳ để phát')
    async def voice_play(self, inter: discord.Interaction, job_id: Optional[str] = None, file: Optional[discord.Attachment] = None):
        ch = await self._need_user_voice(inter)
        if not ch:
            return
        await inter.response.defer(ephemeral=True)
        try:
            if file:
                dest = UPLOAD_DIR / f'play_{uuid.uuid4().hex[:6]}_{safe_name(file.filename, 60)}'
                await file.save(dest)
                title, path = file.filename, dest
            else:
                mine = [h for h in self.history.data if h['user_id'] == inter.user.id]
                rec = next((h for h in mine if h['id'] == (job_id or '').strip()), None) if job_id else (mine[-1] if mine else None)
                if not rec or not os.path.exists(rec['path']):
                    raise Exception('Không tìm thấy cover (chưa tạo hoặc đã hết hạn).')
                title, path = rec['title'], rec['path']
            await self.enqueue_voice(inter.guild, ch, title, path)
            await inter.followup.send(f'🎶 Đã thêm vào hàng phát: **{title[:80]}**', ephemeral=True)
        except Exception as e:
            await inter.followup.send(f'❌ {e}', ephemeral=True)

    @voice_group.command(name='skip', description='Bỏ qua bài đang phát')
    async def voice_skip(self, inter: discord.Interaction):
        vc = inter.guild.voice_client
        if vc and (vc.is_playing() or vc.is_paused()):
            vc.stop()
            await inter.response.send_message('⏭️ Đã bỏ qua.')
        else:
            await inter.response.send_message('Không có gì đang phát.', ephemeral=True)

    @voice_group.command(name='pause', description='Tạm dừng / tiếp tục')
    async def voice_pause(self, inter: discord.Interaction):
        vc = inter.guild.voice_client
        if vc and vc.is_playing():
            vc.pause()
            await inter.response.send_message('⏸️ Đã tạm dừng.')
        elif vc and vc.is_paused():
            vc.resume()
            await inter.response.send_message('▶️ Tiếp tục.')
        else:
            await inter.response.send_message('Không có gì đang phát.', ephemeral=True)

    @voice_group.command(name='stop', description='Dừng phát và xoá hàng phát')
    async def voice_stop(self, inter: discord.Interaction):
        self.player(inter.guild_id).queue.clear()
        vc = inter.guild.voice_client
        if vc:
            vc.stop()
        await inter.response.send_message('⏹️ Đã dừng.')

    @voice_group.command(name='volume', description='Âm lượng phát (0-100)')
    async def voice_volume(self, inter: discord.Interaction, percent: app_commands.Range[int, 0, 100]):
        pl = self.player(inter.guild_id)
        pl.volume = percent / 100
        vc = inter.guild.voice_client
        if vc and vc.source and isinstance(vc.source, discord.PCMVolumeTransformer):
            vc.source.volume = pl.volume
        await inter.response.send_message(f'🔉 Âm lượng: {percent}%')

    @voice_group.command(name='queue', description='Xem hàng phát')
    async def voice_queue(self, inter: discord.Interaction):
        pl = self.player(inter.guild_id)
        lines = [f'▶️ **{pl.current}**'] if pl.current else []
        lines += [f'{i}. {t[:70]}' for i, (t, _) in enumerate(pl.queue, 1)]
        await inter.response.send_message('\n'.join(lines) or 'Hàng phát trống.', ephemeral=True)

    @voice_group.command(name='leave', description='Cho bot rời voice channel')
    async def voice_leave(self, inter: discord.Interaction):
        vc = inter.guild.voice_client
        if vc:
            self.player(inter.guild_id).queue.clear()
            await vc.disconnect()
            await inter.response.send_message('👋 Đã rời voice.')
        else:
            await inter.response.send_message('Bot không ở trong voice.', ephemeral=True)

    # ======================================================================================
    # Tiện ích: /gpu /ping /cleanup /help
    # ======================================================================================
    @app_commands.command(name='gpu', description='Thông tin GPU, ổ đĩa và hàng đợi')
    async def gpu(self, inter: discord.Interaction):
        await inter.response.defer()

        def info():
            lines = []
            try:
                import torch
                if torch.cuda.is_available():
                    i = torch.cuda.current_device()
                    p = torch.cuda.get_device_properties(i)
                    used = torch.cuda.memory_reserved(i) / 1024 ** 3
                    lines.append(f'**GPU:** {p.name} — {p.total_memory / 1024 ** 3:.1f} GB (torch đang giữ {used:.1f} GB)')
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
        text += f'\n**Hàng đợi:** {len(self.pending)} chờ{" + 1 đang chạy" if self.current else ""} • xong {self.stats["done"]} • lỗi {self.stats["failed"]}'
        text += f'\n**Uptime:** {up // 3600}h{up % 3600 // 60:02d}m • **Models:** {len(list_models())}'
        await inter.followup.send(embed=discord.Embed(title='🖥️ Trạng thái hệ thống', description=text, color=0x2ECC71))

    @app_commands.command(name='ping', description='Kiểm tra độ trễ của bot')
    async def ping(self, inter: discord.Interaction):
        await inter.response.send_message(f'🏓 {self.bot.latency * 1000:.0f} ms')

    @app_commands.command(name='cleanup', description='Xoá file tạm / cache trong song_output (admin)')
    @app_commands.describe(all_files='Xoá toàn bộ, kể cả cache stems mới (không chạy khi đang có job)')
    async def cleanup(self, inter: discord.Interaction, all_files: bool = False):
        if not self.is_admin(inter.user):
            return await inter.response.send_message('🚫 Chỉ admin.', ephemeral=True)
        if all_files and (self.current or self.pending):
            return await inter.response.send_message('Đang có job, hãy đợi xong.', ephemeral=True)
        await inter.response.defer(ephemeral=True)
        freed = await run_blocking(self.cleanup_old_files, 0 if all_files else 1, all_files)
        await inter.followup.send(f'🧹 Đã giải phóng {fmt_size(freed)}.', ephemeral=True)

    @app_commands.command(name='help', description='Hướng dẫn sử dụng bot')
    async def help_cmd(self, inter: discord.Interaction):
        e = discord.Embed(title='🎤 AICoverGen Bot', color=0x9B59B6,
                          description='Biến bất kỳ bài hát nào thành bản cover bằng giọng của voice model RVC.')
        e.add_field(name='Tạo cover', value='`/cover song:<link|tên bài>` hoặc đính kèm `file`\n`/search` tìm trên YouTube rồi chọn\n`/separate` chỉ tách vocal/nhạc nền', inline=False)
        e.add_field(name='Cá nhân hoá', value='`/settings set` lưu voice, pitch, reverb... mặc định\n`/settings show` • `/settings reset`', inline=False)
        e.add_field(name='Model', value='`/model list` • `/model info`\n`/model add url name` • `/model upload` • `/model public`\n`/model remove` • `/model rename` (admin)', inline=False)
        e.add_field(name='Hàng đợi', value='`/queue` • `/cancel` • `/history` • `/resend`', inline=False)
        e.add_field(name='Voice channel', value='`/voice join|play|pause|skip|stop|volume|queue|leave`', inline=False)
        e.add_field(name='Khác', value='`/gpu` • `/ping` • `/cleanup` (admin)', inline=False)
        e.set_footer(text=f'Giới hạn: {MAX_DURATION_MIN or "∞"} phút/bài • {MAX_JOBS_PER_USER} job/người')
        await inter.response.send_message(embed=e)


# --------------------------------------------------------------------------------------
# Views
# --------------------------------------------------------------------------------------
class ResultView(discord.ui.View):
    def __init__(self, cog: AICover, job: Job, path):
        super().__init__(timeout=3600)
        self.cog, self.job, self.path = cog, job, str(path)

    @discord.ui.button(label='Phát trong voice', emoji='🔊', style=discord.ButtonStyle.primary)
    async def play(self, inter: discord.Interaction, button: discord.ui.Button):
        if not inter.guild:
            return await inter.response.send_message('Chỉ dùng trong server.', ephemeral=True)
        if not isinstance(inter.user, discord.Member) or not inter.user.voice or not inter.user.voice.channel:
            return await inter.response.send_message('Bạn cần vào một voice channel trước.', ephemeral=True)
        if not os.path.exists(self.path):
            return await inter.response.send_message('File đã hết hạn.', ephemeral=True)
        await inter.response.defer(ephemeral=True)
        try:
            await self.cog.enqueue_voice(inter.guild, inter.user.voice.channel, self.job.title, self.path)
            await inter.followup.send('🎶 Đã thêm vào hàng phát.', ephemeral=True)
        except Exception as e:
            await inter.followup.send(f'❌ {e}', ephemeral=True)

    @discord.ui.button(label='Xoá tin nhắn', emoji='🗑️', style=discord.ButtonStyle.secondary)
    async def delete(self, inter: discord.Interaction, button: discord.ui.Button):
        if inter.user.id != self.job.user_id and not self.cog.is_admin(inter.user):
            return await inter.response.send_message('Chỉ người tạo mới xoá được.', ephemeral=True)
        await inter.message.delete()


class SearchView(discord.ui.View):
    def __init__(self, cog: AICover, user_id, results, voice):
        super().__init__(timeout=180)
        self.cog, self.user_id, self.results, self.voice = cog, user_id, results, voice
        select = discord.ui.Select(placeholder='Chọn bài để tạo cover...', options=[
            discord.SelectOption(label=f'{i}. {r["title"]}'[:100], description=f'{fmt_duration(r["duration"])} • {r["uploader"]}'[:100], value=str(i - 1))
            for i, r in enumerate(results[:25], 1)])
        select.callback = self.on_select
        self.add_item(select)

    async def on_select(self, inter: discord.Interaction):
        if inter.user.id != self.user_id:
            return await inter.response.send_message('Không phải lựa chọn của bạn.', ephemeral=True)
        r = self.results[int(inter.data['values'][0])]
        await inter.response.edit_message(content=f'Đã chọn: **{r["title"][:80]}**', embed=None, view=None)
        try:
            await self.cog.submit(inter, 'cover', r['url'], r['title'], r['duration'], None, [], dict(voice=self.voice))
        except Exception as e:
            await inter.followup.send(f'❌ {e}', ephemeral=True)


class PagerView(discord.ui.View):
    def __init__(self, user_id, embeds):
        super().__init__(timeout=120)
        self.user_id, self.embeds, self.i = user_id, embeds, 0

    async def _go(self, inter, d):
        if inter.user.id != self.user_id:
            return await inter.response.send_message('Không phải của bạn.', ephemeral=True)
        self.i = (self.i + d) % len(self.embeds)
        await inter.response.edit_message(embed=self.embeds[self.i])

    @discord.ui.button(label='◀', style=discord.ButtonStyle.secondary)
    async def prev(self, inter, button):
        await self._go(inter, -1)

    @discord.ui.button(label='▶', style=discord.ButtonStyle.secondary)
    async def next(self, inter, button):
        await self._go(inter, 1)


class ConfirmView(discord.ui.View):
    def __init__(self, user_id, action, done_text):
        super().__init__(timeout=60)
        self.user_id, self.action, self.done_text = user_id, action, done_text

    @discord.ui.button(label='Xác nhận', style=discord.ButtonStyle.danger)
    async def yes(self, inter: discord.Interaction, button):
        if inter.user.id != self.user_id:
            return await inter.response.send_message('Không phải của bạn.', ephemeral=True)
        try:
            await run_blocking(self.action)
            await inter.response.edit_message(content=self.done_text, view=None)
        except Exception as e:
            await inter.response.edit_message(content=f'❌ {e}', view=None)

    @discord.ui.button(label='Huỷ', style=discord.ButtonStyle.secondary)
    async def no(self, inter: discord.Interaction, button):
        await inter.response.edit_message(content='Đã huỷ.', view=None)


# --------------------------------------------------------------------------------------
# Bot
# --------------------------------------------------------------------------------------
class CoverBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        super().__init__(command_prefix=commands.when_mentioned, intents=intents, help_command=None)

    async def setup_hook(self):
        await self.add_cog(AICover(self))
        self.tree.on_error = self.on_tree_error
        if GUILD_IDS:
            for gid in GUILD_IDS:
                g = discord.Object(id=gid)
                self.tree.copy_global_to(guild=g)
                await self.tree.sync(guild=g)
            log.info('Đã sync lệnh cho %d server', len(GUILD_IDS))
        else:
            await self.tree.sync()
            log.info('Đã sync lệnh global (có thể mất tới ~1 giờ để hiện; đặt GUILD_ID để hiện ngay)')

    async def on_ready(self):
        log.info('Đăng nhập: %s (%s) • %d server', self.user, self.user.id, len(self.guilds))
        models = list_models()
        log.info('Voice models: %s', ', '.join(models) or '(chưa có)')

    async def on_tree_error(self, inter: discord.Interaction, error: app_commands.AppCommandError):
        log.exception('Command error', exc_info=error)
        msg = f'❌ Lỗi: {getattr(error, "original", error)}'
        if isinstance(error, app_commands.CommandOnCooldown):
            msg = f'⏱️ Chậm lại, thử lại sau {error.retry_after:.0f}s.'
        try:
            if inter.response.is_done():
                await inter.followup.send(msg, ephemeral=True)
            else:
                await inter.response.send_message(msg, ephemeral=True)
        except discord.HTTPException:
            pass


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
        sys.exit('Chưa có DISCORD_TOKEN (biến môi trường, file .env hoặc Kaggle Secret).')
    check_environment()
    CoverBot().run(token, log_handler=None)


if __name__ == '__main__':
    main()
