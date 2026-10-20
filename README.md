# AICoverGen — Discord Bot & Telegram Bot (không Web UI)

Bot tạo **AI cover** bằng giọng RVC, dựa trên pipeline AICoverGen
(tách vocal MDX-Net → đổi giọng RVC → thêm reverb → trộn lại).
Đã bỏ hoàn toàn Gradio/Web UI nên không còn mở cổng web (nguyên nhân Kaggle kill session).

Có 2 frontend dùng chung một pipeline (`src/core.py`):

| Frontend | File | Chạy |
|---|---|---|
| 🤖 Discord | `src/bot.py` | `bash run_bot.sh` |
| ✈️ Telegram | `src/telegram_bot.py` | `bash run_telegram_bot.sh` |

## 🤖 Telegram bot (mới)

### Cài đặt

1. Mở [@BotFather](https://t.me/BotFather) → `/newbot` → copy token.
2. Tạo file `.env`:

```bash
TELEGRAM_BOT_TOKEN=123456:AAxxxx
OWNER_IDS=123456789      # Telegram user ID của bạn (toàn quyền)
ADMIN_IDS=               # (tuỳ chọn) user ID được quản lý model
```

> Lấy Telegram user ID của bạn: nhắn `/id` với [@userinfobot](https://t.me/userinfobot).

3. Chạy:

```bash
bash setup_venv.sh        # 1 lần: tạo .venv + cài thư viện + tải model
bash run_telegram_bot.sh
```

Trên Kaggle: thêm Secret tên `TELEGRAM_BOT_TOKEN`, rồi:

```python
%cd /kaggle/working/AICoverGen
!bash setup_venv.sh
import os
from kaggle_secrets import UserSecretsClient
os.environ["TELEGRAM_BOT_TOKEN"] = UserSecretsClient().get_secret("TELEGRAM_BOT_TOKEN")
!bash run_telegram_bot.sh
```

### Lệnh

| Nhóm | Lệnh |
|---|---|
| Tạo cover | `/cover <link hoặc tên bài>` (+ `voice=`, `pitch=`, `index=`, `format=`…) • `/search <từ khoá>` (bấm nút chọn) • `/separate` (tách vocal/nhạc nền) • gửi file kèm caption `/cover` |
| Cá nhân hoá | `/settings show` • `/settings set voice=ten pitch=1 index=0.6` • `/settings reset` |
| Model RVC | `/model list/info/add/remove/rename` • gửi file .zip kèm caption `/modeladd <tên>` |
| Hàng đợi | `/queue` • `/cancel` • `/history` • `/resend` |
| Hệ thống | `/gpu` • `/ping` • `/cleanup` (admin) • `/help` |

Vì Telegram không có voice channel cho bot, **kết quả được gửi thẳng dạng file âm thanh**
(mp3/wav/flac) trong chat. File >50 MB sẽ được nén lại, hoặc gửi link nếu bật `UPLOAD_FALLBACK=1`.

## Tính năng (Discord)

| Nhóm | Lệnh |
|---|---|
| Tạo cover | `/cover` (link YouTube/SoundCloud…, tên bài tự tìm, hoặc đính kèm file) • `/search` (chọn từ kết quả) • `/separate` (chỉ tách vocal/nhạc nền) |
| Cá nhân hoá | `/settings set/show/reset` — lưu voice, pitch, f0, index, protect, âm lượng, reverb, định dạng… cho từng người |
| Model RVC | `/model list/info/add/upload/public/remove/rename` (tải từ link zip, HuggingFace, PixelDrain, Dropbox, Google Drive) |
| Hàng đợi | `/queue` `/cancel` `/history` `/resend` — 1 job GPU/lần, có thanh tiến trình, giới hạn job/người |
| Voice channel | `/voice join/play/pause/skip/stop/volume/queue/leave` + nút 🔊 trên kết quả (tuỳ chọn `play_in_voice`) |
| Hệ thống | `/gpu` `/ping` `/cleanup` `/help` — tự dọn file cũ, tự nén mp3 nếu file vượt giới hạn upload |

Mặc định chỉ admin (Manage Server / `OWNER_IDS` / `ADMIN_ROLE_IDS`) được thêm/xoá model; đặt `ALLOW_USER_MODELS=1` để mọi người thêm được.

## Cài đặt trên Kaggle (GPU T4, Internet bật)

1. Tạo bot ở <https://discord.com/developers/applications> → *Bot* → copy token.
   Mời bot với scope `bot applications.commands`, quyền *Send Messages, Attach Files, Embed Links, Connect, Speak*.
   (Không cần bật Privileged Intents.)
2. Notebook Kaggle → *Add-ons → Secrets* → thêm `DISCORD_TOKEN`.
3. Cell chạy (dùng venv Python 3.9, cài tự động bằng `setup_venv.sh`):

```python
%cd /kaggle/working/AICoverGen
!bash setup_venv.sh                      # 1 lần: tạo .venv + cài thư viện + tải model
import os
from kaggle_secrets import UserSecretsClient
os.environ["DISCORD_TOKEN"] = UserSecretsClient().get_secret("DISCORD_TOKEN")
os.environ["GUILD_ID"] = "ID_SERVER_CUA_BAN"   # lệnh / hiện ngay
os.environ["OWNER_IDS"] = "ID_DISCORD_CUA_BAN"
!bash run_bot.sh                         # tự dùng .venv
```

Các lưu ý về onnxruntime-gpu/CUDA 12, sox… xem `SETUP_NOTES.md`.
Cài model giọng bằng Discord: `/model add url:<link zip> name:<tên>`.

## Chạy máy khác / Ubuntu

```bash
cp .env.example .env   # điền DISCORD_TOKEN, OWNER_IDS, GUILD_ID (và TELEGRAM_BOT_TOKEN nếu dùng Telegram)
bash run_bot.sh        # bot Discord — tự đặt LD_LIBRARY_PATH cho CUDA 12 nếu có .venv, tự restart khi sập
bash run_telegram_bot.sh   # bot Telegram
```

## Ghi chú

* **YouTube chặn IP datacenter** (Kaggle/Colab): lỗi *"Sign in to confirm you're not a bot"*.
  Xuất `cookies.txt` từ trình duyệt đã đăng nhập, đặt `YT_COOKIES_FILE=/path/cookies.txt`.
  Hoặc dùng file đính kèm / link SoundCloud. Luôn để `yt-dlp` bản mới nhất.
* **Voice channel** cần PyNaCl + ffmpeg + libopus; Kaggle có thể chặn UDP nên phát voice là *best-effort*.
* Hủy job đang chạy chỉ dừng ở bước kế tiếp (GPU không ngắt giữa chừng).
* Kết quả vượt giới hạn upload của server sẽ được nén mp3 192→48 kbps; nếu vẫn lớn và `UPLOAD_FALLBACK=1`
  thì up lên litterbox.catbox.moe (dịch vụ bên thứ ba, link sống 72 giờ).
* CLI cũ vẫn dùng được: `python src/main.py -i <link|file> -dir <model> -p 0`.

Dựa trên [AICoverGen](https://github.com/SociallyIneptWeeb/AICoverGen) (MIT).
