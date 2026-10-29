# The GPU Relay website (multi-user)

`gpurelay gateway` runs a website where anyone you let in signs up, connects their own
platform accounts, and relays their own jobs. Nobody sees anyone else's jobs or tokens.

## For users: what you need

| # | What | Needed? | What you do |
|---|---|---|---|
| 1 | GPU Relay account | required | Sign up with your email and a password of 12+ characters (and the invite code, if the site asks). Confirm your email if the site sends one. |
| 1b | Your GPU platforms | asked once, skippable | At first sign-in: which hyperscalers (Google Cloud, AWS, Azure), GPU clouds (RunPod, Vast.ai, Lambda, DigitalOcean, CoreWeave, others) and free platforms (Colab, Kaggle, Modal) you have, with optional credentials for the paid ones (sealed in the vault, never shown again, given to no process until GPU Relay can use that platform). Change it any time: **Set up** or Settings → Your GPU platforms. Answering again never removes what you saved; a platform's credentials go only when you remove them. To let GPU Relay use a paid one, turn it on there (confirm its terms, enter the credit you have, optionally a monthly card limit, $0 by default): it is used after free time, cheapest first, never past the credit. Only then does your GPU Relay get that platform's key. Not yet tried on a real paid account: start small. |
| 2 | Hugging Face account + **write** token | required | huggingface.co → Settings → Access Tokens → New token with write access. Paste it on **Set up**. GPU Relay creates (or reuses) your private dataset repo `gpurelay-store`. |
| 3 | Kaggle account, **verified phone**, API token | for automatic free GPUs | Verify your phone on kaggle.com (no GPUs without it). Settings → API → Create New Token; paste it (or the whole kaggle.json). GPU Relay creates a private dataset `gpurelay-secrets` in your Kaggle account so your sessions can reach your storage. |
| 4 | Modal account, **payment method**, token | optional | Add a payment method on modal.com (Modal starts GPUs only then; $30 free a month). Settings → API Tokens → New Token; paste the id and secret and confirm Modal's terms. Paid time only above your limit ($0 by default). |
| 5 | Colab secret | optional | In Colab: key icon → add `GPURELAY_HF_TOKEN` with your Hugging Face token, allow notebook access. When a job needs Colab you get a notice and a notebook: open it, press Run all. |
| 6 | Choices | defaults fine | Colab first or as a fallback, wait for Run all (30 min), paid limit ($0), Kaggle hours a week. |
| 7 | AI auto-fix | optional | On your own computer: install Claude Code, run `claude` and `/login` with your Claude Pro or Max plan (no API key), install GPU Relay (`pip install "gpurelay @ git+https://github.com/sailokesh3229/GPU-Relay"`), make a helper key on **Set up** (step 6) and run the command it shows. A job that stops with an error in its code then offers **Fix it with AI** and **Apply and rerun**; you can also turn on automatic reruns there (limits: fixes per job, GPU hours). Claude Code gets the code and the error in its prompt and no tools: it cannot open files or run commands. Optional: Codex (`npm install -g @openai/codex`, `codex login` with a ChatGPT plan) runs in its read-only sandbox. |
| 8 | Your job | required | A `.py` or `.ipynb` (+ `requirements.txt`). **Open my GPU Relay** → Submit; the resumability check runs first. |

Moving from the single-owner server: sign up, connect the **same** Hugging Face repo
(`gpurelay-store`) and Kaggle account. Your GPU Relay restores your jobs from the backup in
that repo on its first start.

## For the person running the site

### Settings (environment)

| Variable | Needed | Meaning |
|---|---|---|
| `DATABASE_URL` | yes | Postgres URL (`postgresql://...`; Neon or Supabase free plans). Accounts and sealed tokens live there: the server's disk is temporary. |
| `GPURELAY_SESSION_KEY` | yes | 16+ random characters; signs sign-in cookies. Render generates it. |
| `GPURELAY_VAULT_KEY` | yes | Seals users' tokens (AES-256-GCM). 32 random bytes in base64, or any random value of 32+ characters (Render generates it). **Never change it**: stored tokens become unreadable and users must add them again. |
| `GPURELAY_MAX_USERS` | 3 | Accounts allowed in all. |
| `GPURELAY_MAX_ACTIVE` | 2 | Users' GPU Relay running at once (each about 110-150 MB; the free plan has 512 MB). |
| `GPURELAY_MAX_JOBS` | 3 | Unfinished jobs per user. |
| `GPURELAY_INVITE_CODE` | optional | Sign-up needs this code. |
| `GPURELAY_BREVO_API_KEY`, `GPURELAY_MAIL_FROM` | optional | Email through Brevo's HTTPS API (free: 300 mails a day). **Use this on Render's free plan, which blocks outgoing SMTP.** `GPURELAY_MAIL_FROM` must be a sender verified in Brevo. |
| `GPURELAY_SMTP_HOST`, `_PORT` (587), `_USER`, `_PASSWORD`, `GPURELAY_MAIL_FROM` | optional | Email through an SMTP server instead (paid hosting plans). Either way: confirmation links, password reset, job emails. Without email: no verification, and a forgotten password is reset with `gpurelay reset-link <email> --site <url>` run with the site's `DATABASE_URL`. |
| `GPURELAY_SITE_URL` | with email | The site's address, for links in mails. |
| `GPURELAY_USERS_DIR` | optional | Users' folders on the server (the image sets it). |

The server needs **no** platform tokens of its own. Delete old `HF_TOKEN`,
`KAGGLE_API_TOKEN`, `MODAL_TOKEN_*`, `GPURELAY_OWNER_PASSWORD` from the host's settings.

### How it is built

- **Gateway** (`gateway/app.py`): accounts (`accounts/`), the setup pages, `/app/` forwarding.
- **Vault** (`accounts/vault.py`): tokens are sealed per user and name before they reach the
  database, opened only to start a process for their owner, and never shown or logged.
- **Setup checks** (`gateway/setup_worker.py`): each runs in its own process with only that
  user's tokens; a token is kept only after its platform accepted it.
- **Each user's GPU Relay** (`gateway/user_server.py`, `gateway/children.py`): a separate
  process (web app + controller) per user on localhost behind a per-launch key, with an
  environment built from an allowlist (`gateway/userenv.py`): never the server's database URL
  or keys. It starts on a visit, and at site start for users with jobs in flight; it stops
  when idle, never while a job is unfinished. State is backed up to the user's own repo.
- **AI helper** (`gateway/helper_routes.py`, `aifix/`): a user's helper on their laptop asks `/helper/v1/poll` with their helper key (only its hash is stored) for fix requests, and posts answers to `/helper/v1/fixes/<id>`; both go to that user's GPU Relay, which re-checks each answer. Polls never start a stopped GPU Relay.
- **No code runs on the server**: submitted jobs run only on the user's Kaggle, Modal or
  Colab; the server only reads code for the resumability check.

### Limits of the free plan

Render's free plan sleeps after 15 minutes without visits. While asleep no handoffs happen:
GPU sessions keep running and checkpointing, and the next visit wakes the site, which starts
the GPU Relay of every user with jobs in flight. The disk is wiped on restart; accounts are in
Postgres and jobs in each user's storage repo, so nothing is lost.
