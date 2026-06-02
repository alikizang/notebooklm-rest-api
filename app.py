# Google NotebookLM REST API wrapper
# Namhyeon Go <gnh1201@catswords.re.kr>
# https://github.com/gnh1201/notebooklm-rest-api
import os
import time
import uuid
import tempfile
import json
import asyncio
import subprocess
import shutil
import logging
from typing import Any, Optional, Literal, Dict, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
import httpx
from PIL import Image, ImageDraw, ImageFont
import edge_tts
import boto3
from botocore.config import Config as BotoConfig

# Google APIs for YouTube and Drive
from google.oauth2.credentials import Credentials as GoogleCredentials
from googleapiclient.discovery import build as google_build
from googleapiclient.http import MediaFileUpload

from notebooklm import NotebookLMClient, RPCError, VideoFormat, VideoStyle  # notebooklm-py



# ----------------------------
# Config / Security
# ----------------------------
API_KEY = os.environ.get("NOTEBOOKLM_REST_API_KEY", "")  # set this in production
AUTH_STORAGE_PATH = os.environ.get("NOTEBOOKLM_STORAGE_PATH")  # optional override


def require_api_key(x_api_key: Optional[str] = None):
    # Minimal API-key gate. Put this behind a real gateway (Cloudflare, Nginx, etc.) for production.
    if API_KEY:
        # FastAPI header parsing without extra imports (keep simple):
        # Prefer: from fastapi import Header; def require_api_key(x_api_key: str = Header(None)) ...
        # but we keep it minimal and rely on query param fallback too.
        if x_api_key != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")


async def get_client() -> NotebookLMClient:
    """
    Creates a client using notebooklm-py's supported auth precedence:
    - explicit path to from_storage()
    - NOTEBOOKLM_AUTH_JSON
    - NOTEBOOKLM_HOME/storage_state.json
    - ~/.notebooklm/storage_state.json
    :contentReference[oaicite:3]{index=3}
    """
    try:
        if AUTH_STORAGE_PATH:
            return await NotebookLMClient.from_storage(AUTH_STORAGE_PATH)
        return await NotebookLMClient.from_storage()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize NotebookLM client: {e}")


def map_rpc_error(e: RPCError) -> HTTPException:
    # notebooklm-py raises RPCError for API failures :contentReference[oaicite:4]{index=4}
    msg = str(e)
    if "401" in msg or "403" in msg or "auth" in msg.lower():
        return HTTPException(status_code=401, detail=msg)
    if "rate" in msg.lower() or "429" in msg:
        return HTTPException(status_code=429, detail=msg)
    return HTTPException(status_code=502, detail=msg)


# ----------------------------
# Models
# ----------------------------
class AccountAddReq(BaseModel):
    cookies_json: str
    label: Optional[str] = None
    skip_validation: bool = False


class NotebookCreateReq(BaseModel):
    title: str


class NotebookRenameReq(BaseModel):
    new_title: str


class SourceAddUrlReq(BaseModel):
    url: str
    wait: bool = True


class SourceAddTextReq(BaseModel):
    title: str
    content: str


class SourceAddYoutubeReq(BaseModel):
    url: str
    wait: bool = True


class ChatAskReq(BaseModel):
    question: str
    # optional persona fields could be added if you want


class ArtifactGenerateReq(BaseModel):
    # A simple unified generator:
    # audio/video/report/quiz/flashcards/slide_deck/infographic/data_table/mind_map
    type: Literal[
        "audio",
        "video",
        "report",
        "quiz",
        "flashcards",
        "slide_deck",
        "infographic",
        "data_table",
        "mind_map",
    ]
    # Options are passed through as-is to the underlying generate_* calls where applicable.
    # (The library supports many per-type options; keep this generic.)
    options: Dict[str, Any] = {}


class TaskPollResp(BaseModel):
    ok: bool
    status: Any


# ----------------------------
# App
# ----------------------------
app = FastAPI(title="NotebookLM REST API (powered by notebooklm-py)")


@app.get("/health")
async def health():
    return {"ok": True}


# ----------------------------
# Google Accounts (NotebookLM Auth)
# ----------------------------
ACCOUNTS_DIR = os.path.expanduser("~/.notebooklm/accounts")

def get_account_paths(account_id: str):
    os.makedirs(ACCOUNTS_DIR, exist_ok=True)
    cookie_path = os.path.join(ACCOUNTS_DIR, f"account_{account_id}.json")
    meta_path = os.path.join(ACCOUNTS_DIR, f"account_{account_id}.meta.json")
    return cookie_path, meta_path

async def get_client_rotation(job_id: str) -> Optional[NotebookLMClient]:
    """
    Tries to find a working Google Account from our local storage pool.
    Logs progress/errors to jobs_db[job_id]['logs'].
    """
    os.makedirs(ACCOUNTS_DIR, exist_ok=True)
    metas = []
    for f in os.listdir(ACCOUNTS_DIR):
        if f.endswith(".meta.json"):
            try:
                with open(os.path.join(ACCOUNTS_DIR, f), "r", encoding="utf-8") as file:
                    metas.append(json.load(file))
            except Exception:
                pass
                
    if not metas:
        jobs_db[job_id]["logs"].append("No custom Google Accounts connected. Attempting default server credentials...")
        try:
            return await get_client()
        except Exception as e:
            jobs_db[job_id]["logs"].append(f"Default server credentials failed: {e}")
            return None
            
    metas.sort(key=lambda x: x.get("created_at", 0))
    
    for meta in metas:
        acc_id = meta.get("id")
        label = meta.get("label", acc_id)
        cookie_path = os.path.join(ACCOUNTS_DIR, f"account_{acc_id}.json")
        meta_path = os.path.join(ACCOUNTS_DIR, f"account_{acc_id}.meta.json")
        
        if not os.path.exists(cookie_path):
            continue
            
        jobs_db[job_id]["logs"].append(f"Attempting to compile with Google Account: '{label}'...")
        try:
            client = await NotebookLMClient.from_storage(cookie_path)
            # Verify connectivity
            async with client:
                await client.notebooks.list()
                
            jobs_db[job_id]["logs"].append(f"Using Google Account: '{label}' (Connection verified).")
            meta["status"] = "active"
            try:
                with open(meta_path, "w", encoding="utf-8") as file:
                    json.dump(meta, file)
            except Exception:
                pass
            return client
        except Exception as err:
            jobs_db[job_id]["logs"].append(f"Google Account '{label}' failed validation/rate limit: {err}. Rotating...")
            meta["status"] = "rate_limited" if "rate" in str(err).lower() or "429" in str(err) else "expired"
            try:
                with open(meta_path, "w", encoding="utf-8") as file:
                    json.dump(meta, file)
            except Exception:
                pass
                
    jobs_db[job_id]["logs"].append("All custom Google Accounts exhausted or expired.")
    return None

@app.get("/v1/auth/accounts")
async def list_accounts():
    os.makedirs(ACCOUNTS_DIR, exist_ok=True)
    accounts = []
    for f in os.listdir(ACCOUNTS_DIR):
        if f.endswith(".meta.json"):
            meta_path = os.path.join(ACCOUNTS_DIR, f)
            try:
                with open(meta_path, "r", encoding="utf-8") as file:
                    meta = json.load(file)
                accounts.append(meta)
            except Exception:
                pass
    accounts.sort(key=lambda x: x.get("created_at", 0))
    return {"ok": True, "accounts": accounts}

def _preprocess_cookie_editor_json(cookies_data: dict) -> dict:
    """Normalize Cookie-Editor export format for notebooklm-py compatibility.
    
    Cookie-Editor uses 'expirationDate' instead of 'expires', and 'sameSite'
    values like 'no_restriction' instead of 'None'. This normalizes the format
    to match the Playwright storage_state.json schema that notebooklm-py expects.
    """
    if "cookies" not in cookies_data:
        return cookies_data
    
    normalized_cookies = []
    for cookie in cookies_data.get("cookies", []):
        c = dict(cookie)  # shallow copy
        
        # Normalize expirationDate -> expires (Cookie-Editor format)
        if "expirationDate" in c and "expires" not in c:
            c["expires"] = c.pop("expirationDate")
        
        # Normalize sameSite values (Cookie-Editor uses 'no_restriction', 'lax', 'strict')
        same_site = c.get("sameSite", "")
        if isinstance(same_site, str):
            same_site_lower = same_site.lower()
            if same_site_lower in ("no_restriction", "unspecified", "none"):
                c["sameSite"] = "None"
            elif same_site_lower == "lax":
                c["sameSite"] = "Lax"
            elif same_site_lower == "strict":
                c["sameSite"] = "Strict"
        
        # Remove Cookie-Editor specific fields that Playwright doesn't use
        for extra_key in ["hostOnly", "session", "storeId", "id"]:
            c.pop(extra_key, None)
        
        # Ensure domain has leading dot for cross-subdomain cookies
        domain = c.get("domain", "")
        name = c.get("name", "")
        
        # SID-family cookies should always be scoped to .google.com
        google_wide_cookies = {
            "SID", "HSID", "SSID", "APISID", "SAPISID",
            "__Secure-1PSID", "__Secure-3PSID",
            "__Secure-1PAPISID", "__Secure-3PAPISID",
            "__Secure-1PSIDTS", "__Secure-3PSIDTS",
            "LSID"
        }
        if name in google_wide_cookies and domain and "google.com" in domain:
            if domain not in (".google.com", "google.com"):
                c["domain"] = ".google.com"
        
        normalized_cookies.append(c)
    
    cookies_data["cookies"] = normalized_cookies
    return cookies_data


def _validate_cookies_offline(cookies_data: dict) -> tuple[bool, str]:
    """Check that the minimum required cookies are present without making network requests.
    
    Returns (is_valid, error_message). If is_valid is True, error_message is empty.
    """
    cookie_names = set()
    for cookie in cookies_data.get("cookies", []):
        name = cookie.get("name", "")
        if name:
            cookie_names.add(name)
    
    required = {"SID", "__Secure-1PSIDTS"}
    missing = required - cookie_names
    
    if missing:
        found_list = sorted(cookie_names)[:8]
        return False, (
            f"Missing required cookies: {missing}. "
            f"Found: {found_list}{'...' if len(cookie_names) > 8 else ''}. "
            f"Make sure you export ALL cookies from notebooklm.google.com "
            f"(the SID cookie is set on .google.com domain and should be visible)."
        )
    
    # Check for secondary binding (OSID or APISID+SAPISID)
    has_osid = "OSID" in cookie_names
    has_api_pair = {"APISID", "SAPISID"} <= cookie_names
    if not has_osid and not has_api_pair:
        return True, ""  # Warn but don't block - it may still work
    
    return True, ""


@app.post("/v1/auth/accounts")
async def add_account(req: AccountAddReq):
    os.makedirs(ACCOUNTS_DIR, exist_ok=True)
    try:
        cookies_data = json.loads(req.cookies_json)
        # Automatically wrap flat JSON cookie arrays in Playwright's required storage state format
        if isinstance(cookies_data, list):
            cookies_data = {
                "cookies": cookies_data,
                "origins": []
            }
        # Preprocess Cookie-Editor format to Playwright-compatible format
        cookies_data = _preprocess_cookie_editor_json(cookies_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {e}")
    
    # Offline validation — check required cookies are present in the JSON
    is_valid, offline_error = _validate_cookies_offline(cookies_data)
    if not is_valid:
        raise HTTPException(status_code=400, detail=offline_error)
    
    # Determine validation mode
    skip_validation = req.skip_validation
    validation_status = "unverified"
    validation_message = ""
    
    temp_fd, temp_path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
            json.dump(cookies_data, f)
        
        if not skip_validation:
            # Attempt live validation — but gracefully handle IP-mismatch redirects
            try:
                client = await NotebookLMClient.from_storage(temp_path)
                async with client:
                    await client.notebooks.list()
                validation_status = "active"
                validation_message = "Live validation successful."
            except Exception as auth_err:
                err_str = str(auth_err)
                # Detect Google redirect (IP mismatch / session not trusted on this IP)
                if "Redirected to" in err_str and "accounts.google.com" in err_str:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Google rejected the session from this server's IP address. "
                            f"This is normal — Google ties sessions to the IP where they were created. "
                            f"The cookies appear structurally valid (SID and __Secure-1PSIDTS are present). "
                            f"Click 'Save Without Validation' to save the account anyway — "
                            f"the session will be re-authenticated automatically during video rendering."
                        ),
                        headers={"X-Validation-Hint": "skip_validation"}
                    )
                elif "Missing required cookies" in err_str:
                    raise HTTPException(status_code=400, detail=f"Cookie validation failed: {err_str}")
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Google Account validation failed. Error: {err_str}. "
                            f"You can try 'Save Without Validation' to save the cookies anyway."
                        ),
                        headers={"X-Validation-Hint": "skip_validation"}
                    )
        else:
            validation_status = "unverified"
            validation_message = "Saved without live validation (cookies are structurally valid)."
            
        account_id = str(uuid.uuid4())[:8]
        cookie_path, meta_path = get_account_paths(account_id)
        
        with open(cookie_path, "w", encoding="utf-8") as f:
            json.dump(cookies_data, f)
            
        label = req.label.strip() if req.label else f"Google Account ({account_id})"
        meta = {
            "id": account_id,
            "label": label,
            "status": validation_status,
            "created_at": time.time()
        }
        if validation_message:
            meta["message"] = validation_message
        
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)
            
        return {"ok": True, "account": meta}
        
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass

@app.delete("/v1/auth/accounts/{account_id}")
async def delete_account(account_id: str):
    cookie_path, meta_path = get_account_paths(account_id)
    deleted = False
    if os.path.exists(cookie_path):
        os.remove(cookie_path)
        deleted = True
    if os.path.exists(meta_path):
        os.remove(meta_path)
        deleted = True
        
    if not deleted:
        raise HTTPException(status_code=404, detail="Account not found.")
    return {"ok": True, "message": f"Account {account_id} removed successfully."}

@app.post("/v1/auth/accounts/{account_id}/test")
async def test_account(account_id: str):
    cookie_path, meta_path = get_account_paths(account_id)
    if not os.path.exists(cookie_path) or not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="Account not found.")
        
    try:
        client = await NotebookLMClient.from_storage(cookie_path)
        async with client:
            await client.notebooks.list()
            
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["status"] = "active"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)
            
        return {"ok": True, "status": "active", "message": "Connection is valid and operational."}
    except Exception as e:
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta["status"] = "expired"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f)
        except Exception:
            pass
        return {"ok": False, "status": "expired", "message": f"Validation failed: {e}"}


# ----------------------------
# Notebooks
# ----------------------------
@app.get("/v1/notebooks")
async def list_notebooks():
    client = await get_client()
    async with client:
        try:
            nbs = await client.notebooks.list()
            return {"ok": True, "items": [nb.model_dump() if hasattr(nb, "model_dump") else nb.__dict__ for nb in nbs]}
        except RPCError as e:
            raise map_rpc_error(e)


@app.post("/v1/notebooks")
async def create_notebook(req: NotebookCreateReq):
    client = await get_client()
    async with client:
        try:
            nb = await client.notebooks.create(req.title)
            return {"ok": True, "notebook": nb.model_dump() if hasattr(nb, "model_dump") else nb.__dict__}
        except RPCError as e:
            raise map_rpc_error(e)


@app.get("/v1/notebooks/{notebook_id}")
async def get_notebook(notebook_id: str):
    client = await get_client()
    async with client:
        try:
            nb = await client.notebooks.get(notebook_id)
            return {"ok": True, "notebook": nb.model_dump() if hasattr(nb, "model_dump") else nb.__dict__}
        except RPCError as e:
            raise map_rpc_error(e)


@app.delete("/v1/notebooks/{notebook_id}")
async def delete_notebook(notebook_id: str):
    client = await get_client()
    async with client:
        try:
            ok = await client.notebooks.delete(notebook_id)
            return {"ok": True, "deleted": bool(ok)}
        except RPCError as e:
            raise map_rpc_error(e)


@app.patch("/v1/notebooks/{notebook_id}/rename")
async def rename_notebook(notebook_id: str, req: NotebookRenameReq):
    client = await get_client()
    async with client:
        try:
            nb = await client.notebooks.rename(notebook_id, req.new_title)
            return {"ok": True, "notebook": nb.model_dump() if hasattr(nb, "model_dump") else nb.__dict__}
        except RPCError as e:
            raise map_rpc_error(e)


@app.get("/v1/notebooks/{notebook_id}/summary")
async def get_notebook_summary(notebook_id: str):
    client = await get_client()
    async with client:
        try:
            summary = await client.notebooks.get_summary(notebook_id)
            return {"ok": True, "summary": summary}
        except RPCError as e:
            raise map_rpc_error(e)


@app.get("/v1/notebooks/{notebook_id}/description")
async def get_notebook_description(notebook_id: str):
    client = await get_client()
    async with client:
        try:
            desc = await client.notebooks.get_description(notebook_id)
            return {"ok": True, "description": desc.model_dump() if hasattr(desc, "model_dump") else desc.__dict__}
        except RPCError as e:
            raise map_rpc_error(e)


# ----------------------------
# Sources
# ----------------------------
@app.get("/v1/notebooks/{notebook_id}/sources")
async def list_sources(notebook_id: str):
    client = await get_client()
    async with client:
        try:
            items = await client.sources.list(notebook_id)
            return {"ok": True, "items": [s.model_dump() if hasattr(s, "model_dump") else s.__dict__ for s in items]}
        except RPCError as e:
            raise map_rpc_error(e)


@app.post("/v1/notebooks/{notebook_id}/sources/url")
async def add_source_url(notebook_id: str, req: SourceAddUrlReq):
    client = await get_client()
    async with client:
        try:
            src = await client.sources.add_url(notebook_id, req.url, wait=req.wait)
            return {"ok": True, "source": src.model_dump() if hasattr(src, "model_dump") else src.__dict__}
        except TypeError:
            # some versions may not accept wait=; fall back
            try:
                src = await client.sources.add_url(notebook_id, req.url)
                return {"ok": True, "source": src.model_dump() if hasattr(src, "model_dump") else src.__dict__}
            except RPCError as e:
                raise map_rpc_error(e)
        except RPCError as e:
            raise map_rpc_error(e)


@app.post("/v1/notebooks/{notebook_id}/sources/youtube")
async def add_source_youtube(notebook_id: str, req: SourceAddYoutubeReq):
    client = await get_client()
    async with client:
        try:
            src = await client.sources.add_youtube(notebook_id, req.url, wait=req.wait)
            return {"ok": True, "source": src.model_dump() if hasattr(src, "model_dump") else src.__dict__}
        except TypeError:
            try:
                src = await client.sources.add_youtube(notebook_id, req.url)
                return {"ok": True, "source": src.model_dump() if hasattr(src, "model_dump") else src.__dict__}
            except RPCError as e:
                raise map_rpc_error(e)
        except RPCError as e:
            raise map_rpc_error(e)


@app.post("/v1/notebooks/{notebook_id}/sources/text")
async def add_source_text(notebook_id: str, req: SourceAddTextReq):
    client = await get_client()
    async with client:
        try:
            src = await client.sources.add_text(notebook_id, req.title, req.content)
            return {"ok": True, "source": src.model_dump() if hasattr(src, "model_dump") else src.__dict__}
        except RPCError as e:
            raise map_rpc_error(e)


@app.post("/v1/notebooks/{notebook_id}/sources/file")
async def add_source_file(
    notebook_id: str,
    upload: UploadFile = File(...),
    mime_type: Optional[str] = Form(None),
):
    # Save to temp file first
    suffix = os.path.splitext(upload.filename or "")[1] or ".bin"
    tmp_path = os.path.join(tempfile.gettempdir(), f"nb_{uuid.uuid4().hex}{suffix}")
    with open(tmp_path, "wb") as f:
        f.write(await upload.read())

    client = await get_client()
    async with client:
        try:
            src = await client.sources.add_file(notebook_id, tmp_path, mime_type=mime_type)
            return {"ok": True, "source": src.model_dump() if hasattr(src, "model_dump") else src.__dict__}
        except RPCError as e:
            raise map_rpc_error(e)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


@app.get("/v1/notebooks/{notebook_id}/sources/{source_id}/fulltext")
async def get_source_fulltext(notebook_id: str, source_id: str):
    client = await get_client()
    async with client:
        try:
            ft = await client.sources.get_fulltext(notebook_id, source_id)
            return {"ok": True, "fulltext": ft.model_dump() if hasattr(ft, "model_dump") else ft.__dict__}
        except RPCError as e:
            raise map_rpc_error(e)


@app.get("/v1/notebooks/{notebook_id}/sources/{source_id}/guide")
async def get_source_guide(notebook_id: str, source_id: str):
    client = await get_client()
    async with client:
        try:
            guide = await client.sources.get_guide(notebook_id, source_id)
            return {"ok": True, "guide": guide}
        except RPCError as e:
            raise map_rpc_error(e)


@app.delete("/v1/notebooks/{notebook_id}/sources/{source_id}")
async def delete_source(notebook_id: str, source_id: str):
    client = await get_client()
    async with client:
        try:
            ok = await client.sources.delete(notebook_id, source_id)
            return {"ok": True, "deleted": bool(ok)}
        except RPCError as e:
            raise map_rpc_error(e)


# ----------------------------
# Chat
# ----------------------------
@app.post("/v1/notebooks/{notebook_id}/chat/ask")
async def chat_ask(notebook_id: str, req: ChatAskReq):
    client = await get_client()
    async with client:
        try:
            result = await client.chat.ask(notebook_id, req.question)
            # result.answer is shown in docs :contentReference[oaicite:5]{index=5}
            if hasattr(result, "model_dump"):
                return {"ok": True, "result": result.model_dump()}
            return {"ok": True, "result": getattr(result, "__dict__", {"answer": getattr(result, "answer", None)})}
        except RPCError as e:
            raise map_rpc_error(e)


# ----------------------------
# Artifacts: list / generate / poll / download
# ----------------------------
@app.get("/v1/notebooks/{notebook_id}/artifacts")
async def list_artifacts(notebook_id: str, type: Optional[str] = None):
    client = await get_client()
    async with client:
        try:
            items = await client.artifacts.list(notebook_id, type=type) if type else await client.artifacts.list(notebook_id)
            return {"ok": True, "items": [a.model_dump() if hasattr(a, "model_dump") else a.__dict__ for a in items]}
        except RPCError as e:
            raise map_rpc_error(e)


@app.post("/v1/notebooks/{notebook_id}/artifacts/generate")
async def generate_artifact(notebook_id: str, req: ArtifactGenerateReq):
    client = await get_client()
    async with client:
        try:
            t = req.type
            opts = req.options or {}

            if t == "audio":
                status = await client.artifacts.generate_audio(notebook_id, **opts)
            elif t == "video":
                status = await client.artifacts.generate_video(notebook_id, **opts)
            elif t == "report":
                status = await client.artifacts.generate_report(notebook_id, **opts)
            elif t == "quiz":
                status = await client.artifacts.generate_quiz(notebook_id, **opts)
            elif t == "flashcards":
                status = await client.artifacts.generate_flashcards(notebook_id, **opts)
            elif t == "slide_deck":
                status = await client.artifacts.generate_slide_deck(notebook_id, **opts)
            elif t == "infographic":
                status = await client.artifacts.generate_infographic(notebook_id, **opts)
            elif t == "data_table":
                status = await client.artifacts.generate_data_table(notebook_id, **opts)
            elif t == "mind_map":
                # mind_map may return dict directly in docs :contentReference[oaicite:6]{index=6}
                out = await client.artifacts.generate_mind_map(notebook_id, **opts)
                return {"ok": True, "type": t, "result": out}
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported artifact type: {t}")

            # GenerationStatus commonly contains task_id :contentReference[oaicite:7]{index=7}
            payload = status.model_dump() if hasattr(status, "model_dump") else getattr(status, "__dict__", {})
            return {"ok": True, "type": t, "status": payload}
        except RPCError as e:
            raise map_rpc_error(e)


@app.get("/v1/notebooks/{notebook_id}/artifacts/tasks/{task_id}")
async def poll_task(notebook_id: str, task_id: str, wait: bool = False):
    client = await get_client()
    async with client:
        try:
            if wait:
                status = await client.artifacts.wait_for_completion(notebook_id, task_id)
            else:
                status = await client.artifacts.poll_status(notebook_id, task_id)

            payload = status.model_dump() if hasattr(status, "model_dump") else getattr(status, "__dict__", status)
            return {"ok": True, "status": payload}
        except RPCError as e:
            raise map_rpc_error(e)


@app.get("/v1/notebooks/{notebook_id}/artifacts/download")
async def download_artifact(
    notebook_id: str,
    type: Literal[
        "audio",
        "video",
        "infographic",
        "slide_deck",
        "report",
        "mind_map",
        "data_table",
        "quiz",
        "flashcards",
    ],
    artifact_id: Optional[str] = None,
    output_format: Optional[Literal["json", "markdown", "html"]] = None,
):
    """
    Downloads the *first completed* artifact of the given type unless artifact_id is provided.
    notebooklm-py provides type-specific download_* methods. :contentReference[oaicite:8]{index=8}
    """
    suffix_map = {
        "audio": ".mp4",
        "video": ".mp4",
        "infographic": ".png",
        "slide_deck": ".pdf",
        "report": ".md",
        "mind_map": ".json",
        "data_table": ".csv",
        "quiz": ".json" if (output_format in (None, "json")) else (".md" if output_format == "markdown" else ".html"),
        "flashcards": ".json" if (output_format in (None, "json")) else (".md" if output_format == "markdown" else ".html"),
    }
    out_path = os.path.join(tempfile.gettempdir(), f"nlm_{uuid.uuid4().hex}{suffix_map[type]}")

    client = await get_client()
    async with client:
        try:
            if type == "audio":
                await client.artifacts.download_audio(notebook_id, out_path, artifact_id=artifact_id)
            elif type == "video":
                await client.artifacts.download_video(notebook_id, out_path, artifact_id=artifact_id)
            elif type == "infographic":
                await client.artifacts.download_infographic(notebook_id, out_path, artifact_id=artifact_id)
            elif type == "slide_deck":
                await client.artifacts.download_slide_deck(notebook_id, out_path, artifact_id=artifact_id)
            elif type == "report":
                await client.artifacts.download_report(notebook_id, out_path, artifact_id=artifact_id)
            elif type == "mind_map":
                await client.artifacts.download_mind_map(notebook_id, out_path, artifact_id=artifact_id)
            elif type == "data_table":
                await client.artifacts.download_data_table(notebook_id, out_path, artifact_id=artifact_id)
            elif type == "quiz":
                await client.artifacts.download_quiz(
                    notebook_id, out_path, artifact_id=artifact_id, output_format=(output_format or "json")
                )
            elif type == "flashcards":
                await client.artifacts.download_flashcards(
                    notebook_id, out_path, artifact_id=artifact_id, output_format=(output_format or "json")
                )
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported type: {type}")

            filename = os.path.basename(out_path)
            return FileResponse(out_path, filename=filename)
        except RPCError as e:
            # Clean up file if partially created
            try:
                if os.path.exists(out_path):
                    os.remove(out_path)
            except OSError:
                pass
            raise map_rpc_error(e)


# ==============================================================================
# Tutor LMS AI Video Studio Core Extension
# ==============================================================================

# In-memory jobs status database
jobs_db: Dict[str, Dict[str, Any]] = {}
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TutorAIVideo")

# CURATED RATE TABLE (Cost per 1 Million Tokens in USD)
MODEL_RATES = {
    # Gemini
    "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    # OpenRouter Free Tiers
    "meta-llama/llama-3-8b-instruct:free": {"input": 0.0, "output": 0.0},
    "google/gemma-2-9b-it:free": {"input": 0.0, "output": 0.0},
    # OpenRouter Paid Tiers (Defaults if not specified in API response)
    "google/claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "openai/gpt-4o-mini": {"input": 0.150, "output": 0.600},
}

class ScriptGenReq(BaseModel):
    prompt: str
    course_title: str
    lesson_title: str
    lesson_description: str
    api_key: str
    model: str = "gemini-2.5-flash"
    previous_script: Optional[str] = None
    feedback: Optional[str] = None

class S3Config(BaseModel):
    access_key: str
    secret_key: str
    region: str
    bucket: str
    endpoint_url: Optional[str] = None
    cdn_domain: Optional[str] = None

class GoogleCredentialsModel(BaseModel):
    client_id: str
    client_secret: str
    refresh_token: str

class VideoRenderReq(BaseModel):
    script_json: str  # JSON representation of the slide deck
    voice: str = "en-US-GuyNeural"
    style: Optional[Dict[str, Any]] = None
    upload_target: Literal["local", "youtube", "s3", "google_drive"] = "local"
    s3_config: Optional[S3Config] = None
    google_credentials: Optional[GoogleCredentialsModel] = None


# ----------------------------
# 1. Premium Typography Helper
# ----------------------------
def get_premium_font(size: int) -> ImageFont.FreeTypeFont:
    """
    Downloads and caches a premium Google Font (Outfit-Medium) to ensure
    gorgeous typography on any hosting platform (GCP, Azure, Hostinger).
    """
    font_dir = tempfile.gettempdir()
    font_path = os.path.join(font_dir, "Outfit-Medium.ttf")
    
    if not os.path.exists(font_path):
        url = "https://github.com/google/fonts/raw/main/ofl/outfit/static/Outfit-Medium.ttf"
        try:
            logger.info("Downloading Outfit font from Google Fonts GitHub...")
            with httpx.Client(follow_redirects=True) as client:
                resp = client.get(url, timeout=15)
                if resp.status_code == 200:
                    with open(font_path, "wb") as f:
                        f.write(resp.content)
                else:
                    raise Exception("Failed download")
        except Exception as e:
            logger.warning(f"Could not download Outfit font: {e}. Falling back to default.")
            return ImageFont.load_default()
            
    try:
        return ImageFont.truetype(font_path, size)
    except Exception:
        return ImageFont.load_default()


# ----------------------------
# 2. Text Wrapping Utilities
# ----------------------------
def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int, draw: ImageDraw.ImageDraw) -> List[str]:
    """Wraps text cleanly based on visual pixel boundaries for flawless card layout."""
    words = text.split(' ')
    lines = []
    current_line = []
    
    for word in words:
        current_line.append(word)
        test_line = ' '.join(current_line)
        width = draw.textlength(test_line, font=font)
        if width > max_width:
            current_line.pop()
            lines.append(' '.join(current_line))
            current_line = [word]
            
    if current_line:
        lines.append(' '.join(current_line))
    return lines


async def generate_ai_cover_image(image_prompt: str, job_id: str, api_key: str) -> bool:
    """
    Calls Google's Imagen 3 API to generate a high-fidelity course cover.
    Saves it as a valid PNG to the persistent covers folder.
    Returns True if successful, False otherwise.
    """
    import base64
    import io
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-generate-002:predict?key={api_key}"
    payload = {
        "instances": [
            {
                "prompt": image_prompt
            }
        ],
        "parameters": {
            "sampleCount": 1,
            "aspectRatio": "16:9",
            "outputMimeType": "image/jpeg"
        }
    }
    
    try:
        logger.info(f"Invoking Google Imagen 3 API for cover image of job {job_id}...")
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                url, 
                json=payload, 
                headers={"Content-Type": "application/json"}, 
                timeout=60
            )
            if resp.status_code != 200:
                logger.warning(f"Imagen 3 API returned status code {resp.status_code}: {resp.text}")
                return False
            
            data = resp.json()
            predictions = data.get("predictions", [])
            if not predictions:
                logger.warning(f"Imagen 3 response predictions array is empty: {data}")
                return False
                
            b64_data = predictions[0].get("bytesBase64Encoded")
            if not b64_data:
                logger.warning("Imagen 3 prediction contains no bytesBase64Encoded data.")
                return False
                
            # Decode JPEG image bytes
            image_bytes = base64.b64decode(b64_data)
            
            # Save to persistent directory
            persistent_dir = os.path.join(tempfile.gettempdir(), "tutor_ai_videos")
            os.makedirs(persistent_dir, exist_ok=True)
            out_path = os.path.join(persistent_dir, f"{job_id}_cover.png")
            
            # Use Pillow to decode JPEG bytes and convert/save as PNG
            img = Image.open(io.BytesIO(image_bytes))
            img.save(out_path, "PNG")
            
            logger.info(f"Successfully generated high-fidelity AI cover image at: {out_path}")
            return True
            
    except Exception as err:
        logger.error(f"Error during Imagen 3 cover generation: {err}")
        return False


def generate_cover_image(title: str, level: str, job_id: str) -> str:
    """Generates a premium high-resolution cover image for the course using Pillow."""
    width = 1200
    height = 675
    
    # Create canvas
    img = Image.new("RGB", (width, height), "#0f172a")
    draw = ImageDraw.Draw(img)
    
    # 1. Draw modern fluid gradient background (Navy/Blue/Indigo)
    start_color = (15, 23, 42)  # #0f172a (Slate 900)
    end_color = (30, 64, 175)   # #1e40af (Blue 800)
    
    for x in range(width):
        r = int(start_color[0] + (end_color[0] - start_color[0]) * (x / width))
        g = int(start_color[1] + (end_color[1] - start_color[1]) * (x / width))
        b = int(start_color[2] + (end_color[2] - start_color[2]) * (x / width))
        draw.line([(x, 0), (x, height)], fill=(r, g, b))
        
    # 2. Draw modern geometric vectors
    draw.line([(0, 0), (1200, 675)], fill=(37, 99, 235), width=2)   # #2563eb
    draw.line([(0, 200), (800, 675)], fill=(59, 130, 246), width=1)  # #3b82f6
    draw.line([(400, 0), (1200, 475)], fill=(29, 78, 216), width=1)  # #1d4ed8
    
    # 3. Load premium Google Fonts
    font_title = get_premium_font(56)
    font_badge = get_premium_font(20)
    
    # 4. Wrap text and draw title
    lines = wrap_text(title, font_title, 900, draw)
    y_cursor = 240
    for line in lines:
        h = 60 # height fallback
        try:
            bbox = font_title.getbbox(line)
            h = bbox[3] - bbox[1]
        except Exception:
            pass
        draw.text((100, y_cursor), line, fill=(255, 255, 255), font=font_title)
        y_cursor += h + 20
        
    # 5. Draw the solid glowing "✨ AI COURSE STUDIO" badge
    draw.rounded_rectangle([(100, 110), (330, 150)], radius=20, fill=(48, 92, 222))  # #305cde
    draw.text((122, 120), "✨ AI COURSE STUDIO", fill=(255, 255, 255), font=font_badge)
    
    # 6. Draw the outline Difficulty Level badge
    if level:
        level_str = level.upper().replace("_", " ")
        draw.rounded_rectangle([(350, 110), (350 + 20 + len(level_str)*10, 150)], radius=20, outline=(255, 255, 255), width=1)
        draw.text((372, 120), f"⚡ {level_str}", fill=(255, 255, 255), font=font_badge)
        
    # Save cover image to persistent folder
    persistent_dir = os.path.join(tempfile.gettempdir(), "tutor_ai_videos")
    os.makedirs(persistent_dir, exist_ok=True)
    out_path = os.path.join(persistent_dir, f"{job_id}_cover.png")
    img.save(out_path, "PNG")
    
    logger.info(f"Generated gorgeous cover image at: {out_path}")
    return out_path


# ----------------------------
# 3. Premium Pillow Slides Drawing Engine
# ----------------------------
def draw_slide_image(slide: dict, index: int, output_path: str, style: Optional[dict] = None):
    """
    Renders an exceptionally beautiful linear-gradient glassmorphic slide.
    Features: Dark mode, Outfit typography, card layouts, and highlighted bullets.
    """
    width, height = 1920, 1080
    image = Image.new("RGBA", (width, height), (15, 23, 42, 255))
    draw = ImageDraw.Draw(image)
    
    # Sleek Linear Gradient (Navy Blue to Indigo Purple)
    for y in range(height):
        ratio = y / height
        r = int(15 + (15 * ratio))
        g = int(23 - (7 * ratio))
        b = int(42 + (30 * ratio))
        draw.line([(0, y), (width, y)], fill=(r, g, b, 255))
        
    # Glassmorphic Card Container Overlay
    card_margin_x, card_margin_y = 120, 100
    card_w, card_h = width - (card_margin_x * 2), height - (card_margin_y * 2)
    card_x0, card_y0 = card_margin_x, card_margin_y
    card_x1, card_y1 = card_x0 + card_w, card_y0 + card_h
    
    draw.rounded_rectangle(
        [(card_x0, card_y0), (card_x1, card_y1)],
        radius=40,
        fill=(255, 255, 255, 10),
        outline=(255, 255, 255, 30),
        width=2
    )
    
    title_font = get_premium_font(60)
    bullet_font = get_premium_font(42)
    meta_font = get_premium_font(24)
    
    # Metadata Badge
    draw.rounded_rectangle(
        [(card_x0 + 60, card_y0 + 50), (card_x0 + 260, card_y0 + 90)],
        radius=12,
        fill=(168, 85, 247, 50),
        outline=(168, 85, 247, 100),
        width=1
    )
    draw.text((card_x0 + 85, card_y0 + 58), f"SLIDE {index + 1}", font=meta_font, fill=(216, 180, 254, 255))
    
    # Slide Title
    title_text = slide.get("title", "Lesson Slide")
    draw.text(
        (card_x0 + 60, card_y0 + 130),
        title_text,
        font=title_font,
        fill=(255, 255, 255, 255)
    )
    
    # Thin slide accent separator
    draw.line(
        [(card_x0 + 60, card_y0 + 220), (card_x0 + 200, card_y0 + 220)],
        fill=(168, 85, 247, 255),
        width=4
    )
    
    # Render Bullet Points
    bullets = slide.get("points", [])
    current_y = card_y0 + 270
    line_spacing = 70
    
    for bullet in bullets:
        wrapped_lines = wrap_text(bullet, bullet_font, card_w - 180, draw)
        
        bullet_icon_x = card_x0 + 80
        bullet_icon_y = current_y + 18
        draw.ellipse(
            [(bullet_icon_x - 10, bullet_icon_y - 10), (bullet_icon_x + 10, bullet_icon_y + 10)],
            fill=(168, 85, 247, 255),
            outline=(216, 180, 254, 255),
            width=2
        )
        
        for line in wrapped_lines:
            draw.text(
                (card_x0 + 120, current_y),
                line,
                font=bullet_font,
                fill=(241, 245, 249, 255)
            )
            current_y += line_spacing
            
        current_y += 30
        if current_y > card_y1 - 80:
            break
            
    image.save(output_path, "PNG")


# ----------------------------
# 4. Asynchronous Edge-TTS Caller
# ----------------------------
async def generate_speech_audio(text: str, voice: str, output_path: str):
    """Synthesizes high-fidelity audio from slide narration using free neural TTS."""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


# ----------------------------
# 5. Cloud S3 / Cloudflare R2 Uploader
# ----------------------------
def upload_to_s3(local_file: str, s3_cfg: S3Config) -> str:
    """Uploads compiled video directly to AWS S3 or Cloudflare R2 using boto3."""
    config = BotoConfig(
        retries = {'max_attempts': 3, 'mode': 'standard'}
    )
    s3 = boto3.client(
        's3',
        aws_access_key_id=s3_cfg.access_key,
        aws_secret_access_key=s3_cfg.secret_key,
        region_name=s3_cfg.region,
        endpoint_url=s3_cfg.endpoint_url if s3_cfg.endpoint_url else None,
        config=config
    )
    
    file_name = f"tutor_lms_ai_{uuid.uuid4().hex}.mp4"
    logger.info(f"Uploading {local_file} to S3/R2 as {file_name}...")
    
    s3.upload_file(
        local_file,
        s3_cfg.bucket,
        file_name,
        ExtraArgs={'ACL': 'public-read', 'ContentType': 'video/mp4'}
    )
    
    if s3_cfg.cdn_domain:
        return f"https://{s3_cfg.cdn_domain}/{file_name}"
    elif s3_cfg.endpoint_url:
        return f"{s3_cfg.endpoint_url.rstrip('/')}/{s3_cfg.bucket}/{file_name}"
    else:
        return f"https://{s3_cfg.bucket}.s3.{s3_cfg.region}.amazonaws.com/{file_name}"


# ----------------------------
# 6. Cloud Google Drive Uploader
# ----------------------------
def upload_to_google_drive(local_file: str, creds_model: GoogleCredentialsModel) -> str:
    """
    Uploads completed MP4 to user's Google Drive, configures public view-only access,
    and returns a clean, fully-embeddable HTML preview iframe.
    """
    creds = GoogleCredentials(
        token=None,
        refresh_token=creds_model.refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=creds_model.client_id,
        client_secret=creds_model.client_secret
    )
    service = google_build('drive', 'v3', credentials=creds)
    
    file_metadata = {
        'name': f'TutorLMS_AI_Lesson_{uuid.uuid4().hex[:6]}.mp4',
        'mimeType': 'video/mp4'
    }
    media = MediaFileUpload(local_file, mimetype='video/mp4', resumable=True)
    
    logger.info("Uploading file to Google Drive...")
    file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id'
    ).execute()
    file_id = file.get('id')
    
    logger.info(f"Setting public permissions for Google Drive File ID: {file_id}")
    service.permissions().create(
        fileId=file_id,
        body={'type': 'anyone', 'role': 'reader'}
    ).execute()
    
    iframe_code = f'<iframe src="https://drive.google.com/file/d/{file_id}/preview" width="100%" height="450" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>'
    return iframe_code


# ----------------------------
# 7. Cloud YouTube Uploader
# ----------------------------
def upload_to_youtube(local_file: str, creds_model: GoogleCredentialsModel) -> str:
    """Uploads finished lesson video to YouTube channel as unlisted and returns URL."""
    creds = GoogleCredentials(
        token=None,
        refresh_token=creds_model.refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=creds_model.client_id,
        client_secret=creds_model.client_secret
    )
    youtube = google_build('youtube', 'v3', credentials=creds)
    
    body = {
        'snippet': {
            'title': f'AI Lesson Video {uuid.uuid4().hex[:6]}',
            'description': 'Generated automatically by Tutor LMS AI Video Studio add-on.',
            'categoryId': '27'
        },
        'status': {
            'privacyStatus': 'unlisted',
            'selfDeclaredMadeForKids': False
        }
    }
    
    media = MediaFileUpload(local_file, mimetype='video/mp4', resumable=True)
    logger.info("Uploading file to YouTube channel...")
    
    request = youtube.videos().insert(
        part='snippet,status',
        body=body,
        media_body=media
    )
    
    response = request.execute()
    video_id = response.get('id')
    
    return f"https://www.youtube.com/watch?v={video_id}"


# ----------------------------
# 8. Asynchronous Rendering Pipeline Job Task
# ----------------------------
async def async_video_render_pipeline(job_id: str, script_json_str: str, voice: str, style: Optional[dict], upload_target: str, s3_cfg: Optional[S3Config], google_creds: Optional[GoogleCredentialsModel], engine: str = "slides"):
    """Executes video rendering via Slides+TTS or NotebookLM engine based on the engine parameter."""
    temp_dir = tempfile.mkdtemp()
    jobs_db[job_id]["status"] = "processing"
    jobs_db[job_id]["progress"] = 5
    jobs_db[job_id]["logs"].append(f"Started rendering pipeline task. Engine: {engine}")
    
    try:
        script_data = json.loads(script_json_str)
        slides = script_data.get("slides", [])
        if not slides:
            raise Exception("No slides found in script.")

        # =====================================================================
        # NotebookLM Engine Branch
        # =====================================================================
        if engine == "notebooklm":
            jobs_db[job_id]["logs"].append("NotebookLM engine selected. Initializing Google NotebookLM client...")
            try:
                nlm_client = await get_client_rotation(job_id)
                if not nlm_client:
                    jobs_db[job_id]["logs"].append("No working NotebookLM client could be initialized. Falling back to Slides engine.")
                    engine = "slides"
            except Exception as nlm_init_err:
                jobs_db[job_id]["logs"].append(f"NotebookLM client rotation init failed: {nlm_init_err}. Falling back to Slides engine.")
                engine = "slides"  # fall through to slides pipeline below

            if engine == "notebooklm":
                async with nlm_client:
                    # 1. Create a temporary notebook for this lesson
                    lesson_title = slides[0].get("title", "AI Course Lesson") if slides else "AI Course Lesson"
                    jobs_db[job_id]["logs"].append(f"Creating NotebookLM notebook: {lesson_title}")
                    jobs_db[job_id]["progress"] = 10
                    nb = await nlm_client.notebooks.create(lesson_title)
                    notebook_id = nb.id if hasattr(nb, "id") else nb.notebook_id if hasattr(nb, "notebook_id") else str(nb)
                    jobs_db[job_id]["logs"].append(f"Notebook created: {notebook_id}")

                    # 2. Add all slide narrations as a single text source document
                    full_script = "\n\n".join(
                        f"## {s.get('title', f'Slide {i+1}')}\n{s.get('narration', '')}"
                        for i, s in enumerate(slides)
                    )
                    jobs_db[job_id]["logs"].append("Adding lesson script as text source...")
                    jobs_db[job_id]["progress"] = 20
                    await nlm_client.sources.add_text(notebook_id, lesson_title, full_script)
                    jobs_db[job_id]["logs"].append("Source added successfully.")

                    # 3. Generate video overview artifact
                    jobs_db[job_id]["logs"].append("Triggering NotebookLM video generation...")
                    jobs_db[job_id]["progress"] = 30
                    gen_status = await nlm_client.artifacts.generate_video(
                        notebook_id, video_format=VideoFormat.EXPLAINER
                    )
                    task_id = gen_status.task_id if hasattr(gen_status, "task_id") else None

                    # 4. Poll / wait for completion
                    if task_id:
                        jobs_db[job_id]["logs"].append(f"Video generation task started: {task_id}. Waiting for completion...")
                        jobs_db[job_id]["progress"] = 40
                        completed = await nlm_client.artifacts.wait_for_completion(notebook_id, task_id)
                        jobs_db[job_id]["logs"].append("NotebookLM video generation completed.")
                    else:
                        jobs_db[job_id]["logs"].append("Video generation initiated (no task_id returned, assuming synchronous).")

                    jobs_db[job_id]["progress"] = 65

                    # 5. Download the generated video file
                    nlm_output_path = os.path.join(temp_dir, f"nlm_{job_id}.mp4")
                    jobs_db[job_id]["logs"].append("Downloading NotebookLM video overview...")
                    await nlm_client.artifacts.download_video(notebook_id, nlm_output_path)
                    jobs_db[job_id]["logs"].append(f"Downloaded to {nlm_output_path}")
                    jobs_db[job_id]["progress"] = 75

                    # 6. Move to persistent directory (same as slides pipeline)
                    persistent_dir = os.path.join(tempfile.gettempdir(), "tutor_ai_videos")
                    os.makedirs(persistent_dir, exist_ok=True)
                    final_path = os.path.join(persistent_dir, f"{job_id}.mp4")
                    shutil.copy2(nlm_output_path, final_path)
                    jobs_db[job_id]["logs"].append("Copied to persistent storage.")
                    jobs_db[job_id]["progress"] = 80

                    # 7. Handle cloud uploads (reuse same logic as slides)
                    video_url = f"/v1/jobs/{job_id}/download"
                    if upload_target == "s3" and s3_cfg:
                        jobs_db[job_id]["logs"].append("Uploading to S3/R2...")
                        video_url = await upload_to_s3(final_path, f"{job_id}.mp4", s3_cfg)
                    elif upload_target == "youtube" and google_creds:
                        jobs_db[job_id]["logs"].append("Uploading to YouTube...")
                        video_url = await upload_to_youtube(final_path, lesson_title, google_creds)
                    elif upload_target == "google_drive" and google_creds:
                        jobs_db[job_id]["logs"].append("Uploading to Google Drive...")
                        video_url = await upload_to_google_drive(final_path, f"{job_id}.mp4", google_creds)

                    jobs_db[job_id]["progress"] = 100
                    jobs_db[job_id]["status"] = "completed"
                    jobs_db[job_id]["video_url"] = video_url
                    jobs_db[job_id]["logs"].append(f"NotebookLM render complete. URL: {video_url}")

                    # 8. Cleanup notebook
                    try:
                        await nlm_client.notebooks.delete(notebook_id)
                        jobs_db[job_id]["logs"].append("Temporary NotebookLM notebook deleted.")
                    except Exception:
                        pass  # non-critical cleanup

                    # Cleanup temp
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    return

        # =====================================================================
        # Slides Engine (Default) — original pipeline continues below
        # =====================================================================
            
        jobs_db[job_id]["logs"].append(f"Parsing slide script successful. Total slides: {len(slides)}")
        slide_clips = []
        
        for idx, slide in enumerate(slides):
            jobs_db[job_id]["logs"].append(f"Processing slide {idx+1}/{len(slides)}: {slide.get('title')}")
            
            img_path = os.path.join(temp_dir, f"slide_{idx}.png")
            draw_slide_image(slide, idx, img_path, style)
            
            audio_path = os.path.join(temp_dir, f"slide_{idx}.mp3")
            narration = slide.get("narration", "Next slide.")
            await generate_speech_audio(narration, voice, audio_path)
            
            clip_path = os.path.join(temp_dir, f"slide_{idx}.mp4")
            cmd = [
                "ffmpeg", "-y", "-loop", "1", "-i", img_path, "-i", audio_path,
                "-c:v", "libx264", "-tune", "stillimage", "-c:a", "aac",
                "-b:a", "192k", "-pix_fmt", "yuv420p", "-shortest", clip_path
            ]
            
            logger.info(f"Running FFmpeg: {' '.join(cmd)}")
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if proc.returncode != 0:
                err_msg = proc.stderr.decode('utf-8', errors='ignore')
                raise Exception(f"FFmpeg slide clip compilation failed: {err_msg}")
                
            slide_clips.append(clip_path)
            jobs_db[job_id]["progress"] = int(5 + (70 * ((idx + 1) / len(slides))))
            
        jobs_db[job_id]["logs"].append("Concatenating individual slide scenes into master lesson video...")
        concat_txt_path = os.path.join(temp_dir, "concat.txt")
        with open(concat_txt_path, "w") as f:
            for clip in slide_clips:
                f.write(f"file '{clip}'\n")
                
        final_local_mp4 = os.path.join(temp_dir, "final_video.mp4")
        concat_cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_txt_path,
            "-c", "copy", final_local_mp4
        ]
        
        logger.info(f"Running FFmpeg Concat: {' '.join(concat_cmd)}")
        proc_concat = subprocess.run(concat_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc_concat.returncode != 0:
            err_msg = proc_concat.stderr.decode('utf-8', errors='ignore')
            raise Exception(f"FFmpeg master compilation failed: {err_msg}")
            
        jobs_db[job_id]["progress"] = 80
        jobs_db[job_id]["logs"].append("Video compile successful. Starting upload step...")
        
        final_url = ""
        if upload_target == "s3" and s3_cfg:
            jobs_db[job_id]["logs"].append("Uploading video to cloud AWS S3 / Cloudflare R2 bucket...")
            final_url = upload_to_s3(final_local_mp4, s3_cfg)
        elif upload_target == "youtube" and google_creds:
            jobs_db[job_id]["logs"].append("Uploading lesson to YouTube channel...")
            final_url = upload_to_youtube(final_local_mp4, google_creds)
        elif upload_target == "google_drive" and google_creds:
            jobs_db[job_id]["logs"].append("Uploading lesson to Google Drive...")
            final_url = upload_to_google_drive(final_local_mp4, google_creds)
        else:
            jobs_db[job_id]["logs"].append("Saving video locally to FastAPI download storage...")
            persistent_dir = os.path.join(tempfile.gettempdir(), "tutor_ai_videos")
            os.makedirs(persistent_dir, exist_ok=True)
            persistent_path = os.path.join(persistent_dir, f"{job_id}.mp4")
            shutil.copy(final_local_mp4, persistent_path)
            final_url = f"/v1/jobs/{job_id}/download"
            
        jobs_db[job_id]["status"] = "completed"
        jobs_db[job_id]["progress"] = 100
        jobs_db[job_id]["url"] = final_url
        jobs_db[job_id]["logs"].append("Rendering job successfully completed!")
        logger.info(f"Job {job_id} successfully completed. URL: {final_url}")
        
    except Exception as e:
        logger.error(f"Error executing rendering job {job_id}: {e}")
        jobs_db[job_id]["status"] = "failed"
        jobs_db[job_id]["logs"].append(f"FAILED: {str(e)}")
    finally:
        try:
            shutil.rmtree(temp_dir)
        except OSError:
            pass


# ----------------------------
# 9. HTTP REST Controller Routing
# ----------------------------
@app.post("/v1/jobs")
async def start_job(req: Dict[str, Any], background_tasks: BackgroundTasks):
    """
    Central router handling script generation (Gemini/OpenRouter)
    and asynchronous rendering queues.
    """
    job_id = str(uuid.uuid4())
    job_type = req.get("type", "generate_script")
    
    jobs_db[job_id] = {
        "id": job_id,
        "type": job_type,
        "status": "pending",
        "progress": 0,
        "logs": ["Job created, placed in processing queue."],
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost_usd": 0.0,
        "url": "",
        "script": ""
    }
    
    if job_type == "generate_script":
        logger.info(f"Running script generation job: {job_id}")
        jobs_db[job_id]["status"] = "processing"
        jobs_db[job_id]["progress"] = 20
        
        prompt = req.get("prompt", "")
        course_title = req.get("course_title", "")
        lesson_title = req.get("lesson_title", "")
        lesson_desc = req.get("lesson_description", "")
        api_key = req.get("api_key", "")
        model = req.get("model", "gemini-2.5-flash")
        prev_script = req.get("previous_script")
        feedback = req.get("feedback")
        
        is_openrouter = "/" in model or not model.startswith("gemini")
        
        system_instructions = (
            "You are a professional educational curriculum designer. Output a structured, comprehensive "
            "slide presentation layout in 100% clean JSON format. Do NOT wrap it in backticks, markdown, "
            "or include conversational filler text. The output JSON must strictly match this schema:\n"
            "{\n"
            "  \"slides\": [\n"
            "    {\n"
            "      \"title\": \"Slide Title (Clean, 1-6 words)\",\n"
            "      \"points\": [\n"
            "        \"Key educational bullet point (6-14 words)\",\n"
            "        \"Another key takeaway\"\n"
            "      ],\n"
            "      \"narration\": \"Narration script to be read out loud for this slide. Expand on the bullet points in detail, keeping it high-quality and naturally flowing.\"\n"
            "    }\n"
            "  ]\n"
            "}\n"
        )
        
        user_message = ""
        if prev_script and feedback:
            user_message = (
                f"You previously drafted this slide script:\n{prev_script}\n\n"
                f"The user has requested the following corrections/feedback:\n"
                f"\"{feedback}\"\n\n"
                f"Regenerate the slide deck JSON, modifying and correcting the previous output according to the feedback."
            )
        else:
            user_message = (
                f"Create a slideshow lesson about: \"{prompt}\".\n"
                f"Parent Course Title: \"{course_title}\"\n"
                f"Lesson Title: \"{lesson_title}\"\n"
                f"Lesson Description/Goals: \"{lesson_desc}\"\n"
                f"Generate between 3 to 7 comprehensive slides explaining the topic in-depth."
            )
            
        try:
            script_out = ""
            tokens_in = 0
            tokens_out = 0
            
            if is_openrouter:
                logger.info(f"Calling OpenRouter API using model: {model}...")
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                payload = {
                    "model": model,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": system_instructions},
                        {"role": "user", "content": user_message}
                    ]
                }
                async with httpx.AsyncClient() as client:
                    resp = await client.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers, timeout=60)
                    if resp.status_code != 200:
                        raise Exception(f"OpenRouter API returned error code {resp.status_code}: {resp.text}")
                    data = resp.json()
                    script_out = data["choices"][0]["message"]["content"]
                    
                    usage = data.get("usage", {})
                    tokens_in = usage.get("prompt_tokens", 0)
                    tokens_out = usage.get("completion_tokens", 0)
            else:
                logger.info(f"Calling Gemini API using model: {model}...")
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
                payload = {
                    "contents": [{
                        "parts": [{"text": f"{system_instructions}\n\n{user_message}"}]
                    }],
                    "generationConfig": {
                        "responseMimeType": "application/json"
                    }
                }
                async with httpx.AsyncClient() as client:
                    resp = await client.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=60)
                    if resp.status_code != 200:
                        raise Exception(f"Gemini API returned error code {resp.status_code}: {resp.text}")
                    data = resp.json()
                    script_out = data["candidates"][0]["content"]["parts"][0]["text"]
                    
                    usage = data.get("usageMetadata", {})
                    tokens_in = usage.get("promptTokenCount", 0)
                    tokens_out = usage.get("candidatesTokenCount", 0)
            
            script_out = script_out.strip()
            if script_out.startswith("```json"):
                script_out = script_out[7:]
            if script_out.endswith("```"):
                script_out = script_out[:-3]
            script_out = script_out.strip()
            
            json.loads(script_out)
            
            rate = MODEL_RATES.get(model, {"input": 0.5, "output": 1.5})
            cost = ((tokens_in * rate["input"]) + (tokens_out * rate["output"])) / 1000000.0
            
            jobs_db[job_id]["status"] = "completed"
            jobs_db[job_id]["progress"] = 100
            jobs_db[job_id]["prompt_tokens"] = tokens_in
            jobs_db[job_id]["completion_tokens"] = tokens_out
            jobs_db[job_id]["cost_usd"] = cost
            jobs_db[job_id]["script"] = script_out
            jobs_db[job_id]["logs"].append("Script generation completed successfully.")
            
            return {
                "ok": True,
                "job_id": job_id,
                "status": "completed",
                "script": script_out,
                "usage": {
                    "prompt_tokens": tokens_in,
                    "completion_tokens": tokens_out,
                    "cost_usd": cost
                }
            }
            
        except Exception as e:
            logger.error(f"Script generation failed for job {job_id}: {e}")
            jobs_db[job_id]["status"] = "failed"
            jobs_db[job_id]["logs"].append(f"FAILED: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Script drafting failed: {str(e)}")
            
    elif job_type == "generate_full_course":
        logger.info(f"Running full course outline generation job: {job_id}")
        jobs_db[job_id]["status"] = "processing"
        jobs_db[job_id]["progress"] = 20
        
        prompt = req.get("prompt", "")
        api_key = req.get("api_key", "")
        model = req.get("model", "gemini-2.5-flash")
        
        is_openrouter = "/" in model or not model.startswith("gemini")
        
        system_instructions = (
            "You are a professional educational curriculum designer and course creator. "
            "Your task is to generate a complete, structured course outline based on the user's prompt. "
            "You MUST output the result in 100% clean JSON format. Do NOT wrap it in backticks, markdown, or include conversational filler text.\n"
            "For the image cover of the course, write an extremely detailed, professional, artistic visual prompt under the \"image_prompt\" key. "
            "The prompt should describe a stunning 16:9 aspect ratio cover illustration relevant to the course topic (e.g. abstract modern fluid gradients, 3D tech icons/shapes, sleek vector design) and must NOT contain any words, letters, text, or human-written details.\n"
            "The output JSON must strictly match this schema:\n"
            "{\n"
            "  \"title\": \"Course Title\",\n"
            "  \"description\": \"Course Description\",\n"
            "  \"category\": \"A single matching broad educational category name (e.g. Technology, Business, Design, Language, Health, Marketing)\",\n"
            "  \"level\": \"beginner | intermediate | expert | all_levels\",\n"
            "  \"image_prompt\": \"A highly detailed, professional, artistic prompt for an AI image generator to create a stunning 16:9 aspect ratio cover illustration for this course. Be descriptive, focusing on modern abstract, tech, vector, or fluid gradient styles, avoiding any readable text inside the image.\",\n"
            "  \"benefits\": [\"Key takeaway benefit 1\", \"Key takeaway benefit 2\"], \n"
            "  \"requirements\": [\"Prerequisite requirement 1\", \"Prerequisite requirement 2\"],\n"
            "  \"target_audience\": [\"Target audience group 1\", \"Target audience group 2\"],\n"
            "  \"materials_includes\": [\"Material included 1\", \"Material included 2\"],\n"
            "  \"topics\": [\n"
            "    {\n"
            "      \"title\": \"Topic Title\",\n"
            "      \"summary\": \"Brief summary of this topic.\",\n"
            "      \"lessons\": [\n"
            "        {\n"
            "          \"title\": \"Lesson Title\",\n"
            "          \"content\": \"Comprehensive educational reading notes and article content explaining this lesson's concept in detail for the student to read.\",\n"
            "          \"video_script\": {\n"
            "            \"slides\": [\n"
            "              {\n"
            "                \"title\": \"Slide Title (Clean, 1-6 words)\",\n"
            "                \"points\": [\n"
            "                  \"Key bullet point (6-14 words)\",\n"
            "                  \"Another key bullet point\"\n"
            "                ],\n"
            "                \"narration\": \"Detailed narration voiceover script to be read out loud for this slide. Speak naturally, educationally, and in-depth expanding on the bullet points.\"\n"
            "              }\n"
            "            ]\n"
            "          }\n"
            "        }\n"
            "      ],\n"
            "      \"quizzes\": [\n"
            "        {\n"
            "          \"title\": \"Topic Quiz\",\n"
            "          \"description\": \"Short quiz testing this topic's concepts.\",\n"
            "          \"questions\": [\n"
            "            {\n"
            "              \"question_title\": \"Question text?\",\n"
            "              \"question_type\": \"single_choice\",\n"
            "              \"options\": [\n"
            "                { \"option_title\": \"Correct option text\", \"is_correct\": true },\n"
            "                { \"option_title\": \"Incorrect option text\", \"is_correct\": false }\n"
            "              ]\n"
            "            }\n"
            "          ]\n"
            "        }\n"
            "      ],\n"
            "      \"assignments\": [\n"
            "        {\n"
            "          \"title\": \"Topic Assignment\",\n"
            "          \"description\": \"Detailed task description, steps, and expected deliverables for a practical exercise.\",\n"
            "          \"total_marks\": 100,\n"
            "          \"pass_mark\": 50\n"
            "        }\n"
            "      ]\n"
            "    }\n"
            "  ]\n"
            "}\n"
        )
        
        user_message = (
            f"Generate a comprehensive educational course outline matching the JSON schema precisely based on this overview prompt:\n"
            f"\"{prompt}\"\n\n"
            f"Please write 2 to 4 distinct topics. For each topic, generate 2 to 4 in-depth lessons (each with full reading content and slide script slides), "
            f"1 quiz (with 2 to 4 multiple choice questions), and 1 practical assignment."
        )
        
        try:
            course_out = ""
            tokens_in = 0
            tokens_out = 0
            
            if is_openrouter:
                logger.info(f"Calling OpenRouter API using model: {model}...")
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                payload = {
                    "model": model,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": system_instructions},
                        {"role": "user", "content": user_message}
                    ]
                }
                async with httpx.AsyncClient() as client:
                    resp = await client.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers, timeout=60)
                    if resp.status_code != 200:
                        raise Exception(f"OpenRouter API returned error code {resp.status_code}: {resp.text}")
                    data = resp.json()
                    course_out = data["choices"][0]["message"]["content"]
                    
                    usage = data.get("usage", {})
                    tokens_in = usage.get("prompt_tokens", 0)
                    tokens_out = usage.get("completion_tokens", 0)
            else:
                logger.info(f"Calling Gemini API using model: {model}...")
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
                payload = {
                    "contents": [{
                        "parts": [{"text": f"{system_instructions}\n\n{user_message}"}]
                    }],
                    "generationConfig": {
                        "responseMimeType": "application/json"
                    }
                }
                async with httpx.AsyncClient() as client:
                    resp = await client.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=60)
                    if resp.status_code != 200:
                        raise Exception(f"Gemini API returned error code {resp.status_code}: {resp.text}")
                    data = resp.json()
                    course_out = data["candidates"][0]["content"]["parts"][0]["text"]
                    
                    usage = data.get("usageMetadata", {})
                    tokens_in = usage.get("promptTokenCount", 0)
                    tokens_out = usage.get("candidatesTokenCount", 0)
            
            course_out = course_out.strip()
            if course_out.startswith("```json"):
                course_out = course_out[7:]
            if course_out.endswith("```"):
                course_out = course_out[:-3]
            course_out = course_out.strip()
            
            # Parse JSON & generate course cover
            try:
                course_data = json.loads(course_out)
                title = course_data.get("title", "AI Generated Course")
                level = course_data.get("level", "all_levels")
                image_prompt = course_data.get("image_prompt", "")
                
                cover_generated = False
                if image_prompt and api_key and not is_openrouter:
                    try:
                        cover_generated = await generate_ai_cover_image(image_prompt, job_id, api_key)
                    except Exception as imagen_err:
                        logger.error(f"Failed invoking Imagen 3 cover generation: {imagen_err}")
                
                if not cover_generated:
                    logger.info("Falling back to local typographic Pillow cover canvas generation...")
                    generate_cover_image(title, level, job_id)
                    
                cover_url = f"/v1/covers/{job_id}"
            except Exception as img_err:
                logger.warning(f"Could not generate cover image: {img_err}")
                cover_url = ""

            rate = MODEL_RATES.get(model, {"input": 0.5, "output": 1.5})
            cost = ((tokens_in * rate["input"]) + (tokens_out * rate["output"])) / 1000000.0
            
            jobs_db[job_id]["status"] = "completed"
            jobs_db[job_id]["progress"] = 100
            jobs_db[job_id]["prompt_tokens"] = tokens_in
            jobs_db[job_id]["completion_tokens"] = tokens_out
            jobs_db[job_id]["cost_usd"] = cost
            jobs_db[job_id]["script"] = course_out
            jobs_db[job_id]["cover_url"] = cover_url
            jobs_db[job_id]["logs"].append("Course outline generation completed successfully.")
            
            return {
                "ok": True,
                "job_id": job_id,
                "status": "completed",
                "script": course_out,
                "cover_url": cover_url,
                "usage": {
                    "prompt_tokens": tokens_in,
                    "completion_tokens": tokens_out,
                    "cost_usd": cost
                }
            }
            
        except Exception as e:
            logger.error(f"Course generation failed for job {job_id}: {e}")
            jobs_db[job_id]["status"] = "failed"
            jobs_db[job_id]["logs"].append(f"FAILED: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Course generation failed: {str(e)}")
            
    elif job_type == "render_video":
        logger.info(f"Queued video rendering job: {job_id}")
        
        script_json = req.get("script_json", "")
        voice = req.get("voice", "en-US-GuyNeural")
        style = req.get("style", {})
        upload_target = req.get("upload_target", "local")
        engine = req.get("engine", "slides")
        
        s3_cfg = None
        if upload_target == "s3" and "s3_config" in req:
            s3_data = req["s3_config"]
            s3_cfg = S3Config(
                access_key=s3_data.get("access_key", ""),
                secret_key=s3_data.get("secret_key", ""),
                region=s3_data.get("region", "us-east-1"),
                bucket=s3_data.get("bucket", ""),
                endpoint_url=s3_data.get("endpoint_url"),
                cdn_domain=s3_data.get("cdn_domain")
            )
            
        google_creds = None
        if upload_target in ("youtube", "google_drive") and "google_credentials" in req:
            g_data = req["google_credentials"]
            google_creds = GoogleCredentialsModel(
                client_id=g_data.get("client_id", ""),
                client_secret=g_data.get("client_secret", ""),
                refresh_token=g_data.get("refresh_token", "")
            )
            
        background_tasks.add_task(
            async_video_render_pipeline,
            job_id,
            script_json,
            voice,
            style,
            upload_target,
            s3_cfg,
            google_creds,
            engine
        )
        
        return {
            "ok": True,
            "job_id": job_id,
            "status": "pending",
            "message": "Video rendering queue started in background."
        }
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported job type: {job_type}")


@app.get("/v1/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Fetches real-time task statuses, progress bars, logs, and token/cost details."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found.")
    return jobs_db[job_id]


@app.get("/v1/jobs/{job_id}/download")
async def download_job_video(job_id: str):
    """Serves compiled MP4 video output for local server configurations."""
    persistent_dir = os.path.join(tempfile.gettempdir(), "tutor_ai_videos")
    file_path = os.path.join(persistent_dir, f"{job_id}.mp4")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Video file not found or still rendering.")
    return FileResponse(file_path, filename=f"tutor_lms_lesson_{job_id}.mp4", media_type="video/mp4")


@app.get("/v1/covers/{job_id}")
async def download_course_cover(job_id: str):
    """Serves generated PNG course cover for local server configurations."""
    persistent_dir = os.path.join(tempfile.gettempdir(), "tutor_ai_videos")
    file_path = os.path.join(persistent_dir, f"{job_id}_cover.png")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Cover image not found.")
    return FileResponse(file_path, media_type="image/png")

